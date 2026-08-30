"""
RobotDetector: CNOS-based zero-shot robot arm detection.

Produces a 2D bounding box and segmentation mask for the robot arm in a query
image using only rendered templates from robot-renderer as references.

Pipeline:
  1. Template features (consecutive identical joint states reuse the last result)
       renderer.render_templates(joint_angles) -> RGB tensors
       -> CropResizePad(224) -> DINOv2 CLS-token features

  2. Per-frame detection
       FastSAM or SAM (shipped configs use SAM) -> proposals -> DINOv2 features
       -> cosine sim vs templates -> top-scoring proposal -> bbox + mask
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

log = logging.getLogger(__name__)

_CNOS_ROOT: Optional[Path] = None


def _ensure_cnos_path() -> None:
    global _CNOS_ROOT
    if _CNOS_ROOT is not None:
        return

    here = Path(__file__).resolve().parent   # robot-detector/
    cnos = here / "external" / "cnos"
    # NOTE: a clone without --recurse-submodules leaves external/cnos as an
    # EMPTY directory, so checking is_dir() alone is not enough; verify the
    # CNOS package root actually exists.
    if not (cnos / "src" / "utils").is_dir():
        raise RuntimeError(
            f"CNOS submodule missing or not initialised at {cnos} "
            f"(exists={cnos.is_dir()}, but 'src/utils' not found). "
            "Run: git submodule update --init external/cnos"
        )
    if str(cnos) not in sys.path:
        sys.path.insert(0, str(cnos))
    _CNOS_ROOT = cnos


@dataclass
class RobotDetectorConfig:
    segmentor:            str   = "sam"   # "sam" or "fastsam" (release default: SAM)
    fastsam_checkpoint:   Optional[str] = None
    sam_checkpoint:       Optional[str] = None
    dino_model:           str   = "dinov2_vitl14"
    # FastSAM proposal generation: canonical CNOS BOP values (configs/model/
    # segmentor_model/fast_sam.yaml + cnos_fast.yaml segmentor_width_size).
    fastsam_iou:          float = 0.9     # CNOS iou_threshold
    fastsam_conf:         float = 0.05    # CNOS conf_threshold
    fastsam_max_det:      int   = 200     # CNOS max_det
    fastsam_img_size:     int   = 640     # CNOS segmentor_width_size
    # SAM proposal stability filter (SAM only). CNOS's BOP benchmark value is
    # 0.97 (cnos sam.yaml), but the CNOS README recommends lowering it to 0.5
    # for custom (non-BOP) objects; the shipped configs set 0.5 for that reason.
    sam_stability_thresh: float = 0.97
    descriptor_img_size:  int   = 224     # CNOS dinov2.yaml image_size
    top_k_templates:      int   = 5       # CNOS aggregation_function = avg_5
    confidence_threshold: float = 0.15    # CNOS matching_config.confidence_thresh
    # Canonical CNOS proposal size filters (relative to image dimensions/area).
    # Removes tiny noise proposals before scoring. CNOS defaults: 0.05 / 3e-4.
    min_box_size:         float = 0.05
    min_mask_size:        float = 3e-4


@dataclass
class DetectionResult:
    bbox_xyxy: Optional[np.ndarray]   # (4,) float32 xyxy, or None
    mask:      Optional[np.ndarray]   # (H, W) bool, or None
    score:     float = 0.0

    @property
    def found(self) -> bool:
        return self.bbox_xyxy is not None


class RobotDetector:
    def __init__(self, renderer, config: Optional[RobotDetectorConfig] = None,
                 device: Optional[torch.device] = None) -> None:
        _ensure_cnos_path()
        self.renderer = renderer
        self.config   = config or RobotDetectorConfig()
        if self.config.segmentor not in ("sam", "fastsam"):
            raise ValueError(
                f"Unknown segmentor '{self.config.segmentor}': "
                "expected 'sam' or 'fastsam'. Check the detector.segmentor "
                "field in your config."
            )
        self.device   = device or torch.device("cpu")
        self._segmentor   = None
        self._dino        = None
        self._crop_resize = None
        self._normalise   = None
        # Consecutive images can share a joint configuration (notably Baxter,
        # where several camera views are stored per pose). Keep only that last
        # result: continuous trajectories do not benefit from an unbounded cache.
        self._last_joint_key: Optional[bytes] = None
        self._last_template_feats: Optional[torch.Tensor] = None

    def _ensure_models(self) -> None:
        if self._segmentor is not None:
            return
        _ensure_cnos_path()
        from src.utils.bbox_utils import CropResizePad
        from src.model.dinov2 import CustomDINOv2

        log.info("RobotDetector: loading DINOv2 %s ...", self.config.dino_model)
        dino_raw = torch.hub.load(
            "facebookresearch/dinov2", self.config.dino_model, verbose=False
        ).eval().to(self.device)
        self._dino = CustomDINOv2(
            model_name=self.config.dino_model, model=dino_raw,
            token_name="x_norm_clstoken", image_size=self.config.descriptor_img_size,
            chunk_size=16, descriptor_width_size=self.config.descriptor_img_size,
        ).to(self.device)
        self._crop_resize = CropResizePad(self.config.descriptor_img_size)
        self._normalise   = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        if self.config.segmentor == "fastsam":
            self._segmentor = self._load_fastsam()
        else:   # "sam"; anything else was rejected in __init__
            self._segmentor = self._load_sam()
        log.info("RobotDetector: models ready.")

    def _weight_path(self, filename: str) -> Path:
        p = Path.home() / ".cache" / "cnos" / filename
        if not p.exists():
            raise FileNotFoundError(
                f"Weight not found: {p}\n"
                f"Run: make weights   (see Makefile)"
            )
        return p

    def _load_fastsam(self):
        _ensure_cnos_path()
        from src.model.fast_sam import FastSAM
        from omegaconf import OmegaConf
        ckpt = Path(self.config.fastsam_checkpoint) if self.config.fastsam_checkpoint \
               else self._weight_path("FastSAM-x.pt")
        cfg = OmegaConf.create({
            "iou_threshold": self.config.fastsam_iou,
            "conf_threshold": self.config.fastsam_conf,
            "max_det": self.config.fastsam_max_det,
        })
        log.info("RobotDetector: loading FastSAM from %s", ckpt)
        seg = FastSAM(checkpoint_path=str(ckpt), config=cfg,
                      segmentor_width_size=self.config.fastsam_img_size, device=self.device)
        seg.model.setup_model(device=self.device, verbose=False)
        # CNOS's CustomYOLO hardcodes conf=0.25 on line 39, silently overriding the
        # config.conf_threshold we pass. Restore our configured value via the predictor.
        seg.model.predictor.args.conf = self.config.fastsam_conf
        return seg

    def _load_sam(self):
        _ensure_cnos_path()
        from segment_anything import sam_model_registry
        from src.model.sam import CustomSamAutomaticMaskGenerator
        ckpt = Path(self.config.sam_checkpoint) if self.config.sam_checkpoint \
               else self._weight_path("sam_vit_h_4b8939.pth")
        log.info("RobotDetector: loading SAM ViT-H from %s", ckpt)
        # Build the SAM registry directly (the same call CNOS makes internally)
        # and move it to the device; CNOS's own loader does neither for us.
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
        sam.to(device=self.device)
        return CustomSamAutomaticMaskGenerator(
            sam,
            stability_score_thresh=self.config.sam_stability_thresh,
            # Resize before SAM (CNOS's segmentor_width_size, 640): full-resolution
            # SAM is slow and floods the scorer with proposals.
            segmentor_width_size=self.config.fastsam_img_size,
        )

    @staticmethod
    def _template_fg_boxes(zbuf: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Compute tight xyxy bounding boxes around the foreground arm pixels.
        zbuf: (N, H, W), background = -1, arm pixels = positive metric camera-Z.
        Returns (N, 4) long tensor with exclusive x2/y2 bounds, matching
        PIL.Image.getbbox() as used by CNOS's model-based template loader.
        """
        N = zbuf.shape[0]
        fg = zbuf > 0           # (N, H, W) bool
        boxes = zbuf.new_zeros(N, 4, dtype=torch.long)
        for i in range(N):
            ys, xs = torch.where(fg[i])
            if xs.numel() == 0:
                boxes[i] = torch.tensor([0, 0, W, H], device=zbuf.device)
            else:
                boxes[i] = torch.stack([
                    xs.min(), ys.min(), xs.max() + 1, ys.max() + 1,
                ])
        return boxes

    def _get_template_features(self, q: np.ndarray) -> torch.Tensor:
        joint_key = q.tobytes()
        if joint_key == self._last_joint_key and self._last_template_feats is not None:
            return self._last_template_feats

        log.debug("RobotDetector: rendering templates ...")
        result = self.renderer.render_templates(q)
        if result.zbuf is None:
            raise RuntimeError(
                "renderer.render_templates returned no zbuf; the detector needs "
                "the per-view depth buffer to mask and tight-crop templates."
            )
        imgs   = torch.stack([v.image for v in result.views]).to(self.device)
        N, _, H, W = imgs.shape

        # Mirror CustomDINOv2.process_rgb_proposals: normalize, then mask, then
        # crop, so template backgrounds are zero in normalized space, the same as
        # how query proposals are processed.
        zbuf = result.zbuf.to(self.device)                 # (N, H, W)
        fg   = (zbuf > 0).unsqueeze(1).float()             # (N, 1, H, W)
        normed_masked = self._normalise(imgs) * fg
        boxes = self._template_fg_boxes(zbuf, H, W)        # tight crop per view
        # Clipping guard: matching is scale-invariant (both sides tight-cropped
        # to 224), so sphere_distance_factor only has to keep the arm in frame.
        touching = ((boxes[:, 0] == 0) | (boxes[:, 1] == 0)
                    | (boxes[:, 2] >= W) | (boxes[:, 3] >= H))
        if bool(touching.any()):
            log.warning(
                "RobotDetector: %d/%d template views have foreground touching the "
                "render border. The arm may be clipped; increase "
                "renderer.sphere_distance_factor.", int(touching.sum()), N)
        resized = self._crop_resize(normed_masked, boxes)

        with torch.no_grad():
            feats = self._dino.compute_features(resized, token_name="x_norm_clstoken")

        self._last_joint_key = joint_key
        self._last_template_feats = feats
        log.debug("RobotDetector: %d templates, dim=%d.", N, feats.shape[1])
        return feats

    def detect(self, img: torch.Tensor, joint_angles) -> DetectionResult:
        _t = time.perf_counter()
        self._ensure_models()
        q = np.asarray(joint_angles, dtype=np.float64)

        if img.ndim == 4:
            img = img[0]
        if img.is_floating_point():
            img_uint8 = (img.clamp(0.0, 1.0) * 255).to(torch.uint8)
        else:
            img_uint8 = img.to(torch.uint8)
        img_np = img_uint8.permute(1, 2, 0).cpu().numpy()

        ref_feats     = self._get_template_features(q)
        proposals_raw = self._segmentor.generate_masks(img_np)

        if not proposals_raw or len(proposals_raw.get("masks", [])) == 0:
            return DetectionResult(bbox_xyxy=None, mask=None, score=0.0)

        _ensure_cnos_path()
        from src.model.utils import Detections
        proposals = Detections(proposals_raw)

        # Filter tiny / noisy proposals, canonical CNOS post-processing.
        class _SizeFilter:
            min_box_size  = self.config.min_box_size
            min_mask_size = self.config.min_mask_size
        proposals.remove_very_small_detections(_SizeFilter)

        if len(proposals) == 0:
            return DetectionResult(bbox_xyxy=None, mask=None, score=0.0)

        with torch.no_grad():
            query_feats = self._dino.forward(img_np, proposals)

        sims = F.cosine_similarity(
            query_feats.unsqueeze(1), ref_feats.unsqueeze(0), dim=2
        )
        k = min(self.config.top_k_templates, sims.shape[1])
        score_per = torch.topk(sims, k=k, dim=1)[0].mean(dim=1)
        best_score, best_idx = score_per.max(dim=0)
        best_score = float(best_score.item())

        log.debug("RobotDetector: score=%.3f threshold=%.2f proposals=%d elapsed=%.0f ms",
                  best_score, self.config.confidence_threshold, len(proposals),
                  (time.perf_counter() - _t) * 1000)

        if best_score < self.config.confidence_threshold:
            return DetectionResult(bbox_xyxy=None, mask=None, score=best_score)

        return DetectionResult(
            bbox_xyxy=proposals.boxes[best_idx].cpu().numpy().astype(np.float32),
            mask=proposals.masks[best_idx].cpu().numpy().astype(bool),
            score=best_score,
        )
