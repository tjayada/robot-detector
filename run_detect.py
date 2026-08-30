"""
Offline robot arm detection with CNOS (SAM/FastSAM proposals + DINOv2 template matching).

Iterates every frame of a dataset, runs RobotDetector, and writes a single
JSON file with per-frame bboxes and RLE-encoded segmentation masks.

The output JSON is meant to be consumed by a downstream pose estimation
pipeline that needs per-frame detections.

Usage:
    python run_detect.py --config configs/baxter.yaml \
                         --data_folder /path/to/baxter \
                         --output /path/to/detections/baxter.json

    # Override single field without editing the YAML:
    python run_detect.py --config configs/baxter.yaml \
                         --data_folder /path/to/baxter \
                         --output detections/baxter.json \
                         --confidence_threshold 0.3

    # Debug: only run on specific frame indices (0-based):
    python run_detect.py --config configs/craves.yaml \
                         --data_folder /path/to/test_20181024 \
                         --output detections/craves_debug.json \
                         --frames 0 1 5

    # Large dataset: detect only N evenly spaced frames instead of all:
    python run_detect.py --config configs/panda_orb.yaml \
                         --data_folder /path/to/panda-orb \
                         --output detections/panda_orb.json.gz \
                         --subsample 1000

    # Resume after interruption:
    python run_detect.py <same arguments> --resume
"""

import argparse
import gzip
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from detect_utils import (
    scale_K_to_render,
    select_frame_indices,
    validate_frame_indices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args():
    p = argparse.ArgumentParser(description="Run offline CNOS detection on a dataset.")
    p.add_argument("--config",      required=True,  help="Path to dataset YAML config.")
    p.add_argument("--data_folder", required=True,  help="Path to dataset root directory.")
    p.add_argument("--output",      required=True,
                   help="Where to write detections. A '.json.gz' name is written gzipped.")
    p.add_argument("--confidence_threshold", type=float, default=None,
                   help="Override detector.confidence_threshold from config.")
    p.add_argument("--segmentor", choices=["fastsam", "sam"], default=None,
                   help="Override detector.segmentor from config.")
    p.add_argument("--frames", nargs="+", type=int, default=None, metavar="N",
                   help="Only process these frame indices (0-based). Useful for debugging.")
    p.add_argument("--subsample", type=_positive_int, default=None, metavar="N",
                   help="Only process N evenly spaced frames (np.linspace over all "
                        "frames, rounded, duplicates removed). Useful for large "
                        "datasets. Cannot be combined with --frames.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from <output>.partial.jsonl. The saved run settings "
                        "must match the current invocation.")
    p.add_argument("--checkpoint_every", type=_positive_int, default=10, metavar="N",
                   help="Durably sync the append-only checkpoint every N completed "
                        "frames (default: 10).")
    args = p.parse_args()
    if args.frames is not None and args.subsample is not None:
        p.error("--frames and --subsample cannot be combined.")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Load YAML config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]["name"]
    robot_name   = cfg["robot"]["name"]

    # --- Select loader ---
    from loaders import LOADERS
    if dataset_name not in LOADERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {list(LOADERS)}"
        )
    get_camera_info, iter_frames, count_frames = LOADERS[dataset_name]

    # H and W come from the loader, the authoritative source per dataset. All
    # loaders return native resolution (e.g. Baxter 1536x2048, CRAVES 720x1280).
    K_np, img_H, img_W = get_camera_info(args.data_folder)
    n_total = count_frames(args.data_folder)
    if n_total == 0:
        raise ValueError(f"Dataset '{dataset_name}' contains no frames.")

    if args.frames is not None:
        selected_indices = validate_frame_indices(args.frames, n_total)
    else:
        selected_indices = select_frame_indices(n_total, args.subsample)
    selected_set = set(selected_indices)

    # Build and validate the effective detector config before initializing the
    # renderer or loading any model weights.
    from detector import RobotDetector, RobotDetectorConfig

    d_cfg = cfg.get("detector", {})
    defaults = RobotDetectorConfig()
    det_config = RobotDetectorConfig(
        segmentor=args.segmentor or d_cfg.get("segmentor", defaults.segmentor),
        dino_model=d_cfg.get("dino_model", defaults.dino_model),
        confidence_threshold=(
            args.confidence_threshold if args.confidence_threshold is not None
            else d_cfg.get("confidence_threshold", defaults.confidence_threshold)
        ),
        fastsam_checkpoint=d_cfg.get("fastsam_checkpoint"),
        sam_checkpoint=d_cfg.get("sam_checkpoint"),
        fastsam_iou=d_cfg.get("fastsam_iou", defaults.fastsam_iou),
        fastsam_conf=d_cfg.get("fastsam_conf", defaults.fastsam_conf),
        fastsam_max_det=d_cfg.get("fastsam_max_det", defaults.fastsam_max_det),
        fastsam_img_size=d_cfg.get("fastsam_img_size", defaults.fastsam_img_size),
        sam_stability_thresh=d_cfg.get("sam_stability_thresh", defaults.sam_stability_thresh),
        descriptor_img_size=d_cfg.get("descriptor_img_size", defaults.descriptor_img_size),
        top_k_templates=d_cfg.get("top_k_templates", defaults.top_k_templates),
        min_box_size=d_cfg.get("min_box_size", defaults.min_box_size),
        min_mask_size=d_cfg.get("min_mask_size", defaults.min_mask_size),
    )

    # --- Build renderer ---
    import robot_renderer as rr
    from robot_renderer import ViewConfig

    r_cfg = cfg["renderer"]
    num_views = r_cfg["num_views"]
    if not 1 <= det_config.top_k_templates <= num_views:
        raise ValueError(
            "detector.top_k_templates must satisfy "
            f"1 <= top_k_templates <= renderer.num_views ({num_views}), "
            f"got {det_config.top_k_templates}."
        )
    # ViewConfig.fill_frame stays at its default (False): the detector tight-crops
    # templates itself; enabling it would desync template and proposal preprocessing.
    view_cfg = ViewConfig(
        render_size=r_cfg["render_size"],
        viewset=r_cfg.get("viewset", "fibonacci"),
        num_views=r_cfg["num_views"],
        sphere_distance_factor=r_cfg["sphere_distance_factor"],
        elevation_range=tuple(r_cfg["elevation_range"]),
        orientation=r_cfg.get("orientation", "simple_upright"),
        diverse_selection=r_cfg.get("diverse_selection", True),
        anchor_elevation_range=(
            tuple(r_cfg["anchor_elevation_range"])
            if r_cfg.get("anchor_elevation_range") else None
        ),
    )
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale K from the loader's image resolution to the render resolution.
    K_render = scale_K_to_render(K_np, r_cfg["render_size"], img_H, img_W)
    K_tensor = torch.from_numpy(K_render).float().to(device)

    renderer = rr.RobotRenderer(robot_name, K=K_tensor, config=view_cfg, device=device)
    renderer.to(device)
    log.info("Renderer built for robot '%s'.", robot_name)

    detector = RobotDetector(renderer, det_config, device)
    log.info("RobotDetector ready (segmentor=%s, conf_thr=%.2f).",
             det_config.segmentor, det_config.confidence_threshold)

    if args.frames is not None:
        log.info("Frame filter active, processing only indices: %s", selected_indices)
    elif args.subsample is not None:
        log.info("Subsampling: %d of %d frames (evenly spaced).",
                 len(selected_indices), n_total)

    # Each completed frame is appended to a small JSONL journal. This avoids
    # repeatedly rewriting a growing JSON file and lets long jobs resume after
    # preemption or an exception.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path = Path(f"{out_path}.partial.jsonl")
    run_signature = {
        "version": 1,
        "dataset": dataset_name,
        "robot": robot_name,
        "dataset_num_frames": n_total,
        "selection": {
            "frames": selected_indices if args.frames is not None else None,
            "subsample": args.subsample,
        },
        "image_size": [img_H, img_W],
        "camera_K": K_np.tolist(),
        "renderer": r_cfg,
        "detector": asdict(det_config),
    }
    journal, detections = _open_journal(
        journal_path, run_signature, resume=args.resume
    )
    completed_ids = {entry["image_id"] for entry in detections}
    if not completed_ids <= selected_set:
        journal.close()
        raise ValueError(
            f"Checkpoint {journal_path} contains frames outside the current selection."
        )

    remaining = selected_set - completed_ids
    n_found = sum(entry["found"] for entry in detections)
    if detections:
        log.info("Resuming with %d completed frames.", len(detections))

    try:
        if remaining:
            last_selected = selected_indices[-1]
            for image_id, (image, joints, frame_id) in enumerate(
                tqdm(iter_frames(args.data_folder), total=n_total,
                     desc=f"Detecting [{dataset_name}]")
            ):
                if image_id not in remaining:
                    if image_id > last_selected:
                        break
                    continue

                t0     = time.perf_counter()
                result = detector.detect(image, joints)
                ms     = round((time.perf_counter() - t0) * 1000, 1)

                entry = {
                    "image_id": image_id,
                    "frame_id": str(frame_id),
                    "found":    result.found,
                    "score":    round(float(result.score), 4),
                    "time_ms":  ms,
                    "bbox_xyxy":    result.bbox_xyxy.tolist() if result.found else None,
                    "segmentation": _rle(result.mask)         if result.found else None,
                }
                detections.append(entry)
                _append_journal(
                    journal,
                    entry,
                    sync=len(detections) % args.checkpoint_every == 0,
                )
                if result.found:
                    n_found += 1

                remaining.remove(image_id)
                if not remaining:
                    break
        if remaining:
            raise RuntimeError(
                "Dataset loader ended before all selected frames were reached; "
                f"{len(remaining)} frame(s) remain. Checkpoint retained at {journal_path}."
            )
    finally:
        _close_journal(journal)

    rate = n_found / max(len(detections), 1)
    log.info("Done: %d / %d frames detected (%.1f%%).", n_found, len(detections), rate * 100)

    # Write the downstream format atomically, then remove the completed journal.
    output = {
        "dataset":        dataset_name,
        "robot":          robot_name,
        "segmentor":      det_config.segmentor,
        "num_frames":     len(detections),
        "num_detected":   n_found,
        "detection_rate": round(rate, 4),
        "detections":     detections,
    }
    _write_json_atomic(out_path, output)
    journal_path.unlink()
    log.info("Saved detections to %s", out_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_journal(path: Path, signature: dict, resume: bool):
    """Open a new checkpoint journal or validate and resume an existing one."""
    if not resume:
        if path.exists():
            raise FileExistsError(
                f"Checkpoint already exists: {path}. Use --resume to continue it, "
                "or remove it to start over."
            )
        journal = open(path, "x")
        journal.write(json.dumps({"run": signature}, separators=(",", ":")) + "\n")
        journal.flush()
        os.fsync(journal.fileno())
        return journal, []

    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found to resume: {path}")

    detections = []
    with open(path, "rb+") as raw:
        header_line = raw.readline()
        if not header_line:
            raise ValueError(f"Checkpoint is empty: {path}")
        header = json.loads(header_line)
        if header.get("run") != signature:
            raise ValueError(
                f"Checkpoint settings do not match this run: {path}. "
                "Use the original config and frame selection, or start over."
            )

        file_size = os.fstat(raw.fileno()).st_size
        while line := raw.readline():
            line_start = raw.tell() - len(line)
            try:
                detections.append(json.loads(line))
            except json.JSONDecodeError:
                if raw.tell() != file_size:
                    raise ValueError(f"Invalid checkpoint record in {path}") from None
                log.warning("Discarding an incomplete final checkpoint record in %s.", path)
                raw.truncate(line_start)
                break

    image_ids = [entry["image_id"] for entry in detections]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Checkpoint contains duplicate image IDs: {path}")
    return open(path, "a"), detections


def _append_journal(journal, entry: dict, sync: bool) -> None:
    journal.write(json.dumps(entry, separators=(",", ":"), allow_nan=False) + "\n")
    journal.flush()
    if sync:
        os.fsync(journal.fileno())


def _close_journal(journal) -> None:
    if journal.closed:
        return
    journal.flush()
    os.fsync(journal.fileno())
    journal.close()


def _write_json_atomic(path: Path, data: dict) -> None:
    """Replace path only after a complete JSON document is durable.

    A ".gz" suffix gzips and compacts the JSON.
    """
    gz   = path.suffix == ".gz"
    text = (json.dumps(data, separators=(",", ":"), allow_nan=False) if gz
            else json.dumps(data, indent=2, allow_nan=False))
    tmp_path = Path(f"{path}.tmp")
    with open(tmp_path, "wb") as fh:
        fh.write(gzip.compress(text.encode()) if gz else text.encode())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def _rle(mask: np.ndarray) -> dict:
    """Encode a binary mask as COCO column-major RLE using CNOS's own encoder."""
    from detector import _ensure_cnos_path
    _ensure_cnos_path()
    from src.model.utils import mask_to_rle
    return mask_to_rle(mask.astype(np.uint8))


if __name__ == "__main__":
    main()
