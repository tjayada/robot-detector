"""Mask IoU of a detection set against the GT-posed arm silhouette.

Renders the arm at its ground-truth camera pose (T_cam_base) and joint state
with the robot-renderer's PyTorch3D rasterizer, turns the rendered fragments
into a binary silhouette, and reports the mean IoU with the detector's mask.
Lets the SAM-vs-FastSAM choice rest on a number instead of eyeballing.

Hydra only: it is the benchmark whose GT camera pose is available per
measurement (T_cam_base.npy, from hydra-data-calibration). The silhouette is a
true occlusion-correct render (pix_to_face >= 0), not a convex-hull proxy, so
the IoU is an honest segmentation-quality figure, not only a relative one.

Needs the robot-renderer submodule and its PyTorch3D stack (same environment as
run_detect.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# tools/ is a subdirectory; make detect_utils/visualize importable from the
# repo root. robot-renderer comes from the editable install (make install-renderer).
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from robot_renderer.config import ViewConfig  # noqa: E402
from robot_renderer.robot_renderer import RobotRenderer  # noqa: E402

from detect_utils import load_detections  # noqa: E402
from visualize import rle_to_mask  # noqa: E402

# cam-from-base (4,4) -> PyTorch3D (R, T), mirrors robot-renderer's own render path.
_C3 = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


def _pose_to_R_T(T_cam_base: np.ndarray):
    """(4,4) cam-from-base -> PyTorch3D (R, T) camera pair."""
    R = (T_cam_base[:3, :3].T @ _C3).astype(np.float32)
    T = (_C3 @ T_cam_base[:3, 3]).astype(np.float32)
    return R, T


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _measurement_dirs(data_folder: Path):
    return sorted(
        (p for p in data_folder.iterdir()
         if p.is_dir() and p.name.startswith("measurement_")),
        key=lambda p: int(p.name.split("_")[1]),
    )


def _detections_by_frame(path: str) -> dict:
    """frame_id -> decoded bool mask, found=true entries only."""
    data = load_detections(path)
    out = {}
    for det in data["detections"]:
        if det.get("found") and det.get("segmentation") is not None:
            out[str(det["frame_id"])] = rle_to_mask(det["segmentation"]).astype(bool)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True, help="Detections .json or .json.gz")
    ap.add_argument("--data_folder", required=True, help="Hydra robot root (measurement_N/)")
    ap.add_argument("--robot", default=None,
                    help="robot-renderer registry name; default: 'robot' field of the file")
    args = ap.parse_args()

    data_folder = Path(args.data_folder).expanduser()
    meas_dirs = _measurement_dirs(data_folder)
    if not meas_dirs:
        sys.exit(f"no measurement_N/ under {data_folder}")

    robot = args.robot or load_detections(args.detections)["robot"]
    dets = _detections_by_frame(args.detections)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = None
    built_for = None  # (K bytes, H, W) the cameras were last built for

    ious: list[float] = []
    n_no_det = 0
    for meas_dir in meas_dirs:
        m = int(meas_dir.name.split("_")[1])
        K = np.load(meas_dir / "camera_K.npy").astype(np.float32)
        T_cam_base = np.load(meas_dir / "T_cam_base.npy").astype(np.float32)
        with open(meas_dir / "ground_truth.json") as fh:
            gt = json.load(fh)

        for frame_idx in sorted(int(k) for k in gt):
            fid = f"meas{m}_{frame_idx:03d}"
            mask_det = dets.get(fid)
            if mask_det is None:
                n_no_det += 1
                continue
            H, W = mask_det.shape

            if renderer is None:
                renderer = RobotRenderer(
                    name=robot, K=K,
                    config=ViewConfig(render_size=max(H, W)), device=device).to(device)
            pt3d = renderer._renderer
            # Hydra pools measurements with different intrinsics; rebuild on change.
            key = (K.tobytes(), H, W)
            if key != built_for:
                pt3d._build_pt3d_renderers(K, (H, W))
                built_for = key

            joints = np.array(gt[str(frame_idx)]["joints"], dtype=np.float32)
            meshes = pt3d.transform_mesh2robot_config(joints)
            R_np, T_np = _pose_to_R_T(T_cam_base)
            R = torch.from_numpy(R_np).unsqueeze(0).to(device)
            T = torch.from_numpy(T_np).unsqueeze(0).to(device)
            _, fragments = pt3d.phong_render_batched(meshes, R, T)
            sil = (fragments.pix_to_face[0, ..., 0] >= 0).cpu().numpy()
            ious.append(_iou(sil, mask_det))

    if not ious:
        sys.exit("no frames matched between detections and dataset")

    arr = np.asarray(ious)
    print(f"{Path(args.detections).name}  robot={robot}")
    print(f"  frames scored: {len(arr)}  (missing detection: {n_no_det})")
    print(f"  mean IoU:   {arr.mean():.4f}")
    print(f"  median IoU: {np.median(arr):.4f}")


if __name__ == "__main__":
    main()
