"""
Visualize detection results from run_detect.py output JSON.

Overlays the segmentation mask and bounding box on the original images and
saves PNGs to a visualizations/ directory. Pass --side-by-side to also show
the unmodified original next to the overlay.

Usage:
    python visualize.py \
        --detections baxter.json \
        --data_folder /path/to/baxter-real-dataset \
        --num_samples 20 \
        --output_dir visualizations/baxter

    # Show ALL frames (slow for large datasets):
    python visualize.py --detections baxter.json --data_folder /path/to/baxter --all

    # Show specific frame indices (0-based image_id):
    python visualize.py --detections craves.json --data_folder /path/to/test_20181024 \
        --frames 0 1 5 10

    # README-sized figure:
    python visualize.py --detections craves.json --data_folder /path/to/test_20181024 \
        --frames 0 --max_width 720
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from detect_utils import load_detections


# ---------------------------------------------------------------------------
# RLE decode (inverse of CNOS's mask_to_rle)
# ---------------------------------------------------------------------------

def rle_to_mask(rle: dict) -> np.ndarray:
    H, W   = rle["size"]
    flat   = np.zeros(H * W, dtype=bool)
    idx, val = 0, 0
    for count in rle["counts"]:
        flat[idx : idx + count] = bool(val)
        idx += count
        val  = 1 - val
    return flat.reshape(H, W, order="F")


# ---------------------------------------------------------------------------
# Image finding helpers (one per dataset type)
# ---------------------------------------------------------------------------

def find_image(data_folder: Path, dataset_name: str, frame_id: str) -> Path:
    """Return the RGB image path for a given frame_id."""
    if dataset_name == "baxter_real":
        # frame_id = "pose_3_0042": pose_key + image index within pose
        parts    = frame_id.rsplit("_", 1)
        pose_key = parts[0]                   # "pose_3"
        img_idx  = int(parts[1])
        pose_dir = data_folder / pose_key
        imgs     = sorted(pose_dir.glob("*.png"))
        return imgs[img_idx]

    if dataset_name == "panda_orb":
        # frame_id = "000042": digit-prefixed stem
        for ext in (".rgb.jpg", ".rgb.png"):
            p = data_folder / f"{frame_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No RGB image for frame {frame_id} in {data_folder}")

    if dataset_name == "craves":
        # frame_id = "00000037"
        return data_folder / "FusionCameraActor3_2" / "lit" / f"{frame_id}.jpg"

    if dataset_name == "hydra":
        # frame_id = "meas0_000": measurement index + zero-padded frame index.
        # data_folder is the robot root containing measurement_N/images/NNN.png.
        meas_part, frame_part = frame_id.rsplit("_", 1)   # "meas0", "000"
        meas_idx = int(meas_part[len("meas"):])           # 0
        return data_folder / f"measurement_{meas_idx}" / "images" / f"{frame_part}.png"

    if dataset_name == "realsense_franka":
        # frame_id = "1_D415_front_0/000042": view subdir + digit-prefixed stem.
        return data_folder / f"{frame_id}.png"

    raise ValueError(f"Unknown dataset '{dataset_name}'")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_detection(img: Image.Image, det: dict, side_by_side: bool = False,
                     max_width: int | None = None) -> Image.Image:
    """Return the mask+bbox overlay image (or original | overlay if side_by_side)."""
    # Resize image to detection resolution if the mask size differs
    # (e.g. when the detections JSON was generated at a lower resolution).
    if det["found"] and det["segmentation"] is not None:
        mH, mW = det["segmentation"]["size"]
        if (img.height, img.width) != (mH, mW):
            img = img.resize((mW, mH), Image.Resampling.BILINEAR)

    has_mask = det["found"] and det["segmentation"] is not None
    has_bbox = det["found"] and det["bbox_xyxy"] is not None
    mask = rle_to_mask(det["segmentation"]) if has_mask else None   # (H, W) bool
    bbox = list(det["bbox_xyxy"]) if has_bbox else None

    # Before drawing, not after: an outline drawn at native size then shrunk
    # goes sub-pixel.
    if max_width is not None and img.width > max_width:
        scale    = max_width / img.width
        new_size = (max_width, max(1, round(img.height * scale)))
        img      = img.resize(new_size, Image.Resampling.BILINEAR)
        if mask is not None:
            small = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                new_size, Image.Resampling.NEAREST)
            mask = np.array(small) > 127
        if bbox is not None:
            bbox = [v * scale for v in bbox]

    W, H    = img.size
    overlay = img.copy().convert("RGBA")
    draw    = ImageDraw.Draw(overlay)

    if mask is not None:
        # Semi-transparent green fill over masked pixels.
        tint = Image.new("RGBA", (W, H), (0, 200, 80, 0))
        tint_arr = np.array(tint)
        tint_arr[mask, 3] = 120   # alpha only where mask=True
        overlay = Image.alpha_composite(overlay, Image.fromarray(tint_arr))
        draw    = ImageDraw.Draw(overlay)

    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Proportional, so --max_width keeps it visible.
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 0), width=max(2, round(W / 500)))

    score_text = f"score={det['score']:.3f}" if det["found"] else "not found"
    draw.text((6, 6), score_text, fill=(255, 255, 80))

    if not side_by_side:
        return overlay.convert("RGB")

    canvas = Image.new("RGB", (W * 2 + 4, H), (40, 40, 40))
    canvas.paste(img, (0, 0))
    canvas.paste(overlay.convert("RGB"), (W + 4, 0))
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections",   required=True, help="Path to detections JSON.")
    p.add_argument("--data_folder",  required=True, help="Dataset root directory.")
    p.add_argument("--output_dir",   default=None,  help="Where to save PNGs (default: visualizations/<stem>).")
    p.add_argument("--frames",        nargs="+", type=int, default=None, metavar="N",
                                      help="Specific frame indices (image_id) to visualise. "
                                           "Overrides --num_samples and --all.")
    p.add_argument("--num_samples",  type=int, default=20, help="Number of frames to visualise (random).")
    p.add_argument("--all",          action="store_true",  help="Visualise every frame (ignores --num_samples).")
    p.add_argument("--side-by-side", dest="side_by_side", action="store_true",
                                     help="Show the unmodified original next to the overlay.")
    p.add_argument("--max_width",    type=int, default=None, metavar="N",
                                     help="Downscale to at most N pixels wide before drawing.")
    p.add_argument("--seed",         type=int, default=0)
    args = p.parse_args()

    if args.max_width is not None and args.max_width < 1:
        p.error("--max_width must be at least 1")

    det_path     = Path(args.detections)
    det_name     = det_path.name.removesuffix(".gz").removesuffix(".json")
    output_dir   = Path(args.output_dir) if args.output_dir else Path("visualizations") / det_name
    data_folder  = Path(args.data_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_detections(det_path)

    dataset_name = data["dataset"]
    detections   = data["detections"]
    n_found      = data["num_detected"]
    n_total      = data["num_frames"]
    print(f"Dataset : {dataset_name}")
    print(f"Detected: {n_found} / {n_total} ({data['detection_rate']*100:.1f}%)")

    if args.frames is not None:
        id_set  = set(args.frames)
        samples = [d for d in detections if d["image_id"] in id_set]
        missing = id_set - {d["image_id"] for d in samples}
        if missing:
            print(f"[warn] Frame indices not found in detections JSON: {sorted(missing)}")
    elif args.all:
        samples = detections
    else:
        random.seed(args.seed)
        samples = random.sample(detections, min(args.num_samples, len(detections)))

    saved = 0
    for det in samples:
        try:
            img_path = find_image(data_folder, dataset_name, det["frame_id"])
            img      = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, IndexError) as e:
            print(f"  [warn] {det['frame_id']}: {e}")
            continue

        vis      = render_detection(img, det, side_by_side=args.side_by_side,
                                    max_width=args.max_width)
        # frame_id may contain "/" (realsense_franka view prefix); flatten for the filename.
        safe_id  = det["frame_id"].replace("/", "_")
        out_path = output_dir / f"{det['image_id']:05d}_{safe_id}.png"
        vis.save(out_path)
        saved += 1

    print(f"Saved {saved} visualizations to {output_dir}/")


if __name__ == "__main__":
    main()
