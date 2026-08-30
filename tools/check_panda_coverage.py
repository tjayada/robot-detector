"""Check that a panda-orb detections file covers exactly the frames the run
targeted: the whole split, or the N-frame evenly spaced selection of
`run_detect.py --subsample N`.

Cheap by design: lists the digit-prefixed annotation JSONs in the dataset
folder (the same files the loader iterates, in the same sorted order), computes
the expected index selection, and compares frame_id sets. No images are loaded
and no torch is needed.

Usage (after the panda-orb detection run):
    python tools/check_panda_coverage.py \
        --detections detections/sam/panda_orb.json.gz \
        --data_folder /path/to/panda-orb
"""

import argparse
import sys
from pathlib import Path

# tools/ is a subdirectory; make detect_utils importable from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detect_utils import load_detections, select_frame_indices  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="Verify a panda-orb detections file covers exactly the "
                    "expected frame selection.")
    p.add_argument("--detections",  required=True, help="Path to detections JSON (.json or .json.gz).")
    p.add_argument("--data_folder", required=True, help="panda-orb dataset root.")
    p.add_argument("--num_frames",  type=int, default=None,
                   help="The --subsample value the run used. Omit for a full-split run.")
    args = p.parse_args()

    # Same listing the DREAM loader uses: sorted digit-prefixed .json stems.
    stems = sorted(
        p.stem for p in Path(args.data_folder).iterdir()
        if p.suffix == ".json" and p.stem[0].isdigit()
    )
    if not stems:
        sys.exit(f"No digit-prefixed .json files found in {args.data_folder}")

    indices  = select_frame_indices(len(stems), args.num_frames)
    expected = {stems[i] for i in indices}

    data = load_detections(args.detections)
    got  = {d["frame_id"] for d in data["detections"]}

    print(f"Dataset frames:     {len(stems)}")
    print(f"Expected selection: {len(expected)} frame_ids")
    print(f"In detections:      {len(got)} frame_ids")

    missing = expected - got
    extra   = got - expected
    if missing:
        print(f"MISSING from detections JSON ({len(missing)}): "
              f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")
    if extra:
        print(f"UNEXPECTED in detections JSON ({len(extra)}): "
              f"{sorted(extra)[:10]}{' ...' if len(extra) > 10 else ''}")
    if missing or extra:
        sys.exit("FAIL: coverage mismatch.")
    print("OK: detections JSON covers exactly the expected frame selection.")


if __name__ == "__main__":
    main()
