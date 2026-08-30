#!/bin/bash
# Regenerate every shipped detection JSON (all datasets x both segmentors).
#
# This script is the provenance record for the JSONs under detections/:
# running it against the same dataset copies reproduces them.
#
# Usage: edit the paths below for your machine, then:
#   conda activate cnos
#   ./run_all_detections.sh
#
# Budget ~5.5 days, nearly all of it panda-orb (~32k frames). If a run is
# interrupted, re-issue the single dataset command with --resume (see README.md)
# instead of restarting this script.
#
# Tip: keep your edited version as run_all_detections.local.sh. That name is
# gitignored, so your machine-specific paths never end up in a commit.
set -e

# Path to this repo. The script changes into it first, so it can be started
# from anywhere; output JSONs land in <repo>/detections/.
PROJECT_ROOT=/path/to/robot-detector

# Python with all dependencies installed (see README.md). Plain "python" works
# when the conda env is activated; otherwise paste the full interpreter path.
PYTHON=python

# SAM / DINOv2 weights cache (several GB). Uncomment to move it, e.g. to a local scratch.
#export XDG_CACHE_HOME=/local/scratch/.cache

# One line per dataset:  output-filename  config  dataset-folder  [extra run_detect.py args]
# All shipped gzipped (.json.gz) for a uniform format; panda-orb is the full
# ~32k-frame split.
DATASETS=(
    "baxter.json.gz     configs/baxter.yaml     /path/to/baxter-real-dataset"
    "craves.json.gz     configs/craves.yaml     /path/to/test_20181024"
    "lbr_med7.json.gz   configs/lbr_med7.yaml   /path/to/hydra_eval/lbr"
    "xarm.json.gz       configs/xarm7.yaml      /path/to/hydra_eval/xarm"
    "meca.json.gz       configs/meca500.yaml    /path/to/hydra_eval/meca"
    "panda_orb.json.gz  configs/panda_orb.yaml  /path/to/panda-orb"
    "realsense_franka.json.gz  configs/realsense_franka.yaml  /path/to/RealSense-Franka"
)

cd "$PROJECT_ROOT"

for segmentor in sam fastsam; do
    for entry in "${DATASETS[@]}"; do
        read -r name config data extra <<< "$entry"
        out="detections/$segmentor/$name"
        echo "=== $name ($segmentor) -> $out"
        # $extra is intentionally unquoted so multi-word extras split into args.
        "$PYTHON" run_detect.py \
            --config "$config" \
            --data_folder "$data" \
            --output "$out" \
            --segmentor "$segmentor" \
            $extra
    done
done

echo "All detection JSONs regenerated under detections/."
