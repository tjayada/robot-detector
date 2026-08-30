#!/bin/bash
# Mask IoU of each detection set vs the GT-posed arm silhouette, for both
# segmentors across the Hydra datasets. Prints the mean IoU per (segmentor,
# robot) - the appendix number backing the SAM-over-FastSAM choice.
#
# Hydra only: it is the benchmark whose GT camera pose ships per measurement
# (T_cam_base.npy from hydra-data-calibration). Needs the robot-renderer stack
# (same environment as run_detect.py).
#
# Usage: edit the paths below for your machine, then:
#   conda activate cnos
#   ./tools/segmentor_iou.sh
#
# Tip: keep your edited version as segmentor_iou.local.sh (gitignored).
set -e

PROJECT_ROOT=/path/to/robot-detector
PYTHON=python

# One line per dataset:  detections-filename   dataset-folder
DATASETS=(
    "lbr_med7.json.gz    /path/to/hydra_eval/lbr"
    "xarm.json.gz        /path/to/hydra_eval/xarm"
    "meca.json.gz        /path/to/hydra_eval/meca"
)

cd "$PROJECT_ROOT"

for segmentor in sam fastsam; do
    for entry in "${DATASETS[@]}"; do
        read -r name data <<< "$entry"
        "$PYTHON" tools/segmentor_iou.py \
            --detections "detections/$segmentor/$name" \
            --data_folder "$data"
    done
done
