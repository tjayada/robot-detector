# robot-renderer is a git submodule at external/robot-renderer (fetched by
# `make submodules`). Override to point at a local checkout if you have one.
ROBOT_RENDERER_PATH ?= external/robot-renderer
WEIGHTS_DIR        ?= $(HOME)/.cache/cnos
ENV_NAME           ?= cnos

# PyTorch3D wheel: must match Python + CUDA + PyTorch exactly.
# Override on the CLI if your setup differs, e.g.:
#   make install-pt3d PT3D_WHEEL=https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt241/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl
PT3D_WHEEL ?= https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl

# --- Step-by-step setup ---
#
# Recommended flow (matches the README setup steps):
#
#   conda create -n cnos python=3.10 pip -c conda-forge -y
#   conda activate cnos
#   make submodules
#   make install-torch
#   make install-pt3d
#   make install-cnos
#   make install-sam
#   make install-fastsam
#   make install-renderer
#   make weights
#   make check
#
# Or all at once after activating the env:
#   make setup

.PHONY: submodules install-torch install-pt3d install-cnos install-sam install-fastsam \
        install-renderer fix-pt3d-deps weights setup check test lint

submodules:
	git submodule update --init --recursive

install-torch:
	pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
	    --index-url https://download.pytorch.org/whl/cu121

install-pt3d:
	pip install fvcore==0.1.5.post20221221 iopath==0.1.10
	pip install $(PT3D_WHEEL)

install-cnos:
	# setuptools<70 ships the full pkg_resources module needed by
	# pytorch-lightning==1.8.6 and torchmetrics==0.10.3.
	pip install "setuptools<70"
	# Core CNOS dependencies (from external/cnos/environment.yml pip section),
	# plus tqdm and pyyaml which run_detect.py imports directly.
	pip install \
	    "pytorch-lightning==1.8.6" \
	    "torchmetrics==0.10.3" \
	    numpy omegaconf tqdm pyyaml \
	    opencv-python pycocotools matplotlib scipy pandas \
	    hydra-core hydra-colorlog ruamel.yaml wandb distinctipy

install-sam:
	pip install git+https://github.com/facebookresearch/segment-anything.git

install-fastsam:
	pip install "ultralytics==8.0.135"

install-renderer:
	pip install -e $(ROBOT_RENDERER_PATH)

fix-pt3d-deps:
	# Re-pin fvcore + iopath after all other packages are installed,
	# in case a transitive dep (e.g. ultralytics, pytorch-lightning)
	# silently upgraded them to an incompatible version.
	pip install fvcore==0.1.5.post20221221 iopath==0.1.10

# Downloads both segmentor checkpoints into WEIGHTS_DIR (~/.cache/cnos):
#   FastSAM-x.pt          - FastSAM segmentor (detector.segmentor: fastsam)
#   sam_vit_h_4b8939.pth  - SAM ViT-H (detector.segmentor: sam; CNOS-canonical)
# Safe to re-run: each download is skipped if already present. SAM ViT-H is ~2.4 GB.
weights:
	mkdir -p $(WEIGHTS_DIR)
	@if [ -f "$(WEIGHTS_DIR)/FastSAM-x.pt" ]; then \
		echo "FastSAM-x.pt already present - skipping."; \
	else \
		echo "Downloading FastSAM-x.pt..."; \
		pip install gdown; \
		gdown --no-cookies --no-check-certificate -O $(WEIGHTS_DIR)/FastSAM-x.pt 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv; \
	fi
	@if [ -f "$(WEIGHTS_DIR)/sam_vit_h_4b8939.pth" ]; then \
		echo "sam_vit_h_4b8939.pth already present - skipping."; \
	else \
		echo "Downloading SAM ViT-H checkpoint (~2.4 GB)..."; \
		wget -O $(WEIGHTS_DIR)/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth; \
	fi

setup: submodules install-torch install-pt3d install-cnos install-sam install-fastsam install-renderer fix-pt3d-deps weights

check:
	python -c "import torch; print('torch:', torch.__version__)"
	python -c "import pytorch_lightning; print('pytorch_lightning: ok')"
	python -c "import ultralytics; print('ultralytics: ok')"
	python -c "from segment_anything import sam_model_registry; print('segment_anything: ok')"
	python -c "import robot_renderer; print('robot_renderer: ok')"
	python -c "from detector import RobotDetector; print('detector: ok')"
	@echo "All checks passed."

# Unit tests: config consistency, K-scaling, frame selection, checkpoint/resume,
# RLE round-trip, and synthetic-dataset loaders. No weights, GPU, or real dataset
# needed. Deps: numpy, PIL, PyYAML, torch/torchvision, pytest.
test:
	python -m pytest tests/ -v

lint:
	python -m ruff check .
