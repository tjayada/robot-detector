# Third-party components

robot-detector's own code is MIT-licensed (see `LICENSE`). It builds on the
following third-party components. None of their code is copied into this
repository: they are pulled in as a git submodule, pip dependencies, or
downloaded weights, and remain under their own licenses.

| Component | How it is used | License | Notes |
|---|---|---|---|
| [CNOS](https://github.com/nv-nguyen/cnos) | git submodule at `external/cnos`, pinned to commit `298d1f3366171464ca271659f0e2f7a6eb8e39b4` | MIT | Proposal scoring pipeline (segmentor wrappers, DINOv2 descriptor model, crop utilities). If you re-clone this repo without history, re-pin the submodule to this exact commit. |
| [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) | pip dependency + ViT-H weights (`sam_vit_h_4b8939.pth`) | Apache-2.0 | Default proposal segmentor in the shipped configs (`segmentor: sam`). Weights are Apache-2.0 and downloaded by `make weights`. |
| [DINOv2](https://github.com/facebookresearch/dinov2) | loaded via `torch.hub` (`dinov2_vitl14`) at runtime | Apache-2.0 | Template/proposal descriptor backbone. Weights are Apache-2.0. |
| [ultralytics / FastSAM](https://github.com/ultralytics/ultralytics) | optional pip dependency (`ultralytics==8.0.135`) + `FastSAM-x.pt` weights | **AGPL-3.0** | Only used when `segmentor: fastsam` is selected; the shipped configs default to SAM, so this dependency is strictly optional. No ultralytics code is copied into this repo. FastSAM weights are distributed via CNOS's Google Drive link (`make weights`). |
| [robot-renderer](https://github.com/tjayada/robot-renderer) | git submodule at `external/robot-renderer`, installed editable (`pip install -e external/robot-renderer`) | MIT | Template rendering. Robot mesh/URDF assets inside it carry their own upstream licenses; see its `MESH_LICENSES/`. If you re-clone this repo without history, re-pin the submodule to the released commit. |
| PyTorch3D | pinned wheel (see `Makefile`) | BSD-3-Clause | Rendering backend required by robot-renderer. |

The detection JSONs shipped with this repository are derived outputs of the
pipeline (model *outputs*, not copies of any of the above components) and are
covered by this repository's MIT license.
