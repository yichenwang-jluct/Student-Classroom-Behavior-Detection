# GBH-YOLO — Dense Student Behaviour Detection in University Classrooms

Code accompanying the manuscript *"Edge-Deployable Multi-Scale Visual Sensing for
Dense Student Behaviour Detection in University Classrooms"* (Yichen Wang,
Shuangyuan Li, Ji Ge, Yuli He).

GBH-YOLO extends YOLOv8s with a GFPN neck (queen fusion + CSPStage), bi-level
routing attention at P4/P5, a fourth 160×160 detection head, depthwise-separable
detection heads at all four scales, and a dynamic adaptive Focaler-CIoU (DF-CIoU)
regression loss that is active only during training.

## Contents

| Path | Contents |
|---|---|
| `ultralytics/nn/Addmodules/RepGFPN.py` | `CSPStage` — GFPN neck block |
| `ultralytics/nn/Addmodules/Biformer.py` | `BiLevelRoutingAttention` — BRA |
| `ultralytics/nn/Addmodules/LightHead.py` | `Detect_Light` — depthwise-separable decoupled head |
| `ultralytics/nn/Addmodules/__init__.py` | exports the three modules above |
| `ultralytics/nn/tasks.py` | Ultralytics 8.2.18 model parser with the three modules registered |
| `ultralytics/utils/loss.py` | `BboxLoss` with the DF-CIoU branch |
| `ultralytics/utils/df_ciou.py` | `DFState` — DF-CIoU switch, d0/u0 and training progress |
| `ultralytics/cfg/models/Add/GBH-YOLO.yaml` | full model |
| `ultralytics/cfg/models/Add/yolov8s.yaml` | baseline used in every comparison |
| `ultralytics/cfg/models/Add/ablation/` | stock-head and CSPStage-depth variants (Table 3 †, Table 6) |
| `ultralytics/cfg/datasets/UCB.yaml` | dataset config (images not distributed) |
| `scripts/` | training, evaluation, inference, parameter and FLOP counting |
| `splits/` | exact train / val / test file lists (filenames only) |
| `weights/` | trained GBH-YOLO checkpoint (seed 0) |

## Installation

The files are drop-in replacements for a stock Ultralytics 8.2.18 tree.

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/ultralytics/ultralytics.git -b v8.2.18 ultralytics-8.2.18
# copy this repository's ultralytics/ folder into the package folder of the clone
cp -r Student-Classroom-Behavior-Detection/ultralytics/* ultralytics-8.2.18/ultralytics/
pip install -e ultralytics-8.2.18
pip install einops thop==0.1.1        # einops is required by Biformer.py
```

Run the scripts from the root of this repository.

Environment used for every experiment in the paper:

| | |
|---|---|
| OS | Windows 11 |
| CPU | Intel Core i7-13650HX |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB |
| RAM | 16 GB DDR5-4800 |
| Python | 3.10.16 |
| PyTorch | 1.12.1 + CUDA 11.6 |
| Ultralytics | 8.2.18 |
| thop (FLOPs) | 0.1.1 |

## Usage

```bash
# full model (DF-CIoU is switched on automatically for GBH-YOLO configs)
python scripts/train.py --model ultralytics/cfg/models/Add/GBH-YOLO.yaml --seed 0

# baseline (standard CIoU)
python scripts/train.py --model ultralytics/cfg/models/Add/yolov8s.yaml --seed 0

# evaluation: six-class and five-class mAP, precision, recall
python scripts/val.py --weights runs/train/gbh_seed0/weights/best.pt --split val
python scripts/val.py --weights runs/train/gbh_seed0/weights/best.pt --split test

# parameters / FLOPs, before and after Ultralytics' fuse()
python scripts/count_params_flops.py --model ultralytics/cfg/models/Add/GBH-YOLO.yaml
```

The paper reports means over seeds 0, 42 and 2024 (`--seed`). Training is
single-GPU; see the DF-CIoU section.

## Parameter and FLOP counts

`count_params_flops.py` prints both counts. The paper quotes the values after
`fuse()`: 11,127,906 parameters for YOLOv8s with the six-class head and
10,778,280 for GBH-YOLO (10,783,752 before `fuse()`). `fuse()` folds BatchNorm
into Ultralytics' own `Conv` layers only; `CSPStage` (with its `RepConv`
branches) and `Detect_Light` are defined in `Addmodules` and are left untouched,
so the multi-branch form of `CSPStage` is what is counted (Section 3.1).

## Released checkpoint

`weights/GBH-YOLO_best.pt` is the seed-0 run of the full model (Ultralytics
8.2.18, 200 epochs; best epoch 198). On the validation split it reaches
P 0.931, R 0.836, mAP@0.5 0.8485 and mAP@0.5:0.95 0.7267; the figures
tabulated in the paper are means over seeds 0, 42 and 2024. The checkpoint also
stores the run's training arguments and per-epoch metrics:

```python
import torch
ckpt = torch.load('weights/GBH-YOLO_best.pt', map_location='cpu')
ckpt['train_args'], ckpt['train_metrics'], ckpt['train_results']
```

The YOLOv8s baseline is not shipped as a checkpoint: it is the stock
architecture and is reproduced directly from the configuration file included
here, so releasing its weights would add nothing that the configuration and the
training script do not already provide.

## The DF-CIoU loss

* **Switch.** `scripts/train.py --df-ciou auto|on|off` (default `auto`: on for
  configurations whose file name contains "GBH", off otherwise). It sets
  `DFState.enabled`, which `BboxLoss` reads; `off` gives standard CIoU.
* **Schedule.** `d0 = 0.2`, `u0 = 0.95` (`DFState`). With `t = epoch / epochs`
  the interval narrows linearly from [0, 1] at the start of training, where the
  loss equals standard CIoU, to [d0, u0] at the end.
* **Callback.** The `on_train_epoch_start` callback in `scripts/train.py` is
  required: it is the only thing that advances `DFState.epoch`; without it `t`
  stays at 0 and the loss is numerically identical to standard CIoU. `BboxLoss`
  prints one `[DF-PROBE]` line per epoch so the schedule can be verified from the
  training log — if `d` stays at `0.0000`, the callback is not firing.
* **Single GPU.** `DFState` lives in the training process; DDP worker processes
  would not see it.

## Ablation configurations

| Run | Command (`python scripts/train.py ...`) |
|---|---|
| YOLOv8s (Table 3, row 1) | `--model ultralytics/cfg/models/Add/yolov8s.yaml` |
| YOLOv8s + DF-CIoU (Table 3, row 5; Table 11) | `--model ultralytics/cfg/models/Add/yolov8s.yaml --df-ciou on` |
| GFPN + BRA + fourth head, no DF-CIoU (Table 3, row 7) | `--model ultralytics/cfg/models/Add/GBH-YOLO.yaml --df-ciou off` |
| same with stock `Detect` heads (Table 3, row †) | `--model ultralytics/cfg/models/Add/ablation/GBH-YOLO-stdhead.yaml --df-ciou off` |
| full GBH-YOLO (Table 3, last row) | `--model ultralytics/cfg/models/Add/GBH-YOLO.yaml` |
| `CSPStage` depth n = 2 / 3 (Table 6) | `--model ultralytics/cfg/models/Add/ablation/GBH-YOLO-csp-n2.yaml --df-ciou off` (or `-n3`) |

`CSPStage` depth is n = 1 at all six positions of `GBH-YOLO.yaml`.

## Data availability

The UCB-Dataset involves human subjects and is **not** included here. A
face-anonymised version is available under a data-use agreement on reasonable
request to the corresponding author. `splits/` lists the exact files of each
partition; `splits/README.md` explains how they relate to the 982 annotated
frames. All reported results were computed on the original (non-anonymised)
images.

## Licence

Ultralytics 8.2.18 is distributed under AGPL-3.0. The modifications in this
repository are derivative works of it and are released under the same licence.

## Citation

The manuscript is under review; its reference will be added on publication.
Archived code: Zenodo, https://doi.org/10.5281/zenodo.21931934
