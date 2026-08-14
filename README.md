# GBH-YOLO — Dense Student Behaviour Detection in University Classrooms

Code accompanying the manuscript *"Edge-Deployable Multi-Scale Visual Sensing for
Dense Student Behaviour Detection in University Classrooms"*.

GBH-YOLO extends YOLOv8s with a GFPN neck (queen-fusion + CSPStage), bi-level
routing attention at P4/P5, a fourth 160×160 detection head, lightweight
depthwise-separable detection heads, and a dynamic adaptive Focaler-CIoU
regression loss that is active only during training.

## Contents

| Path | Contents |
|---|---|
| `ultralytics/nn/Addmodules/RepGFPN.py` | `CSPStage` — GFPN neck block |
| `ultralytics/nn/Addmodules/Biformer.py` | `BiLevelRoutingAttention` — BRA |
| `ultralytics/nn/Addmodules/LightHead.py` | `Detect_Light` — depthwise-separable decoupled head |
| `ultralytics/nn/tasks.py` | model parser with the modules above registered |
| `ultralytics/utils/loss.py` | `BboxLoss` with the DF-CIoU branch |
| `ultralytics/utils/df_ciou.py` | `DFState` — training-progress state for d(t), u(t) |
| `ultralytics/cfg/models/Add/GBH-YOLO.yaml` | full model |
| `ultralytics/cfg/models/Add/yolov8s.yaml` | baseline used in every comparison |
| `ultralytics/cfg/datasets/UCB.yaml` | dataset config (images not distributed) |
| `scripts/` | training, evaluation, inference, parameter counting |
| `splits/` | exact train / val / test file lists (filenames only) |
| `weights/` | trained GBH-YOLO checkpoint (seed 0) |

## Installation

These files are drop-in replacements for a stock Ultralytics 8.2.18 tree.

```bash
git clone https://github.com/ultralytics/ultralytics.git -b v8.2.18
cd ultralytics
# copy the ultralytics/ directory of this repository over the clone,
# then install in editable mode
pip install -e .
```

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

## Usage

```bash
# full model
python scripts/train.py --model ultralytics/cfg/models/Add/GBH-YOLO.yaml --seed 0

# baseline
python scripts/train.py --model ultralytics/cfg/models/Add/yolov8s.yaml --seed 0

# evaluation
python scripts/val.py --weights runs/train/gbh_seed0/weights/best.pt --split val
python scripts/val.py --weights runs/train/gbh_seed0/weights/best.pt --split test

# parameters / FLOPs (the paper quotes the values after fuse())
python scripts/count_params_flops.py --model ultralytics/cfg/models/Add/GBH-YOLO.yaml
```

## Released checkpoint

`weights/GBH-YOLO_best.pt` is the seed-0 run of the full model; the figures
tabulated in the paper are means over seeds 0, 42 and 2024. The YOLOv8s baseline
is not shipped as a checkpoint: it is the stock architecture and is reproduced
directly from the configuration file included here,

```bash
python scripts/train.py --model ultralytics/cfg/models/Add/yolov8s.yaml --seed 0
```

so releasing its weights would add nothing that the configuration and the
training script do not already provide.

## The DF-CIoU loss schedule

`BboxLoss.use_df_ciou` in `ultralytics/utils/loss.py` switches the loss:

* `True` — dynamic adaptive Focaler-CIoU, the full GBH-YOLO configuration
* `False` — standard CIoU, used for the ablation rows without DF-CIoU

The interval endpoints are `d0 = 0.2`, `u0 = 0.95`; the interval widens linearly
with training progress `t = epoch / epochs`.

**The `on_train_epoch_start` callback in `scripts/train.py` is required.** It is
the only thing that advances `DFState.epoch`; without it `t` stays at 0 and the
loss is numerically identical to standard CIoU. `BboxLoss` prints one
`[DF-PROBE]` line per epoch so the schedule can be verified from the training
log — if `d` stays at `0.0000`, the callback is not firing.

## Ablation configurations

The rows of the ablation table are obtained from the two YAMLs above:

| Component | How it is toggled |
|---|---|
| GFPN neck | `CSPStage` vs. the stock `C2f` neck blocks |
| BRA | remove layers 25 and 29 of `GBH-YOLO.yaml` |
| Fourth detection head | drop layer 18 from the `Detect_Light` input list at layer 30 |
| Lightweight heads | `Detect_Light` vs. the stock `Detect` at layer 30 |
| DF-CIoU | `use_df_ciou` in `loss.py` |

`CSPStage` depth is `n = 1` at all six positions in the released configuration.

## Data availability

The UCB-Dataset involves human subjects and is **not** included here. A
face-anonymised version is available under a data-use agreement on reasonable
request to the corresponding author. `splits/` lists the exact filenames used
for each partition, so the partitions in the paper can be reconstructed once
access is granted. All reported results were computed on the original
(non-anonymised) images.

## Licence

Ultralytics 8.2.18 is distributed under AGPL-3.0. The modifications in this
repository are derivative works of it and are released under the same licence.

## Citation

To be added once the DOI is issued.
