# -*- coding: utf-8 -*-
"""
GBH-YOLO training script (single GPU).

    python scripts/train.py --model ultralytics/cfg/models/Add/GBH-YOLO.yaml --seed 0   # full model
    python scripts/train.py --model ultralytics/cfg/models/Add/yolov8s.yaml --seed 0    # baseline

--df-ciou selects the bounding-box regression loss:
    auto (default)  DF-CIoU for GBH-YOLO configurations, standard CIoU otherwise
    on / off        force DF-CIoU / standard CIoU (e.g. the ablation rows without DF-CIoU)

The `on_train_epoch_start` callback registered below is REQUIRED for DF-CIoU: it
advances DFState.epoch so that the interval [d(t), u(t)] moves with training.
Without it the loss silently degenerates to standard CIoU.

Run from the repository root.
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from ultralytics import YOLO
from ultralytics.utils.df_ciou import DFState

DEFAULT_MODEL = 'ultralytics/cfg/models/Add/GBH-YOLO.yaml'
DEFAULT_DATA = 'ultralytics/cfg/datasets/UCB.yaml'


def sync_df_state(trainer):
    """Publish training progress to DFState so BboxLoss can compute d(t), u(t)."""
    DFState.epoch = trainer.epoch
    DFState.epochs = trainer.epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help='GBH-YOLO.yaml for the full model, yolov8s.yaml for the baseline')
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--df-ciou', choices=['auto', 'on', 'off'], default='auto',
                    help='auto: on for GBH-YOLO configs, off otherwise')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='0')
    ap.add_argument('--name', default='gbh')
    args = ap.parse_args()

    if args.df_ciou == 'auto':
        DFState.enabled = 'gbh' in Path(args.model).stem.lower()
    else:
        DFState.enabled = args.df_ciou == 'on'
    print(f'[DF-CIoU] {"ON  (d0=%.2f, u0=%.2f)" % (DFState.d0, DFState.u0) if DFState.enabled else "OFF (standard CIoU)"}',
          flush=True)

    model = YOLO(args.model)
    model.add_callback('on_train_epoch_start', sync_df_state)   # <-- required for DF-CIoU

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,          # 4 on an 8 GB GPU with the 160x160 fourth head
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,                  # final lr = lr0 * lrf = 1e-4
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=False,              # linear decay
        close_mosaic=0,            # mosaic stays on for the whole schedule
        seed=args.seed,
        deterministic=True,        # not fully effective under torch 1.12.1 + CUDA
        single_cls=False,
        cache=False,
        workers=0,
        device=args.device,
        amp=False,
        project='runs/train',
        name=f'{args.name}_seed{args.seed}',
    )


if __name__ == '__main__':
    main()

# ---------------------------------------------------------------------------
# Checking that DF-CIoU is active
#
#   The script prints "[DF-CIoU] ON" at start-up, and BboxLoss then prints one
#   [DF-PROBE] line per epoch, e.g.
#     [DF-PROBE] epoch=0/200    t=0.0000  d=0.0000  u=1.0000
#     [DF-PROBE] epoch=100/200  t=0.5000  d=0.1000  u=0.9750
#     [DF-PROBE] epoch=199/200  t=0.9950  d=0.1990  u=0.9503
#   If d stays at 0.0000 the callback is not firing — stop and fix it.
#
# Reproducing the paper's runs: seeds 0, 42 and 2024 (--seed).
# ---------------------------------------------------------------------------
