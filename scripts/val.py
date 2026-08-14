# -*- coding: utf-8 -*-
"""
Evaluate a trained checkpoint.

    python scripts/val.py --weights runs/train/gbh_seed0/weights/best.pt
    python scripts/val.py --weights ... --split test

Also reports mAP over the five well-represented classes, i.e. excluding
bowing_the_head (31 instances in the whole dataset; 7 in validation, 4 in test),
whose AP is dominated by a handful of detections.
"""
import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import YOLO

RARE = 'bowing_the_head'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', default='ultralytics/cfg/datasets/UCB.yaml')
    ap.add_argument('--split', default='val', choices=['val', 'test', 'train'])
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    kw = dict(data=args.data, split=args.split, imgsz=args.imgsz,
              batch=args.batch, rect=False, verbose=True,
              project='runs/val', name=args.split)
    if args.device is not None:
        kw['device'] = args.device
    r = model.val(**kw)

    ap50 = {model.names[c]: float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}
    ap95 = {model.names[c]: float(r.box.ap[i]) for i, c in enumerate(r.box.ap_class_index)}
    k5 = [c for c in ap50 if c != RARE]

    print()
    print(f'six classes  mAP@0.5 = {r.box.map50:.4f}   mAP@0.5:0.95 = {r.box.map:.4f}')
    if k5:
        print(f'five classes mAP@0.5 = {sum(ap50[c] for c in k5) / len(k5):.4f}   '
              f'mAP@0.5:0.95 = {sum(ap95[c] for c in k5) / len(k5):.4f}   (excluding {RARE})')
    print(f'P = {r.box.mp:.4f}   R = {r.box.mr:.4f}')


if __name__ == '__main__':
    main()
