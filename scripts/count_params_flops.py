# -*- coding: utf-8 -*-
"""
Report parameters and FLOPs, both unfused and after Ultralytics' native fuse().
The values quoted in the paper are the native-fuse ones.

    python scripts/count_params_flops.py --model ultralytics/cfg/models/Add/GBH-YOLO.yaml
    python scripts/count_params_flops.py --weights runs/train/gbh_seed0/weights/best.pt
"""
import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help='model yaml')
    ap.add_argument('--weights', default=None, help='trained .pt (takes precedence)')
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    src = args.weights or args.model
    if src is None:
        raise SystemExit('give --model or --weights')

    m = YOLO(src)
    print('--- unfused ---')
    m.info(detailed=False, imgsz=args.imgsz)
    m.fuse()
    print('--- after fuse() ---')
    m.info(detailed=False, imgsz=args.imgsz)


if __name__ == '__main__':
    main()
