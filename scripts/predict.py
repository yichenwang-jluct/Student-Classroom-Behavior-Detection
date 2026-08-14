# -*- coding: utf-8 -*-
"""
Run inference on images or a video.

    python scripts/predict.py --weights runs/train/gbh_seed0/weights/best.pt --source path/to/images
"""
import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--source', required=True, help='image, directory, or video')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--device', default=None)
    ap.add_argument('--save', action='store_true', help='write annotated images')
    args = ap.parse_args()

    kw = dict(source=args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
              save=args.save, project='runs/predict', name='exp')
    if args.device is not None:
        kw['device'] = args.device
    YOLO(args.weights).predict(**kw)


if __name__ == '__main__':
    main()
