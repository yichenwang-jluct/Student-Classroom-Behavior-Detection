# -*- coding: utf-8 -*-
"""
DF-CIoU - Dynamic Adaptive Focaler-CIoU (Section 3.4, Eqs. 5-7).
Applied during training only; no inference cost.

Location: ultralytics/utils/df_ciou.py
"""
import math
import torch

__all__ = ["DFState", "df_ciou_components"]


class DFState:
    """Holds training progress so the loss can compute d(t) and u(t).
    `epoch` is advanced by the on_train_epoch_start callback in scripts/train.py.

    Note: at training time the effective d(t)/u(t) are computed in
    BboxLoss._df_interval() of ultralytics/utils/loss.py, using self.df_d0 /
    self.df_u0. The values here are kept identical and are used only when
    df_ciou_components() is called directly.
    """
    epoch = 0
    epochs = 200
    d0 = 0.2
    u0 = 0.95      # Section 3.4

    @classmethod
    def interval(cls):
        t = max(0.0, min(1.0, cls.epoch / max(1, cls.epochs)))
        d = cls.d0 * t
        u = 1.0 - (1.0 - cls.u0) * t
        return d, u


def _ciou_parts(pred, target, eps=1e-7):
    """Boxes in xyxy. Returns (iou, penalty) with CIoU = iou - penalty,
    where penalty is the geometric term of Eq. 7."""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    pw, ph = (px2 - px1).clamp(0), (py2 - py1).clamp(0)
    tw, th = (tx2 - tx1).clamp(0), (ty2 - ty1).clamp(0)
    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) * \
            (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
    union = pw * ph + tw * th - inter + eps
    iou = inter / union
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw ** 2 + ch ** 2 + eps
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    v = (4 / math.pi ** 2) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    penalty = rho2 / c2 + alpha * v
    return iou, penalty


def df_ciou_components(pred, target, d=None, u=None, eps=1e-7):
    """
    Returns (loss_per_box, iou).
    loss_per_box = 1 - IoU_DF + penalty   (Eq. 7); IoU_DF is the linear interval
    mapping of Eq. 6.
    `iou` is the plain IoU, reused for DFL and the quality weights.
    pred, target: (..., 4) xyxy。
    """
    if d is None or u is None:
        d, u = DFState.interval()
    iou, penalty = _ciou_parts(pred, target, eps)
    iou_df = ((iou - d) / (u - d + eps)).clamp(0.0, 1.0)   # Eq. 6
    loss = (1.0 - iou_df) + penalty                         # Eq. 7
    return loss, iou
