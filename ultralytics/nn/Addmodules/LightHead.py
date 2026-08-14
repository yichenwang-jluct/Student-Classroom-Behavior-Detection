# -*- coding: utf-8 -*-
"""
Detect_Light - lightweight decoupled detection head (Section 3.3 of the paper).

Each 3x3 convolution in the regression and classification branches of the standard
YOLOv8 Detect head is replaced by a depthwise-separable pair (1x1 pointwise followed
by a 3x3 depthwise convolution). This mainly reduces the cost of the high-resolution
heads, above all P2 at 160x160. The output format is identical to the standard Detect
head, so training, inference, export, DFL and the task-aligned assigner are unchanged.

Location: ultralytics/nn/Addmodules/LightHead.py
Register with: from .LightHead import *   in Addmodules/__init__.py
and add Detect_Light to the Detect group in tasks.py.
"""
import math
import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ['Detect_Light']


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depthwise convolution (groups = gcd)."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DSConv(nn.Module):
    """Depthwise-separable block: 1x1 pointwise projection + kxk depthwise convolution.
    Drop-in replacement for a standard 3x3 Conv."""
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.pw = Conv(c1, c2, 1, 1)          # pointwise
        self.dw = DWConv(c2, c2, k, 1)         # depthwise
    def forward(self, x):
        return self.dw(self.pw(x))


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect_Light(nn.Module):
    """YOLOv8 decoupled detection head with depthwise-separable convolutions."""
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))
        # both branches: DSConv replaces the two 3x3 Conv of the standard Detect head
        self.cv2 = nn.ModuleList(
            nn.Sequential(DSConv(x, c2, 3), DSConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(DSConv(x, c3, 3), DSConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
