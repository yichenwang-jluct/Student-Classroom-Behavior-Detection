# GBH-YOLO：基于改进 YOLOv8 的课堂学生行为检测

本仓库为论文《（请填写你的论文标题）》的官方代码实现。我们在 YOLOv8s 的基础上进行结构改进（GBH-YOLO），用于课堂场景下的学生行为检测，可识别 6 类行为：低头（bowing_head）、举手（hand_raising）、学习（studying）、阅读（reading）、玩手机（using_phone）、书写（writing）。

## 1. 主要结果

在自建课堂行为数据集上训练 200 个 epoch 后的验证集表现：

| 指标 | 数值 |
| --- | --- |
| Precision | 0.934 |
| Recall | 0.834 |
| mAP@0.5 | 0.848 |
| mAP@0.5:0.95 | 0.726 |

各类别 AP@0.5：

| 类别 | AP@0.5 |
| --- | --- |
| writing | 0.988 |
| reading | 0.977 |
| hand_raising | 0.952 |
| studying | 0.941 |
| using_phone | 0.935 |
| bowing_head | 0.298 |

> 注：bowing_head 类样本量极少（见 `labels.jpg`），属于类别极度不平衡，是后续改进的方向之一。

训练曲线、PR 曲线等可视化结果见 `runs/train/exp101/` 目录。

## 2. 环境依赖

```bash
# 建议使用 conda 创建独立环境
conda create -n gbhyolo python=3.8 -y
conda activate gbhyolo

# 安装依赖（根据你的 CUDA 版本调整 PyTorch）
pip install -r requirements.txt
```

主要依赖：Python 3.8+、PyTorch（建议 1.8+ 并支持 CUDA）、ultralytics（本仓库为改进版，请使用仓库内代码而非 pip 安装的官方版本）。

## 3. 数据集准备

数据集采用 YOLO 格式，目录结构如下：

```
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

并在 `ultralytics/cfg/datasets/A_my_data.yaml` 中配置数据集路径与类别名称。

## 4. 训练

```bash
python train.py
```

或使用命令行：

```bash
yolo detect train \
  model=ultralytics/cfg/models/Add/GBH-YOLO.yaml \
  data=ultralytics/cfg/datasets/A_my_data.yaml \
  epochs=200 batch=4 imgsz=640 device=0 optimizer=SGD lr0=0.01
```

完整训练超参数见 `runs/train/exp101/args.yaml`。

## 5. 验证 / 推理

```bash
# 验证
yolo detect val model=runs/train/exp101/weights/best.pt data=ultralytics/cfg/datasets/A_my_data.yaml

# 推理
yolo detect predict model=runs/train/exp101/weights/best.pt source=你的图片或视频路径
```

预训练权重 `best.pt` 可在本仓库 [Releases](../../releases) 页面下载。

## 6. 引用

如果本工作对你的研究有帮助，请引用：

```bibtex
@article{your2025gbhyolo,
  title   = {你的论文标题},
  author  = {你的姓名 and 合作者},
  journal = {期刊/会议名称},
  year    = {2025}
}
```

## 7. 致谢

本项目基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 开发，感谢其开源贡献。
