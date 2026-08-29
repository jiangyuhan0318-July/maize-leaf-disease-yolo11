# Maize Leaf Disease Detection with YOLO11n

Experiments for the study *Lightweight Visual Content Analysis for Maize Leaf Disease Detection: A Comparative Study of YOLO11n Structural Improvements*.

This repository contains the training notebooks, trained weights, inference scripts, training curves, and result figures for a systematic comparison of five structural improvement directions applied to YOLO11n, together with a C2PSA ablation study.

## Study Overview

- **Baseline:** YOLO11n (2.6M parameters), 640×640 input, three detection scales
- **Dataset:** Corn Leaf Disease (Roboflow), 1,830 images, 3 classes (common rust, blight, healthy), 7:2:1 split before augmentation; training split augmented 3× (flip, rotation ±10°, shear ±2°, brightness ±12%, exposure ±10%, blur ≤0.5 px, noise ≤0.18%)
- **Five modification directions:** CBAM at the detection head, CBAM in the neck, SE in the backbone, P2 high-resolution detection head, Focal Loss
- **C2PSA ablation:** with/without the built-in C2PSA block × with/without external attention modules

### Main Results

| Experiment | Best mAP50 | vs. Baseline |
|---|---|---|
| Baseline YOLO11n | 0.4367 | — |
| CBAM (head) | 0.4161 | −0.0206 |
| CBAM (neck) | 0.4290 | −0.0077 |
| SE (backbone) | 0.4343 | −0.0024 |
| P2 head | 0.4397 | +0.0030 |
| Focal Loss | 0.3197 | −0.1170 |

The full ablation results (with/without C2PSA, with/without external attention) are in the paper and partially in `results/`.

## Repository Structure

```
.
├── notebooks/          # Kaggle notebooks for each experiment
├── weights/            # Trained checkpoints (best.pt, best_100.pt, best_150.pt)
├── predict.py          # Inference script
├── predict_150.py      # Inference script (150-epoch checkpoint)
├── test_corn.jpg       # Example test image
├── results/            # Training curves, PR curves, confusion matrices per experiment
├── metrics/            # results_100/150.csv and result.xlsx
└── figures/            # Confusion matrix, loss curves, validation previews
```

## Dataset

- Source: [Corn Leaf Disease Dataset on Roboflow Universe](https://universe.roboflow.com/ilikecorn/corn-leaf-disease-zsljc)
- 1,830 images across three classes: common rust, blight, healthy
- Split 7:2:1 (1,281 / 366 / 183) before augmentation; augmentation applied to the training split only

## Experiments (notebooks)

| Notebook | Experiment |
|---|---|
| `notebooks/01_CBAM-head.ipynb` | CBAM modules before each detection head (P3/P4/P5) |
| `notebooks/02_CBAM-neck.ipynb` | CBAM modules after each Concat in the PAN-FPN neck |
| `notebooks/03_SE.ipynb` | SE module after C2PSA at the backbone tail |
| `notebooks/04_P2.ipynb` | Additional P2 high-resolution detection branch |
| `notebooks/05_Focalloss.ipynb` | Focal Loss (γ=2.0, α=0.25) for classification |
| `notebooks/noC2PSA-001.ipynb` | No-C2PSA control baseline |
| `notebooks/noC2PSA-002.ipynb` | No-C2PSA + CBAM (neck) |
| `notebooks/noC2PSA-003.ipynb` | No-C2PSA + SE (backbone) |

Note: the baseline notebook was not saved; the baseline training configuration is reproduced in the other notebooks and in `results/00_baseline/`.

The modified Ultralytics code used by the notebooks (custom model YAMLs and attention modules) is published as a Kaggle dataset: [ultralytics-cbam-project](https://www.kaggle.com/datasets/roxyjiang12180318/ultralytics-cbam-project).

## Inference

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
model.predict("test_corn.jpg", save=True)
```

or simply run:

```bash
python predict.py
```

## Environment

- Python 3 + [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO11)
- Training hardware: NVIDIA Tesla T4 GPU (Kaggle)
- Training settings: batch size 16, up to 200 epochs, AdamW (lr 1e-3, weight decay 5e-4), early stopping with validation patience 30

## Notes

- Full training archives (including intermediate checkpoints) are several hundred MB each and exceed GitHub file-size limits; the lightweight parts (curves, plots, metrics) are included under `results/`. The complete archives are available on request.
- The dataset images are not stored in this repository; download them from the Roboflow link above and follow the split described in the notebooks.

---

# 玉米叶病害检测（YOLO11n）

*Lightweight Visual Content Analysis for Maize Leaf Disease Detection: A Comparative Study of YOLO11n Structural Improvements* 一文的实验仓库。

本仓库包含训练 notebook、训练权重、推理脚本、训练曲线和结果图表：对 YOLO11n 的五个结构改进方向做了系统对比，并进行了 C2PSA 消融实验。

## 研究概况

- **基线模型：** YOLO11n（260 万参数），输入 640×640，三个检测尺度
- **数据集：** Corn Leaf Disease（Roboflow），1,830 张图像，3 类（玉米锈病、枯萎病、健康），先按 7:2:1 划分再做增强；仅训练集做 3 倍增强（翻转、旋转 ±10°、剪切 ±2°、亮度 ±12%、曝光 ±10%、模糊 ≤0.5 px、噪声 ≤0.18%）
- **五个改进方向：** 检测头 CBAM、颈部 CBAM、骨干 SE、P2 高分辨率检测分支、Focal Loss
- **C2PSA 消融：** 有/无内置 C2PSA × 有/无外部注意力模块

### 主要结果

| 实验 | 最佳 mAP50 | 相对基线 |
|---|---|---|
| 基线 YOLO11n | 0.4367 | — |
| CBAM（检测头） | 0.4161 | −0.0206 |
| CBAM（颈部） | 0.4290 | −0.0077 |
| SE（骨干） | 0.4343 | −0.0024 |
| P2 检测头 | 0.4397 | +0.0030 |
| Focal Loss | 0.3197 | −0.1170 |

完整的 C2PSA 消融结果（有/无 C2PSA，有/无外部注意力）见论文及 `results/` 目录。

## 仓库结构

```
.
├── notebooks/          # 每个实验对应的 Kaggle notebook
├── weights/            # 训练好的权重（best.pt、best_100.pt、best_150.pt）
├── predict.py          # 推理脚本
├── predict_150.py      # 推理脚本（150 epoch 权重）
├── test_corn.jpg       # 示例测试图片
├── results/            # 各实验的训练曲线、PR 曲线、混淆矩阵
├── metrics/            # results_100/150.csv 与 result.xlsx
└── figures/            # 混淆矩阵、损失曲线、验证集预览
```

## 数据集

- 来源：[Roboflow Universe 上的 Corn Leaf Disease 数据集](https://universe.roboflow.com/ilikecorn/corn-leaf-disease-zsljc)
- 1,830 张图像，三个类别：玉米锈病、枯萎病、健康
- 先按 7:2:1 划分（1,281 / 366 / 183），再仅对训练集做数据增强

## 实验（notebook）

| Notebook | 实验 |
|---|---|
| `notebooks/01_CBAM-head.ipynb` | 三个检测头（P3/P4/P5）前各加 CBAM |
| `notebooks/02_CBAM-neck.ipynb` | PAN-FPN 颈部每个 Concat 后加 CBAM |
| `notebooks/03_SE.ipynb` | 骨干尾部 C2PSA 之后加 SE |
| `notebooks/04_P2.ipynb` | 增加 P2 高分辨率检测分支 |
| `notebooks/05_Focalloss.ipynb` | 分类损失换为 Focal Loss（γ=2.0，α=0.25） |
| `notebooks/noC2PSA-001.ipynb` | 无 C2PSA 对照基线 |
| `notebooks/noC2PSA-002.ipynb` | 无 C2PSA + CBAM（颈部） |
| `notebooks/noC2PSA-003.ipynb` | 无 C2PSA + SE（骨干） |

说明：基线 notebook 没有保存；基线训练配置与其他 notebook 一致，训练曲线见 `results/00_baseline/`。

notebook 使用的修改版 Ultralytics 代码（自定义模型 YAML 与注意力模块）已发布为 Kaggle 数据集：[ultralytics-cbam-project](https://www.kaggle.com/datasets/roxyjiang12180318/ultralytics-cbam-project)。

## 推理

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
model.predict("test_corn.jpg", save=True)
```

或直接运行：

```bash
python predict.py
```

## 环境

- Python 3 + [Ultralytics](https://github.com/ultralytics/ultralytics)（YOLO11）
- 训练硬件：NVIDIA Tesla T4 GPU（Kaggle）
- 训练设置：batch size 16，最多 200 epochs，AdamW（lr 1e-3，weight decay 5e-4），验证集早停 patience 30

## 说明

- 完整训练归档（含中间 checkpoint）每个有数百 MB，超出 GitHub 单文件大小限制；仓库中只保留轻量部分（曲线、图表、指标），完整归档可按需提供。
- 数据集图片不存放在本仓库中，请从上方 Roboflow 链接下载，并按 notebook 中描述的划分方式使用。
