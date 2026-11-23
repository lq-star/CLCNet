# CLCNet

**C**onfidence-gated **L**idar-**C**amera calibration **Net**

在 LCCNet 主干上增加置信度头与门控迭代精修，用于相机–LiDAR 外参**在线标定**（专利方向）。

## 目录

```text
CLCNet/
├── METHOD.md
├── train_quality.sh
├── configs/vod_quality.yaml
├── clcnet/          # 模型 / 损失 / 门控迭代
├── tools/           # train.py / train_5stage.py / evaluate.py
└── outputs/
```

依赖同级目录：`../LCCNet`（DatasetVoD / backbone）、`../data/VoD`。

## 方法摘要

```text
扰动外参 → LiDAR 投影深度图
        ↓
   LCCNet 双分支 + Cost Volume
        ↓
   预测 (Δt, Δq, confidence)
        ↓
   conf 门控更新 + 重投影（可多步）
```

## 质量训练（目标超过 LCCNet）

```bash
cd /home/abc/liu/calib/CLCNet
bash train_quality.sh
# 或前台：
PYTHONPATH=. python tools/train_5stage.py --quality --gpus 0,1 --batch_size 120
```

- Stage1 热启动：`../LCCNet/pretrained/vod_quality/vod_iter1.tar`
- Epoch：200/80/80/80/80；`train_iters=2`；双卡 batch 120
- 权重：`outputs/checkpoints/vod_quality/iter*/clcnet_best.pth`
- 日志：`outputs/logs/vod_quality/train_iter*.log`

## 评估

```bash
PYTHONPATH=. python tools/evaluate.py \
  --weights outputs/checkpoints/vod_quality/iter5/clcnet_best.pth \
  --eval_iters 3 --conf_threshold 0.5
```
