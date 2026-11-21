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

## 单张 RTX 4090 训练

在服务器上将 `CLCNet` 和 `LCCNet` 放在同一目录。使用已装好 PyTorch、torchvision 和 LCCNet 依赖的环境。默认从 `../data/VoD` 读数据，从 `../LCCNet/pretrained/vod_quality/vod_iter1.tar` 加载第一阶段的预训练权重。
本地主干在构造时会读取 torchvision 的 ResNet18 权重，离线服务器需沿用已有 LCCNet 环境的 Torch 缓存。

```bash
# 在 CLCNet 源码目录运行
conda activate lccnet
bash train_quality.sh
```

如果路径不同，可以在启动时指定。

```bash
DATA_FOLDER=/path/to/VoD PRETRAINED=/path/to/vod_iter1.tar bash train_quality.sh
```

脚本默认使用 GPU 0、batch 8、4 个数据加载进程和 5e-5 的微调学习率，可通过 `GPUS`、`BATCH_SIZE`、`NUM_WORKERS` 和 `LR` 覆盖。五个阶段依次训练 50、40、40、40、50 轮。每次启动会打印本次训练的日志和权重目录。

每个训练样本做两步局部残差监督，第二步使用第一步软门控更新后的点云。验证时最多接受两步更新。日志中的 `first_t_med`、`first_r_med` 是同一批验证帧经过第一步的中位误差，`t_med`、`r_med` 是最终中位误差，单位分别为厘米和度。各阶段按最终 `t_med` 保存最佳权重。

## 第一阶段对照

将数据和权重准备好后，可以先单独训练第 1 阶段。

```bash
END_STAGE=1 bash train_quality.sh
```

将 `RUN_DIR` 设为启动时打印的 checkpoints 路径。下面的命令在同一批 VoD 验证帧上运行 LCCNet、关闭门控的 CLCNet、开启门控的 CLCNet。三组使用相同的初始扰动、深度输入和最多两步更新。评估报告平移与旋转的均值、中位数、P90、误差最大 10% 帧的均值，以及实际接受的更新步数。平移、旋转各自排序。

```bash
RUN_DIR=/path/to/outputs/checkpoints/vod_4090_run
BASE=../LCCNet/pretrained/vod_quality
PYTHONPATH=. python tools/evaluate.py \
  --weights "$RUN_DIR/iter1/clcnet_best.pth" --baseline_weights "$BASE/vod_iter1.tar" \
  --max_r 20 --max_t 1.5 --eval_iters 2
```

## 五阶段级联

第一阶段单独训练后，可在同一目录接着训练第 2 至 5 阶段。运行前激活训练环境。自定义数据路径时，训练和评估都要传入 `--data_folder`。

```bash
PYTHONPATH=. python tools/train_5stage.py --quality --start_stage 2 --checkpoints "$RUN_DIR"
```

五阶段评估需要两套按粗到细排列的权重。以下 Bash 命令在相同的初始大扰动上依次执行五个模型，每个阶段最多更新一次。某个阶段拒绝更新时，后面的细量程模型仍可尝试。

```bash
PYTHONPATH=. python tools/evaluate.py \
  --weights "$RUN_DIR"/iter{1..5}/clcnet_best.pth \
  --baseline_weights "$BASE"/vod_iter{1..5}.tar --max_r 20 --max_t 1.5
```

单独评估 CLCNet 时，只传 `--weights`，可用 `--no_gate` 关闭门控。新模型训练默认先投影到 VoD 的 384×1280 画布，再双线性缩放到 256×512。单独评估旧 CLCNet 权重时，缺少 `depth_mode` 元数据会沿用旧版直接投影。三组对照默认统一使用 LCCNet 预处理，并在日志里打印实际设置。
