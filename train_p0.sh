#!/usr/bin/env bash
# CLCNet P0：逐档 LCCNet quality 初始化 + 冻骨干训 conf + 小 lr 解冻
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p outputs/logs/vod_p0 outputs/checkpoints/vod_p0

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lccnet
PY="$(command -v python)"

echo "[CLCNet P0] python=$PY"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

nohup "$PY" -u tools/train_5stage.py \
  --p0 \
  --data_folder /home/abc/liu/calib/data/VoD \
  --checkpoints /home/abc/liu/calib/CLCNet/outputs/checkpoints/vod_p0 \
  --lccnet_quality_dir /home/abc/liu/calib/LCCNet/pretrained/vod_quality \
  --batch_size 160 \
  --gpus 0,1 \
  --num_workers 12 \
  --lr 3e-4 \
  --backbone_lr_mult 0.05 \
  --max_points 16384 \
  > outputs/logs/vod_p0/train_5stage.out 2>&1 &

echo $! > outputs/logs/vod_p0/train_5stage.pid
echo "Started PID=$(cat outputs/logs/vod_p0/train_5stage.pid)"
echo "Tail: tail -f $ROOT/outputs/logs/vod_p0/train_iter1.log"
