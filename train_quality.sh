#!/usr/bin/env bash
# CLCNet 热启动加速配方：更大 batch + 更短 epoch，填满双卡并尽快超过 LCCNet。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p outputs/logs/vod_quality outputs/checkpoints/vod_quality

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PY="${PYTHON:-python}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lccnet 2>/dev/null || true
  PY="$(command -v python)"
fi

echo "[CLCNet] python=$PY"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

nohup "$PY" -u tools/train_5stage.py \
  --quality \
  --data_folder /home/abc/liu/calib/data/VoD \
  --checkpoints /home/abc/liu/calib/CLCNet/outputs/checkpoints/vod_quality \
  --lccnet_quality_dir /home/abc/liu/calib/LCCNet/pretrained/vod_quality \
  --batch_size 180 \
  --gpus 0,1 \
  --train_iters 2 \
  --max_points 16384 \
  --num_workers 12 \
  --lr 3e-4 \
  > outputs/logs/vod_quality/train_5stage.out 2>&1 &

echo $! > outputs/logs/vod_quality/train_5stage.pid
echo "Started PID=$(cat outputs/logs/vod_quality/train_5stage.pid)"
echo "Tail: tail -f $ROOT/outputs/logs/vod_quality/train_iter1.log"
