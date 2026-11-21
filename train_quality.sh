#!/usr/bin/env bash
# Five-stage VoD fine-tuning on one RTX 4090.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PY="${PYTHON:-python}"
DATA="${DATA_FOLDER:-$ROOT/../data/VoD}"
INIT="${PRETRAINED:-$ROOT/../LCCNet/pretrained/vod_quality/vod_iter1.tar}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
for file in "$DATA/lidar/ImageSets/train.txt" "$DATA/lidar/ImageSets/val.txt" "$INIT"; do
  if [[ ! -s "$file" ]]; then
    echo "Missing training input: $file (check DATA_FOLDER and PRETRAINED)" >&2
    exit 1
  fi
done
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

mkdir -p outputs/logs
LOGDIR="$(mktemp -d "$ROOT/outputs/logs/vod_4090_$(date +%Y%m%d_%H%M%S)_XXXX")"
CKPT="$ROOT/outputs/checkpoints/${LOGDIR##*/}"
nohup "$PY" -u tools/train_5stage.py \
  --quality --data_folder "$DATA" --pretrained "$INIT" --checkpoints "$CKPT" \
  --gpus "$CUDA_VISIBLE_DEVICES" --batch_size "${BATCH_SIZE:-8}" --num_workers "${NUM_WORKERS:-4}" \
  --train_iters 2 --max_points 16384 --lr "${LR:-5e-5}" --end_stage "${END_STAGE:-5}" \
  > "$LOGDIR/train_5stage.out" 2>&1 &
echo $! > "$LOGDIR/train_5stage.pid"
echo "Started PID=$!  checkpoints=$CKPT"
echo "Stage 1: tail -F '$LOGDIR/train_iter1.log'"
echo "Launcher: tail -f '$LOGDIR/train_5stage.out'"
