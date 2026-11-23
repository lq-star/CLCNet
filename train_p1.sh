#!/usr/bin/env bash
# CLCNet P1：冲超 LCCNet quality 迭代精度
# 相对 P0：更长冻骨干 / 更小 backbone_lr / 全点云 / tr_med 选模 / 细档 multi-step
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p outputs/logs/vod_p1 outputs/checkpoints/vod_p1

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lccnet
PY="$(command -v python)"

echo "[CLCNet P1] python=$PY"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

nohup "$PY" -u tools/train_5stage.py \
  --p1 \
  --data_folder /home/abc/liu/calib/data/VoD \
  --checkpoints /home/abc/liu/calib/CLCNet/outputs/checkpoints/vod_p1 \
  --lccnet_quality_dir /home/abc/liu/calib/LCCNet/pretrained/vod_quality \
  --batch_size 120 \
  --gpus 0,1 \
  --num_workers 12 \
  --lr 2e-4 \
  --backbone_lr_mult 0.02 \
  --lambda_conf 0.25 \
  --lambda_gate 0.03 \
  --max_points 0 \
  --ckpt_metric tr_med \
  --ckpt_r_weight 10 \
  > outputs/logs/vod_p1/train_5stage.out 2>&1 &

echo $! > outputs/logs/vod_p1/train_5stage.pid
PID="$(cat outputs/logs/vod_p1/train_5stage.pid)"
echo "Started PID=$PID"
echo "Tail: tail -f $ROOT/outputs/logs/vod_p1/train_iter1.log"

# 训完后自动评估（与 driver 同生命周期的旁路等待脚本）
nohup bash -c "
  while kill -0 $PID 2>/dev/null; do sleep 60; done
  echo '[P1] training finished, running eval...'
  PYTHONPATH=$ROOT $PY -u tools/evaluate_quality.py \
    --ckpt_dir $ROOT/outputs/checkpoints/vod_p1 \
    --out_json $ROOT/outputs/checkpoints/vod_p1/eval_metrics.json \
    --gpus 0 \
    > $ROOT/outputs/logs/vod_p1/eval_quality.log 2>&1
  echo '[P1] eval done' >> $ROOT/outputs/logs/vod_p1/train_5stage.out
" > outputs/logs/vod_p1/eval_waiter.out 2>&1 &
echo $! > outputs/logs/vod_p1/eval_waiter.pid
echo "Eval waiter PID=$(cat outputs/logs/vod_p1/eval_waiter.pid)"
