#!/usr/bin/env bash
# CLCNet P2：对齐五档级联的单步选模，重点重训薄弱的 iter3/4
# - 复用 P1 已强的 iter1/2/5
# - iter3/4 从 LCCNet quality 重训（iters=1, tr_med, 更长冻骨干）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p outputs/logs/vod_p2 outputs/checkpoints/vod_p2

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lccnet
PY="$(command -v python)"

echo "[CLCNet P2] python=$PY"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

nohup "$PY" -u tools/train_5stage.py \
  --p2 \
  --start_stage 3 \
  --end_stage 4 \
  --seed_clcnet_dir "$ROOT/outputs/checkpoints/vod_p1" \
  --seed_stages 1,2,5 \
  --data_folder /home/abc/liu/calib/data/VoD \
  --checkpoints "$ROOT/outputs/checkpoints/vod_p2" \
  --lccnet_quality_dir /home/abc/liu/calib/LCCNet/pretrained/vod_quality \
  --batch_size 120 \
  --gpus 0,1 \
  --num_workers 12 \
  --lr 2e-4 \
  --backbone_lr_mult 0.015 \
  --lambda_conf 0.25 \
  --lambda_gate 0.03 \
  --max_points 0 \
  --ckpt_metric tr_med \
  --ckpt_r_weight 15 \
  > outputs/logs/vod_p2/train_5stage.out 2>&1 &

echo $! > outputs/logs/vod_p2/train_5stage.pid
PID="$(cat outputs/logs/vod_p2/train_5stage.pid)"
echo "Started PID=$PID"
echo "Tail: tail -f $ROOT/outputs/logs/vod_p2/train_iter3.log"

nohup bash -c "
  while kill -0 $PID 2>/dev/null; do sleep 60; done
  if [[ -f $ROOT/outputs/checkpoints/vod_p2/iter3/clcnet_best.pth \
     && -f $ROOT/outputs/checkpoints/vod_p2/iter4/clcnet_best.pth \
     && -f $ROOT/outputs/checkpoints/vod_p2/iter5/clcnet_best.pth ]]; then
    echo '[P2] training finished, running eval...'
    PYTHONPATH=$ROOT $PY -u tools/evaluate_quality.py \
      --ckpt_dir $ROOT/outputs/checkpoints/vod_p2 \
      --out_json $ROOT/outputs/checkpoints/vod_p2/eval_metrics.json \
      --gpus 0 \
      > $ROOT/outputs/logs/vod_p2/eval_quality.log 2>&1
    echo '[P2] eval done' >> $ROOT/outputs/logs/vod_p2/train_5stage.out
  else
    echo '[P2] missing ckpts after exit' >> $ROOT/outputs/logs/vod_p2/train_5stage.out
  fi
" > outputs/logs/vod_p2/eval_waiter.out 2>&1 &
echo $! > outputs/logs/vod_p2/eval_waiter.pid
echo "Eval waiter PID=$(cat outputs/logs/vod_p2/eval_waiter.pid)"
