#!/usr/bin/env bash
# 从 Stage1 best 续训（修复 DP hook 后），再跑完 iter2–iter5。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p outputs/logs/vod_quality

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lccnet
PY="$(command -v python)"

BEST="$ROOT/outputs/checkpoints/vod_quality/iter1/clcnet_best.pth"
STAGE1="$ROOT/outputs/checkpoints/vod_quality/iter1"
LOGDIR="$ROOT/outputs/logs/vod_quality"

echo "[resume] finish stage1 from $BEST (47 more epochs)"
nohup bash -c "
set -e
'$PY' -u tools/train.py \
  --data_folder /home/abc/liu/calib/data/VoD \
  --checkpoints '$STAGE1' \
  --pretrained '$BEST' \
  --epochs 47 \
  --batch_size 160 \
  --num_workers 12 \
  --lr 3e-4 \
  --max_r 20.0 --max_t 1.5 \
  --train_iters 2 \
  --lambda_conf 0.2 --lambda_gate 0.05 \
  --print_freq 10 \
  --gpus 0,1 \
  --ckpt_metric t_med \
  --max_points 16384 \
  > '$LOGDIR/train_iter1_resume.log' 2>&1

ln -sfn '$STAGE1/clcnet_best.pth' '$ROOT/outputs/checkpoints/vod_quality/clcnet_iter1.pth'

'$PY' -u tools/train_5stage.py \
  --quality \
  --start_stage 2 \
  --data_folder /home/abc/liu/calib/data/VoD \
  --checkpoints '$ROOT/outputs/checkpoints/vod_quality' \
  --lccnet_quality_dir /home/abc/liu/calib/LCCNet/pretrained/vod_quality \
  --batch_size 160 \
  --gpus 0,1 \
  --train_iters 2 \
  --max_points 16384 \
  --num_workers 12 \
  --lr 3e-4 \
  >> '$LOGDIR/train_5stage.out' 2>&1
" > "$LOGDIR/resume_wrapper.out" 2>&1 &

echo $! > "$LOGDIR/train_5stage.pid"
echo "Started resume PID=$(cat "$LOGDIR/train_5stage.pid")"
echo "Stage1 log: tail -f $LOGDIR/train_iter1_resume.log"
