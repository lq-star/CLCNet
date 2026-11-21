"""
CLCNet 五档课程训练（对齐 LCCNet 官方迭代设定）:

  Stage1: max_r=20°, max_t=1.5m
  Stage2: max_r=10°, max_t=1.0m
  Stage3: max_r=5°,  max_t=0.5m
  Stage4: max_r=2°,  max_t=0.2m
  Stage5: max_r=1°,  max_t=0.1m

质量配方（默认）：
  - 单张 4090，batch=8，微调学习率 5e-5
  - Stage1 从 LCCNet vod_quality/vod_iter1.tar 热启动
  - 后续档用上一档 CLCNet best 级联
  - train_iters=2（软门控多步）
  - epoch: 50/40/40/40/50

Example:
  PYTHONPATH=. python tools/train_5stage.py --quality
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if os.environ.get(_k, "0") in ("", "0"):
        os.environ[_k] = "8"

from clcnet.paths import (
    CHECKPOINT_ROOT,
    DATA_ROOT,
    LCCNET_QUALITY,
    LOG_ROOT,
    METHOD_ROOT,
    ensure_method_on_path,
)

ensure_method_on_path()

STAGES = [
    {"name": "iter1", "max_r": 20.0, "max_t": 1.5},
    {"name": "iter2", "max_r": 10.0, "max_t": 1.0},
    {"name": "iter3", "max_r": 5.0, "max_t": 0.5},
    {"name": "iter4", "max_r": 2.0, "max_t": 0.2},
    {"name": "iter5", "max_r": 1.0, "max_t": 0.1},
]

# Warm-start from LCCNet quality: no need for 200-from-scratch epochs.
# Conf head + light backbone fine-tune; later stages keep cascading.
QUALITY_EPOCHS = [50, 40, 40, 40, 50]


def parse_args():
    p = argparse.ArgumentParser(description="5-stage CLCNet curriculum on VoD")
    p.add_argument("--data_folder", type=str, default=str(DATA_ROOT))
    p.add_argument("--checkpoints", type=str, default=str(CHECKPOINT_ROOT / "vod_quality"))
    p.add_argument("--quality", action="store_true", help="VoD-quality recipe to beat LCCNet")
    p.add_argument("--epochs_per_stage", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train_iters", type=int, default=2)
    p.add_argument("--lambda_conf", type=float, default=0.2)
    p.add_argument("--lambda_gate", type=float, default=0.05)
    p.add_argument("--max_points", type=int, default=16384, help="0=full cloud; 16k≈quality speed tradeoff")
    p.add_argument("--print_freq", type=int, default=10)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument("--end_stage", type=int, default=5, choices=[1, 2, 3, 4, 5])
    p.add_argument("--pretrained", type=str, default="", help="optional init for stage1")
    p.add_argument(
        "--lccnet_quality_dir",
        type=str,
        default=str(LCCNET_QUALITY),
        help="LCCNet vod_quality dir (vod_iter*.tar) for stage1 warm-start",
    )
    args = p.parse_args()
    if args.start_stage > args.end_stage:
        p.error("--start_stage must not exceed --end_stage")
    return args


def stage_epochs(args, stage_idx: int) -> int:
    if args.quality:
        return QUALITY_EPOCHS[stage_idx - 1]
    return args.epochs_per_stage


def default_stage1_pretrained(args) -> str:
    if args.pretrained:
        return args.pretrained
    cand = Path(args.lccnet_quality_dir) / "vod_iter1.tar"
    if cand.exists():
        return str(cand)
    return ""


def main():
    args = parse_args()
    if args.quality:
        args.checkpoints = args.checkpoints or str(CHECKPOINT_ROOT / "vod_quality")
        args.train_iters = max(args.train_iters, 2)
        args.gpus = args.gpus or "0"

    os.makedirs(args.checkpoints, exist_ok=True)
    log_dir = LOG_ROOT / Path(args.checkpoints).name
    log_dir.mkdir(parents=True, exist_ok=True)

    prev_best = default_stage1_pretrained(args)
    if args.start_stage > 1:
        previous_stage = STAGES[args.start_stage - 2]["name"]
        checkpoint = Path(args.checkpoints) / previous_stage / "clcnet_best.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing previous stage checkpoint: {checkpoint}")
        prev_best = str(checkpoint)
    for i, stage in enumerate(STAGES, start=1):
        if i < args.start_stage or i > args.end_stage:
            continue

        stage_dir = Path(args.checkpoints) / stage["name"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        epochs = stage_epochs(args, i)
        log_file = log_dir / f"train_{stage['name']}.log"
        cmd = [
            sys.executable,
            "-u",
            str(METHOD_ROOT / "tools" / "train.py"),
            "--data_folder",
            args.data_folder,
            "--checkpoints",
            str(stage_dir),
            "--epochs",
            str(epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--lr",
            str(args.lr),
            "--max_r",
            str(stage["max_r"]),
            "--max_t",
            str(stage["max_t"]),
            "--train_iters",
            str(args.train_iters),
            "--lambda_conf",
            str(args.lambda_conf),
            "--lambda_gate",
            str(args.lambda_gate),
            "--print_freq",
            str(args.print_freq),
            "--gpus",
            args.gpus,
            "--ckpt_metric",
            "t_med",
            "--max_points",
            str(args.max_points),
        ]
        if prev_best:
            cmd += ["--pretrained", prev_best]

        print("=" * 60)
        print(f"Stage {i}/5  {stage['name']}  max_r={stage['max_r']}  max_t={stage['max_t']}  epochs={epochs}")
        print(f"pretrained: {prev_best or '(scratch)'}")
        print(f"log: {log_file}")
        print(" ".join(cmd))
        print("=" * 60)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(METHOD_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS") if env.get("OMP_NUM_THREADS") not in ("", "0") else "8"
        env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS") if env.get("MKL_NUM_THREADS") not in ("", "0") else "8"
        with open(log_file, "w") as f:
            ret = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(METHOD_ROOT), env=env
            )
        if ret.returncode != 0:
            print(f"Stage {stage['name']} failed with code {ret.returncode}. See {log_file}")
            sys.exit(ret.returncode)

        best = stage_dir / "clcnet_best.pth"
        if not best.exists():
            print(f"Missing best checkpoint after stage {stage['name']}: {best}")
            sys.exit(1)
        prev_best = str(best)
        alias = Path(args.checkpoints) / f"clcnet_{stage['name']}.pth"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        os.symlink(best.resolve(), alias)
        print(f"Stage {stage['name']} done -> {best}")

    print(f"Finished stages {args.start_stage} through {args.end_stage}.")
    print(f"Final model: {prev_best}")


if __name__ == "__main__":
    main()
