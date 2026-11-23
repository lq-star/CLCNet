"""
CLCNet 五档课程训练。

P0（--p0）: 逐档 LCCNet init + 冻骨干训 conf + 小 lr 解冻；选模 t_med
P1（--p1）: 更长冻骨干 / 全点云 / tr_med；细档 multi-step（级联偏弱）
P2（--p2，推荐冲迭代）:
  - 针对 P1：中档 multi-step 选模与五档级联（每档只 forward 1 次）错位
  - 全程 train/eval_iters=1，与 cascade / per-range single 对齐
  - 加长 iter3/4，更高 r 权重护旋转；更小 backbone_lr
  - 可 --seed_clcnet_dir 复用 P1 已强的档（默认复用 1/2/5）
  - 输出 outputs/checkpoints/vod_p2

Example:
  PYTHONPATH=. python tools/train_5stage.py --p2 --start_stage 3 --end_stage 4
"""

from __future__ import annotations

import argparse
import os
import shutil
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

P0_EPOCHS = [30, 30, 30, 40, 40]
P0_FREEZE = [12, 12, 12, 15, 15]

P1_EPOCHS = [40, 40, 45, 60, 60]
P1_FREEZE = [20, 20, 22, 30, 30]
P1_TRAIN_ITERS = [1, 1, 2, 2, 3]
P1_EVAL_ITERS = [1, 1, 2, 2, 3]

# P2: single-shot only; invest epochs in weak mid stages (iter3/4)
P2_EPOCHS = [40, 40, 70, 80, 50]
P2_FREEZE = [22, 22, 40, 45, 28]
P2_TRAIN_ITERS = [1, 1, 1, 1, 1]
P2_EVAL_ITERS = [1, 1, 1, 1, 1]
# When seeding from a prior CLCNet run, reuse these stages unless overwritten.
P2_DEFAULT_SEED_STAGES = (1, 2, 5)


def parse_args():
    p = argparse.ArgumentParser(description="5-stage CLCNet curriculum on VoD")
    p.add_argument("--data_folder", type=str, default=str(DATA_ROOT))
    p.add_argument("--checkpoints", type=str, default="")
    p.add_argument("--p0", action="store_true", help="P0 recipe")
    p.add_argument("--p1", action="store_true", help="P1 recipe")
    p.add_argument("--p2", action="store_true", help="P2 recipe: cascade-aligned single-shot")
    p.add_argument("--quality", action="store_true", help="legacy quality recipe")
    p.add_argument("--epochs_per_stage", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=160)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train_iters", type=int, default=1)
    p.add_argument("--eval_iters", type=int, default=1)
    p.add_argument("--lambda_conf", type=float, default=0.3)
    p.add_argument("--lambda_gate", type=float, default=0.05)
    p.add_argument("--max_points", type=int, default=16384)
    p.add_argument("--freeze_backbone_epochs", type=int, default=-1, help="-1: use recipe schedule")
    p.add_argument("--backbone_lr_mult", type=float, default=0.05)
    p.add_argument("--ckpt_metric", type=str, default="t_med", choices=["val_loss", "t_med", "tr_med"])
    p.add_argument("--ckpt_r_weight", type=float, default=10.0)
    p.add_argument("--print_freq", type=int, default=10)
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument("--end_stage", type=int, default=5, choices=[1, 2, 3, 4, 5])
    p.add_argument("--pretrained", type=str, default="", help="override stage1 init")
    p.add_argument(
        "--lccnet_quality_dir",
        type=str,
        default=str(LCCNET_QUALITY),
        help="LCCNet vod_quality dir (vod_iter*.tar)",
    )
    p.add_argument(
        "--seed_clcnet_dir",
        type=str,
        default="",
        help="Prior CLCNet ckpt dir to seed strong stages (e.g. vod_p1)",
    )
    p.add_argument(
        "--seed_stages",
        type=str,
        default="1,2,5",
        help="Comma stages to copy from seed_clcnet_dir before training",
    )
    return p.parse_args()


def recipe_flags(args):
    return sum(bool(x) for x in (args.p0, args.p1, args.p2))


def stage_epochs(args, stage_idx: int) -> int:
    if args.p2:
        return P2_EPOCHS[stage_idx - 1]
    if args.p1:
        return P1_EPOCHS[stage_idx - 1]
    if args.p0:
        return P0_EPOCHS[stage_idx - 1]
    if args.quality:
        return [50, 40, 40, 40, 50][stage_idx - 1]
    return args.epochs_per_stage


def stage_freeze(args, stage_idx: int) -> int:
    if args.freeze_backbone_epochs >= 0:
        return args.freeze_backbone_epochs
    if args.p2:
        return P2_FREEZE[stage_idx - 1]
    if args.p1:
        return P1_FREEZE[stage_idx - 1]
    if args.p0:
        return P0_FREEZE[stage_idx - 1]
    return 0


def stage_iters(args, stage_idx: int) -> tuple[int, int]:
    if args.p2:
        return P2_TRAIN_ITERS[stage_idx - 1], P2_EVAL_ITERS[stage_idx - 1]
    if args.p1:
        return P1_TRAIN_ITERS[stage_idx - 1], P1_EVAL_ITERS[stage_idx - 1]
    return args.train_iters, args.eval_iters


def stage_pretrained(args, stage_idx: int) -> str:
    """Per-stage LCCNet quality init; P2 may prefer seeded CLCNet on some stages."""
    if args.pretrained and stage_idx == 1:
        return args.pretrained

    if args.p2 and args.seed_clcnet_dir:
        seed = Path(args.seed_clcnet_dir) / f"iter{stage_idx}" / "clcnet_best.pth"
        # Only use seed as init when we are (re)training that stage and file exists.
        # Seeding-copy is handled separately; here prefer LCCNet for weak mid stages.
        prefer_lcc = stage_idx in (3, 4)
        if (not prefer_lcc) and seed.exists():
            return str(seed)

    if args.p0 or args.p1 or args.p2:
        cand = Path(args.lccnet_quality_dir) / f"vod_iter{stage_idx}.tar"
        if cand.exists():
            return str(cand)
        raise FileNotFoundError(cand)

    if stage_idx == 1:
        if args.pretrained:
            return args.pretrained
        cand = Path(args.lccnet_quality_dir) / "vod_iter1.tar"
        return str(cand) if cand.exists() else ""
    return ""


def seed_stages_from_prior(args):
    """Copy strong stages from a prior CLCNet run into the new checkpoint root."""
    if not args.seed_clcnet_dir:
        return
    src_root = Path(args.seed_clcnet_dir)
    dst_root = Path(args.checkpoints)
    stages = []
    for tok in args.seed_stages.split(","):
        tok = tok.strip()
        if tok:
            stages.append(int(tok))
    for i in stages:
        src = src_root / f"iter{i}" / "clcnet_best.pth"
        if not src.exists():
            print(f"[seed] skip iter{i}: missing {src}")
            continue
        dst_dir = dst_root / f"iter{i}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "clcnet_best.pth"
        shutil.copy2(src, dst)
        alias = dst_root / f"clcnet_iter{i}.pth"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        os.symlink(dst.resolve(), alias)
        print(f"[seed] iter{i}: {src} -> {dst}")


def main():
    args = parse_args()
    if recipe_flags(args) > 1:
        print("Choose only one of --p0 / --p1 / --p2")
        sys.exit(2)
    if args.end_stage < args.start_stage:
        print("--end_stage must be >= --start_stage")
        sys.exit(2)

    if args.p2:
        if not args.checkpoints:
            args.checkpoints = str(CHECKPOINT_ROOT / "vod_p2")
        args.lambda_conf = max(args.lambda_conf, 0.25)
        args.lambda_gate = 0.03 if args.lambda_gate <= 0 else min(args.lambda_gate, 0.03)
        args.backbone_lr_mult = min(args.backbone_lr_mult, 0.015)
        args.max_points = 0
        args.ckpt_metric = "tr_med"
        if args.ckpt_r_weight <= 10.0:
            args.ckpt_r_weight = 15.0
        if args.lr >= 3e-4:
            args.lr = 2e-4
        if args.batch_size > 120:
            args.batch_size = 120
        if not args.seed_clcnet_dir:
            cand = CHECKPOINT_ROOT / "vod_p1"
            if cand.exists():
                args.seed_clcnet_dir = str(cand)
    elif args.p1:
        if not args.checkpoints:
            args.checkpoints = str(CHECKPOINT_ROOT / "vod_p1")
        args.lambda_conf = max(args.lambda_conf, 0.25)
        args.lambda_gate = min(args.lambda_gate, 0.03) if args.lambda_gate > 0 else 0.03
        args.backbone_lr_mult = min(args.backbone_lr_mult, 0.02)
        args.max_points = 0
        args.ckpt_metric = "tr_med"
        if args.lr >= 3e-4:
            args.lr = 2e-4
        if args.batch_size > 120:
            args.batch_size = 120
    elif args.p0:
        if not args.checkpoints:
            args.checkpoints = str(CHECKPOINT_ROOT / "vod_p0")
        args.train_iters = 1
        args.eval_iters = 1
        args.lambda_conf = max(args.lambda_conf, 0.3)
        args.ckpt_metric = "t_med"
    elif args.quality:
        if not args.checkpoints:
            args.checkpoints = str(CHECKPOINT_ROOT / "vod_quality")
    else:
        if not args.checkpoints:
            args.checkpoints = str(CHECKPOINT_ROOT / "vod_5stage")

    os.makedirs(args.checkpoints, exist_ok=True)
    log_dir = LOG_ROOT / Path(args.checkpoints).name
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.p2:
        seed_stages_from_prior(args)

    prev_best = ""
    for i, stage in enumerate(STAGES, start=1):
        if i < args.start_stage or i > args.end_stage:
            cand = Path(args.checkpoints) / stage["name"] / "clcnet_best.pth"
            if cand.exists():
                prev_best = str(cand)
            continue

        stage_dir = Path(args.checkpoints) / stage["name"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        for stale in stage_dir.glob("clcnet_*.pth"):
            stale.unlink()

        epochs = stage_epochs(args, i)
        freeze_n = stage_freeze(args, i)
        train_iters, eval_iters = stage_iters(args, i)
        if args.p0 or args.p1 or args.p2:
            pretrained = stage_pretrained(args, i)
        else:
            pretrained = stage_pretrained(args, i) if i == 1 else (prev_best or stage_pretrained(args, i))

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
            str(train_iters),
            "--eval_iters",
            str(eval_iters),
            "--lambda_conf",
            str(args.lambda_conf),
            "--lambda_gate",
            str(args.lambda_gate),
            "--print_freq",
            str(args.print_freq),
            "--gpus",
            args.gpus,
            "--ckpt_metric",
            args.ckpt_metric,
            "--ckpt_r_weight",
            str(args.ckpt_r_weight),
            "--max_points",
            str(args.max_points),
            "--freeze_backbone_epochs",
            str(freeze_n),
            "--backbone_lr_mult",
            str(args.backbone_lr_mult),
            "--no_inherit_best",
        ]
        if pretrained:
            cmd += ["--pretrained", pretrained]

        print("=" * 60)
        print(
            f"Stage {i}/5  {stage['name']}  max_r={stage['max_r']}  max_t={stage['max_t']}  "
            f"epochs={epochs} freeze={freeze_n} iters={train_iters}/{eval_iters}"
        )
        print(f"pretrained: {pretrained or '(scratch)'}")
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

    print("All requested stages finished.")
    print(f"Last model: {prev_best}")


if __name__ == "__main__":
    main()
