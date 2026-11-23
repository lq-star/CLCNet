"""
Train CLCNet (Confidence-gated LCCNet) on VoD.

Example:
  cd /root/autodl-tmp/CLCNet/CLCNet
  python tools/train.py \\
    --data_folder /root/autodl-tmp/CLCNet/data/VoD \\
    --pretrained ../LCCNet/checkpoints/vod/xxx.pth \\
    --epochs 40 --batch_size 8 --train_iters 1 --max_r 1.0 --max_t 0.2
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

# AutoDL often ships OMP/MKL=0 which tanks CPU preprocess / DataLoader.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if os.environ.get(_k, "0") in ("", "0"):
        os.environ[_k] = "8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from clcnet.paths import (
    CHECKPOINT_ROOT,
    DATA_ROOT,
    ensure_lccnet_on_path,
    ensure_method_on_path,
)

ensure_method_on_path()
ensure_lccnet_on_path()

from DatasetVoD import DatasetVoD  # noqa: E402
from quaternion_distances import quaternion_distance  # noqa: E402
from utils import merge_inputs  # noqa: E402

from clcnet.losses import ConfGatedCombinedLoss  # noqa: E402
from clcnet.models import ConfLCCNet  # noqa: E402
from clcnet.models.gated_refine import compose_delta, soft_gate_pose  # noqa: E402
from clcnet.utils.geometry import (  # noqa: E402
    lidar_project_depth,
    miscalibrate_cloud,
    project_batch_after_rt,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train confidence-gated LCCNet on VoD")
    p.add_argument("--data_folder", type=str, default=str(DATA_ROOT))
    p.add_argument("--checkpoints", type=str, default=str(CHECKPOINT_ROOT / "vod"))
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_r", type=float, default=1.0)
    p.add_argument("--max_t", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=80.0)
    p.add_argument("--train_iters", type=int, default=1, help="soft-gated refinement steps in training")
    p.add_argument("--eval_iters", type=int, default=1, help="val refine steps; 1=LCCNet-style single-shot")
    p.add_argument("--lambda_conf", type=float, default=0.3)
    p.add_argument("--lambda_gate", type=float, default=0.05)
    p.add_argument("--weight_point_cloud", type=float, default=0.5)
    p.add_argument("--rescale_rot", type=float, default=1.0)
    p.add_argument("--rescale_transl", type=float, default=2.0)
    p.add_argument("--max_points", type=int, default=0, help="0=full cloud like official CombinedLoss")
    p.add_argument(
        "--ckpt_metric",
        type=str,
        default="t_med",
        choices=["val_loss", "t_med", "tr_med"],
        help="tr_med = t_med + r_weight * r_med (protect rotation while selecting)",
    )
    p.add_argument(
        "--ckpt_r_weight",
        type=float,
        default=10.0,
        help="degrees→cm weight for tr_med (10 ⇒ 0.1° ≈ 1 cm)",
    )
    p.add_argument("--freeze_backbone_epochs", type=int, default=0, help="train conf head only for N epochs")
    p.add_argument("--backbone_lr_mult", type=float, default=0.05, help="backbone lr = lr * mult after unfreeze")
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--no_inherit_best", action="store_true", help="do not inherit old best score in ckpt dir")
    return p.parse_args()


def ckpt_score(metrics: dict, metric: str, r_weight: float = 10.0) -> float:
    if metric == "t_med":
        return float(metrics["t_med"])
    if metric == "tr_med":
        return float(metrics["t_med"]) + float(r_weight) * float(metrics["r_med"])
    return float(metrics["val_loss"])


def _raw_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def set_backbone_trainable(model, trainable: bool):
    raw = _raw_model(model)
    for p in raw.backbone.parameters():
        p.requires_grad = trainable
    # Keep BN in train() even when weights are frozen: CLCNet depth projection
    # differs slightly from LCCNet, so running stats need to adapt. Forcing
    # backbone.eval() collapses pose (~1.5m errors).
    for p in raw.fc1_conf.parameters():
        p.requires_grad = True
    for p in raw.fc2_conf.parameters():
        p.requires_grad = True


def build_optimizer(model, lr, backbone_lr_mult, weight_decay=5e-6):
    raw = _raw_model(model)
    conf_params = list(raw.fc1_conf.parameters()) + list(raw.fc2_conf.parameters())
    backbone_params = [p for p in raw.backbone.parameters() if p.requires_grad]
    groups = [{"params": conf_params, "lr": lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})
    return optim.Adam(groups, weight_decay=weight_decay)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_depth_batch(sample, img_shape, max_r_unused, max_depth, device, net_hw=(256, 512)):
    """Miscalibrate + project depth on GPU.

    Project directly at network resolution with scaled K (avoids 384x1280 sparse
    canvas + bilinear on zeros). RGB still pad→bilinear like LCCNet.
    """
    net_h, net_w = net_hw
    canvas_h, canvas_w = img_shape
    sy = net_h / float(canvas_h)
    sx = net_w / float(canvas_w)

    rgb_input, lidar_input, pcs_mis, shapes, calibs = [], [], [], [], []
    for idx in range(len(sample["rgb"])):
        pc_rot = miscalibrate_cloud(
            sample["point_cloud"][idx],
            sample["rot_error"][idx],
            sample["tr_error"][idx],
            device,
        )
        if max_depth < 80.0:
            pc_rot = pc_rot[:, pc_rot[0, :] < max_depth].clone()

        K = sample["calib"][idx]
        if not isinstance(K, torch.Tensor):
            K = torch.as_tensor(K, dtype=torch.float32)
        K = K.to(device=device, dtype=torch.float32).clone()
        # Dataset already scaled K to canvas; rescale to network resolution.
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        depth = lidar_project_depth(pc_rot, K, (net_h, net_w), max_depth=max_depth)
        depth = depth / max_depth

        rgb = sample["rgb"][idx].to(device, non_blocking=True)
        shape_pad = [0, 0, 0, 0]
        shape_pad[3] = canvas_h - rgb.shape[1]
        shape_pad[1] = canvas_w - rgb.shape[2]
        rgb = F.pad(rgb, shape_pad)

        rgb_input.append(rgb)
        lidar_input.append(depth)
        pcs_mis.append(pc_rot)
        shapes.append((net_h, net_w))
        calibs.append(K)

    rgb_b = F.interpolate(torch.stack(rgb_input), size=[net_h, net_w], mode="bilinear")
    depth_b = torch.stack(lidar_input)
    return rgb_b, depth_b, pcs_mis, shapes, calibs


def train_one_epoch(model, loader, optimizer, loss_fn, args, img_shape, device, epoch):
    model.train()
    total = 0.0
    n = 0
    t0 = time.time()
    for batch_idx, sample in enumerate(loader):
        sample["tr_error"] = sample["tr_error"].to(device)
        sample["rot_error"] = sample["rot_error"].to(device)
        # CombinedLoss expects clouds on the same device as predictions.
        pcs_gt = [pc.to(device) for pc in sample["point_cloud"]]
        rgb, depth, pcs_mis, shapes, calibs = build_depth_batch(
            sample, img_shape, args.max_r, args.max_depth, device
        )

        optimizer.zero_grad()
        if args.train_iters <= 1:
            transl, rot, conf = model(rgb, depth)
            losses = loss_fn(
                pcs_gt,
                sample["tr_error"],
                sample["rot_error"],
                transl,
                rot,
                conf,
            )
        else:
            # Soft-gated multi-step: supervise accumulated pose + last-step confidence.
            acc = torch.eye(4, device=device).unsqueeze(0).repeat(rgb.shape[0], 1, 1)
            depth_k = depth
            conf_last = None
            transl_last = rot_last = None
            for it in range(args.train_iters):
                transl, rot, conf = model(rgb, depth_k)
                transl_g, rot_g = soft_gate_pose(transl, rot, conf)
                delta = compose_delta(transl_g, rot_g)
                acc = delta.bmm(acc)
                conf_last, transl_last, rot_last = conf, transl, rot
                if it + 1 < args.train_iters:
                    # Reproject miscalibrated clouds with current accumulated correction.
                    # Network predicts residual error w.r.t. camera; after correction,
                    # depth should approach the GT-aligned projection.
                    depth_full = project_batch_after_rt(
                        pcs_mis, calibs, shapes, acc, args.max_depth, device
                    )
                    depth_k = F.interpolate(depth_full, size=[256, 512], mode="bilinear")
            # Convert accumulated RT to transl/quat for pose loss against full GT error.
            from clcnet.models.gated_refine import mat_to_transl_quat

            transl_acc, rot_acc = mat_to_transl_quat(acc)
            losses = loss_fn(
                pcs_gt,
                sample["tr_error"],
                sample["rot_error"],
                transl_acc,
                rot_acc,
                conf_last,
            )

        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = rgb.shape[0]
        total += losses["total_loss"].item() * bs
        n += bs
        if batch_idx % args.print_freq == 0:
            print(
                f"Epoch {epoch} Iter {batch_idx}/{len(loader)} "
                f"loss={losses['total_loss'].item():.4f} "
                f"conf={losses['conf_mean'].item():.3f} "
                f"conf_tgt={losses['conf_tgt_mean'].item():.3f} "
                f"t={time.time()-t0:.1f}s"
            )
            t0 = time.time()
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, args, img_shape, device, conf_threshold=0.5, eval_iters=3):
    model.eval()
    total_loss = 0.0
    t_errs, r_errs = [], []
    steps_all = []
    reject_n = 0
    n = 0

    for sample in loader:
        sample["tr_error"] = sample["tr_error"].to(device)
        sample["rot_error"] = sample["rot_error"].to(device)
        pcs_gt = [pc.to(device) for pc in sample["point_cloud"]]
        rgb, depth, pcs_mis, shapes, calibs = build_depth_batch(
            sample, img_shape, args.max_r, args.max_depth, device
        )
        B = rgb.shape[0]
        acc = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        depth_k = depth
        active = torch.ones(B, dtype=torch.bool, device=device)
        steps = torch.zeros(B, dtype=torch.long, device=device)
        conf_last = torch.zeros(B, 1, device=device)

        for it in range(max(eval_iters, 1)):
            transl, rot, conf = model(rgb, depth_k)
            conf_last = conf
            if eval_iters <= 1:
                delta = compose_delta(transl, rot)
                acc = delta.bmm(acc)
                steps += 1
                break
            gate = (conf.view(-1) >= conf_threshold) & active
            if not gate.any() and it > 0:
                reject_n += int((~gate).sum().item())
                break
            eye = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            delta = compose_delta(transl, rot)
            delta_m = torch.where(gate.view(B, 1, 1), delta, eye)
            acc = delta_m.bmm(acc)
            steps = steps + gate.long()
            active = active & gate
            if it + 1 < eval_iters and active.any():
                depth_full = project_batch_after_rt(
                    pcs_mis, calibs, shapes, acc, args.max_depth, device
                )
                depth_k = F.interpolate(depth_full, size=[256, 512], mode="bilinear")

        from clcnet.models.gated_refine import mat_to_transl_quat

        pred_t, pred_r = mat_to_transl_quat(acc)
        losses = loss_fn(
            pcs_gt,
            sample["tr_error"],
            sample["rot_error"],
            pred_t,
            pred_r,
            conf_last,
        )
        total_loss += losses["total_loss"].item() * B
        n += B
        t_errs.append(torch.norm(sample["tr_error"] - pred_t, dim=1).cpu() * 100.0)
        r_errs.append(
            (quaternion_distance(sample["rot_error"], pred_r, pred_r.device) * 180.0 / math.pi).cpu()
        )
        steps_all.append(steps.cpu())

    t_all = torch.cat(t_errs).numpy()
    r_all = torch.cat(r_errs).numpy()
    steps_np = torch.cat(steps_all).numpy()
    return {
        "val_loss": total_loss / max(n, 1),
        "t_mean": float(t_all.mean()),
        "t_med": float(np.median(t_all)),
        "r_mean": float(r_all.mean()),
        "r_med": float(np.median(r_all)),
        "steps_mean": float(steps_np.mean()),
        "reject_n": reject_n,
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.benchmark = True
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except RuntimeError:
            pass
    torch.set_num_threads(8)
    img_shape = (384, 1280)
    input_size = (256, 512)

    os.makedirs(args.checkpoints, exist_ok=True)
    train_set = DatasetVoD(args.data_folder, max_r=args.max_r, max_t=args.max_t, split="train")
    val_set = DatasetVoD(args.data_folder, max_r=args.max_r, max_t=args.max_t, split="val")
    loader_kw = dict(
        num_workers=args.num_workers,
        collate_fn=merge_inputs,
        pin_memory=device.type == "cuda",
    )
    if args.num_workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kw,
    )
    print(
        f"Train {len(train_set)}  Val {len(val_set)}  "
        f"batch={args.batch_size} workers={args.num_workers} device={device}"
    )

    model = ConfLCCNet(image_size=input_size, use_feat_from=1, md=4)
    if args.pretrained:
        try:
            ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.pretrained, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        else:
            state = ckpt
        # Strip DataParallel prefix if present.
        cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        # Sacred / older dumps may nest tensors under non-tensor junk; keep tensors only.
        if cleaned and not any(isinstance(v, torch.Tensor) for v in cleaned.values()):
            raise RuntimeError(f"No tensor weights found in {args.pretrained}")
        if any(k.startswith("backbone.") for k in cleaned) or any(
            k.startswith("fc1_conf") for k in cleaned
        ):
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"Loaded CLCNet weights from {args.pretrained}")
        else:
            missing, unexpected = model.load_lccnet_weights(cleaned)
            print(f"Loaded plain LCCNet backbone from {args.pretrained}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loss_fn = ConfGatedCombinedLoss(
        rescale_trans=args.rescale_transl,
        rescale_rot=args.rescale_rot,
        weight_point_cloud=args.weight_point_cloud,
        lambda_conf=args.lambda_conf,
        lambda_gate=args.lambda_gate,
        t_scale=args.max_t,
        r_scale_deg=args.max_r,
        max_points=args.max_points,
    )

    freeze_n = max(0, int(args.freeze_backbone_epochs))
    if freeze_n > 0:
        set_backbone_trainable(model, False)
        print(f"Freeze backbone for first {freeze_n} epochs (conf head only)")
    else:
        set_backbone_trainable(model, True)

    optimizer = build_optimizer(model, args.lr, args.backbone_lr_mult if freeze_n == 0 else 1.0)
    e = max(int(args.epochs), 1)
    milestones = sorted({max(1, int(e * 0.35)), max(2, int(e * 0.60)), max(3, int(e * 0.80))})
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    print(
        f"scheduler milestones={milestones} ckpt_metric={args.ckpt_metric} "
        f"ckpt_r_weight={args.ckpt_r_weight} "
        f"train_iters={args.train_iters} eval_iters={args.eval_iters} "
        f"backbone_lr_mult={args.backbone_lr_mult} max_points={args.max_points}"
    )

    best = 1e9
    best_path = os.path.join(args.checkpoints, "clcnet_best.pth")
    if (not args.no_inherit_best) and args.pretrained and os.path.isfile(best_path):
        try:
            prev = torch.load(best_path, map_location="cpu", weights_only=False)
        except TypeError:
            prev = torch.load(best_path, map_location="cpu")
        prev_m = prev.get("metrics") or {}
        if prev_m and (
            (args.ckpt_metric == "tr_med" and "t_med" in prev_m and "r_med" in prev_m)
            or (args.ckpt_metric == "t_med" and "t_med" in prev_m)
            or (args.ckpt_metric == "val_loss" and "val_loss" in prev_m)
        ):
            best = ckpt_score(prev_m, args.ckpt_metric, args.ckpt_r_weight)
            print(f"Inherited best {args.ckpt_metric}={best:.4f} from existing checkpoint")

    unfrozen = freeze_n == 0
    for epoch in range(args.epochs):
        if (not unfrozen) and epoch >= freeze_n:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, args.lr, args.backbone_lr_mult)
            # rebuild scheduler remaining steps
            remain = max(args.epochs - epoch, 1)
            ms = sorted(
                {
                    max(1, int(remain * 0.35)),
                    max(2, int(remain * 0.60)),
                    max(3, int(remain * 0.80)),
                }
            )
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=0.5)
            unfrozen = True
            print(
                f"Unfreeze backbone at epoch {epoch}: "
                f"conf_lr={args.lr} backbone_lr={args.lr * args.backbone_lr_mult} milestones={ms}"
            )

        phase = "conf-only" if (freeze_n > 0 and epoch < freeze_n) else "joint"
        print(f"===== Epoch {epoch} ({phase}) =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, args, img_shape, device, epoch)
        metrics = validate(
            model,
            val_loader,
            loss_fn,
            args,
            img_shape,
            device,
            eval_iters=args.eval_iters,
        )
        scheduler.step()
        print(
            f"train_loss={train_loss:.4f} val_loss={metrics['val_loss']:.4f} "
            f"t_mean={metrics['t_mean']:.3f}cm t_med={metrics['t_med']:.3f} "
            f"r_mean={metrics['r_mean']:.3f}deg r_med={metrics['r_med']:.3f} "
            f"steps={metrics['steps_mean']:.2f}"
        )
        raw = _raw_model(model)
        ckpt_path = os.path.join(
            args.checkpoints,
            f"clcnet_r{args.max_r:.2f}_t{args.max_t:.2f}_e{epoch}_{metrics['val_loss']:.3f}.pth",
        )
        payload = {
            "epoch": epoch,
            "model": raw.state_dict(),
            "state_dict": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        }
        score = ckpt_score(metrics, args.ckpt_metric, args.ckpt_r_weight)
        if score < best:
            best = score
            torch.save(payload, best_path)
            print(
                f"Saved best ({args.ckpt_metric}={best:.4f} "
                f"t_med={metrics['t_med']:.3f} r_med={metrics['r_med']:.3f}) -> {best_path}"
            )
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            torch.save(payload, ckpt_path)
            print(f"Saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
