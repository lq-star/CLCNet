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
import time

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
from clcnet.models import ConfLCCNet, GatedIterativeCalibrator  # noqa: E402
from clcnet.models.gated_refine import append_update, compose_delta, mat_to_transl_quat, soft_gate_pose  # noqa: E402
from clcnet.utils.geometry import (  # noqa: E402
    network_depth,
    miscalibrate_cloud,
    project_batch_after_rt,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train confidence-gated LCCNet on VoD")
    p.add_argument("--data_folder", type=str, default=str(DATA_ROOT))
    p.add_argument("--checkpoints", type=str, default=str(CHECKPOINT_ROOT / "vod"))
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_r", type=float, default=1.0)
    p.add_argument("--max_t", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=80.0)
    p.add_argument("--depth_mode", choices=["lccnet", "direct"], default="lccnet")
    p.add_argument("--train_iters", type=int, default=2, help="soft-gated refinement steps in training")
    p.add_argument("--lambda_conf", type=float, default=0.2)
    p.add_argument("--lambda_gate", type=float, default=0.05)
    p.add_argument("--weight_point_cloud", type=float, default=0.5)
    p.add_argument("--rescale_rot", type=float, default=1.0)
    p.add_argument("--rescale_transl", type=float, default=2.0)
    p.add_argument("--max_points", type=int, default=0, help="0=full cloud like official CombinedLoss")
    p.add_argument("--ckpt_metric", type=str, default="t_med", choices=["val_loss", "t_med"])
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--allow_tf32", action="store_true")
    args = p.parse_args()
    if args.train_iters < 1:
        p.error("--train_iters must be at least one")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_depth_batch(sample, img_shape, max_r_unused, max_depth, device, net_hw=(256, 512),
                      depth_mode="lccnet"):
    """Project VoD's resized camera-frame clouds using the selected training convention."""
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

        K = sample["calib"][idx]
        if not isinstance(K, torch.Tensor):
            K = torch.as_tensor(K, dtype=torch.float32)
        K = K.to(device=device, dtype=torch.float32).clone()
        shape = img_shape
        if depth_mode == "direct":
            K[0] *= sx
            K[1] *= sy
            shape = net_hw
        elif max_depth < 80.0:
            # Match the baseline's optional pre-projection filter once, on the
            # miscalibrated cloud retained for subsequent refinements.
            pc_rot = pc_rot[:, pc_rot[0] < max_depth]
        depth = network_depth(pc_rot, K, shape, max_depth, net_hw, depth_mode)

        rgb = sample["rgb"][idx].to(device, non_blocking=True)
        shape_pad = [0, 0, 0, 0]
        shape_pad[3] = canvas_h - rgb.shape[1]
        shape_pad[1] = canvas_w - rgb.shape[2]
        rgb = F.pad(rgb, shape_pad)

        rgb_input.append(rgb)
        lidar_input.append(depth)
        pcs_mis.append(pc_rot)
        shapes.append(shape)
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
            sample, img_shape, args.max_r, args.max_depth, device, depth_mode=args.depth_mode
        )

        optimizer.zero_grad()
        # At step k the observed cloud is inv(acc) @ GT @ pc_gt. Its
        # supervised residual is therefore inv(acc) @ GT, in that order.
        target_t, target_q = sample["tr_error"], sample["rot_error"]
        if args.train_iters > 1:
            gt = compose_delta(target_t, target_q)
            acc = torch.eye(4, device=device).expand(rgb.shape[0], -1, -1)
        batch_loss = 0.0
        for it in range(args.train_iters):
            transl, rot, conf = model(rgb, depth)
            losses = loss_fn(pcs_gt, target_t, target_q, transl, rot, conf)
            step_loss = losses["total_loss"] / args.train_iters
            step_loss.backward()
            batch_loss += step_loss.detach().item()
            if it + 1 < args.train_iters:
                # Form the next training example from the model's own update.
                with torch.no_grad():
                    gated_t, gated_q = soft_gate_pose(transl, rot, conf)
                    acc = append_update(acc, compose_delta(gated_t, gated_q))
                    target_t, target_q = mat_to_transl_quat(acc.inverse().bmm(gt))
                    depth = project_batch_after_rt(
                        pcs_mis, calibs, shapes, acc, args.max_depth, device, rgb.shape[-2:], args.depth_mode
                    )

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = rgb.shape[0]
        total += batch_loss * bs
        n += bs
        if batch_idx % args.print_freq == 0:
            print(
                f"Epoch {epoch} Iter {batch_idx}/{len(loader)} "
                f"loss={batch_loss:.4f} "
                f"conf={losses['conf_mean'].item():.3f} "
                f"conf_tgt={losses['conf_tgt_mean'].item():.3f} "
                f"t={time.time()-t0:.1f}s"
            )
            t0 = time.time()
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, args, img_shape, device, conf_threshold=0.5, eval_iters=None):
    model.eval()
    eval_iters = args.train_iters if eval_iters is None else eval_iters
    calibrator = GatedIterativeCalibrator(model, eval_iters, conf_threshold)
    total_loss = 0.0
    t_errs, r_errs = [], []
    first_t_errs, first_r_errs = [], []
    steps_all = []
    reject_n = 0
    n = 0

    for sample in loader:
        sample["tr_error"] = sample["tr_error"].to(device)
        sample["rot_error"] = sample["rot_error"].to(device)
        pcs_gt = [pc.to(device) for pc in sample["point_cloud"]]
        rgb, depth, pcs_mis, shapes, calibs = build_depth_batch(
            sample, img_shape, args.max_r, args.max_depth, device, depth_mode=args.depth_mode
        )
        B = rgb.shape[0]
        result = calibrator.refine_eval(
            rgb, depth,
            lambda acc: project_batch_after_rt(pcs_mis, calibs, shapes, acc, args.max_depth,
                                               device, rgb.shape[-2:], args.depth_mode),
        )
        pred_t, pred_r = result["transl"], result["rot"]
        steps = result["steps_used"]
        reject_n += int((steps < eval_iters).sum().item())
        # Final pose uses all accepted steps; confidence is supervised only
        # against the local proposal which produced it.
        pose_loss, _, _, _ = loss_fn.pose_terms(
            pcs_gt, sample["tr_error"], sample["rot_error"], pred_t, pred_r
        )
        total_loss += pose_loss.item() * B
        n += B
        t_errs.append(torch.norm(sample["tr_error"] - pred_t, dim=1).cpu() * 100.0)
        first_t, first_r = mat_to_transl_quat(result["first_RT"])
        first_t_errs.append(torch.norm(sample["tr_error"] - first_t, dim=1).cpu() * 100.0)
        first_r_errs.append(torch.rad2deg(quaternion_distance(sample["rot_error"], first_r, device)).cpu())
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
        "t_p90": float(np.percentile(t_all, 90)),
        "r_p90": float(np.percentile(r_all, 90)),
        "first_t_med": float(np.median(torch.cat(first_t_errs).numpy())),
        "first_r_med": float(np.median(torch.cat(first_r_errs).numpy())),
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
        is_clcnet, _, _ = model.load_checkpoint(args.pretrained)
        kind = "CLCNet" if is_clcnet else "LCCNet"
        print(f"Loaded {kind} weights from {args.pretrained}")

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

    params = model.module.parameters() if isinstance(model, nn.DataParallel) else model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=5e-6)
    # Scale MultiStep to epoch budget (LCCNet-style decay, VoD-quality lengths).
    e = max(int(args.epochs), 1)
    milestones = sorted({max(1, int(e * 0.35)), max(2, int(e * 0.60)), max(3, int(e * 0.80))})
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    print(f"scheduler milestones={milestones} ckpt_metric={args.ckpt_metric} train_iters={args.train_iters}")

    best = 1e9
    best_path = os.path.join(args.checkpoints, "clcnet_best.pth")
    # Keep prior best when resuming so a worse first epoch cannot clobber it.
    if args.pretrained and os.path.isfile(best_path):
        try:
            prev = torch.load(best_path, map_location="cpu", weights_only=False)
        except TypeError:
            prev = torch.load(best_path, map_location="cpu")
        prev_m = prev.get("metrics") or {}
        if args.ckpt_metric in prev_m:
            best = float(prev_m[args.ckpt_metric])
        print(f"Inherited best {args.ckpt_metric}={best:.4f} from existing checkpoint")

    for epoch in range(args.epochs):
        print(f"===== Epoch {epoch} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, args, img_shape, device, epoch)
        metrics = validate(model, val_loader, loss_fn, args, img_shape, device)
        scheduler.step()
        print(
            f"train_loss={train_loss:.4f} val_loss={metrics['val_loss']:.4f} "
            f"t_mean={metrics['t_mean']:.3f}cm t_med={metrics['t_med']:.3f} "
            f"r_mean={metrics['r_mean']:.3f}deg r_med={metrics['r_med']:.3f} "
            f"t_p90={metrics['t_p90']:.3f}cm r_p90={metrics['r_p90']:.3f}deg "
            f"first_t_med={metrics['first_t_med']:.3f}cm first_r_med={metrics['first_r_med']:.3f}deg "
            f"steps={metrics['steps_mean']:.2f}"
        )
        raw = model.module if isinstance(model, nn.DataParallel) else model
        ckpt_path = os.path.join(
            args.checkpoints,
            f"clcnet_r{args.max_r:.2f}_t{args.max_t:.2f}_e{epoch}_{metrics['val_loss']:.3f}.pth",
        )
        payload = {
            "epoch": epoch,
            "eval_iters": args.train_iters,
            "model": raw.state_dict(),
            "state_dict": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        }
        score = metrics["t_med"] if args.ckpt_metric == "t_med" else metrics["val_loss"]
        if score < best:
            best = score
            torch.save(payload, best_path)
            print(f"Saved best ({args.ckpt_metric}={best:.4f}) -> {best_path}")
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            torch.save(payload, ckpt_path)
            print(f"Saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
