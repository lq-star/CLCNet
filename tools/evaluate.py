"""
Evaluate CLCNet with hard confidence-gated iteration on VoD val.

Example:
  python evaluate_vod.py \\
    --weights outputs/checkpoints/vod/clcnet_best.pth \\
    --conf_threshold 0.5 --eval_iters 3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clcnet.paths import DATA_ROOT, METHOD_ROOT, ensure_lccnet_on_path, ensure_method_on_path

ensure_method_on_path()
ensure_lccnet_on_path()
sys.path.insert(0, str(METHOD_ROOT / "tools"))

from DatasetVoD import DatasetVoD  # noqa: E402
from quaternion_distances import quaternion_distance  # noqa: E402
from utils import merge_inputs  # noqa: E402

from clcnet.models import ConfLCCNet  # noqa: E402
from clcnet.models.gated_refine import compose_delta, mat_to_transl_quat  # noqa: E402
from clcnet.utils.geometry import project_batch_after_rt  # noqa: E402
from train import build_depth_batch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", type=str, default=str(DATA_ROOT))
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_r", type=float, default=1.0)
    p.add_argument("--max_t", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=80.0)
    p.add_argument("--eval_iters", type=int, default=3)
    p.add_argument("--conf_threshold", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpus", type=str, default="0")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_shape = (384, 1280)

    val_set = DatasetVoD(args.data_folder, max_r=args.max_r, max_t=args.max_t, split="val")
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=merge_inputs,
    )

    model = ConfLCCNet(image_size=(256, 512), use_feat_from=1, md=4)
    try:
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.weights, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    # Support both ConfLCCNet and plain LCCNet weights.
    if any(k.startswith("backbone.") for k in cleaned) or any(k.startswith("fc1_conf") for k in cleaned):
        model.load_state_dict(cleaned, strict=False)
    else:
        model.load_lccnet_weights(cleaned)
    model = model.to(device).eval()

    t_errs, r_errs, steps_all, confs = [], [], [], []
    for sample in loader:
        sample["tr_error"] = sample["tr_error"].to(device)
        sample["rot_error"] = sample["rot_error"].to(device)
        rgb, depth, pcs_mis, shapes, calibs = build_depth_batch(
            sample, img_shape, args.max_r, args.max_depth, device
        )
        B = rgb.shape[0]
        acc = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        depth_k = depth
        active = torch.ones(B, dtype=torch.bool, device=device)
        steps = torch.zeros(B, dtype=torch.long, device=device)

        for it in range(args.eval_iters):
            transl, rot, conf = model(rgb, depth_k)
            gate = (conf.view(-1) >= args.conf_threshold) & active
            if it > 0 and not gate.any():
                break
            eye = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            delta = compose_delta(transl, rot)
            delta_m = torch.where(gate.view(B, 1, 1), delta, eye)
            acc = delta_m.bmm(acc)
            steps = steps + gate.long()
            active = active & gate
            confs.append(conf.detach().cpu())
            if it + 1 < args.eval_iters and active.any():
                depth_full = project_batch_after_rt(
                    pcs_mis, calibs, shapes, acc, args.max_depth, device
                )
                depth_k = F.interpolate(depth_full, size=[256, 512], mode="bilinear")

        pred_t, pred_r = mat_to_transl_quat(acc)
        t_errs.append(torch.norm(sample["tr_error"] - pred_t, dim=1).cpu() * 100.0)
        r_errs.append(
            (quaternion_distance(sample["rot_error"], pred_r, pred_r.device) * 180.0 / math.pi).cpu()
        )
        steps_all.append(steps.cpu())

    t_all = torch.cat(t_errs).numpy()
    r_all = torch.cat(r_errs).numpy()
    steps_np = torch.cat(steps_all).numpy()
    print("===== CLCNet VoD Val (gated iterative) =====")
    print(f"iters={args.eval_iters}  conf_threshold={args.conf_threshold}")
    print(f"translation  mean={t_all.mean():.3f} cm  median={np.median(t_all):.3f} cm")
    print(f"rotation     mean={r_all.mean():.3f} deg median={np.median(r_all):.3f} deg")
    print(f"steps used   mean={steps_np.mean():.3f}  hist={np.bincount(steps_np, minlength=args.eval_iters+1)}")
    if confs:
        c = torch.cat(confs).numpy().reshape(-1)
        print(f"confidence   mean={c.mean():.3f}  <thr={(c < args.conf_threshold).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
