"""
CLCNet VoD quality eval — aligned with LCCNet scripts/eval_vod_quality.sh:

  1) per-range single (each stage weight on its own disturbance)
  2) iterative multi_range from ±1.5m/20° cascading iter1→iter5 with conf gate

Writes outputs/checkpoints/vod_quality/eval_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clcnet.paths import (
    CHECKPOINT_ROOT,
    DATA_ROOT,
    METHOD_ROOT,
    ensure_lccnet_on_path,
    ensure_method_on_path,
)

ensure_method_on_path()
ensure_lccnet_on_path()

from DatasetVoD import DatasetVoD  # noqa: E402
from quaternion_distances import quaternion_distance  # noqa: E402
from utils import merge_inputs  # noqa: E402

from clcnet.models import ConfLCCNet  # noqa: E402
from clcnet.models.gated_refine import compose_delta, mat_to_transl_quat  # noqa: E402
from clcnet.utils.geometry import project_batch_after_rt  # noqa: E402

import sys

sys.path.insert(0, str(METHOD_ROOT / "tools"))
from train import build_depth_batch  # noqa: E402

RANGES = [
    ("iter1", 1.5, 20.0),
    ("iter2", 1.0, 10.0),
    ("iter3", 0.5, 5.0),
    ("iter4", 0.2, 2.0),
    ("iter5", 0.1, 1.0),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", type=str, default=str(DATA_ROOT))
    p.add_argument(
        "--ckpt_dir",
        type=str,
        default=str(CHECKPOINT_ROOT / "vod_quality"),
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--conf_threshold", type=float, default=0.5)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--max_depth", type=float, default=80.0)
    p.add_argument(
        "--out_json",
        type=str,
        default=str(CHECKPOINT_ROOT / "vod_quality" / "eval_metrics.json"),
    )
    return p.parse_args()


def load_model(weight_path: str, device):
    model = ConfLCCNet(image_size=(256, 512), use_feat_from=1, md=4)
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt if isinstance(ckpt, dict) else ckpt
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    if any(k.startswith("backbone.") for k in cleaned) or any(k.startswith("fc1_conf") for k in cleaned):
        model.load_state_dict(cleaned, strict=False)
    else:
        model.load_lccnet_weights(cleaned)
    return model.to(device).eval()


def summarize(t_cm: np.ndarray, r_deg: np.ndarray):
    return {
        "t_cm_mean": float(t_cm.mean()),
        "t_cm_median": float(np.median(t_cm)),
        "r_deg_mean": float(r_deg.mean()),
        "r_deg_median": float(np.median(r_deg)),
        "n": int(len(t_cm)),
    }


@torch.no_grad()
def eval_single(model, loader, max_t, max_r, device, max_depth):
    img_shape = (384, 1280)
    t_errs, r_errs = [], []
    for sample in loader:
        sample["tr_error"] = sample["tr_error"].to(device)
        sample["rot_error"] = sample["rot_error"].to(device)
        rgb, depth, _, _, _ = build_depth_batch(sample, img_shape, max_r, max_depth, device)
        transl, rot, _conf = model(rgb, depth)
        t_errs.append(torch.norm(sample["tr_error"] - transl, dim=1).cpu() * 100.0)
        r_errs.append(
            (quaternion_distance(sample["rot_error"], rot, rot.device) * 180.0 / math.pi).cpu()
        )
    return summarize(torch.cat(t_errs).numpy(), torch.cat(r_errs).numpy())


@torch.no_grad()
def eval_multirange(models, loader, device, max_depth, conf_threshold):
    """Cascade models[0]..models[-1] from coarse disturbance, gated each step."""
    img_shape = (384, 1280)
    t_errs, r_errs, steps_all = [], [], []
    for sample in loader:
        sample["tr_error"] = sample["tr_error"].to(device)
        sample["rot_error"] = sample["rot_error"].to(device)
        rgb, depth, pcs_mis, shapes, calibs = build_depth_batch(
            sample, img_shape, 20.0, max_depth, device
        )
        B = rgb.shape[0]
        acc = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        depth_k = depth
        active = torch.ones(B, dtype=torch.bool, device=device)
        steps = torch.zeros(B, dtype=torch.long, device=device)

        for model in models:
            transl, rot, conf = model(rgb, depth_k)
            gate = (conf.view(-1) >= conf_threshold) & active
            eye = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            delta = compose_delta(transl, rot)
            # Always apply first stage; later stages gated (match online refine intent).
            # For fair multi_range vs LCCNet (always updates), also allow force-update:
            # use gate only to skip when conf low AFTER step 0.
            delta_m = torch.where(gate.view(B, 1, 1), delta, eye)
            # If nothing would update on first usable step, still apply (like LCCNet).
            if not gate.any():
                delta_m = delta
                gate = active.clone()
            acc = delta_m.bmm(acc)
            steps = steps + gate.long()
            active = active & gate
            depth_full = project_batch_after_rt(pcs_mis, calibs, shapes, acc, max_depth, device)
            depth_k = F.interpolate(depth_full, size=[256, 512], mode="bilinear")

        pred_t, pred_r = mat_to_transl_quat(acc)
        t_errs.append(torch.norm(sample["tr_error"] - pred_t, dim=1).cpu() * 100.0)
        r_errs.append(
            (quaternion_distance(sample["rot_error"], pred_r, pred_r.device) * 180.0 / math.pi).cpu()
        )
        steps_all.append(steps.cpu())

    t = torch.cat(t_errs).numpy()
    r = torch.cat(r_errs).numpy()
    s = torch.cat(steps_all).numpy()
    out = summarize(t, r)
    out["steps_mean"] = float(s.mean())
    return out


@torch.no_grad()
def eval_multirange_always(models, loader, device, max_depth):
    """LCCNet-style multi_range: always update each stage (no conf gate)."""
    img_shape = (384, 1280)
    t_errs, r_errs = [], []
    for sample in loader:
        sample["tr_error"] = sample["tr_error"].to(device)
        sample["rot_error"] = sample["rot_error"].to(device)
        rgb, depth, pcs_mis, shapes, calibs = build_depth_batch(
            sample, img_shape, 20.0, max_depth, device
        )
        B = rgb.shape[0]
        acc = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        depth_k = depth
        for model in models:
            transl, rot, _conf = model(rgb, depth_k)
            delta = compose_delta(transl, rot)
            acc = delta.bmm(acc)
            depth_full = project_batch_after_rt(pcs_mis, calibs, shapes, acc, max_depth, device)
            depth_k = F.interpolate(depth_full, size=[256, 512], mode="bilinear")
        pred_t, pred_r = mat_to_transl_quat(acc)
        t_errs.append(torch.norm(sample["tr_error"] - pred_t, dim=1).cpu() * 100.0)
        r_errs.append(
            (quaternion_distance(sample["rot_error"], pred_r, pred_r.device) * 180.0 / math.pi).cpu()
        )
    return summarize(torch.cat(t_errs).numpy(), torch.cat(r_errs).numpy())


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    weights = []
    for name, _, _ in RANGES:
        w = ckpt_dir / name / "clcnet_best.pth"
        if not w.exists():
            raise FileNotFoundError(w)
        weights.append(str(w))

    print("Weights:")
    for w in weights:
        print(" ", w)

    per_range = {}
    for (name, max_t, max_r), w in zip(RANGES, weights):
        key = f"t{max_t}_r{max_r}"
        print(f"\n=== per-range single {key} ===")
        model = load_model(w, device)
        ds = DatasetVoD(args.data_folder, max_r=max_r, max_t=max_t, split="val")
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=merge_inputs,
        )
        metrics = eval_single(model, loader, max_t, max_r, device, args.max_depth)
        per_range[key] = metrics
        print(
            f"  t_med={metrics['t_cm_median']:.3f}cm t_mean={metrics['t_cm_mean']:.3f} "
            f"r_med={metrics['r_deg_median']:.3f} r_mean={metrics['r_deg_mean']:.3f}"
        )
        del model
        torch.cuda.empty_cache()

    print("\n=== iterative multi_range from ±1.5m/20° (always update, LCCNet-style) ===")
    models = [load_model(w, device) for w in weights]
    ds = DatasetVoD(args.data_folder, max_r=20.0, max_t=1.5, split="val")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=merge_inputs,
    )
    iter_always = eval_multirange_always(models, loader, device, args.max_depth)
    print(
        f"  always: t_med={iter_always['t_cm_median']:.3f}cm t_mean={iter_always['t_cm_mean']:.3f} "
        f"r_med={iter_always['r_deg_median']:.3f} r_mean={iter_always['r_deg_mean']:.3f}"
    )

    print("\n=== iterative multi_range (confidence-gated) ===")
    iter_gated = eval_multirange(
        models, loader, device, args.max_depth, args.conf_threshold
    )
    print(
        f"  gated:  t_med={iter_gated['t_cm_median']:.3f}cm t_mean={iter_gated['t_cm_mean']:.3f} "
        f"r_med={iter_gated['r_deg_median']:.3f} r_mean={iter_gated['r_deg_mean']:.3f} "
        f"steps={iter_gated['steps_mean']:.2f}"
    )

    # Fine-range gated refine on iter5 only (online single-model multi-step)
    print("\n=== iter5 gated refine on ±0.1m/1° (eval_iters=3) ===")
    model5 = models[-1]
    ds5 = DatasetVoD(args.data_folder, max_r=1.0, max_t=0.1, split="val")
    loader5 = DataLoader(
        ds5,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=merge_inputs,
    )
    # reuse gated loop with single model repeated 3 times
    fine = eval_multirange(
        [model5, model5, model5], loader5, device, args.max_depth, args.conf_threshold
    )
    print(
        f"  fine3:  t_med={fine['t_cm_median']:.3f}cm t_mean={fine['t_cm_mean']:.3f} "
        f"r_med={fine['r_deg_median']:.3f} r_mean={fine['r_deg_mean']:.3f}"
    )

    obj = {
        "ckpts": weights,
        "per_range": per_range,
        "iterative": {
            "t1.5_r20_always": iter_always,
            "t1.5_r20_gated": iter_gated,
            "t0.1_r1_gated3": fine,
        },
        "lccnet_quality_ref": {
            "per_range_t0.1_r1.0": {"t_cm_median": 6.0752, "r_deg_median": 0.466},
            "iterative_t1.5_r20": {"t_cm_median": 8.6058, "r_deg_median": 0.9974},
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
