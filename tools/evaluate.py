"""Evaluate one VoD model or a coarse-to-fine cascade on fixed validation perturbations."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from clcnet.paths import DATA_ROOT, METHOD_ROOT, ensure_lccnet_on_path

ensure_lccnet_on_path()
sys.path.insert(0, str(METHOD_ROOT / "tools"))

from DatasetVoD import DatasetVoD
from quaternion_distances import quaternion_distance
from utils import merge_inputs

from clcnet.models import ConfLCCNet, GatedIterativeCalibrator
from clcnet.utils.geometry import project_batch_after_rt
from train import build_depth_batch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_folder", default=str(DATA_ROOT))
    p.add_argument("--weights", nargs="+", required=True, help="one checkpoint or ordered cascade")
    p.add_argument("--baseline_weights", nargs="+", help="matching plain LCCNet weights for a three-way comparison")
    p.add_argument("--no_gate", action="store_true", help="accept every CLCNet proposal")
    p.add_argument("--depth_mode", choices=["lccnet", "direct"], help="override checkpoint preprocessing")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_r", type=float)
    p.add_argument("--max_t", type=float)
    p.add_argument("--max_depth", type=float)
    p.add_argument("--eval_iters", type=int, help="repeats for one model; a cascade always uses each stage once")
    p.add_argument("--conf_threshold", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpus", default="0")
    args = p.parse_args()
    if args.baseline_weights and (args.no_gate or len(args.weights) != len(args.baseline_weights)):
        p.error("Comparison requires one LCCNet checkpoint per CLCNet checkpoint and runs both gate settings")
    if args.eval_iters is not None and (args.eval_iters < 1 or
            (len(args.weights) > 1 and args.eval_iters != len(args.weights))):
        p.error("--eval_iters must be positive and must match the number of stages in a cascade")
    if not 0 <= args.conf_threshold <= 1:
        p.error("--conf_threshold must be in [0, 1]")
    return args


def load_stages(paths, device):
    models, modes = [], []
    first_kind = None
    for path in paths:
        wrapper = ConfLCCNet(image_size=(256, 512), use_feat_from=1, md=4)
        is_clcnet, saved, iters = wrapper.load_checkpoint(path)
        if first_kind is None:
            first_kind, first_args, first_iters = is_clcnet, saved, iters
        if is_clcnet != first_kind:
            raise ValueError("Every stage of a cascade must use the same model family")
        model = wrapper if is_clcnet else wrapper.backbone
        models.append(model.to(device).eval())
        modes.append(saved.get("depth_mode", "direct" if is_clcnet else "lccnet"))
        print(f"Loaded {'CLCNet' if is_clcnet else 'LCCNet'} stage {len(models)}: {path}")
    return models, first_kind, first_args, first_iters, modes


def describe_errors(errors):
    errors = torch.cat(errors).numpy()
    tail_count = max(1, int(np.ceil(0.1 * len(errors))))
    tail = np.partition(errors, len(errors) - tail_count)[-tail_count:]
    return (f"mean={errors.mean():.3f} median={np.median(errors):.3f} "
            f"P90={np.percentile(errors, 90):.3f} worst10_mean={tail.mean():.3f}")


@torch.no_grad()
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, is_clcnet, saved, saved_iters, modes = load_stages(args.weights, device)
    cascade = len(models) > 1
    iterations = len(models) if cascade else (args.eval_iters or saved_iters or (3 if is_clcnet else 1))
    for name, default in (("max_r", 20. if cascade else 1.), ("max_t", 1.5 if cascade else 0.2),
                          ("max_depth", 80.)):
        if getattr(args, name) is None:
            setattr(args, name, saved.get(name, default))

    if args.baseline_weights:
        if not is_clcnet:
            raise ValueError("--weights must contain CLCNet checkpoints for a comparison")
        baseline, baseline_is_clcnet, _, _, _ = load_stages(args.baseline_weights, device)
        if baseline_is_clcnet:
            raise ValueError("--baseline_weights requires plain LCCNet checkpoints")
        runs = [("LCCNet", baseline, False), ("CLCNet gate off", models, False),
                ("CLCNet gate on", models, True)]
        depth_mode = args.depth_mode or "lccnet"
    else:
        if args.depth_mode is None and len(set(modes)) > 1:
            raise ValueError("Cascade has mixed depth modes; specify --depth_mode explicitly")
        depth_mode = args.depth_mode or modes[0]
        gate = is_clcnet and not args.no_gate
        label = "LCCNet"
        if is_clcnet:
            label = "CLCNet gate on" if gate else "CLCNet gate off"
        runs = [(label, models, gate)]

    val_set = DatasetVoD(args.data_folder, max_r=args.max_r, max_t=args.max_t, split="val")
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=merge_inputs)
    print(f"VoD val frames={len(val_set)} max_r={args.max_r}deg max_t={args.max_t}m "
          f"max_depth={args.max_depth} depth={depth_mode} steps={iterations} threshold={args.conf_threshold}")
    calibrators = {name: GatedIterativeCalibrator(stages[0], iterations, args.conf_threshold)
                   for name, stages, gate in runs}
    results = {name: {"t": [], "r": [], "steps": []} for name, _, _ in runs}
    for sample in loader:
        gt_t, gt_q = sample["tr_error"].to(device), sample["rot_error"].to(device)
        rgb, depth, pcs, shapes, calibs = build_depth_batch(
            sample, (384, 1280), args.max_r, args.max_depth, device, depth_mode=depth_mode
        )
        def reproject(acc):
            return project_batch_after_rt(pcs, calibs, shapes, acc, args.max_depth,
                                          device, rgb.shape[-2:], depth_mode)

        for name, stages, gate in runs:
            result = calibrators[name].refine_eval(
                rgb, depth, reproject, stage_models=stages if cascade else None, gate_enabled=gate
            )
            values = results[name]
            values["t"].append((100 * (gt_t - result["transl"]).norm(dim=1)).cpu())
            values["r"].append(torch.rad2deg(quaternion_distance(gt_q, result["rot"], device)).cpu())
            values["steps"].append(result["steps_used"].cpu())

    for name, _, _ in runs:
        values = results[name]
        steps = torch.cat(values["steps"]).numpy()
        print(f"===== {name} =====")
        print(f"translation (cm) {describe_errors(values['t'])}")
        print(f"rotation (deg)   {describe_errors(values['r'])}")
        print(f"accepted steps mean={steps.mean():.3f} hist={np.bincount(steps, minlength=iterations+1)}")


if __name__ == "__main__":
    main()

