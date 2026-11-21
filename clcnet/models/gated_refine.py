"""Confidence-gated iterative extrinsic refinement."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from clcnet.paths import ensure_lccnet_on_path

ensure_lccnet_on_path()
from utils import quat2mat, quaternion_from_matrix, tvector2mat  # noqa: E402


def compose_delta(transl, rot):
    mats = []
    for i in range(transl.shape[0]):
        R = quat2mat(rot[i])
        T = tvector2mat(transl[i])
        mats.append(torch.mm(T, R))
    return torch.stack(mats, dim=0)


def mat_to_transl_quat(mat):
    B = mat.shape[0]
    transl = mat[:, :3, 3]
    quats = [quaternion_from_matrix(mat[i]) for i in range(B)]
    return transl, torch.stack(quats, dim=0)


def append_update(acc, delta):
    # Reprojection applies inverse(acc) to the original cloud. The next
    # correction acts on that cloud as inverse(delta) @ inverse(acc).
    return acc.bmm(delta)


def soft_gate_pose(transl, rot, conf, identity_rot=None):
    if identity_rot is None:
        identity_rot = torch.zeros_like(rot)
        identity_rot[:, 0] = 1.0
    conf = conf.view(-1, 1)
    rot = torch.where((rot * identity_rot).sum(1, keepdim=True) < 0, -rot, rot)
    transl_g = conf * transl
    rot_g = F.normalize((1.0 - conf) * identity_rot + conf * rot, dim=1)
    return transl_g, rot_g


class GatedIterativeCalibrator:
    """Hard gate at eval, soft gate at train."""

    def __init__(self, model, max_iters: int = 3, conf_threshold: float = 0.5, soft_gate: bool = True):
        self.model = model
        if max_iters < 1:
            raise ValueError("max_iters must be at least one")
        self.max_iters = max_iters
        self.conf_threshold = conf_threshold
        self.soft_gate = soft_gate

    @torch.no_grad()
    def refine_eval(self, rgb, depth0, reproject_fn, stage_models=None, gate_enabled=True) -> Dict:
        """Repeat one model, or try each supplied specialist once in order."""
        if stage_models is not None and len(stage_models) != self.max_iters:
            raise ValueError("Supply one model per cascade stage")
        models = stage_models if stage_models is not None else [self.model] * self.max_iters
        B = rgb.shape[0]
        device = rgb.device
        acc = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        depth = depth0
        conf_hist: List[torch.Tensor] = []
        updated = torch.ones(B, dtype=torch.bool, device=device)
        steps_used = torch.zeros(B, dtype=torch.long, device=device)
        first_rt = acc

        final_conf = torch.zeros(B, 1, device=device)
        for it, model in enumerate(models):
            if model.training:
                model.eval()
            outputs = model(rgb, depth)
            transl, rot = outputs[:2]
            gate = updated
            if len(outputs) == 3:
                conf = outputs[2]
                conf_hist.append(conf.detach())
                final_conf = torch.where(updated[:, None], conf, final_conf)
                if gate_enabled:
                    gate = updated & (conf.view(-1) >= self.conf_threshold)
            elif gate_enabled:
                raise ValueError("Confidence gating requires a CLCNet checkpoint")
            if not gate.any():
                if stage_models is not None:
                    continue
                break
            delta = compose_delta(transl, rot)
            eye = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            delta_masked = torch.where(gate.view(B, 1, 1), delta, eye)
            acc = append_update(acc, delta_masked)
            if it == 0:
                first_rt = acc
            steps_used = steps_used + gate.long()
            # A rejection stops repeated use of this model. A different,
            # finer-range specialist can still improve the same frame.
            if stage_models is None:
                updated = gate
            if it + 1 < self.max_iters:
                depth = reproject_fn(acc)

        transl_out, rot_out = mat_to_transl_quat(acc)
        return {
            "transl": transl_out,
            "rot": rot_out,
            "RT": acc,
            "first_RT": first_rt,
            "conf_hist": conf_hist,
            "steps_used": steps_used,
            "final_conf": final_conf,
        }

    def refine_train_step(self, rgb, depth0, reproject_fn) -> Dict:
        self.model.train()
        B = rgb.shape[0]
        device = rgb.device
        acc = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        depth = depth0
        preds = []
        for it in range(self.max_iters):
            transl, rot, conf = self.model(rgb, depth)
            transl_g, rot_g = soft_gate_pose(transl, rot, conf)
            delta = compose_delta(transl_g, rot_g)
            acc = append_update(acc, delta)
            preds.append(
                {"transl": transl, "rot": rot, "conf": conf, "transl_g": transl_g, "rot_g": rot_g}
            )
            if it + 1 < self.max_iters:
                depth = reproject_fn(acc)
        transl_out, rot_out = mat_to_transl_quat(acc)
        return {
            "transl": transl_out,
            "rot": rot_out,
            "RT": acc,
            "steps": preds,
            "final_conf": preds[-1]["conf"],
        }
