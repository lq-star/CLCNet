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


def soft_gate_pose(transl, rot, conf, identity_rot=None):
    if identity_rot is None:
        identity_rot = torch.zeros_like(rot)
        identity_rot[:, 0] = 1.0
    conf = conf.view(-1, 1)
    transl_g = conf * transl
    rot_g = F.normalize((1.0 - conf) * identity_rot + conf * rot, dim=1)
    return transl_g, rot_g


class GatedIterativeCalibrator:
    """Hard gate at eval, soft gate at train."""

    def __init__(self, model, max_iters: int = 3, conf_threshold: float = 0.5, soft_gate: bool = True):
        self.model = model
        self.max_iters = max_iters
        self.conf_threshold = conf_threshold
        self.soft_gate = soft_gate

    @torch.no_grad()
    def refine_eval(self, rgb, depth0, reproject_fn) -> Dict:
        self.model.eval()
        B = rgb.shape[0]
        device = rgb.device
        acc = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        depth = depth0
        conf_hist: List[torch.Tensor] = []
        updated = torch.ones(B, dtype=torch.bool, device=device)
        steps_used = torch.zeros(B, dtype=torch.long, device=device)

        for _ in range(self.max_iters):
            transl, rot, conf = self.model(rgb, depth)
            conf_hist.append(conf.detach())
            gate = (conf.view(-1) >= self.conf_threshold) & updated
            if not gate.any():
                break
            delta = compose_delta(transl, rot)
            eye = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            delta_masked = torch.where(gate.view(B, 1, 1), delta, eye)
            acc = delta_masked.bmm(acc)
            steps_used = steps_used + gate.long()
            updated = updated & gate
            depth = reproject_fn(acc)

        transl_out, rot_out = mat_to_transl_quat(acc)
        return {
            "transl": transl_out,
            "rot": rot_out,
            "RT": acc,
            "conf_hist": conf_hist,
            "steps_used": steps_used,
            "final_conf": conf_hist[-1] if conf_hist else torch.zeros(B, 1, device=device),
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
            acc = delta.bmm(acc)
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
