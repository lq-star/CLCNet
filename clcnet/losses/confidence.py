"""Pose + confidence losses for CLCNet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from clcnet.paths import ensure_lccnet_on_path

ensure_lccnet_on_path()
from quaternion_distances import quaternion_distance  # noqa: E402
from utils import quat2mat, rotate_forward, tvector2mat  # noqa: E402


def pose_error_target_conf(pred_transl, pred_rot, gt_transl, gt_rot, t_scale=0.2, r_scale_deg=1.0):
    t_err = torch.norm(pred_transl - gt_transl, dim=1) / max(t_scale, 1e-6)
    r_err = quaternion_distance(pred_rot, gt_rot, pred_rot.device) * 180.0 / math.pi
    r_err = r_err / max(r_scale_deg, 1e-6)
    e = 0.5 * (t_err + r_err)
    return torch.exp(-e).unsqueeze(1)


def _points_distance_loss(point_clouds, target_transl, target_rot, transl_err, rot_err, max_points=8192):
    """Same math as LCCNet CombinedLoss, with optional GPU subsample for speed."""
    device = transl_err.device
    total = torch.zeros((), device=device, dtype=transl_err.dtype)
    B = target_transl.shape[0]
    for i in range(len(point_clouds)):
        pc = point_clouds[i].to(device, non_blocking=True)
        if pc.shape[0] != 4 and pc.shape[1] == 4:
            pc = pc.t()
        n = pc.shape[1]
        if max_points and n > max_points:
            sel = torch.randint(0, n, (max_points,), device=device)
            pc = pc[:, sel]
        RT_target = tvector2mat(target_transl[i]) @ quat2mat(target_rot[i])
        RT_pred = tvector2mat(transl_err[i]) @ quat2mat(rot_err[i])
        RT_total = RT_target.inverse() @ RT_pred
        pc_out = rotate_forward(pc, RT_total)
        total = total + (pc_out - pc).norm(dim=0).mean()
    return total / max(B, 1)


class ConfGatedCombinedLoss(nn.Module):
    def __init__(
        self,
        rescale_trans=1.0,
        rescale_rot=1.0,
        weight_point_cloud=0.5,
        lambda_conf=0.2,
        lambda_gate=0.05,
        t_scale=0.2,
        r_scale_deg=1.0,
        max_points=0,
    ):
        super().__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.weight_point_cloud = weight_point_cloud
        self.lambda_conf = lambda_conf
        self.lambda_gate = lambda_gate
        self.t_scale = t_scale
        self.r_scale_deg = r_scale_deg
        # 0 / negative => full cloud (match official LCCNet CombinedLoss).
        self.max_points = max_points
        self.transl_loss = nn.SmoothL1Loss(reduction="none")
        self.bce = nn.BCELoss(reduction="mean")
        self.losses = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err, conf):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean() if self.rescale_trans else transl_err.new_zeros(())
        loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean() if self.rescale_rot else transl_err.new_zeros(())
        pose_loss = self.rescale_trans * loss_transl + self.rescale_rot * loss_rot
        pc_loss = _points_distance_loss(
            point_clouds,
            target_transl,
            target_rot,
            transl_err,
            rot_err,
            max_points=self.max_points if self.max_points and self.max_points > 0 else None,
        )
        pose_total = (1.0 - self.weight_point_cloud) * pose_loss + self.weight_point_cloud * pc_loss

        with torch.no_grad():
            conf_tgt = pose_error_target_conf(
                transl_err.detach(),
                rot_err.detach(),
                target_transl,
                target_rot,
                t_scale=self.t_scale,
                r_scale_deg=self.r_scale_deg,
            )
        conf = conf.clamp(1e-4, 1.0 - 1e-4)
        conf_loss = self.bce(conf, conf_tgt)

        transl_mag = torch.norm(transl_err, dim=1, keepdim=True)
        rot_id = torch.zeros_like(rot_err)
        rot_id[:, 0] = 1.0
        rot_mag = quaternion_distance(rot_err, rot_id, rot_err.device).unsqueeze(1)
        gate_pen = ((1.0 - conf) * (transl_mag / self.t_scale + rot_mag)).mean()

        total = pose_total + self.lambda_conf * conf_loss + self.lambda_gate * gate_pen
        self.losses = {
            "total_loss": total,
            "transl_loss": loss_transl,
            "rot_loss": loss_rot,
            "point_clouds_loss": pc_loss,
            "conf_loss": conf_loss,
            "gate_pen": gate_pen,
            "conf_mean": conf.mean(),
            "conf_tgt_mean": conf_tgt.mean(),
        }
        return self.losses
