"""Pose + confidence losses for CLCNet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from clcnet.paths import ensure_lccnet_on_path

ensure_lccnet_on_path()
from quaternion_distances import quaternion_distance  # noqa: E402
from utils import quat2mat, rotate_forward, tvector2mat  # noqa: E402


@torch.no_grad()
def update_gain(pred_transl, pred_rot, gt_transl, gt_rot, t_scale, r_scale_deg):
    """Relative reduction of the normalized local pose error after a full update."""
    identity = torch.zeros_like(gt_rot)
    identity[:, 0] = 1
    t_scale = max(t_scale, 1e-6)
    r_scale = max(math.radians(r_scale_deg), 1e-6)
    before = gt_transl.norm(dim=1) / t_scale
    before += quaternion_distance(identity, gt_rot, gt_rot.device) / r_scale
    # inv(pred) @ GT has translation norm ||gt_t - pred_t||.
    after = (gt_transl - pred_transl).norm(dim=1) / t_scale
    after += quaternion_distance(pred_rot, gt_rot, gt_rot.device) / r_scale
    return ((before - after) / torch.maximum(before, after).clamp_min(1e-6)).unsqueeze(1)


def pose_error_target_conf(pred_transl, pred_rot, gt_transl, gt_rot, t_scale=0.2, r_scale_deg=1.0):
    gain = update_gain(pred_transl, pred_rot, gt_transl, gt_rot, t_scale, r_scale_deg)
    return (gain > 1e-6).to(pred_transl.dtype)


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

    def pose_terms(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
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
        return pose_total, loss_transl, loss_rot, pc_loss

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err, conf):
        pose_total, loss_transl, loss_rot, pc_loss = self.pose_terms(
            point_clouds, target_transl, target_rot, transl_err, rot_err
        )
        gain = update_gain(transl_err, rot_err, target_transl, target_rot, self.t_scale, self.r_scale_deg)
        conf_tgt = (gain > 1e-6).to(conf.dtype)
        conf = conf.clamp(1e-4, 1.0 - 1e-4)
        conf_loss = self.bce(conf, conf_tgt)

        # Penalize rejecting a helpful update or accepting a harmful one.
        gate_pen = ((1.0 - conf) * gain.clamp_min(0) + conf * (-gain).clamp_min(0)).mean()

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
