"""Projection helpers (GPU-native, matching LCCNet last-write depth)."""

from __future__ import annotations

import torch

from clcnet.paths import ensure_lccnet_on_path

ensure_lccnet_on_path()
from utils import quat2mat, rotate_back, rotate_forward, tvector2mat  # noqa: E402


def lidar_project_depth(pc, cam_calib, img_shape, max_depth=80.0):
    """Project LiDAR to depth image on the same device as ``pc`` (no CPU roundtrip)."""
    if not isinstance(pc, torch.Tensor):
        pc = torch.as_tensor(pc, dtype=torch.float32)
    pc3 = pc[:3, :]
    device = pc3.device
    dtype = pc3.dtype

    if isinstance(cam_calib, torch.Tensor):
        K = cam_calib.to(device=device, dtype=dtype)
    else:
        K = torch.as_tensor(cam_calib, device=device, dtype=dtype)

    uvz = K @ pc3
    z = uvz[2]
    inv_z = 1.0 / (z + 1e-10)
    u = uvz[0] * inv_z
    v = uvz[1] * inv_z

    h, w = int(img_shape[0]), int(img_shape[1])
    mask = (u > 0) & (u < w) & (v > 0) & (v < h) & (z > 0) & (z < max_depth)
    depth = torch.zeros(h, w, device=device, dtype=dtype)
    if mask.any():
        # Same semantics as numpy: last write wins for colliding pixels.
        depth[v[mask].long(), u[mask].long()] = z[mask]
    return depth.unsqueeze(0)


def miscalibrate_cloud(pc, rot_quat, transl, device):
    """Match LCCNet train: pc_rotated = rotate_back(pc, T*R)."""
    pc = pc.to(device, non_blocking=True)
    q = rot_quat.to(device=device, dtype=torch.float32)
    t = transl.to(device=device, dtype=torch.float32)
    RT = tvector2mat(t) @ quat2mat(q)
    return rotate_back(pc, RT)


def project_batch_after_rt(point_clouds, calibs, shapes, RT, max_depth, device):
    depths = []
    for i, pc in enumerate(point_clouds):
        pc = pc.to(device, non_blocking=True)
        pc_corr = rotate_forward(pc, RT[i])
        depth = lidar_project_depth(pc_corr, calibs[i], shapes[i], max_depth=max_depth)
        depths.append(depth / max_depth)
    return torch.stack(depths, dim=0)
