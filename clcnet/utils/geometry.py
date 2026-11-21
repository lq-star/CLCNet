"""Projection helpers (GPU-native, matching LCCNet last-write depth)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

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
    u, v = uvz[:2] / (z + 1e-10)

    h, w = int(img_shape[0]), int(img_shape[1])
    mask = (u > 0) & (u < w) & (v > 0) & (v < h) & (z > 0) & (z < max_depth)
    depth = torch.zeros(h, w, device=device, dtype=dtype)
    pixels = v[mask].long() * w + u[mask].long()
    # NumPy's CPU projector keeps the last input point per pixel. Repeated
    # indices in a single CUDA indexed write would race.
    order = torch.argsort(pixels, stable=True)
    pixels = pixels[order]
    last = torch.ones_like(pixels, dtype=torch.bool)
    last[:-1] = pixels[:-1] != pixels[1:]
    depth.view(-1)[pixels[last]] = z[mask][order[last]]
    return depth.unsqueeze(0)


def miscalibrate_cloud(pc, rot_quat, transl, device):
    """Match LCCNet train: pc_rotated = rotate_back(pc, T*R)."""
    pc = pc.to(device, non_blocking=True)
    q = rot_quat.to(device=device, dtype=torch.float32)
    t = transl.to(device=device, dtype=torch.float32)
    RT = tvector2mat(t) @ quat2mat(q)
    return rotate_back(pc, RT)


def network_depth(pc, K, shape, max_depth, net_hw, depth_mode):
    # LCCNet uses max_depth for normalization, including points beyond 80 m.
    z_limit = float("inf") if depth_mode == "lccnet" else max_depth
    depth = lidar_project_depth(pc, K, shape, z_limit) / max_depth
    if net_hw is not None and tuple(shape) != tuple(net_hw):
        depth = F.interpolate(depth[None], size=net_hw, mode="bilinear", align_corners=False)[0]
    return depth


def project_batch_after_rt(point_clouds, calibs, shapes, RT, max_depth, device,
                           net_hw=None, depth_mode="direct"):
    depths = []
    for i, pc in enumerate(point_clouds):
        pc = pc.to(device, non_blocking=True)
        pc_corr = rotate_forward(pc, RT[i])
        depths.append(network_depth(pc_corr, calibs[i], shapes[i], max_depth, net_hw, depth_mode))
    return torch.stack(depths, dim=0)
