"""LCCNet backbone + confidence head."""

from __future__ import annotations

import torch
import torch.nn as nn

from clcnet.paths import ensure_lccnet_on_path

ensure_lccnet_on_path()
from models.LCCNet import LCCNet  # noqa: E402


class _CaptureLinear(nn.Module):
    """Linear wrapper that stores last output (DataParallel-safe, no forward hooks)."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.inner = linear
        self.last_out = None

    def forward(self, x):
        self.last_out = self.inner(x)
        return self.last_out


class ConfLCCNet(nn.Module):
    """
    Confidence-gated LCCNet (CLCNet).

    Dual-encoder + cost-volume trunk (LCCNet), plus a scalar confidence in (0, 1)
    that scores whether the current prediction should update the extrinsic.
    """

    def __init__(
        self,
        image_size=(256, 512),
        use_feat_from=1,
        md=4,
        use_reflectance=False,
        dropout=0.0,
        res_num=18,
    ):
        super().__init__()
        self.backbone = LCCNet(
            image_size=image_size,
            use_feat_from=use_feat_from,
            md=md,
            use_reflectance=use_reflectance,
            dropout=dropout,
            Action_Func="leakyrelu",
            attention=False,
            res_num=res_num,
        )
        # Avoid register_forward_hook: races / wrong-device under nn.DataParallel.
        if not isinstance(self.backbone.fc1, _CaptureLinear):
            self.backbone.fc1 = _CaptureLinear(self.backbone.fc1)

        # Confidence head: start near ~0.73 so soft-gate keeps warm-started residuals early.
        self.fc1_conf = nn.Linear(512, 256)
        self.fc2_conf = nn.Linear(256, 1)
        nn.init.xavier_normal_(self.fc1_conf.weight, gain=0.1)
        nn.init.zeros_(self.fc1_conf.bias)
        nn.init.zeros_(self.fc2_conf.weight)
        nn.init.constant_(self.fc2_conf.bias, 1.0)

    def _remap_backbone_key(self, key: str) -> str:
        # Plain LCCNet: fc1.weight → wrapped: fc1.inner.weight
        if key.startswith("fc1.") and not key.startswith("fc1.inner."):
            if not key.startswith("fc1_"):
                return "fc1.inner." + key[len("fc1.") :]
        return key

    def load_lccnet_weights(self, state_dict, strict_backbone=False):
        cleaned = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith("module.") else k
            if key.startswith("backbone."):
                key = key[len("backbone.") :]
            key = self._remap_backbone_key(key)
            cleaned[key] = v
        return self.backbone.load_state_dict(cleaned, strict=strict_backbone)

    def load_state_dict(self, state_dict, strict=True):
        # Accept both wrapped (fc1.inner.*) and unwrapped (fc1.weight) checkpoints.
        remapped = {}
        for k, v in state_dict.items():
            key = k
            if key.startswith("module."):
                key = key[7:]
            if key.startswith("backbone.fc1.") and not key.startswith("backbone.fc1.inner."):
                key = "backbone.fc1.inner." + key[len("backbone.fc1.") :]
            remapped[key] = v
        return super().load_state_dict(remapped, strict=strict)

    def load_checkpoint(self, path):
        """Load full CLCNet or LCCNet weights and return small evaluation metadata."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch before weights_only
            ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        keys = [k[7:] if k.startswith("module.") else k for k in state]
        is_clcnet = any(k.startswith(("backbone.", "fc1_conf.")) for k in keys)
        if is_clcnet:
            self.load_state_dict(state, strict=True)
        else:
            self.load_lccnet_weights(state, strict_backbone=True)
        return is_clcnet, ckpt.get("args") or ckpt.get("config") or {}, ckpt.get("eval_iters")

    def forward(self, rgb, lidar, return_feat=False):
        transl, rot = self.backbone(rgb, lidar)
        x = getattr(self.backbone.fc1, "last_out", None)
        if x is None:
            raise RuntimeError("Failed to capture LCCNet embedding for confidence head")
        # Guard against rare device skew under DataParallel.
        if x.device != self.fc1_conf.weight.device:
            x = x.to(self.fc1_conf.weight.device, non_blocking=True)
        x = self.backbone.leakyRELU(x)
        conf = torch.sigmoid(self.fc2_conf(self.backbone.leakyRELU(self.fc1_conf(x))))
        if return_feat:
            return transl, rot, conf, x
        return transl, rot, conf
