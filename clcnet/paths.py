"""Repository path helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# CLCNet/ (method root)
METHOD_ROOT = Path(__file__).resolve().parents[1]
# workspace: /home/abc/liu/calib
WORKSPACE_ROOT = METHOD_ROOT.parent
LCCNET_ROOT = WORKSPACE_ROOT / "LCCNet"
DATA_ROOT = WORKSPACE_ROOT / "data" / "VoD"
OUTPUT_ROOT = METHOD_ROOT / "outputs"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
LOG_ROOT = OUTPUT_ROOT / "logs"
LCCNET_QUALITY = WORKSPACE_ROOT / "LCCNet" / "pretrained" / "vod_quality"


def ensure_lccnet_on_path() -> Path:
    """Put baseline LCCNet on sys.path for DatasetVoD / utils / models.LCCNet."""
    root = str(LCCNET_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return LCCNET_ROOT


def ensure_method_on_path() -> Path:
    """Allow `import clcnet...` from tools/ scripts."""
    root = str(METHOD_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return METHOD_ROOT
