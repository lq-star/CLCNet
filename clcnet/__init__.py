"""CLCNet: Confidence-gated Lidar-Camera calibration Network."""

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy imports avoid circular path issues when only reading paths.
    if name == "ConfLCCNet":
        from .models.network import ConfLCCNet

        return ConfLCCNet
    if name == "GatedIterativeCalibrator":
        from .models.gated_refine import GatedIterativeCalibrator

        return GatedIterativeCalibrator
    raise AttributeError(name)


__all__ = ["ConfLCCNet", "GatedIterativeCalibrator"]
