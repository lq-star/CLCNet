import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

from tools import evaluate


class ConstantPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.translation = 0.

    def forward(self, rgb, depth):
        t = rgb.new_tensor([[self.translation, 0., 0.]]).expand(len(rgb), -1)
        q = rgb.new_tensor([[1., 0., 0., 0.]]).expand(len(rgb), -1)
        return t, q


class FixtureModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = ConstantPose()

    def load_checkpoint(self, name):
        stage = int(name[-1])
        self.backbone.translation = stage / 100.
        self.confidence = float(stage != 1)
        return name.startswith("clc"), {"max_r": 20., "max_t": 1.5, "depth_mode": "lccnet"}, 2

    def forward(self, rgb, depth):
        t, q = self.backbone(rgb, depth)
        return t, q, rgb.new_full((len(rgb), 1), self.confidence)


class ComparisonTest(unittest.TestCase):
    def test_five_stage_comparison_uses_all_weights_on_one_sample(self):
        sample = {
            "rgb": torch.zeros(3, 384, 1280),
            "calib": torch.eye(3),
            "tr_error": torch.tensor([0.15, 0., 0.]),
            "rot_error": torch.tensor([1., 0., 0., 0.]),
            "point_cloud": torch.tensor([[5.], [5.], [2.], [1.]]),
        }
        argv = ["evaluate", "--weights", *[f"clc{i}" for i in range(1, 6)],
                "--baseline_weights", *[f"base{i}" for i in range(1, 6)], "--num_workers", "0"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(evaluate, "ConfLCCNet", FixtureModel), \
             patch.object(evaluate, "DatasetVoD", return_value=[sample]) as dataset, \
             patch.object(torch.cuda, "is_available", return_value=False), redirect_stdout(output):
            evaluate.main()
        dataset.assert_called_once()
        text = output.getvalue()
        for label in ("LCCNet", "CLCNet gate off"):
            self.assertIn(f"===== {label} =====\ntranslation (cm) mean=0.000", text)
        self.assertIn("===== CLCNet gate on =====\ntranslation (cm) mean=1.000", text)
        self.assertEqual(text.count("accepted steps mean=5.000"), 2)
        self.assertIn("accepted steps mean=4.000", text)
