import math
import unittest

import torch
import torch.nn.functional as F

from clcnet.losses.confidence import pose_error_target_conf, update_gain
from clcnet.models.gated_refine import (
    GatedIterativeCalibrator,
    compose_delta,
    soft_gate_pose,
)
from clcnet.utils.geometry import lidar_project_depth, network_depth, project_batch_after_rt


class RefinementTest(unittest.TestCase):
    def test_confidence_labels_measure_improvement(self):
        gt = torch.tensor([[0.02, 0., 0.]]).expand(4, -1)
        pred = torch.tensor([[0.02, 0., 0.], [-0.02, 0., 0.], [0., 0., 0.], [0.01, 0., 0.]])
        q = torch.tensor([[1., 0., 0., 0.]]).expand(4, -1)
        labels = pose_error_target_conf(pred, -q, gt, q, 1.5, 20.)
        self.assertEqual(labels.flatten().tolist(), [1, 0, 0, 1])
        torch.testing.assert_close(update_gain(pred, q, gt, q, 1.5, 20.).flatten(),
                                   torch.tensor([1., -0.5, 0., 0.5]))
        rotated = torch.tensor([[0.8, 0., 0.6, 0.]])
        origin = torch.zeros(1, 3)
        self.assertEqual(pose_error_target_conf(origin, -rotated, origin, rotated).item(), 1)
        self.assertEqual(pose_error_target_conf(origin, rotated, origin, q[:1]).item(), 0)

    def test_canvas_depth_matches_last_point_then_resize(self):
        K = torch.eye(3)
        # Both project to (2,2); the later return is beyond the 80 m divisor.
        cloud = torch.tensor([[10., 200.], [10., 200.], [5., 100.], [1., 1.]])
        canvas = torch.zeros(1, 1, 8, 8)
        canvas[0, 0, 2, 2] = 100. / 80.
        expected = F.interpolate(canvas, size=(4, 4), mode="bilinear", align_corners=False)
        torch.testing.assert_close(network_depth(cloud, K, (8, 8), 80., (4, 4), "lccnet"), expected[0])
        reproj = project_batch_after_rt([cloud], [K], [(8, 8)], torch.eye(4)[None], 80., cloud.device,
                                        (4, 4), "lccnet")
        torch.testing.assert_close(reproj, expected)

    def test_cascade_rejection_leaves_next_specialist_active(self):
        class Specialist(torch.nn.Module):
            def __init__(self, t, conf):
                super().__init__()
                self.t, self.conf = t, conf

            def forward(self, rgb, depth):
                return torch.tensor([[self.t, 0., 0.]]), torch.tensor([[1., 0., 0., 0.]]), torch.tensor([[self.conf]])

        stages = [Specialist(1., 0.), Specialist(0.2, 1.)]
        image = torch.zeros(1, 1, 8, 8)
        calibrator = GatedIterativeCalibrator(stages[0], 2)
        gated = calibrator.refine_eval(image, image, lambda _: image, stages)
        ungated = calibrator.refine_eval(image, image, lambda _: image, stages, gate_enabled=False)
        self.assertEqual(gated["steps_used"].item(), 1)
        self.assertEqual(ungated["steps_used"].item(), 2)
        torch.testing.assert_close(gated["transl"], torch.tensor([[0.2, 0., 0.]]))
        torch.testing.assert_close(ungated["transl"], torch.tensor([[1.2, 0., 0.]]))

    def test_two_noncommuting_updates_and_rejected_sample(self):
        s, c = math.sin(math.pi / 8), math.cos(math.pi / 8)
        t1, q1 = torch.tensor([[0.4, 0., 0.]]), torch.tensor([[c, 0., 0., s]])
        t2, q2 = torch.tensor([[0., 0.2, 0.]]), torch.tensor([[c, 0., s, 0.]])
        first, second = compose_delta(t1, q1), compose_delta(t2, q2)
        target = first @ second
        self.assertFalse(torch.allclose(target, second @ first))
        pc = torch.tensor([[0.12, 0.8], [0.16, 0.9], [5., 6.], [1., 1.]])
        K = torch.tensor([[8., 0., 8.], [0., 8., 8.], [0., 0., 1.]])
        pc_mis = target[0] @ pc

        class TwoSteps(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.outputs = iter([(t1, q1, [1., 0.]), (t2, q2, [1., 1.])])

            def forward(self, rgb, depth):
                t, q, confidence = next(self.outputs)
                return t.expand(2, -1), q.expand(2, -1), torch.tensor(confidence)[:, None]

        def reproject(acc):
            return project_batch_after_rt([pc_mis] * 2, [K] * 2, [(16, 16)] * 2,
                                          acc, 80., pc.device)

        initial = reproject(torch.eye(4).expand(2, -1, -1))
        result = GatedIterativeCalibrator(TwoSteps(), max_iters=2).refine_eval(initial, initial, reproject)
        torch.testing.assert_close(result["RT"][0], target[0])
        torch.testing.assert_close(result["RT"][1], torch.eye(4))
        self.assertEqual(result["steps_used"].tolist(), [2, 0])
        self.assertEqual(result["final_conf"].flatten().tolist(), [1., 0.])
        torch.testing.assert_close(reproject(result["RT"])[0], lidar_project_depth(pc, K, (16, 16)) / 80.)

    def test_soft_gate_takes_same_rotation_for_both_quaternion_signs(self):
        q = torch.tensor([[0.8, 0., 0.6, 0.]])
        t = torch.tensor([[1., 0., 0.]])
        confidence = torch.tensor([[0.5]])
        pos = compose_delta(*soft_gate_pose(t, q, confidence))
        neg = compose_delta(*soft_gate_pose(t, -q, confidence))
        torch.testing.assert_close(pos, neg)


if __name__ == "__main__":
    unittest.main()
