import unittest

import torch
from monai.data.meta_tensor import MetaTensor

from linguine.prompt_selectors.metrics import ClickValidityMetrics


class TestClickValidityMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = ClickValidityMetrics()

    def test_initialization(self):
        """Test that metrics are properly initialized"""
        self.assertEqual(self.metrics.correctly_kept, 0)
        self.assertEqual(self.metrics.falsely_kept, 0)
        self.assertEqual(self.metrics.out_of_bounds, 0)
        self.assertEqual(len(self.metrics.distances), 0)

    def test_perfect_metrics(self):
        """Test metrics for a perfect click propagation"""
        self.metrics.update_perfect()
        self.assertEqual(len(self.metrics.valid_click_per_lesion_log), 1)
        self.assertEqual(len(self.metrics.invalid_click_per_lesion_log), 1)
        self.assertEqual(self.metrics.valid_click_per_lesion_log[0], 1)
        self.assertEqual(self.metrics.invalid_click_per_lesion_log[0], 0)
        self.assertEqual(self.metrics.distances[0], 0.0)

    def test_metrics_computation(self):
        """Test computation of accuracy"""
        # Simulate some metrics data
        self.metrics.correctly_kept = 5  # True positives
        self.metrics.falsely_kept = 2  # False positives

        # Test accuracy
        self.assertAlmostEqual(self.metrics._compute_accuracy(), 5 / 7)

    def test_update_with_clicks(self):
        """Test updating metrics with filtered clicks"""
        # Create a simple target label
        target = torch.zeros((1, 1, 3, 3, 3))
        target[0, 0, 1, 1, 1] = 1  # One valid location
        target = MetaTensor(target)

        clicks = [(1, 1, 1), (0, 0, 0)]  # One valid, one invalid

        self.metrics.update(clicks=clicks, target_label=target)

        self.assertEqual(self.metrics.correctly_kept, 1)  # One valid
        self.assertEqual(self.metrics.falsely_kept, 1)  # One invalid

    def test_metrics_computation_edge_cases(self):
        """Test metric computation with edge cases (zero denominators)"""
        # All zeros - should return None
        self.assertIsNone(self.metrics._compute_accuracy())

        # Only correctly_kept
        self.metrics.correctly_kept = 5
        self.assertAlmostEqual(self.metrics._compute_accuracy(), 1.0)


if __name__ == "__main__":
    unittest.main()
