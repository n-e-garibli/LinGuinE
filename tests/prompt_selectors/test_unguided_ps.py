import unittest
from unittest.mock import Mock

import torch
from monai.data.meta_tensor import MetaTensor

from linguine.constants import FILE_ID
from linguine.prompt_selectors.unguided_ps import PromptSelector, UnguidedModelPS


class TestUnguidedModelPS(unittest.TestCase):
    def setUp(self):
        self.mock_inferer = Mock()
        self.mock_inferer.spacing = (1.0, 1.0, 1.0)
        self.ps = UnguidedModelPS(inferer=self.mock_inferer)

    def test_initialization(self):
        """Test UnguidedModelPS initialization"""
        self.assertEqual(self.ps.inferer, self.mock_inferer)

    def test_get_pred_with_provided_predictions(self):
        """Test _get_pred with pre-computed predictions"""
        pred = torch.rand((5, 5, 5))
        result = self.ps._get_pred(target_img=None, target_dict=None, target_pred=pred)
        self.assertTrue(torch.equal(result, pred))

    def test_get_pred_with_image_inference(self):
        """Test _get_pred by running inference on image"""
        expected_pred = torch.rand((5, 5, 5))

        self.mock_inferer.infer.return_value = expected_pred

        target_dict = {"path": "test.nii", FILE_ID: "3"}
        result = self.ps._get_pred(target_img=MetaTensor(torch.rand((1, 1, 5, 5, 5))), target_dict=target_dict)

        self.mock_inferer.infer.assert_called_once()
        self.assertTrue(torch.equal(result, expected_pred))

    def test_get_pred_with_provided_image(self):
        """Test _get_pred with provided image tensor"""
        img_tensor = MetaTensor(torch.rand((1, 1, 5, 5, 5)))
        expected_pred = torch.rand((5, 5, 5))

        self.mock_inferer.infer.return_value = expected_pred

        result = self.ps._get_pred(target_dict=None, target_img=img_tensor)

        self.mock_inferer.infer.assert_called_once()
        self.assertTrue(torch.equal(result, expected_pred))

    def test_get_pred_assertion_errors(self):
        """Test assertion errors in _get_pred"""
        # No inferer
        ps_no_inferer = UnguidedModelPS()
        with self.assertRaises(AssertionError):
            ps_no_inferer._get_pred(None, None)

        # Wrong prediction dimensions
        bad_pred = torch.rand((5, 5))  # 2D instead of 3D
        with self.assertRaises(AssertionError):
            self.ps._get_pred(target_img=None, target_dict=None, target_pred=bad_pred)

    def test_get_best_clicks_ranking(self):
        """Test get_best_clicks with"""
        pred = torch.zeros((5, 5, 5))
        pred[1, 1, 1] = 0.3
        pred[2, 2, 2] = 0.8
        pred[3, 3, 3] = 0.6

        clicks = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
        result = self.ps.get_best_clicks(clicks, None, target_pred=pred, n_clicks=3)

        expected = [(2, 2, 2), (3, 3, 3), (1, 1, 1)]
        self.assertEqual(result, expected)

    def test_get_best_clicks_empty_input(self):
        """Test get_best_clicks with empty click list"""
        pred = torch.rand((5, 5, 5))
        result = self.ps.get_best_clicks([], None, target_pred=pred, n_clicks=2)
        self.assertEqual(result, [])

    def test_get_best_clicks_out_of_bounds_filtering(self):
        """Test that get_best_clicks filters out of bounds clicks"""
        pred = torch.ones((5, 5, 5))  # All high probabilities
        clicks = [(1, 1, 1), (10, 10, 10), (2, 2, 2)]  # One out of bounds

        result = self.ps.get_best_clicks(clicks, None, target_pred=pred, n_clicks=3)

        # Should exclude the out of bounds click, return last n after sorting
        expected = set([(1, 1, 1), (2, 2, 2)])  # Last 2 after sorting (same probability)
        self.assertEqual(set(result), expected)

    def test_get_best_clicks_n_clicks_limit(self):
        """Test get_best_clicks respects n_clicks limit"""
        pred = torch.zeros((5, 5, 5))
        pred[1, 1, 1] = 0.6  # Lowest
        pred[2, 2, 2] = 0.75  # Second lowest
        pred[3, 3, 3] = 0.8  # Second highest
        pred[4, 4, 4] = 0.9  # Highest

        clicks = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
        result = self.ps.get_best_clicks(clicks, None, target_pred=pred, n_clicks=2)

        # Should return only top 2 clicks (last 2 after sorting)
        self.assertEqual(len(result), 2)
        expected = [(4, 4, 4), (3, 3, 3)]  # These are the top to from highest prob to lowest
        self.assertEqual(result, expected)

    def test_get_best_clicks_with_inference(self):
        """Test get_best_clicks using inference instead of pre-computed predictions"""
        img_tensor = MetaTensor(torch.rand((1, 1, 5, 5, 5)))
        pred = torch.zeros((5, 5, 5))
        pred[1, 1, 1] = 0.3
        pred[2, 2, 2] = 0.8

        self.mock_inferer.infer.return_value = pred

        clicks = [(1, 1, 1), (2, 2, 2)]
        target_dict = {"path": "test.nii", FILE_ID: "2"}

        result = self.ps.get_best_clicks(
            target_clicks=clicks, target_img=img_tensor, target_dict=target_dict, n_clicks=2
        )
        expected = [(2, 2, 2), (1, 1, 1)]
        self.assertEqual(result, expected)

    def test_inheritance_from_base(self):
        """Test that UnguidedModelPS inherits from PromptSelector"""
        self.assertIsInstance(self.ps, PromptSelector)

        # Test static method access
        clicks = [(1, 1, 1), (10, 10, 10)]
        shape = (5, 5, 5)
        result = self.ps.filter_out_of_bounds_clicks(clicks, shape)
        self.assertEqual(result, [(1, 1, 1)])

    def test_edge_cases(self):
        """Test various edge cases"""
        # Test with single click
        clicks = [(1, 1, 1)]
        pred = torch.zeros((5, 5, 5))
        pred[1, 1, 1] = 0.8

        result = self.ps.get_best_clicks(clicks, None, target_pred=pred, n_clicks=1)
        self.assertEqual(result, [(1, 1, 1)])

    def test_tiny_cache_stores_prediction(self):
        """Test that tiny cache stores prediction for non-ROI based inferers"""
        # Setup non-ROI based inferer
        self.ps.inferer_is_roi_based = False

        img_tensor = MetaTensor(torch.rand((1, 1, 5, 5, 5)))
        pred = torch.rand((5, 5, 5))

        self.mock_inferer.infer.return_value = pred

        target_dict = {FILE_ID: "test_file_1"}

        # First call should run inference and cache result
        result1 = self.ps._get_pred(img_tensor, target_dict)
        self.mock_inferer.infer.assert_called_once()
        self.assertTrue(torch.equal(result1, pred))
        self.assertIn("test_file_1", self.ps._tiny_cache)
        self.assertTrue(torch.equal(self.ps._tiny_cache["test_file_1"], pred))

        # Second call should use cache
        self.mock_inferer.reset_mock()
        result2 = self.ps._get_pred(img_tensor, target_dict)
        self.mock_inferer.infer.assert_not_called()
        self.assertTrue(torch.equal(result2, pred))

    def test_tiny_cache_clears_on_new_file(self):
        """Test that tiny cache only stores one prediction at a time"""
        # Setup non-ROI based inferer
        self.ps.inferer_is_roi_based = False

        img_tensor = MetaTensor(torch.rand((1, 1, 5, 5, 5)))
        pred1 = torch.rand((5, 5, 5))
        pred2 = torch.rand((5, 5, 5))

        # First file
        self.mock_inferer.infer.return_value = pred1
        target_dict1 = {FILE_ID: "test_file_1"}
        self.ps._get_pred(img_tensor, target_dict1)

        self.assertEqual(len(self.ps._tiny_cache), 1)
        self.assertIn("test_file_1", self.ps._tiny_cache)
        self.assertTrue(torch.equal(self.ps._tiny_cache["test_file_1"], pred1))

        # Second file - should clear cache and store new prediction
        self.mock_inferer.infer.return_value = pred2
        target_dict2 = {FILE_ID: "test_file_2"}
        self.ps._get_pred(img_tensor, target_dict2)

        # Cache should now only contain the second file
        self.assertEqual(len(self.ps._tiny_cache), 1)
        self.assertNotIn("test_file_1", self.ps._tiny_cache)
        self.assertIn("test_file_2", self.ps._tiny_cache)
        self.assertTrue(torch.equal(self.ps._tiny_cache["test_file_2"], pred2))

    def test_tiny_cache_not_used_for_roi_based_inferer(self):
        """Test that tiny cache is not used for ROI-based inferers"""
        # Setup ROI-based inferer
        from linguine.inferers.roi_inferer import ROIInferer

        self.ps.inferer = Mock(spec=ROIInferer)
        self.ps.inferer.spacing = (1.0, 1.0, 1.0)
        self.ps.inferer_is_roi_based = True

        img_tensor = MetaTensor(torch.rand((1, 1, 5, 5, 5)))
        pred = torch.rand((5, 5, 5))

        self.ps.inferer.infer.return_value = pred

        target_dict = {FILE_ID: "test_file_1"}
        target_clicks = [(1, 1, 1)]

        # Call twice with same file - should run inference both times
        self.ps._get_pred(img_tensor, target_dict, target_clicks=target_clicks)
        self.ps._get_pred(img_tensor, target_dict, target_clicks=target_clicks)

        # Should have called inference twice (no caching for ROI-based)
        self.assertEqual(self.ps.inferer.infer.call_count, 2)
        self.assertEqual(len(self.ps._tiny_cache), 0)  # Cache should remain empty

    def test_tiny_cache_memory_efficiency(self):
        """Test that cache only holds one file at a time for memory efficiency"""
        # Setup non-ROI based inferer
        self.ps.inferer_is_roi_based = False

        img_tensor = MetaTensor(torch.rand((1, 1, 5, 5, 5)))

        # Process multiple files sequentially
        for i in range(5):
            pred = torch.rand((5, 5, 5))
            self.mock_inferer.infer.return_value = pred
            target_dict = {FILE_ID: f"test_file_{i}"}

            self.ps._get_pred(img_tensor, target_dict)

            # Cache should always contain exactly one item
            self.assertEqual(len(self.ps._tiny_cache), 1)
            self.assertIn(f"test_file_{i}", self.ps._tiny_cache)

            # Previous files should not be in cache
            for j in range(i):
                self.assertNotIn(f"test_file_{j}", self.ps._tiny_cache)


if __name__ == "__main__":
    unittest.main()
