import unittest

import torch

from linguine.prompt_selectors import PromptSelector, ThresholdPS


class TestThresholdPS(unittest.TestCase):
    def setUp(self):
        self.l_threshold = -200
        self.u_threshold = 350
        self.ps = ThresholdPS(l_threshold=self.l_threshold, u_threshold=self.u_threshold)

    def test_initialization(self):
        """Test ThresholdPS initialization"""
        self.assertEqual(self.ps.l_threshold, self.l_threshold)
        self.assertEqual(self.ps.u_threshold, self.u_threshold)

    def test_initialization_invalid_thresholds(self):
        """Test ThresholdPS initialization raises assertion error for invalid thresholds"""
        with self.assertRaises(AssertionError):
            ThresholdPS(l_threshold=100, u_threshold=50)  # u_threshold < l_threshold

    def test_filter_invalid_clicks_with_image_tensor(self):
        """Test filtering clicks with provided image tensor"""
        # Create a 3D image with different intensities
        img = torch.zeros((5, 5, 5))
        img[1, 1, 1] = -100  # Above threshold
        img[2, 2, 2] = -300  # Below threshold
        img[3, 3, 3] = -150  # Above threshold
        img = img.unsqueeze(0).unsqueeze(0)
        clicks = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
        result = self.ps._filter_invalid_clicks(clicks, target_img=img)

        # Should keep clicks above threshold
        expected = [(1, 1, 1), (3, 3, 3)]
        self.assertEqual(result, expected)

    def test_filter_invalid_clicks_wrong_image_dimensions(self):
        """Test assertion error for wrong image dimensions"""
        clicks = [(1, 1, 1)]
        wrong_img = torch.zeros((5, 5))  # 2D instead of 5D

        with self.assertRaises(AssertionError):
            self.ps._filter_invalid_clicks(clicks, target_img=wrong_img)

    def test_filter_invalid_clicks_out_of_bounds(self):
        """Test that out of bounds clicks are filtered"""
        img = torch.full((5, 5, 5), -100)  # All above threshold
        img = img.unsqueeze(0).unsqueeze(0)
        clicks = [(1, 1, 1), (10, 10, 10), (2, 2, 2)]  # One out of bounds

        result = self.ps._filter_invalid_clicks(clicks, target_img=img)
        expected = [(1, 1, 1), (2, 2, 2)]
        self.assertEqual(result, expected)

    def test_filter_invalid_clicks_exact_threshold(self):
        """Test behavior at exact threshold value"""
        img = torch.zeros((5, 5, 5))
        img[1, 1, 1] = self.l_threshold  # Exactly at threshold
        img[2, 2, 2] = self.l_threshold - 1  # Just below threshold
        img[3, 3, 3] = self.l_threshold + 1  # Just above threshold
        img = img.unsqueeze(0).unsqueeze(0)

        clicks = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
        result = self.ps._filter_invalid_clicks(clicks, target_img=img)

        # Should keep clicks >= threshold
        expected = [(1, 1, 1), (3, 3, 3)]
        self.assertEqual(result, expected)

    def test_filter_invalid_clicks_upper_threshold_filtering(self):
        """Test filtering clicks with upper threshold boundary"""
        # Create a 3D image with different intensities
        img = torch.zeros((5, 5, 5))
        img[1, 1, 1] = -100  # Within range (above l_threshold, below u_threshold)
        img[2, 2, 2] = -300  # Below l_threshold
        img[3, 3, 3] = 400  # Above u_threshold
        img[4, 4, 4] = 200  # Within range
        img = img.unsqueeze(0).unsqueeze(0)

        clicks = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
        result = self.ps._filter_invalid_clicks(clicks, target_img=img)

        # Should keep clicks within the threshold range [l_threshold, u_threshold]
        expected = [(1, 1, 1), (4, 4, 4)]
        self.assertEqual(result, expected)

    def test_threshold_boundary_values(self):
        """Test behavior at exact threshold boundary values"""
        img = torch.zeros((5, 5, 5))
        img[1, 1, 1] = self.l_threshold  # Exactly at lower threshold
        img[2, 2, 2] = self.u_threshold  # Exactly at upper threshold
        img[3, 3, 3] = self.l_threshold - 1  # Just below lower threshold
        img[4, 4, 4] = self.u_threshold + 1  # Just above upper threshold
        img = img.unsqueeze(0).unsqueeze(0)

        clicks = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
        result = self.ps._filter_invalid_clicks(clicks, target_img=img)

        # Should keep clicks exactly at boundaries (inclusive)
        expected = [(1, 1, 1), (2, 2, 2)]
        self.assertEqual(result, expected)

    def test_get_best_clicks_with_invalid_click(self):
        """Test get_best_clicks when one click is invalid."""
        img = torch.zeros((5, 5, 5))
        img[1, 1, 1] = -100  # Above threshold
        img[2, 2, 2] = -300  # Below threshold
        img[3, 3, 3] = -150  # Above threshold
        img = img.unsqueeze(0).unsqueeze(0)

        clicks = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]

        result = self.ps.get_best_clicks(target_clicks=clicks, target_img=img, n_clicks=2)
        self.assertEqual(result, [(1, 1, 1), (3, 3, 3)])

    def test_get_best_clicks_fewer_than_requested(self):
        """Test get_best_clicks when fewer valid clicks than requested"""
        img = torch.full((5, 5, 5), -400)
        img[1, 1, 1] = -100  # Above threshold
        img = img.unsqueeze(0).unsqueeze(0)

        clicks = [(1, 1, 1), (6, 6, 6)]  # Only one valid

        result = self.ps.get_best_clicks(clicks, target_img=img, n_clicks=5)
        self.assertEqual(result, [(1, 1, 1)])

    def test_get_best_clicks_empty_input(self):
        """Test get_best_clicks with empty click list"""
        img = torch.rand((5, 5, 5))
        img = img.unsqueeze(0).unsqueeze(0)
        result = self.ps.get_best_clicks([], target_img=img, n_clicks=2)
        self.assertEqual(len(result), 0)

    def test_different_threshold_values(self):
        """Test with different threshold values"""
        # Test with very low threshold
        ps_low = ThresholdPS(l_threshold=-1000, u_threshold=1000)
        img = torch.full((3, 3, 3), -500)
        img = img.unsqueeze(0).unsqueeze(0)
        clicks = [(1, 1, 1), (2, 2, 2)]

        result = ps_low._filter_invalid_clicks(clicks, target_img=img)
        self.assertEqual(result, clicks)  # All should pass

        # Test with high threshold
        ps_high = ThresholdPS(l_threshold=100, u_threshold=200)
        result = ps_high._filter_invalid_clicks(clicks, target_img=img)
        self.assertEqual(result, [])  # None should pass

    def test_inheritance_from_base(self):
        """Test that ThresholdPS inherits from ClickValidityClassifier"""

        self.assertIsInstance(self.ps, PromptSelector)

        # Test static method access
        clicks = [(1, 1, 1), (10, 10, 10)]
        shape = (5, 5, 5)
        result = self.ps.filter_out_of_bounds_clicks(clicks, shape)
        self.assertEqual(result, [(1, 1, 1)])


if __name__ == "__main__":
    unittest.main()
