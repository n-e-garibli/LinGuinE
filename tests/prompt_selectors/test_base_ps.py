import unittest

import torch

from linguine.prompt_selectors.base_ps import PromptSelector


class TestPromptSelector(unittest.TestCase):
    def setUp(self):
        self.ps = PromptSelector()
        self.img = torch.zeros(size=(1, 1, 10, 10, 10))

    def test_filter_out_of_bounds_clicks_valid(self):
        """Test filtering with all valid clicks"""
        clicks = [(1, 2, 3), (5, 5, 5), (9, 9, 9)]
        result = self.ps.filter_out_of_bounds_clicks(clicks, self.img.shape[2:])
        self.assertEqual(result, clicks)

    def test_filter_out_of_bounds_clicks_invalid(self):
        """Test filtering with invalid clicks"""
        clicks = [(-1, 2, 3), (5, 5, 5), (10, 9, 9), (5, -1, 5)]
        expected = [(5, 5, 5)]
        result = self.ps.filter_out_of_bounds_clicks(clicks, self.img.shape[2:])
        self.assertEqual(result, expected)

    def test_filter_out_of_bounds_clicks_empty(self):
        """Test filtering with empty click list"""
        clicks = []
        result = self.ps.filter_out_of_bounds_clicks(clicks, self.img.shape[2:])
        self.assertEqual(result, [])

    def test_filter_out_of_bounds_clicks_all_invalid(self):
        """Test filtering with all invalid clicks"""
        clicks = [(-1, -1, -1), (10, 10, 10), (15, 5, 5)]
        result = self.ps.filter_out_of_bounds_clicks(clicks, self.img.shape[2:])
        self.assertEqual(result, [])

    def test_filter_out_of_bounds_clicks_edge_cases(self):
        """Test edge cases for bounds checking"""
        # Test exact boundary values
        clicks = [(0, 0, 0), (9, 9, 9), (0, 9, 5)]
        result = self.ps.filter_out_of_bounds_clicks(clicks, self.img.shape[2:])
        self.assertEqual(result, clicks)

        # Test just outside boundaries
        clicks = [(10, 5, 5), (5, 10, 5), (5, 5, 10)]
        result = self.ps.filter_out_of_bounds_clicks(clicks, self.img.shape[2:])
        self.assertEqual(result, [])

    def test_get_best_clicks_sufficient_valid(self):
        """Test get_best_clicks with sufficient valid clicks"""
        clicks = [(1, 2, 3), (5, 5, 5), (9, 9, 9)]
        result = self.ps.get_best_clicks(clicks, self.img, n_clicks=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result, [(1, 2, 3), (5, 5, 5)])

    def test_get_best_clicks_insufficient_valid(self):
        """Test get_best_clicks with insufficient valid clicks"""
        clicks = [(1, 2, 3), (-1, 2, 3)]  # Only one valid
        result = self.ps.get_best_clicks(clicks, self.img, n_clicks=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [(1, 2, 3)])

    def test_get_best_clicks_no_valid(self):
        """Test get_best_clicks with no valid clicks"""
        clicks = [(-1, -1, -1), (10, 10, 10)]
        result = self.ps.get_best_clicks(clicks, self.img, n_clicks=2)
        self.assertEqual(result, [])

    def test_get_best_clicks_zero_n_clicks(self):
        """Test get_best_clicks with n_clicks=0"""
        clicks = [(1, 2, 3), (5, 5, 5)]
        result = self.ps.get_best_clicks(clicks, self.img, n_clicks=0)
        self.assertEqual(result, [])

    def test_different_image_shapes(self):
        """Test with different image shapes"""
        # Test with non-cubic shape
        shape = (5, 10, 8)
        clicks = [(0, 0, 0), (4, 9, 7), (5, 5, 5), (3, 10, 4)]
        expected = [(0, 0, 0), (4, 9, 7)]
        result = self.ps.filter_out_of_bounds_clicks(clicks, shape)
        self.assertEqual(result, expected)

        # Test with very small shape
        shape = (1, 1, 1)
        clicks = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        expected = [(0, 0, 0)]
        result = self.ps.filter_out_of_bounds_clicks(clicks, shape)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
