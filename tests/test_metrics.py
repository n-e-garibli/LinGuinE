"""Unit tests for metrics module."""

import unittest

import numpy as np

from linguine.metrics import (
    MetricsBundle,
    dice,
    get_surface_points,
    mean_euclidean_distance,
    surface_distances,
)


class TestDiceFunction(unittest.TestCase):
    """Test cases for dice coefficient function."""

    def test_perfect_overlap(self):
        """Test dice coefficient with perfect overlap."""
        mask = np.array([[[1, 1], [1, 0]], [[0, 1], [1, 1]]])
        result = dice(mask, mask)
        self.assertEqual(result, 1.0)

    def test_no_overlap(self):
        """Test dice coefficient with no overlap."""
        mask1 = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
        mask2 = np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]])
        result = dice(mask1, mask2)
        self.assertEqual(result, 0.0)

    def test_partial_overlap(self):
        """Test dice coefficient with partial overlap."""
        mask1 = np.array([[[1, 1], [1, 0]], [[0, 1], [1, 1]]])
        mask2 = np.array([[[1, 0], [1, 1]], [[0, 1], [0, 1]]])
        # mask1 has 6 ones, mask2 has 5 ones, intersection has 4 ones
        # dice = 2 * 4 / (6 + 5) = 8/11
        expected = 8.0 / 11.0
        result = dice(mask1, mask2)
        self.assertAlmostEqual(result, expected, places=6)

    def test_empty_masks(self):
        """Test dice coefficient with empty masks."""
        mask1 = np.zeros((2, 2, 2))
        mask2 = np.zeros((2, 2, 2))
        result = dice(mask1, mask2)
        self.assertEqual(result, 0.0)

    def test_one_empty_mask(self):
        """Test dice coefficient when one mask is empty."""
        mask1 = np.array([[[1, 1], [1, 0]], [[0, 1], [1, 1]]])
        mask2 = np.zeros((2, 2, 2))
        result = dice(mask1, mask2)
        self.assertEqual(result, 0.0)


class TestGetSurfacePoints(unittest.TestCase):
    """Test cases for surface point extraction function."""

    def test_simple_cube(self):
        """Test surface point extraction from a simple 3x3x3 cube."""
        # Create a 3x3x3 cube with 1s in the center
        mask = np.zeros((5, 5, 5))
        mask[1:4, 1:4, 1:4] = 1

        surface_points = get_surface_points(mask)

        # Surface points should be non-empty
        self.assertGreater(len(surface_points), 0)

        # All surface points should be within the original mask
        for point in surface_points:
            self.assertEqual(mask[tuple(point)], 1)

    def test_single_voxel(self):
        """Test surface point extraction from a single voxel."""
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1

        surface_points = get_surface_points(mask)

        # Single voxel should be its own surface
        self.assertEqual(len(surface_points), 1)
        np.testing.assert_array_equal(surface_points[0], [1, 1, 1])

    def test_empty_mask(self):
        """Test surface point extraction from empty mask."""
        mask = np.zeros((3, 3, 3))
        surface_points = get_surface_points(mask)
        self.assertEqual(len(surface_points), 0)

    def test_2d_slice(self):
        """Test surface point extraction from a 2D slice."""
        mask = np.zeros((3, 3, 3))
        mask[:, :, 1] = 1  # Fill middle slice

        surface_points = get_surface_points(mask)

        # Should have surface points
        self.assertGreater(len(surface_points), 0)

        # All surface points should be on the filled slice
        for point in surface_points:
            self.assertEqual(point[2], 1)


class TestSurfaceDistances(unittest.TestCase):
    """Test cases for surface distance calculation function."""

    def test_identical_masks(self):
        """Test surface distances with identical masks."""
        mask = np.zeros((5, 5, 5))
        mask[1:4, 1:4, 1:4] = 1

        assd, hd95, nsd = surface_distances(mask, mask)

        # Distances should be 0 for identical masks
        self.assertEqual(assd, 0.0)
        self.assertEqual(hd95, 0.0)
        self.assertEqual(nsd, 1.0)

    def test_adjacent_cubes(self):
        """Test surface distances with adjacent cubes."""
        mask1 = np.zeros((6, 3, 3))
        mask1[0:3, :, :] = 1

        mask2 = np.zeros((6, 3, 3))
        mask2[3:6, :, :] = 1

        assd, hd95, nsd = surface_distances(mask1, mask2)

        # Should have positive distances since cubes don't overlap
        self.assertGreater(assd, 0.0)
        self.assertGreater(hd95, 0.0)

        # HD95 should be >= ASSD
        self.assertGreaterEqual(hd95, assd)

        # NSD should be between 0 and 1
        self.assertGreaterEqual(nsd, 0.0)
        self.assertLessEqual(nsd, 1.0)

    def test_empty_masks(self):
        """Test surface distances with empty masks."""
        mask1 = np.zeros((3, 3, 3))
        mask2 = np.zeros((3, 3, 3))

        assd, hd95, nsd = surface_distances(mask1, mask2)

        self.assertTrue(np.isnan(assd))
        self.assertTrue(np.isnan(hd95))
        self.assertTrue(np.isnan(nsd))

    def test_one_empty_mask(self):
        """Test surface distances when one mask is empty."""
        mask1 = np.zeros((3, 3, 3))
        mask1[1, 1, 1] = 1
        mask2 = np.zeros((3, 3, 3))

        assd, hd95, nsd = surface_distances(mask1, mask2)

        self.assertTrue(np.isnan(assd))
        self.assertTrue(np.isnan(hd95))
        self.assertTrue(np.isnan(nsd))

    def test_single_voxels(self):
        """Test surface distances between single voxels."""
        mask1 = np.zeros((5, 5, 5))
        mask1[1, 1, 1] = 1

        mask2 = np.zeros((5, 5, 5))
        mask2[3, 3, 3] = 1

        assd, hd95, nsd = surface_distances(mask1, mask2)

        # Distance should be sqrt((3-1)^2 + (3-1)^2 + (3-1)^2) = sqrt(12)
        expected_distance = np.sqrt(12)
        self.assertAlmostEqual(assd, expected_distance, places=6)
        self.assertAlmostEqual(hd95, expected_distance, places=6)

        # With default tolerance of 2.0, sqrt(12) ~= 3.46 > 2.0, so NSD should be 0
        self.assertEqual(nsd, 0.0)

    def test_tolerance_effect(self):
        """Test that different tolerances affect NSD in surface_distances."""
        mask1 = np.zeros((6, 3, 3))
        mask1[0:3, :, :] = 1

        mask2 = np.zeros((6, 3, 3))
        mask2[3:6, :, :] = 1

        assd1, hd951, nsd_small = surface_distances(mask1, mask2, tolerance=1.0)
        assd2, hd952, nsd_large = surface_distances(mask1, mask2, tolerance=5.0)

        # ASSD and HD95 should be the same regardless of tolerance
        self.assertEqual(assd1, assd2)
        self.assertEqual(hd951, hd952)

        # Larger tolerance should give higher or equal NSD
        self.assertGreaterEqual(nsd_large, nsd_small)


class TestNormalizedSurfaceDistance(unittest.TestCase):
    """Test cases for normalized surface distance calculation via surface_distances function."""

    def test_identical_masks(self):
        """Test NSD with identical masks."""
        mask = np.zeros((5, 5, 5))
        mask[1:4, 1:4, 1:4] = 1

        _, _, nsd = surface_distances(mask, mask, tolerance=2.0)

        # NSD should be 1.0 for identical masks
        self.assertEqual(nsd, 1.0)

    def test_adjacent_cubes_within_tolerance(self):
        """Test NSD with adjacent cubes that are within tolerance."""
        mask1 = np.zeros((6, 3, 3))
        mask1[0:3, :, :] = 1

        mask2 = np.zeros((6, 3, 3))
        mask2[2:5, :, :] = 1  # One voxel overlap

        _, _, nsd = surface_distances(mask1, mask2, tolerance=2.0)

        # Some surfaces should be within tolerance
        self.assertGreater(nsd, 0.0)
        self.assertLessEqual(nsd, 1.0)

    def test_far_apart_cubes(self):
        """Test NSD with cubes that are far apart."""
        mask1 = np.zeros((10, 3, 3))
        mask1[0:2, :, :] = 1

        mask2 = np.zeros((10, 3, 3))
        mask2[7:9, :, :] = 1

        _, _, nsd = surface_distances(mask1, mask2, tolerance=2.0)

        # Most or all surfaces should be outside tolerance
        self.assertGreaterEqual(nsd, 0.0)
        # With large distance, NSD should be close to 0
        self.assertLess(nsd, 0.5)

    def test_empty_masks(self):
        """Test NSD with empty masks."""
        mask1 = np.zeros((3, 3, 3))
        mask2 = np.zeros((3, 3, 3))

        _, _, nsd = surface_distances(mask1, mask2, tolerance=2.0)

        self.assertTrue(np.isnan(nsd))

    def test_one_empty_mask(self):
        """Test NSD when one mask is empty."""
        mask1 = np.zeros((3, 3, 3))
        mask1[1, 1, 1] = 1
        mask2 = np.zeros((3, 3, 3))

        _, _, nsd = surface_distances(mask1, mask2, tolerance=2.0)

        self.assertTrue(np.isnan(nsd))

    def test_tolerance_effect(self):
        """Test that increasing tolerance increases NSD."""
        mask1 = np.zeros((6, 3, 3))
        mask1[0:3, :, :] = 1

        mask2 = np.zeros((6, 3, 3))
        mask2[3:6, :, :] = 1

        _, _, nsd_small = surface_distances(mask1, mask2, tolerance=1.0)
        _, _, nsd_large = surface_distances(mask1, mask2, tolerance=5.0)

        # Larger tolerance should give higher or equal NSD
        self.assertGreaterEqual(nsd_large, nsd_small)


class TestMeanEuclideanDistance(unittest.TestCase):
    """Test cases for mean euclidean distance calculation function."""

    def test_identical_masks(self):
        """Test MED with identical masks."""
        mask = np.zeros((5, 5, 5))
        mask[1:4, 1:4, 1:4] = 1

        med = mean_euclidean_distance(mask, mask)

        # MED should be 0 for identical masks
        self.assertEqual(med, 0.0)

    def test_shifted_masks(self):
        """Test MED with shifted masks."""
        mask1 = np.zeros((10, 10, 10))
        mask1[2:4, 2:4, 2:4] = 1  # Center at (3, 3, 3)

        mask2 = np.zeros((10, 10, 10))
        mask2[5:7, 5:7, 5:7] = 1  # Center at (6, 6, 6)

        med = mean_euclidean_distance(mask1, mask2)

        # Distance should be sqrt((6-3)^2 + (6-3)^2 + (6-3)^2) = sqrt(27)
        expected_distance = np.sqrt(27)
        self.assertAlmostEqual(med, expected_distance, places=6)

    def test_single_voxels(self):
        """Test MED with single voxels."""
        mask1 = np.zeros((5, 5, 5))
        mask1[1, 1, 1] = 1

        mask2 = np.zeros((5, 5, 5))
        mask2[3, 3, 3] = 1

        med = mean_euclidean_distance(mask1, mask2)

        # Distance should be sqrt((3-1)^2 + (3-1)^2 + (3-1)^2) = sqrt(12)
        expected_distance = np.sqrt(12)
        self.assertAlmostEqual(med, expected_distance, places=6)

    def test_empty_masks(self):
        """Test MED with empty masks."""
        mask1 = np.zeros((3, 3, 3))
        mask2 = np.zeros((3, 3, 3))

        med = mean_euclidean_distance(mask1, mask2)

        self.assertTrue(np.isnan(med))

    def test_one_empty_mask(self):
        """Test MED when one mask is empty."""
        mask1 = np.zeros((3, 3, 3))
        mask1[1, 1, 1] = 1
        mask2 = np.zeros((3, 3, 3))

        med = mean_euclidean_distance(mask1, mask2)

        self.assertTrue(np.isnan(med))

    def test_asymmetric_masks(self):
        """Test MED with asymmetric masks."""
        mask1 = np.zeros((10, 10, 10))
        mask1[2:5, 2:5, 2:5] = 1  # Larger cube

        mask2 = np.zeros((10, 10, 10))
        mask2[6, 6, 6] = 1  # Single voxel

        med = mean_euclidean_distance(mask1, mask2)

        # Should have a positive distance
        self.assertGreater(med, 0.0)

    def test_different_axis_shifts(self):
        """Test MED with shifts along different axes."""
        mask1 = np.zeros((10, 10, 10))
        mask1[2:4, 2:4, 2:4] = 1

        # Shift only along x-axis
        mask2 = np.zeros((10, 10, 10))
        mask2[6:8, 2:4, 2:4] = 1

        med = mean_euclidean_distance(mask1, mask2)

        # Distance should be approximately 4 (shift in x-axis)
        self.assertAlmostEqual(med, 4.0, places=1)


class TestMetricsBundle(unittest.TestCase):
    """Test cases for MetricsBundle class."""

    def test_default_initialization(self):
        """Test MetricsBundle default initialization."""
        metrics = MetricsBundle()

        # All metrics should be NaN by default
        self.assertTrue(np.isnan(metrics.dice))
        self.assertTrue(np.isnan(metrics.recall))
        self.assertTrue(np.isnan(metrics.precision))
        self.assertTrue(np.isnan(metrics.hd95))
        self.assertTrue(np.isnan(metrics.assd))
        self.assertTrue(np.isnan(metrics.nsd))
        self.assertTrue(np.isnan(metrics.med))

    def test_custom_initialization(self):
        """Test MetricsBundle with custom values."""
        metrics = MetricsBundle(dice=0.8, recall=0.9, precision=0.7, hd95=2.5, assd=1.2, nsd=0.95, med=3.5)

        self.assertEqual(metrics.dice, 0.8)
        self.assertEqual(metrics.recall, 0.9)
        self.assertEqual(metrics.precision, 0.7)
        self.assertEqual(metrics.hd95, 2.5)
        self.assertEqual(metrics.assd, 1.2)
        self.assertEqual(metrics.nsd, 0.95)
        self.assertEqual(metrics.med, 3.5)

    def test_compute_metrics_perfect_overlap(self):
        """Test compute_metrics with perfect overlap."""
        mask = np.array([[[1, 1], [1, 0]], [[0, 1], [1, 1]]])
        metrics = MetricsBundle()

        metrics.compute_metrics(mask, mask)

        # Perfect overlap should give perfect scores
        self.assertEqual(metrics.dice, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.hd95, 0.0)
        self.assertEqual(metrics.assd, 0.0)
        self.assertEqual(metrics.nsd, 1.0)
        self.assertEqual(metrics.med, 0.0)

    def test_compute_metrics_no_overlap(self):
        """Test compute_metrics with no overlap."""
        mask1 = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
        mask2 = np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]])
        metrics = MetricsBundle()

        metrics.compute_metrics(mask1, mask2)

        # No overlap should give zero dice, recall, precision
        self.assertEqual(metrics.dice, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.precision, 0.0)
        # Surface distances should be positive
        self.assertGreater(metrics.hd95, 0.0)
        self.assertGreater(metrics.assd, 0.0)
        # NSD should be between 0 and 1
        self.assertGreaterEqual(metrics.nsd, 0.0)
        self.assertLessEqual(metrics.nsd, 1.0)
        # MED can be 0 even with no overlap if centers coincide (as in this symmetric case)
        self.assertGreaterEqual(metrics.med, 0.0)

    def test_compute_metrics_partial_overlap(self):
        """Test compute_metrics with partial overlap."""
        mask1 = np.array([[[1, 1], [1, 0]], [[0, 1], [1, 1]]])  # 6 ones
        mask2 = np.array([[[1, 0], [1, 1]], [[0, 1], [0, 1]]])  # 5 ones, 4 overlap
        metrics = MetricsBundle()

        metrics.compute_metrics(mask1, mask2)

        # Check dice coefficient: 2 * 4 / (6 + 5) = 8/11
        expected_dice = 8.0 / 11.0
        self.assertAlmostEqual(metrics.dice, expected_dice, places=6)

        # Check recall: TP / (TP + FN) = 4 / (4 + 1) = 4/5 = 0.8
        # FN = sum((1-mask1) * mask2) = sum of mask2 where mask1 is 0 = 1
        expected_recall = 4.0 / 5.0
        self.assertAlmostEqual(metrics.recall, expected_recall, places=6)

        # Check precision: TP / (TP + FP) = 4 / (4 + 2) = 2/3
        # FP = sum(mask1 * (1-mask2)) = sum of mask1 where mask2 is 0 = 2
        expected_precision = 4.0 / 6.0
        self.assertAlmostEqual(metrics.precision, expected_precision, places=6)

        # Surface distances should be >= 0
        self.assertGreaterEqual(metrics.hd95, 0.0)
        self.assertGreaterEqual(metrics.assd, 0.0)

        # NSD should be between 0 and 1
        self.assertGreaterEqual(metrics.nsd, 0.0)
        self.assertLessEqual(metrics.nsd, 1.0)

        # MED should be >= 0
        self.assertGreaterEqual(metrics.med, 0.0)

    def test_compute_metrics_empty_ground_truth(self):
        """Test compute_metrics when ground truth is empty."""
        mask1 = np.array([[[1, 1], [1, 0]], [[0, 1], [1, 1]]])
        mask2 = np.zeros((2, 2, 2))
        metrics = MetricsBundle()

        metrics.compute_metrics(mask1, mask2)

        # No true positives, no false negatives (GT is empty)
        self.assertEqual(metrics.dice, 0.0)
        self.assertTrue(np.isnan(metrics.recall))
        self.assertEqual(metrics.precision, 0.0)  # 0/(0+5) = 0
        self.assertTrue(np.isnan(metrics.hd95))
        self.assertTrue(np.isnan(metrics.assd))
        self.assertTrue(np.isnan(metrics.nsd))
        self.assertTrue(np.isnan(metrics.med))

    def test_compute_metrics_empty_prediction(self):
        """Test compute_metrics when prediction is empty."""
        mask1 = np.zeros((2, 2, 2))
        mask2 = np.array([[[1, 1], [1, 0]], [[0, 1], [1, 1]]])
        metrics = MetricsBundle()

        metrics.compute_metrics(mask1, mask2)

        # No true positives, no false positives (prediction is empty)
        self.assertEqual(metrics.dice, 0.0)
        self.assertEqual(metrics.recall, 0.0)  # 0/(0+6) = 0
        self.assertTrue(np.isnan(metrics.precision))  # 0/0 case handled
        self.assertTrue(np.isnan(metrics.hd95))
        self.assertTrue(np.isnan(metrics.assd))
        self.assertTrue(np.isnan(metrics.nsd))
        self.assertTrue(np.isnan(metrics.med))

    def test_compute_metrics_both_empty(self):
        """Test compute_metrics when both masks are empty."""
        mask1 = np.zeros((2, 2, 2))
        mask2 = np.zeros((2, 2, 2))
        metrics = MetricsBundle()

        metrics.compute_metrics(mask1, mask2)

        # Almost all metrics should be undefined for empty masks
        self.assertEqual(metrics.dice, 0.0)
        self.assertTrue(np.isnan(metrics.recall))
        self.assertTrue(np.isnan(metrics.precision))
        self.assertTrue(np.isnan(metrics.hd95))
        self.assertTrue(np.isnan(metrics.assd))
        self.assertTrue(np.isnan(metrics.nsd))
        self.assertTrue(np.isnan(metrics.med))

    def test_compute_metrics_large_masks(self):
        """Test compute_metrics with larger realistic masks."""
        # Create two 10x10x10 cubes with some overlap
        mask1 = np.zeros((15, 15, 15))
        mask1[2:12, 2:12, 2:12] = 1  # 10x10x10 cube

        mask2 = np.zeros((15, 15, 15))
        mask2[5:15, 5:15, 5:15] = 1  # 10x10x10 cube, shifted

        metrics = MetricsBundle()
        metrics.compute_metrics(mask1, mask2)

        # Should have partial overlap
        self.assertGreater(metrics.dice, 0.0)
        self.assertLess(metrics.dice, 1.0)

        # All metrics should be computed
        self.assertFalse(np.isnan(metrics.dice))
        self.assertFalse(np.isnan(metrics.recall))
        self.assertFalse(np.isnan(metrics.precision))
        self.assertFalse(np.isnan(metrics.hd95))
        self.assertFalse(np.isnan(metrics.assd))
        self.assertFalse(np.isnan(metrics.nsd))
        self.assertFalse(np.isnan(metrics.med))

        # Surface distances should be positive due to partial overlap
        self.assertGreater(metrics.hd95, 0.0)
        self.assertGreater(metrics.assd, 0.0)

        # NSD should be between 0 and 1
        self.assertGreaterEqual(metrics.nsd, 0.0)
        self.assertLessEqual(metrics.nsd, 1.0)

        # MED should be positive (cubes are shifted)
        self.assertGreater(metrics.med, 0.0)

    def test_compute_metrics_with_spacing(self):
        """Test compute_metrics with different spacing to verify mm conversion."""
        # Create two simple masks with known distance
        mask1 = np.zeros((10, 10, 10))
        mask1[2:4, 2:4, 2:4] = 1  # Center at (3, 3, 3)

        mask2 = np.zeros((10, 10, 10))
        mask2[5:7, 5:7, 5:7] = 1  # Center at (6, 6, 6)

        # Test with default spacing (1.0, 1.0, 1.0)
        metrics_default = MetricsBundle()
        metrics_default.compute_metrics(mask1, mask2, spacing=(1.0, 1.0, 1.0))

        # Test with custom spacing (2.0, 2.0, 2.0) - distances should double
        metrics_custom = MetricsBundle()
        metrics_custom.compute_metrics(mask1, mask2, spacing=(2.0, 2.0, 2.0))

        # With spacing doubled, all distance metrics should approximately double
        # MED should be exactly doubled (center distance: sqrt(27) * 2)
        self.assertAlmostEqual(metrics_custom.med, metrics_default.med * 2.0, places=5)

        # HD95 and ASSD should also approximately double
        self.assertGreater(metrics_custom.hd95, metrics_default.hd95 * 1.9)
        self.assertLess(metrics_custom.hd95, metrics_default.hd95 * 2.1)
        self.assertGreater(metrics_custom.assd, metrics_default.assd * 1.9)
        self.assertLess(metrics_custom.assd, metrics_default.assd * 2.1)

        # NSD should remain the same if we also scale the tolerance
        metrics_default_tol2 = MetricsBundle()
        metrics_default_tol2.compute_metrics(mask1, mask2, tolerance=2.0, spacing=(1.0, 1.0, 1.0))
        metrics_custom_tol4 = MetricsBundle()
        metrics_custom_tol4.compute_metrics(mask1, mask2, tolerance=4.0, spacing=(2.0, 2.0, 2.0))
        self.assertAlmostEqual(metrics_default_tol2.nsd, metrics_custom_tol4.nsd, places=2)

    def test_surface_distances_with_spacing(self):
        """Test surface_distances function with spacing parameter."""
        mask1 = np.zeros((5, 5, 5))
        mask1[1, 1, 1] = 1

        mask2 = np.zeros((5, 5, 5))
        mask2[3, 3, 3] = 1

        # With spacing (1, 1, 1), distance should be sqrt(12)
        assd1, hd951, nsd1 = surface_distances(mask1, mask2, spacing=(1.0, 1.0, 1.0))
        expected_distance = np.sqrt(12)
        self.assertAlmostEqual(assd1, expected_distance, places=5)
        self.assertAlmostEqual(hd951, expected_distance, places=5)

        # With spacing (2, 2, 2), distance should be 2*sqrt(12)
        assd2, hd952, nsd2 = surface_distances(mask1, mask2, spacing=(2.0, 2.0, 2.0))
        self.assertAlmostEqual(assd2, expected_distance * 2.0, places=5)
        self.assertAlmostEqual(hd952, expected_distance * 2.0, places=5)

        # With spacing (1, 2, 3), distance should be sqrt((2^2 + 4^2 + 6^2)) = sqrt(56)
        assd3, hd953, nsd3 = surface_distances(mask1, mask2, spacing=(1.0, 2.0, 3.0))
        expected_distance_aniso = np.sqrt(56)
        self.assertAlmostEqual(assd3, expected_distance_aniso, places=5)
        self.assertAlmostEqual(hd953, expected_distance_aniso, places=5)

    def test_mean_euclidean_distance_with_spacing(self):
        """Test mean_euclidean_distance function with spacing parameter."""
        mask1 = np.zeros((10, 10, 10))
        mask1[2:4, 2:4, 2:4] = 1  # Center at (3, 3, 3)

        mask2 = np.zeros((10, 10, 10))
        mask2[5:7, 5:7, 5:7] = 1  # Center at (6, 6, 6)

        # With spacing (1, 1, 1), distance should be sqrt(27)
        med1 = mean_euclidean_distance(mask1, mask2, spacing=(1.0, 1.0, 1.0))
        expected_distance = np.sqrt(27)
        self.assertAlmostEqual(med1, expected_distance, places=5)

        # With spacing (2, 2, 2), distance should be 2*sqrt(27)
        med2 = mean_euclidean_distance(mask1, mask2, spacing=(2.0, 2.0, 2.0))
        self.assertAlmostEqual(med2, expected_distance * 2.0, places=5)

        # With spacing (1, 2, 3), distance should be sqrt((3^2 + 6^2 + 9^2)) = sqrt(126)
        med3 = mean_euclidean_distance(mask1, mask2, spacing=(1.0, 2.0, 3.0))
        expected_distance_aniso = np.sqrt(126)
        self.assertAlmostEqual(med3, expected_distance_aniso, places=5)


if __name__ == "__main__":
    unittest.main()
