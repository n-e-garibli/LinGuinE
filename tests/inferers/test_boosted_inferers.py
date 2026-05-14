# Copyright AstraZeneca 2025

import unittest.mock as mock

import numpy as np
import torch

from linguine.inferers.boosted_inferers import (
    BasicBoostedInferer,
    MergeProbabilitiesBoostedInferer,
    OrientationEnsembleInferer,
    ResampleAdditiveBoostedInferer,
)
from linguine.utils.bounding_boxes import BBox3D, Point3D


class MockInferer:
    """Mock inferer for testing purposes."""

    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self._spacing = spacing
        self.call_count = 0
        self.call_history = []

    @property
    def spacing(self):
        return self._spacing

    def infer(self, img, fg_clicks, bg_clicks, bboxes, return_probs=False, filter_pred=True):
        self.call_count += 1
        call_info = {
            "fg_clicks": fg_clicks,
            "bg_clicks": bg_clicks,
            "bboxes": bboxes,
            "return_probs": return_probs,
            "filter_pred": filter_pred,
        }
        self.call_history.append(call_info)

        # Return different predictions based on call count to simulate boosting
        if self.call_count == 1:
            # First call - return initial prediction
            if return_probs:
                return torch.tensor([[[0.8, 0.6], [0.4, 0.2]]], dtype=torch.float32)
            else:
                return np.array([[[1, 1], [0, 0]]], dtype=np.int32)
        else:
            # Second call - return boosted prediction
            if return_probs:
                return torch.tensor([[[0.9, 0.7], [0.3, 0.1]]], dtype=torch.float32)
            else:
                return np.array([[[1, 1], [1, 0]]], dtype=np.int32)


class TestBasicBoostedInferer:
    """Test cases for BasicBoostedInferer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_inferer = MockInferer()
        self.boosted_inferer = BasicBoostedInferer(self.mock_inferer)
        self.test_img = torch.randn(1, 1, 2, 2, 2)
        self.test_fg_clicks = [(0, 0, 0)]
        self.test_bg_clicks = [(1, 1, 1)]
        self.test_bboxes = [BBox3D(Point3D(0, 0, 0), Point3D(1, 1, 1))]

    def test_spacing_property(self):
        """Test that spacing property is delegated to base inferer."""
        assert self.boosted_inferer.spacing == (1.0, 1.0, 1.0)

    @mock.patch("linguine.inferers.boosted_inferers.find_mask_center")
    def test_infer_with_valid_center(self, mock_find_center):
        """Test inference when a valid center is found."""
        # Mock find_mask_center to return a valid center
        mock_find_center.return_value = (0, 1, 0)

        result = self.boosted_inferer.infer(self.test_img, self.test_fg_clicks, self.test_bg_clicks, self.test_bboxes)

        # Should call base inferer twice
        assert self.mock_inferer.call_count == 2

        # First call should use original fg_clicks
        first_call = self.mock_inferer.call_history[0]
        assert first_call["fg_clicks"] == self.test_fg_clicks
        assert first_call["filter_pred"] is True
        assert first_call["return_probs"] is False

        # Second call should use resampled center as fg_clicks
        second_call = self.mock_inferer.call_history[1]
        assert second_call["fg_clicks"] == [(0, 1, 0)]
        assert second_call["bg_clicks"] == self.test_bg_clicks
        assert second_call["bboxes"] == self.test_bboxes

        # Result should be from second inference
        expected_result = np.array([[[1, 1], [1, 0]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected_result)

    @mock.patch("linguine.inferers.boosted_inferers.find_mask_center")
    def test_infer_with_empty_center(self, mock_find_center):
        """Test inference when no center is found."""
        # Mock find_mask_center to return empty tuple
        mock_find_center.return_value = ()

        result = self.boosted_inferer.infer(self.test_img, self.test_fg_clicks, self.test_bg_clicks, self.test_bboxes)

        # Should only call base inferer once
        assert self.mock_inferer.call_count == 1

        # Result should be from first inference
        expected_result = np.array([[[1, 1], [0, 0]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected_result)

    @mock.patch("linguine.inferers.boosted_inferers.find_mask_center")
    def test_infer_with_return_probs(self, mock_find_center):
        """Test inference with return_probs=True."""
        mock_find_center.return_value = (0, 1, 0)

        self.boosted_inferer.infer(
            self.test_img, self.test_fg_clicks, self.test_bg_clicks, self.test_bboxes, return_probs=True
        )

        # Should call base inferer twice
        assert self.mock_inferer.call_count == 2

        # Second call should have return_probs=True
        second_call = self.mock_inferer.call_history[1]
        assert second_call["return_probs"] is True


class TestResampleAdditiveBoostedInferer:
    """Test cases for ResampleAdditiveBoostedInferer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_inferer = MockInferer()
        self.boosted_inferer = ResampleAdditiveBoostedInferer(self.mock_inferer)
        self.test_img = torch.randn(1, 1, 2, 2, 2)
        self.test_fg_clicks = [(0, 0, 0), (1, 0, 0)]
        self.test_bg_clicks = [(1, 1, 1)]
        self.test_bboxes = [BBox3D(Point3D(0, 0, 0), Point3D(1, 1, 1))]

    @mock.patch("linguine.inferers.boosted_inferers.find_mask_center")
    def test_infer_additive_fg_clicks(self, mock_find_center):
        """Test that fg_clicks are additive (not replaced)."""
        mock_find_center.return_value = (0, 1, 0)

        self.boosted_inferer.infer(self.test_img, self.test_fg_clicks, self.test_bg_clicks, self.test_bboxes)

        # Should call base inferer twice
        assert self.mock_inferer.call_count == 2

        # First call should use original fg_clicks
        first_call = self.mock_inferer.call_history[0]
        assert first_call["fg_clicks"] == self.test_fg_clicks

        # Second call should include both resampled center AND original fg_clicks
        second_call = self.mock_inferer.call_history[1]
        expected_fg_clicks = [(0, 1, 0)] + self.test_fg_clicks
        assert second_call["fg_clicks"] == expected_fg_clicks

    @mock.patch("linguine.inferers.boosted_inferers.find_mask_center")
    def test_infer_with_empty_original_fg_clicks(self, mock_find_center):
        """Test additive behavior with empty original fg_clicks."""
        mock_find_center.return_value = (0, 1, 0)

        self.boosted_inferer.infer(
            self.test_img,
            [],  # Empty fg_clicks
            self.test_bg_clicks,
            self.test_bboxes,
        )

        # Second call should only have the resampled center
        second_call = self.mock_inferer.call_history[1]
        assert second_call["fg_clicks"] == [(0, 1, 0)]


class TestMergeProbabilitiesBoostedInferer:
    """Test cases for MergeProbabilitiesBoostedInferer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_inferer = MockInferer()
        self.boosted_inferer = MergeProbabilitiesBoostedInferer(self.mock_inferer)
        self.test_img = torch.randn(1, 1, 2, 2, 2)
        self.test_fg_clicks = [(0, 0, 0)]
        self.test_bg_clicks = [(1, 1, 1)]
        self.test_bboxes = [BBox3D(Point3D(0, 0, 0), Point3D(1, 1, 1))]

    @mock.patch("linguine.inferers.boosted_inferers.find_mask_center")
    @mock.patch("linguine.inferers.boosted_inferers.filter_prediction")
    def test_infer_probability_merging(self, mock_filter, mock_find_center):
        """Test that probabilities are correctly merged."""
        mock_find_center.return_value = (0, 1, 0)
        mock_filter.return_value = (np.array([[[1, 1], [0, 0]]]), None)

        # Mock the base inferer to return specific probabilities
        def mock_infer(img, fg_clicks, bg_clicks, bboxes, return_probs=False, filter_pred=True):
            if return_probs:
                # Second call returns probabilities
                return torch.tensor([[[0.9, 0.7], [0.3, 0.1]]], dtype=torch.float32)
            else:
                # First call returns binary prediction converted to probabilities
                return torch.tensor([[[0.8, 0.6], [0.4, 0.2]]], dtype=torch.float32)

        self.mock_inferer.infer = mock_infer

        result = self.boosted_inferer.infer(
            self.test_img, self.test_fg_clicks, self.test_bg_clicks, self.test_bboxes, return_probs=True
        )

        # Expected merged probabilities: (original + second) / 2
        expected_probs = torch.tensor([[[0.85, 0.65], [0.35, 0.15]]], dtype=torch.float32)
        torch.testing.assert_close(result, expected_probs)

    @mock.patch("linguine.inferers.boosted_inferers.find_mask_center")
    @mock.patch("linguine.inferers.boosted_inferers.filter_prediction")
    def test_infer_binary_output_with_filtering(self, mock_filter, mock_find_center):
        """Test binary output with prediction filtering."""
        mock_find_center.return_value = (0, 1, 0)
        filtered_pred = np.array([[[1, 1], [0, 0]]], dtype=np.int32)
        mock_filter.return_value = (filtered_pred, None)

        # Mock the base inferer to return specific probabilities
        def mock_infer(img, fg_clicks, bg_clicks, bboxes, return_probs=False, filter_pred=True):
            if return_probs:
                return torch.tensor([[[0.9, 0.7], [0.3, 0.1]]], dtype=torch.float32)
            else:
                return torch.tensor([[[0.8, 0.6], [0.4, 0.2]]], dtype=torch.float32)

        self.mock_inferer.infer = mock_infer

        result = self.boosted_inferer.infer(
            self.test_img,
            self.test_fg_clicks,
            self.test_bg_clicks,
            self.test_bboxes,
            return_probs=False,
            filter_pred=True,
        )

        # Should call filter_prediction with merged probabilities > 0.5
        mock_filter.assert_called_once()

        # Result should be the filtered prediction
        np.testing.assert_array_equal(result, filtered_pred)

    @mock.patch("linguine.inferers.boosted_inferers.find_mask_center")
    def test_infer_binary_output_without_filtering(self, mock_find_center):
        """Test binary output without prediction filtering."""
        mock_find_center.return_value = (0, 1, 0)

        # Mock the base inferer to return specific probabilities
        def mock_infer(img, fg_clicks, bg_clicks, bboxes, return_probs=False, filter_pred=True):
            if return_probs:
                return torch.tensor([[[0.9, 0.7], [0.3, 0.1]]], dtype=torch.float32)
            else:
                return torch.tensor([[[0.8, 0.6], [0.4, 0.2]]], dtype=torch.float32)

        self.mock_inferer.infer = mock_infer

        result = self.boosted_inferer.infer(
            self.test_img,
            [],  # Empty fg_clicks to skip filtering
            self.test_bg_clicks,
            self.test_bboxes,
            return_probs=False,
            filter_pred=False,
        )

        # Expected binary result from merged probabilities > 0.5
        expected_binary = np.array([[[1, 1], [0, 0]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected_binary)

    @mock.patch("linguine.inferers.boosted_inferers.find_mask_center")
    def test_infer_with_empty_center(self, mock_find_center):
        """Test behavior when no center is found."""
        mock_find_center.return_value = ()

        # Mock to return a binary prediction when no center found
        def mock_infer(img, fg_clicks, bg_clicks, bboxes, return_probs=False, filter_pred=True):
            return torch.tensor([[[1, 0], [0, 1]]], dtype=torch.int32)

        self.mock_inferer.infer = mock_infer

        result = self.boosted_inferer.infer(self.test_img, self.test_fg_clicks, self.test_bg_clicks, self.test_bboxes)

        # Should return the original prediction without boosting
        expected_result = np.array([[[1, 0], [0, 1]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected_result)


class TestBoostedInfererIntegration:
    """Integration tests for boosted inferers."""

    def test_all_inferers_inherit_from_basic(self):
        """Test that all boosted inferers properly inherit from BasicBoostedInferer."""
        mock_inferer = MockInferer()

        basic_inferer = BasicBoostedInferer(mock_inferer)
        additive_inferer = ResampleAdditiveBoostedInferer(mock_inferer)
        merge_inferer = MergeProbabilitiesBoostedInferer(mock_inferer)

        assert isinstance(additive_inferer, BasicBoostedInferer)
        assert isinstance(merge_inferer, BasicBoostedInferer)

        # All should have the same spacing property
        assert basic_inferer.spacing == additive_inferer.spacing == merge_inferer.spacing

    def test_different_spacing_values(self):
        """Test that inferers work with different spacing values."""
        custom_spacing = (2.0, 1.5, 0.5)
        mock_inferer = MockInferer(spacing=custom_spacing)
        boosted_inferer = BasicBoostedInferer(mock_inferer)

        assert boosted_inferer.spacing == custom_spacing


class TestOrientationEnsembleInferer:
    """Test cases for OrientationEnsembleInferer."""

    def setup_method(self):
        self.mock_inferer = MockInferer()
        self.orientations = [0, 1, 2]
        self.boosted_inferer = OrientationEnsembleInferer(self.mock_inferer, transforms=self.orientations)
        self.test_img = torch.randn(1, 1, 2, 2, 2)
        self.test_fg_clicks = [(0, 0, 0)]
        self.test_bg_clicks = [(1, 1, 0)]
        self.test_bboxes = [BBox3D(Point3D(0, 0, 0), Point3D(1, 1, 0))]

    def test_spacing_property(self):
        """Test that spacing property is delegated to base inferer."""
        assert self.boosted_inferer.spacing == (1.0, 1.0, 1.0)

    def test_infer_probabilities_mean_aggregation(self):
        """
        Return probabilities from base inferer; ensemble should:
        - rotate inputs per orientation
        - invert predictions back
        - average the probabilities across orientations (float32)
        This test uses three independent tensors p0, p1_raw, p2_raw.
        We feed the ensemble rotated versions pred_1, pred_2 so that,
        after inversion by the ensemble, they become p1_raw and p2_raw.
        The expected output is the mean of (p0, p1_raw, p2_raw).
        """
        # Probability tensors after inversion (for aggregation)
        p0 = torch.randn(1, 1, 2, 2, 1, dtype=torch.float32)
        p1_raw = torch.randn(1, 1, 2, 2, 1, dtype=torch.float32)
        p2_raw = torch.randn(1, 1, 2, 2, 1, dtype=torch.float32)

        # Probability tensors before inversion (results of base inferer)
        pred_0 = p0
        pred_1 = torch.rot90(p1_raw, k=1, dims=(-3, -2))
        pred_2 = torch.rot90(p2_raw, k=2, dims=(-3, -2))

        def mock_infer(img, fg_clicks, bg_clicks, bboxes, return_probs=False, filter_pred=True):
            # OrientationEnsembleInferer should request probabilities here
            assert return_probs is True
            c = self.mock_inferer.call_count
            self.mock_inferer.call_count += 1
            return [pred_0, pred_1, pred_2][c]

        self.mock_inferer.infer = mock_infer
        self.mock_inferer.call_count = 0

        # Run ensemble
        result = self.boosted_inferer.infer(
            img=self.test_img,
            fg_clicks=self.test_fg_clicks,
            bg_clicks=self.test_bg_clicks,
            bboxes=self.test_bboxes,
            return_probs=True,
            filter_pred=False,
        )

        # Called once per orientation
        assert self.mock_inferer.call_count == len(self.orientations)
        # Output type/shape
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        expected = (p0 + p1_raw + p2_raw) / 3.0
        torch.testing.assert_close(result, expected)
