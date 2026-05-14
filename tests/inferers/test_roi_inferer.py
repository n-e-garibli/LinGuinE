# Copyright AstraZeneca 2025

import pytest
import torch

from linguine.inferers.roi_inferer import ROIInferer
from linguine.utils.bounding_boxes import BBox3D, Point3D


class ConcreteROIInferer(ROIInferer):
    """Concrete implementation of ROIInferer for testing purposes."""

    def __init__(self, roi_size=(32, 32, 32)):
        self._roi_size = roi_size
        self.call_history = []
        self.oob_roi_handling = "shift"

    @property
    def roi(self) -> tuple[float, float, float]:
        """Return the ROI size."""
        return self._roi_size

    def infer_on_roi(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> torch.Tensor:
        """Mock implementation that records call parameters and returns a simple prediction."""
        call_info = {
            "img_shape": img.shape,
            "fg_clicks": fg_clicks,
            "bg_clicks": bg_clicks,
            "bboxes": bboxes,
            "return_probs": return_probs,
            "filter_pred": filter_pred,
        }
        self.call_history.append(call_info)

        # Create a simple prediction based on ROI size
        roi_x, roi_y, roi_z = self.roi
        if return_probs:
            # Return probabilities - higher in center, lower at edges
            pred = torch.ones((roi_x, roi_y, roi_z), dtype=img.dtype, device=img.device) * 0.2
            pred[roi_x // 4 : 3 * roi_x // 4, roi_y // 4 : 3 * roi_y // 4, roi_z // 4 : 3 * roi_z // 4] = 0.9
        else:
            # Return binary prediction - 1 in center, 0 at edges
            pred = torch.zeros((roi_x, roi_y, roi_z), dtype=img.dtype, device=img.device)
            pred[roi_x // 4 : 3 * roi_x // 4, roi_y // 4 : 3 * roi_y // 4, roi_z // 4 : 3 * roi_z // 4] = 1

        return pred


class TestROIInferer:
    """Test cases for ROIInferer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.roi_inferer = ConcreteROIInferer(roi_size=(32, 32, 32))
        self.test_img = torch.randn(1, 1, 64, 64, 64)  # Larger than ROI
        self.test_fg_clicks = [(32, 32, 32), (30, 30, 30)]
        self.test_bg_clicks = [(10, 10, 10)]
        self.test_bboxes = [BBox3D(Point3D(20, 20, 20), Point3D(40, 40, 40))]

    def test_roi_property(self):
        """Test that the roi property returns expected size."""
        assert self.roi_inferer.roi == (32, 32, 32)

    def test_extract_roi_around_click_centered(self):
        """Test ROI extraction when click is well within image bounds."""
        click = (32, 32, 32)
        roi_tensor, roi_offset = self.roi_inferer._extract_roi_around_click(self.test_img, click)

        # Check ROI tensor shape
        assert roi_tensor.shape == (1, 1, 32, 32, 32)

        # Check offset (should be click - roi_size//2)
        expected_offset = (16, 16, 16)  # 32 - 32//2 = 16
        assert roi_offset == expected_offset

    def test_extract_roi_around_click_near_boundary(self):
        """Test ROI extraction when click is near image boundary."""
        # Click near the edge
        click = (10, 10, 10)
        roi_tensor, roi_offset = self.roi_inferer._extract_roi_around_click(self.test_img, click)

        # ROI should still be correct size
        assert roi_tensor.shape == (1, 1, 32, 32, 32)

        # Offset should be adjusted to fit within image
        assert roi_offset == (0, 0, 0)

    def test_extract_roi_with_padding_required(self):
        """Test ROI extraction when image is smaller than ROI in some dimension."""
        small_img = torch.randn(1, 1, 20, 20, 20)  # Smaller than ROI
        click = (10, 10, 10)
        roi_tensor, roi_offset = self.roi_inferer._extract_roi_around_click(small_img, click)

        # ROI should be padded to correct size
        assert roi_tensor.shape == (1, 1, 32, 32, 32)

        # Should have some padding (zeros) due to small image
        assert torch.sum(roi_tensor == 0) > 0

    def test_format_guidance_data_to_roi(self):
        """Test conversion of guidance data from full image to ROI coordinates."""
        roi_offset = (16, 16, 16)

        # Original clicks and bboxes in full image coordinates
        fg_clicks = [(20, 20, 20), (30, 30, 30)]  # Both should be in ROI
        bg_clicks = [(10, 10, 10)]  # Should be outside ROI after offset
        bboxes = [BBox3D(Point3D(18, 18, 18), Point3D(25, 25, 25))]

        formatted_fg, formatted_bg, formatted_bboxes = self.roi_inferer._format_guidance_data_to_roi(
            fg_clicks, bg_clicks, bboxes, roi_offset
        )

        # Check foreground clicks conversion
        assert len(formatted_fg) == 2
        assert (4, 4, 4) in formatted_fg  # 20 - 16 = 4
        assert (14, 14, 14) in formatted_fg  # 30 - 16 = 14

        # Check background clicks (should be empty as click is outside ROI bounds)
        assert len(formatted_bg) == 0

        # Check bounding box conversion
        assert len(formatted_bboxes) == 1
        bbox = formatted_bboxes[0]
        assert bbox.x_min == 2  # 18 - 16 = 2
        assert bbox.y_min == 2
        assert bbox.z_min == 2
        assert bbox.x_max == 9  # 25 - 16 = 9
        assert bbox.y_max == 9
        assert bbox.z_max == 9

    def test_map_roi_prediction_to_full_image(self):
        """Test mapping ROI prediction back to full image space."""
        roi_prediction = torch.ones((32, 32, 32))
        roi_offset = (16, 16, 16)
        original_shape = (64, 64, 64)

        full_prediction = self.roi_inferer._map_roi_prediction_to_full_image(roi_prediction, roi_offset, original_shape)

        # Check output shape
        assert full_prediction.shape == original_shape

        # Check that ROI region has prediction values
        assert torch.all(full_prediction[16:48, 16:48, 16:48] == 1)

        # Check that areas outside ROI are zeros
        assert torch.all(full_prediction[0:16, :, :] == 0)
        assert torch.all(full_prediction[48:64, :, :] == 0)

    def test_infer_with_valid_clicks(self):
        """Test main infer method with valid foreground clicks."""
        result = self.roi_inferer.infer(
            self.test_img,
            self.test_fg_clicks,
            self.test_bg_clicks,
            self.test_bboxes,
            return_probs=False,
            filter_pred=True,
        )

        # Check that infer_on_roi was called
        assert len(self.roi_inferer.call_history) == 1
        call_info = self.roi_inferer.call_history[0]

        # Check call parameters
        assert call_info["img_shape"] == (1, 1, 32, 32, 32)
        assert call_info["return_probs"] is False
        assert call_info["filter_pred"] is True

        # Check result shape matches original image
        assert result.shape == (64, 64, 64)

        # Check that result has some non-zero values where ROI was placed
        assert torch.sum(result) > 0

    def test_infer_with_no_fg_clicks_different_size(self):
        """Test infer method with no foreground clicks or bboxes when image doesn't match ROI size."""
        with pytest.raises(ValueError, match="At least one foreground guidance click or bbox is required"):
            self.roi_inferer.infer(
                self.test_img,
                [],  # No foreground clicks
                self.test_bg_clicks,
                [],
            )

    def test_infer_with_no_fg_clicks_matching_size(self):
        """Test infer method with no foreground clicks when image matches ROI size."""
        roi_sized_img = torch.randn(1, 1, 32, 32, 32)

        result = self.roi_inferer.infer(
            roi_sized_img,
            [],  # No foreground clicks
            self.test_bg_clicks,
            self.test_bboxes,
        )

        # Should work without error
        assert result.shape == (32, 32, 32)

    def test_infer_unguided_mode(self):
        """Test infer method with unguided=True."""
        result = self.roi_inferer.infer(
            self.test_img, self.test_fg_clicks, self.test_bg_clicks, self.test_bboxes, unguided=True
        )

        # Check that infer_on_roi was called with empty guidance
        call_info = self.roi_inferer.call_history[0]
        assert call_info["fg_clicks"] == []
        assert call_info["bg_clicks"] == []
        assert call_info["bboxes"] == []
        assert call_info["filter_pred"] is False

        assert result.shape == (64, 64, 64)

    def test_device_consistency(self):
        """Test that ROI extraction maintains device consistency."""
        if torch.cuda.is_available():
            cuda_img = self.test_img.cuda()
            cuda_roi_inferer = ConcreteROIInferer(roi_size=(32, 32, 32))

            result = cuda_roi_inferer.infer(cuda_img, self.test_fg_clicks, self.test_bg_clicks, self.test_bboxes)

            # Result should be on same device as input
            assert result.device == cuda_img.device

    def test_different_roi_sizes(self):
        """Test ROI inferer with different ROI sizes."""
        # Test with asymmetric ROI
        asymmetric_inferer = ConcreteROIInferer(roi_size=(16, 24, 32))

        result = asymmetric_inferer.infer(self.test_img, self.test_fg_clicks, self.test_bg_clicks, self.test_bboxes)

        # Should still work and return full image size
        assert result.shape == (64, 64, 64)

        # Check that ROI had correct asymmetric size
        call_info = asymmetric_inferer.call_history[0]
        assert call_info["img_shape"] == (1, 1, 16, 24, 32)
