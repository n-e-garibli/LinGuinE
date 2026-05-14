# Copyright AstraZeneca 2026
"""
ROI-based inference base class for LinGuinE.

This module provides the ROIInferer abstract base class, to facilitate the enablement of
segmentation models that operate on fixed-size Regions of Interest (ROIs) within LinGuinE.
Within the LinGuinE pipeline such an inferer would produce a segmentation only on the ROI
of the correct size defined by the propagated click in the target images.
"""

from abc import abstractmethod

import torch

from linguine.inferers.base_inferer import AbstractInferer
from linguine.utils.bounding_boxes import BBox3D, Point3D


class ROIInferer(AbstractInferer):
    """Abstract base class for inferers that operate on a Region of Interest (ROI) rather than full images.

    This class allows users to define segmentation models that work on fixed-size ROIs while still
    performing inference on full-sized images. The class handles:
    - Extracting ROIs around guidance clicks
    - Converting guidance data to ROI coordinate space
    - Mapping ROI predictions back to full image space

    Subclasses must implement the `roi` property and `infer_on_roi` method.
    """

    def __init__(self, out_of_bounds_roi_handling: str = "pad"):
        """Constructor.

        Args:
            out_of_bounds_roi_handling: one of "pad" or "shift". Determines what to do
                with an ROI crop that doesn't fall entirely within the image bounds.
        """
        assert out_of_bounds_roi_handling in ["pad", "shift"], (
            f"Unknown roi out of bounds handling method: {out_of_bounds_roi_handling}"
        )
        self.oob_roi_handling = out_of_bounds_roi_handling

    @property
    @abstractmethod
    def roi(self) -> tuple[float, float, float]:
        """Defines the image ROI that inferer expects."""
        pass

    @abstractmethod
    def infer_on_roi(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> torch.Tensor:
        """Perform inference on a single ROI.

        Args:
            img: ROI image tensor of shape (B, C, D, H, W).
            fg_clicks: Foreground click coordinates in ROI space.
            bg_clicks: Background click coordinates in ROI space.
            bboxes: Bounding boxes in ROI coordinate space.
            return_probs: Whether to return probabilities instead of binary segmentation.
            filter_pred: Whether to filter prediction to largest connected component.

        Returns:
            ROI prediction tensor.
        """
        pass

    def _extract_roi_around_click(
        self, img: torch.Tensor, click_coords: tuple[int, int, int]
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Extract ROI centered around a guidance click.

        Args:
            img: Input image tensor of shape (B, C, X, Y, Z).
            click_coords: Click coordinates (x, y, z).

        Returns:
            Tuple of (roi_tensor, roi_offset) where roi_offset is the top-left corner of the ROI.
        """
        if self.oob_roi_handling == "pad":
            return self._extract_roi_around_click_pad(img, click_coords)
        elif self.oob_roi_handling == "shift":
            return self._extract_roi_around_click_shift(img, click_coords)
        raise ValueError(f"Unknown out of bounds roi handling mode: {self.oob_roi_handling}")

    def _extract_roi_around_click_shift(
        self, img: torch.Tensor, click_coords: tuple[int, int, int]
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Extract ROI centered around a guidance click.

        Shifts the ROI to ensure it's entirely inside the image bounds, if necessary.

        Args:
            img: Input image tensor of shape (B, C, X, Y, Z).
            click_coords: Click coordinates (x, y, z).

        Returns:
            Tuple of (roi_tensor, roi_offset) where roi_offset is the top-left corner of the ROI.
        """
        # (B, C, X, Y, Z) - remove batch dimension for processing
        img = img.squeeze(0)
        _, x_shape, y_shape, z_shape = img.shape
        x, y, z = click_coords

        # Calculate ROI bounds centered on the click
        roi_x, roi_y, roi_z = int(self.roi[0]), int(self.roi[1]), int(self.roi[2])

        # Calculate ROI bounds with centering
        z_start = max(0, z - roi_z // 2)
        z_end = min(z_shape, z_start + roi_z)
        z_start = max(0, z_end - roi_z)  # Adjust start if we hit the boundary

        y_start = max(0, y - roi_y // 2)
        y_end = min(y_shape, y_start + roi_y)
        y_start = max(0, y_end - roi_y)

        x_start = max(0, x - roi_x // 2)
        x_end = min(x_shape, x_start + roi_x)
        x_start = max(0, x_end - roi_x)

        # Extract ROI
        roi = img[:, x_start:x_end, y_start:y_end, z_start:z_end]

        # Pad if necessary to ensure exact ROI size
        pad_x = roi_x - roi.shape[1]
        pad_y = roi_y - roi.shape[2]
        pad_z = roi_z - roi.shape[3]

        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            # PyTorch pad order: (left, right) for each dimension in reverse order (Z, Y, X)
            padding = (0, pad_z, 0, pad_y, 0, pad_x)
            roi = torch.nn.functional.pad(roi, padding, mode="constant", value=0)

        # Add batch dimension back
        roi = roi.unsqueeze(0)  # (1, C, D, H, W)

        roi_offset = (x_start, y_start, z_start)
        return roi, roi_offset

    def _extract_roi_around_click_pad(
        self, img: torch.Tensor, click_coords: tuple[int, int, int]
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Extract ROI centered around a guidance click.

        Pads regions that fall outside the image with 0s.

        Args:
            img: Input image tensor of shape (B, C, X, Y, Z).
            click_coords: Click coordinates (x, y, z).

        Returns:
            Tuple of (roi_tensor, roi_offset) where roi_offset is the top-left corner of the ROI.
        """
        # (B, C, X, Y, Z) - remove batch dimension for processing
        img = img.squeeze(0)
        _, x_shape, y_shape, z_shape = img.shape
        x, y, z = click_coords

        # Calculate ROI bounds centered on the click
        roi_x, roi_y, roi_z = int(self.roi[0]), int(self.roi[1]), int(self.roi[2])

        # Calculate ROI bounds with centering (allow negative starts for out-of-bounds)
        x_start = x - roi_x // 2
        x_end = x_start + roi_x

        y_start = y - roi_y // 2
        y_end = y_start + roi_y

        z_start = z - roi_z // 2
        z_end = z_start + roi_z

        # Calculate padding needed for out-of-bounds regions
        pad_left_x = max(0, -x_start)
        pad_right_x = max(0, x_end - x_shape)
        pad_left_y = max(0, -y_start)
        pad_right_y = max(0, y_end - y_shape)
        pad_left_z = max(0, -z_start)
        pad_right_z = max(0, z_end - z_shape)

        # Adjust extraction bounds to stay within image
        extract_x_start = max(0, x_start)
        extract_x_end = min(x_shape, x_end)
        extract_y_start = max(0, y_start)
        extract_y_end = min(y_shape, y_end)
        extract_z_start = max(0, z_start)
        extract_z_end = min(z_shape, z_end)

        # Extract the valid region from the image
        roi = img[:, extract_x_start:extract_x_end, extract_y_start:extract_y_end, extract_z_start:extract_z_end]

        # Apply padding to achieve the exact ROI size
        if any([pad_left_x, pad_right_x, pad_left_y, pad_right_y, pad_left_z, pad_right_z]):
            # PyTorch pad order: (left, right) for each dimension in reverse order (Z, Y, X)
            padding = (pad_left_z, pad_right_z, pad_left_y, pad_right_y, pad_left_x, pad_right_x)
            roi = torch.nn.functional.pad(roi, padding, mode="constant", value=0)

        # Add batch dimension back
        roi = roi.unsqueeze(0)  # (1, C, X, Y, Z)

        # ROI offset is the top-left corner of the centered ROI (can be negative)
        roi_offset = (x_start, y_start, z_start)
        return roi, roi_offset

    def _format_guidance_data_to_roi(
        self,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        roi_offset: tuple[int, int, int],
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[BBox3D]]:
        """Format guidance data for ROI inference by converting coordinates to ROI space.

        Args:
            fg_clicks: Foreground click coordinates in original image space.
            bg_clicks: Background click coordinates in original image space.
            bboxes: Bounding boxes in original image space.
            roi_offset: Offset of the ROI from the original image (x_offset, y_offset, z_offset).

        Returns:
            Tuple of (formatted_fg_clicks, formatted_bg_clicks, formatted_bboxes) in ROI coordinates.
            Only guidance data that falls within the ROI bounds is included.
        """
        x_offset, y_offset, z_offset = roi_offset

        # Convert clicks to ROI coordinates
        formatted_fg_clicks = []
        formatted_bg_clicks = []

        # Process foreground clicks
        for x, y, z in fg_clicks:
            roi_x = x - x_offset
            roi_y = y - y_offset
            roi_z = z - z_offset
            # Only include clicks that fall within the ROI
            if 0 <= roi_x < self.roi[0] and 0 <= roi_y < self.roi[1] and 0 <= roi_z < self.roi[2]:
                formatted_fg_clicks.append((roi_x, roi_y, roi_z))

        # Process background clicks
        for x, y, z in bg_clicks:
            roi_x = x - x_offset
            roi_y = y - y_offset
            roi_z = z - z_offset
            # Only include clicks that fall within the ROI
            if 0 <= roi_x < self.roi[0] and 0 <= roi_y < self.roi[1] and 0 <= roi_z < self.roi[2]:
                formatted_bg_clicks.append((roi_x, roi_y, roi_z))

        # Process bounding boxes (convert to ROI coordinates)
        formatted_bboxes = []
        for bbox in bboxes:
            # Convert bbox coordinates to ROI space
            min_x = max(0, bbox.x_min - x_offset)
            min_y = max(0, bbox.y_min - y_offset)
            min_z = max(0, bbox.z_min - z_offset)
            max_x = min(self.roi[0] - 1, bbox.x_max - x_offset)
            max_y = min(self.roi[1] - 1, bbox.y_max - y_offset)
            max_z = min(self.roi[2] - 1, bbox.z_max - z_offset)

            # Only include bbox if it has valid bounds in ROI space
            if min_x <= max_x and min_y <= max_y and min_z <= max_z:
                roi_bbox = BBox3D(
                    min_point=Point3D(int(min_x), int(min_y), int(min_z)),
                    max_point=Point3D(int(max_x), int(max_y), int(max_z)),
                )
                formatted_bboxes.append(roi_bbox)

        return formatted_fg_clicks, formatted_bg_clicks, formatted_bboxes

    def _map_roi_prediction_to_full_image(
        self,
        roi_prediction: torch.Tensor,
        roi_offset: tuple[int, int, int],
        original_image_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Map ROI prediction back to the original image shape.

        Args:
            roi_prediction: Prediction tensor from ROI inference of shape (X, Y, Z).
            roi_offset: Offset of the ROI from the original image (x_offset, y_offset, z_offset).
            original_image_shape: Shape of the original input image (X, Y, Z).

        Returns:
            Full-size prediction tensor with ROI prediction placed at correct location.
            Areas outside the ROI are filled with zeros.
        """
        if self.oob_roi_handling == "shift":
            return self._map_roi_prediction_to_full_image_shift(roi_prediction, roi_offset, original_image_shape)
        elif self.oob_roi_handling == "pad":
            return self._map_roi_prediction_to_full_image_pad(roi_prediction, roi_offset, original_image_shape)
        raise ValueError(f"Unknown out of bounds roi handling mode: {self.oob_roi_handling}")

    def _map_roi_prediction_to_full_image_shift(
        self,
        roi_prediction: torch.Tensor,
        roi_offset: tuple[int, int, int],
        original_image_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Map ROI prediction back to the original image shape.

        Logic is valid if self.oob_roi_handling == shift.

        Args:
            roi_prediction: Prediction tensor from ROI inference of shape (X, Y, Z).
            roi_offset: Offset of the ROI from the original image (x_offset, y_offset, z_offset).
            original_image_shape: Shape of the original input image (X, Y, Z).

        Returns:
            Full-size prediction tensor with ROI prediction placed at correct location.
            Areas outside the ROI are filled with zeros.
        """
        # Create full-size tensor filled with zeros
        full_prediction = torch.zeros(original_image_shape, dtype=roi_prediction.dtype, device=roi_prediction.device)

        roi_x, roi_y, roi_z = roi_prediction.shape
        # Calculate the region in the full image where the ROI prediction should be placed
        x_start, y_start, z_start = roi_offset
        z_end = z_start + roi_z
        y_end = y_start + roi_y
        x_end = x_start + roi_x

        # Ensure we don't exceed original image bounds
        orig_x, orig_y, orig_z = original_image_shape
        z_end = min(z_end, orig_z)
        y_end = min(y_end, orig_y)
        x_end = min(x_end, orig_x)

        # Calculate how much of the ROI prediction to use (in case ROI was padded)
        roi_z_end = z_end - z_start
        roi_y_end = y_end - y_start
        roi_x_end = x_end - x_start

        # Place the ROI prediction in the correct location
        full_prediction[x_start:x_end, y_start:y_end, z_start:z_end] = roi_prediction[
            :roi_x_end, :roi_y_end, :roi_z_end
        ]

        return full_prediction

    def _map_roi_prediction_to_full_image_pad(
        self,
        roi_prediction: torch.Tensor,
        roi_offset: tuple[int, int, int],
        original_image_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Map ROI prediction back to the original image shape.

        Logic is valid if self.oob_roi_handling == pad.

        Args:
            roi_prediction: Prediction tensor from ROI inference of shape (X, Y, Z).
            roi_offset: Offset of the ROI from the original image (x_offset, y_offset, z_offset).
            original_image_shape: Shape of the original input image (X, Y, Z).

        Returns:
            Full-size prediction tensor with ROI prediction placed at correct location.
            Areas outside the ROI are filled with zeros.
        """
        # Create full-size tensor filled with zeros
        full_prediction = torch.zeros(original_image_shape, dtype=roi_prediction.dtype, device=roi_prediction.device)

        roi_x, roi_y, roi_z = roi_prediction.shape
        orig_x, orig_y, orig_z = original_image_shape
        x_offset, y_offset, z_offset = roi_offset

        # Calculate overlapping region between ROI and original image
        # Destination bounds in full image (clamp to valid range)
        dest_x_start = max(0, x_offset)
        dest_x_end = min(orig_x, x_offset + roi_x)
        dest_y_start = max(0, y_offset)
        dest_y_end = min(orig_y, y_offset + roi_y)
        dest_z_start = max(0, z_offset)
        dest_z_end = min(orig_z, z_offset + roi_z)

        # Source bounds in ROI prediction (account for negative offsets)
        src_x_start = max(0, -x_offset)
        src_x_end = src_x_start + (dest_x_end - dest_x_start)
        src_y_start = max(0, -y_offset)
        src_y_end = src_y_start + (dest_y_end - dest_y_start)
        src_z_start = max(0, -z_offset)
        src_z_end = src_z_start + (dest_z_end - dest_z_start)

        # Only copy if there's an overlapping region
        if dest_x_end > dest_x_start and dest_y_end > dest_y_start and dest_z_end > dest_z_start:
            full_prediction[dest_x_start:dest_x_end, dest_y_start:dest_y_end, dest_z_start:dest_z_end] = roi_prediction[
                src_x_start:src_x_end, src_y_start:src_y_end, src_z_start:src_z_end
            ]
        return full_prediction

    def infer(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
        unguided: bool = False,
    ) -> torch.Tensor:
        """Generate a model prediction on the full image using ROI-based inference.

        This method extracts an ROI around the first foreground click or bbox, performs inference
        on that ROI, and maps the result back to the full image dimensions.

        Args:
            img: Input image tensor of shape (B, C, D, H, W) where B must be 1.
            fg_clicks: List of foreground guidance clicks in full image coordinates.
                       At least one click is required unless image already matches ROI size.
            bg_clicks: List of background guidance clicks in full image coordinates.
            bboxes: List of bounding boxes to use as guidance in full image coordinates.
            return_probs: Whether to return foreground class probabilities instead
                         of binary segmentation.
            filter_pred: Whether to filter the prediction to only keep the
                        connected component at the click location.
            unguided: If True, performs unguided inference (clicks only used for ROI definition).

        Returns:
            Full-size prediction tensor matching the spatial dimensions of the input image.
        """
        # Store original image shape for output mapping
        original_shape = img.shape
        assert original_shape[0] == 1, f"Currently only supported for batch size 1, got {original_shape[0]}"

        if not fg_clicks:
            if len(bboxes) == 0:
                if not all(img_shape == roi_shape for img_shape, roi_shape in zip(img.shape[2:], self.roi)):
                    raise ValueError(
                        "At least one foreground guidance click or bbox is required for this inferer if the image is not shaped like the ROI."
                    )
                return self.infer_on_roi(
                    img=img, fg_clicks=[], bg_clicks=[], bboxes=[], return_probs=return_probs, filter_pred=False
                )
            else:
                # Use center of first bbox to define click to crop ROI around.
                first_bbox = bboxes[0].to_bounds()
                roi_center = tuple([(x + j) // 2 for x, j in first_bbox])
        else:
            # Will extract ROI around the first guidance click
            roi_center = fg_clicks[0]

        roi_tensor, roi_offset = self._extract_roi_around_click(img, roi_center)

        # Format guidance data for the ROI
        formatted_fg_clicks, formatted_bg_clicks, formatted_bboxes = self._format_guidance_data_to_roi(
            fg_clicks, bg_clicks, bboxes, roi_offset
        )

        # Move tensors to correct device and add expected batch dimensions where necessary.
        roi_tensor = roi_tensor.to(img.device)
        if unguided:
            roi_prediction = self.infer_on_roi(
                img=roi_tensor, fg_clicks=[], bg_clicks=[], bboxes=[], return_probs=return_probs, filter_pred=False
            )
        else:
            roi_prediction = self.infer_on_roi(
                img=roi_tensor,
                fg_clicks=formatted_fg_clicks,
                bg_clicks=formatted_bg_clicks,
                bboxes=formatted_bboxes,
                return_probs=return_probs,
                filter_pred=filter_pred,
            )
        full_prediction = self._map_roi_prediction_to_full_image(roi_prediction, roi_offset, original_shape[-3:])
        return full_prediction
