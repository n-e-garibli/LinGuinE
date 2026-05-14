# Copyright AstraZeneca 2026
"""Disappearance detectors for filtering false positive lesion predictions.

This module contains classes designed to detect false positive predictions in areas where
tumours have actually disappeared. These detectors help improve lesion prediction pipeline
accuracy by filtering out unreasonable predictions based on various criteria like size,
intensity values, and composite logic.
"""

import numpy as np
import pandas as pd
import torch
from monai.data.meta_tensor import MetaTensor

from linguine.utils.data import get_spacing_from_metatensor


class DisappearanceDetector:
    """Base class for detecting false positive lesion predictions where tumours have actually disappeared."""

    def tumour_present(
        self,
        lesion_pred: torch.Tensor,
        target_image: torch.Tensor,
        *args,
        **kwargs,
    ) -> bool:
        """Check if a tumour is present in the predicted lesion region.

        Base implementation always returns True. Subclasses should override
        to implement specific detection logic.

        Args:
            lesion_pred: 5D binary tensor indicating predicted lesion locations
            target_image: 5D image tensor
            *args, **kwargs: For easy extension/compatibility of interfaces.

        Returns:
            True if tumour is considered present, False if likely disappeared
        """
        return True


class CompositeDisappearanceDetector(DisappearanceDetector):
    """Combines multiple disappearance detectors using AND logic - all must agree tumour is present."""

    def __init__(self, dds: list[DisappearanceDetector]):
        """Initialize with a list of disappearance detectors.

        Args:
            dds: List of DisappearanceDetector instances to combine
        """
        self.dds = dds

    def tumour_present(self, *args, **kwargs) -> bool:
        """Check if tumour is present using all constituent detectors.

        All detectors must agree that the tumour is present for this method
        to return True.

        Args:
            *args: Arguments passed to each detector's tumour_present method
            **kwargs: Keyword arguments passed to each detector's tumour_present method

        Returns:
            True only if all detectors agree tumour is present
        """
        for dd in self.dds:
            if not dd.tumour_present(*args, **kwargs):
                return False
        return True


class SizeFilter(DisappearanceDetector):
    """Filters out predictions that are too small based on axial length measurements."""

    def __init__(self, minimum_axial_length_mm: float):
        """Constructor.

        Args:
            minimum_axial_length: Smallest acceptable size in mm along the axial axis to be considered "big enough".
        """
        self.minimum_axial_length_mm = minimum_axial_length_mm

    def tumour_present(self, lesion_pred: torch.Tensor, target_image: MetaTensor, *args, **kwargs) -> bool:
        """Check if predicted lesion is large enough to be considered a real tumour.

        Args:
            lesion_pred: 5D binary tensor indicating predicted lesion locations
            target_image: 5D image metatensor with spacing metadata

        Returns:
            True if lesion exceeds minimum axial length threshold
        """
        numpy_3D_pred = lesion_pred.cpu().numpy()[0]
        spacing = get_spacing_from_metatensor(target_image)
        return self._is_tumour_big_enough(voxel_spacing=spacing, label_array=numpy_3D_pred, label_array_value=1.0)

    @staticmethod
    def maximal_axial_length(
        voxel_spacing: tuple[float, float, float],
        label_array: np.ndarray,
        label_array_value: int,
    ) -> float:
        """
        Maximal axial length is an attempt to capture the same metric used universally by RECIST radiologists to assess tumour burden.
        Rads will measure a tumours longest axis (shortest axis for nodal disease) along the axial axis.
        This measurement is then used to monitor tumour burden throughout a drug trial.
        Being able to compute this is useful as it will help with validation of data/other features and
        it will allow us to communicate, more effectively, model performance to stakeholders (by using vol features they are familiar with)

        Args:
            voxel_spacing:  array of voxel_spacing = [x_spacing, y_spacing, z_spacing]
            label_array:  3d segmentation array with elements = lesion. Format of label_array.shape = [x_dim, y_dim, z_dim]
            label_array_value:  Lesion value for measurement computation.

        Returns:
            The maximum length of a lesion within the axial view plane. This should be equivalent to an accurate RECIST measurement.

        Note: Ensure that the spacing and label shape matches the above format for proper function.
        """
        # NOTE(Nadine): This function is complex for reasons that has nothing to do with this repo (look at the
        # docstring lol). Can simplify and stop using pandas.

        if not np.any(label_array == label_array_value):
            print(f"No lesion of value {label_array_value}.")
            return 0

        # Define spacing for clarity
        # Ensure voxel_spacing elements are floats
        voxel_spacing = [float(spacing) for spacing in voxel_spacing]
        x_spacing = voxel_spacing[0]
        y_spacing = voxel_spacing[1]

        # Find all the label coordinates
        coordinate_ls = np.array(np.where(label_array == label_array_value)).transpose().tolist()

        # Organise according to intercepting plane
        coord_dataframe = pd.DataFrame(coordinate_ls, columns=["Sagittal", "Coronal", "Axial"])

        volume_max_axial_distance_mm = 0

        # Check for the maximal lesion distance in the orthogonal plane
        for axial_slice in coord_dataframe["Axial"].unique():
            slice_df = coord_dataframe[coord_dataframe.Axial == axial_slice]
            x_max = np.max(slice_df.Sagittal)
            x_min = np.min(slice_df.Sagittal)
            y_max = np.max(slice_df.Coronal)
            y_min = np.min(slice_df.Coronal)

            x_dist = (x_max - x_min) * x_spacing
            y_dist = (y_max - y_min) * y_spacing

            max_longitudinal_distance_mm = np.sqrt(np.square(x_dist) + np.square(y_dist))
            if max_longitudinal_distance_mm > volume_max_axial_distance_mm:
                volume_max_axial_distance_mm = max_longitudinal_distance_mm

        # Return the largest
        return volume_max_axial_distance_mm

    def _is_tumour_big_enough(
        self,
        voxel_spacing: tuple[float, float, float],
        label_array: np.ndarray,
        label_array_value: int,
    ) -> bool:
        """Checks whether a tumour is bigger than the length specified.
        Args:
            voxel_spacing: array of voxel_spacing = (x_spacing, y_spacing, z_spacing)
            label_array: 3d segmentation array with elements = lesion. Format of label_array.shape = [x_dim, y_dim, z_dim]
            label_array_value: Lesion value for measurement computation.

        Returns:
            A bool specifying whether the input lesion is bigger than the minimum_axial_length.
        """
        if self.minimum_axial_length_mm <= 0:
            return True

        size = self.maximal_axial_length(
            voxel_spacing=voxel_spacing,
            label_array=label_array,
            label_array_value=label_array_value,
        )
        if size > self.minimum_axial_length_mm:
            return True
        return False


class MeanValueInRange(DisappearanceDetector):
    """Filters predictions based on mean intensity values within the predicted lesion region."""

    def __init__(
        self,
        l_threshold: float | int = -650,
        u_threshold: float | int = 350,
    ):
        """Initialize the threshold-based disappearance detector.

        Args:
            l_threshold: The threshold above which a voxel is considered
                foreground.
            u_threshold: The upper threshold for foreground voxels.
        """
        self.l_threshold = l_threshold
        self.u_threshold = u_threshold
        assert l_threshold < u_threshold, "Invalid thresholds: l_threshold must be less than u_threshold."

    def tumour_present(self, lesion_pred: torch.Tensor, target_image: torch.Tensor, *args, **kwargs) -> bool:
        """Check if mean intensity in predicted lesion region is within expected range.

        Args:
            lesion_pred: 5D binary tensor indicating predicted lesion locations
            target_image: 5D Image tensor

        Returns:
            True if mean intensity is within the defined threshold range
        """
        assert target_image.ndim == 5, f"Expect 5D image, got {target_image.ndim}"
        assert lesion_pred.ndim == 5, f"Expect 5D prediction, got {lesion_pred.ndim}"
        value = target_image[lesion_pred.bool()].mean()
        if self.u_threshold >= value >= self.l_threshold:
            return True
        return False
