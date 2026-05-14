# Copyright AstraZeneca 2026
"""Contains a MetricsBundle object to compute and store segmentation metrics.

Metrics include Dice, Precision, Recall, 95th percentile Hausdorff Distance,
Average Symmetric Surface Distance, Normalized Surface Distance, and Mean Euclidean Distance."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, center_of_mass


def dice(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Dice coefficient between two binary masks.

    The Dice coefficient is a measure of overlap between two sets, commonly used
    in medical image segmentation to evaluate the similarity between predicted
    and ground truth segmentations.

    Args:
        x: Binary mask (predicted segmentation)
        y: Binary mask (ground truth segmentation)

    Returns:
        float: Dice coefficient value between 0 and 1, where 1 indicates perfect overlap
    """
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


def get_surface_points(mask: np.ndarray) -> np.ndarray:
    """
    Extract surface points from a binary mask.

    Surface points are the boundary voxels of the segmented object, obtained by
    finding the difference between the original mask and its morphologically
    eroded version.

    Args:
        mask: 3D binary mask where 1 represents the segmented object

    Returns:
        np.ndarray: Array of shape (N, 3) containing the coordinates of N surface points
    """
    # Convert to boolean for bitwise operations
    mask_bool = mask.astype(bool)
    # Find the boundary by subtracting eroded mask from original
    eroded = binary_erosion(mask_bool)
    boundary = mask_bool & ~eroded
    # Get coordinates of boundary points
    points = np.argwhere(boundary)
    return points


def surface_distances(
    mask: np.ndarray,
    label: np.ndarray,
    tolerance: float = 2.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float, float]:
    """
    Calculate ASSD, HD95, and NSD between two binary masks simultaneously.

    This function efficiently computes Average Symmetric Surface Distance,
    95th percentile Hausdorff Distance, and Normalized Surface Distance by
    calculating surface distances once and applying different statistical measures.

    Args:
        mask: 3D binary mask (predicted segmentation)
        label: 3D binary mask (ground truth segmentation)
        tolerance: Distance threshold in mm for NSD calculation (default: 2.0)
        spacing: Voxel spacing in mm for each dimension (x, y, z) (default: (1.0, 1.0, 1.0))

    Returns:
        A tuple containing (ASSD, HD95, NSD) values.
        ASSD and HD95 are in mm.
        NSD is a proportion between 0 and 1, where 1 indicates all surface points are within tolerance.
    """
    # Get surface points for both masks
    mask_points = get_surface_points(mask)
    label_points = get_surface_points(label)

    if len(mask_points) == 0 or len(label_points) == 0:
        return np.nan, np.nan, np.nan

    # Convert spacing to numpy array for efficient computation
    spacing_array = np.array(spacing)

    # Calculate distances from mask surface to label surface
    distances_1 = []
    for point in mask_points:
        # Apply spacing to convert voxel distances to mm
        diff = (label_points - point) * spacing_array
        min_dist = np.min(np.sqrt(np.sum(diff**2, axis=1)))
        distances_1.append(min_dist)

    # Calculate distances from label surface to mask surface
    distances_2 = []
    for point in label_points:
        # Apply spacing to convert voxel distances to mm
        diff = (mask_points - point) * spacing_array
        min_dist = np.min(np.sqrt(np.sum(diff**2, axis=1)))
        distances_2.append(min_dist)

    # Combine all distances
    all_distances = np.array(distances_1 + distances_2)

    # Calculate metrics from distance array
    assd = np.mean(all_distances)
    hd95 = np.percentile(all_distances, 95)
    nsd = np.sum(all_distances <= tolerance) / len(all_distances)

    return assd, hd95, nsd


def mean_euclidean_distance(
    mask: np.ndarray, label: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Calculate Mean Euclidean Distance (MED) between centers of two binary masks.

    MED measures the Euclidean distance between the centers of mass of the predicted
    and ground truth segmentations. This metric is useful for evaluating the overall
    location accuracy of lesion detection.

    Args:
        mask: 3D binary mask (predicted segmentation)
        label: 3D binary mask (ground truth segmentation)
        spacing: Voxel spacing in mm for each dimension (x, y, z) (default: (1.0, 1.0, 1.0))

    Returns:
        float: Euclidean distance between centers in mm
    """
    # Check if either mask is empty
    if not np.any(mask) or not np.any(label):
        return np.nan

    # Calculate center of mass for both masks
    mask_center = center_of_mass(mask)
    label_center = center_of_mass(label)

    # Calculate Euclidean distance between centers with spacing applied
    spacing_array = np.array(spacing)
    center_diff = (np.array(mask_center) - np.array(label_center)) * spacing_array
    med = np.sqrt(np.sum(center_diff**2))

    return med


@dataclass
class MetricsBundle:
    """
    A dataclass for storing and computing medical image segmentation metrics.

    This class provides a convenient way to calculate and store common evaluation
    metrics used in medical image segmentation tasks, including overlap-based,
    volume-based, and surface-based measures.

    Attributes:
        dice (float): Dice coefficient (0-1), measures overlap similarity
        recall (float): Recall/Sensitivity (0-1), measures true positive rate
        precision (float): Precision (0-1), measures positive predictive value
        hd95 (float): 95th percentile Hausdorff distance in mm, measures worst-case surface error
        assd (float): Average Symmetric Surface Distance in mm, measures average surface error
        nsd (float): Normalized Surface Distance (0-1), proportion of surface points within tolerance
        med (float): Mean Euclidean Distance in mm between predicted and ground truth centers
        confidence (float): Optional confidence score for the segmentation (default: NaN)
        detected_disappearance (float): 1.0 if both pred and label are empty, 0.0 otherwise (or NaN if not computed.)
    """

    dice: float = np.nan
    recall: float = np.nan
    precision: float = np.nan
    hd95: float = np.nan
    assd: float = np.nan
    nsd: float = np.nan
    med: float = np.nan
    confidence: float = np.nan
    detected_disappearance: float = np.nan

    def compute_metrics(
        self,
        mask: np.ndarray,
        label: np.ndarray,
        tolerance: float = 2.0,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """
        Compute all segmentation metrics from two 3D binary masks.

        This method calculates and stores all metrics in the class attributes.
        It computes overlap metrics (dice, recall, precision) and surface-based
        metrics (hd95, assd, nsd) to provide a comprehensive evaluation.

        Args:
            mask: 3D binary mask representing the predicted segmentation
            label: 3D binary mask representing the ground truth segmentation
            tolerance: Distance threshold in mm for NSD calculation (default: 2.0)
            spacing: Voxel spacing in mm for each dimension (x, y, z) (default: (1.0, 1.0, 1.0))
        """
        self.dice = dice(mask, label)

        # Calculate True Positives, False Positives, False Negatives
        tp = np.sum(mask * label)
        fp = np.sum(mask * (1 - label))
        fn = np.sum((1 - mask) * label)

        # Calculate recall (sensitivity)
        if (tp + fn) != 0:
            self.recall = tp / (tp + fn)

        # Calculate precision
        if (tp + fp) != 0:
            self.precision = tp / (tp + fp)

        # Calculate surface distance metrics efficiently (now in mm)
        self.assd, self.hd95, self.nsd = surface_distances(mask, label, tolerance=tolerance, spacing=spacing)

        # Calculate Mean Euclidean Distance between centers (now in mm)
        self.med = mean_euclidean_distance(mask, label, spacing=spacing)

        lesion_present = label.any()

        if not lesion_present and not mask.any():
            # Successfully didn't predict anything
            self.detected_disappearance = 1.0
        elif not lesion_present:
            # No lesion, but there was a prediction
            self.detected_disappearance = 0.0
        # Otherwise keep as nan.
