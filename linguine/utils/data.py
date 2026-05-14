# Copyright AstraZeneca 2026
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Spacing

from linguine.utils.bounding_boxes import BBox2D, BBox3D, Point3D


def get_path_from_data_dict_entry(data_dict: dict[str, Any], key: str) -> Path:
    """Extract a Path object from a data dictionary entry.

    Args:
        data_dict: Dictionary containing data entries.
        key: The key to look up in the data dictionary.

    Returns:
        Path object extracted from the dictionary entry, or None if key not found
        or value cannot be converted to a Path.
    """
    if key not in data_dict:
        return None
    value = data_dict[key]
    if isinstance(value, Path):
        return value
    if isinstance(value, tuple):
        return Path(value[0])
    elif isinstance(value, str):
        return Path(value)
    else:
        return None


def get_spacing_from_metatensor(
    img: MetaTensor,
) -> tuple[float, float, float] | None:
    """Extracts metatensor spacing if possible.

    Args:
        img: a monai MetaTensor.

    Returns:
        The (x,y,z) image spacing if successfully found in meta or affine, otherwise None.
        For affine matrices, spacing is calculated as the magnitude of each column vector
        to correctly handle both canonical and non-canonical image orientations.
    """
    if "spacing" in img.meta:
        # This accounts for Monai sometimes having a "batch" dimension in the
        # meta and sometimes not...
        if len(img.meta["spacing"]) == 3:
            spacing = [t.item() if isinstance(t, torch.Tensor) else t for t in img.meta["spacing"]]
        else:
            spacing = [t.item() if isinstance(t, torch.Tensor) else t for t in img.meta["spacing"][0]]
        spacing = tuple(spacing)
    elif hasattr(img, "affine") and img.affine is not None:
        # Fallback: extract spacing from affine matrix
        # For non-canonical orientations, spacing is the magnitude of each column vector
        try:
            if img.affine.ndim == 2:
                affine_matrix = img.affine[:3, :3]  # Extract 3x3 rotation/scaling matrix
            elif img.affine.ndim == 3:
                affine_matrix = img.affine[0][:3, :3]  # Extract 3x3 rotation/scaling matrix
            else:
                raise ValueError(f"Invalid affine matrix: {img.affine}")
            spacing = tuple(float(np.linalg.norm(affine_matrix[:, i])) for i in range(3))
        except (AttributeError, IndexError, TypeError):
            spacing = None
    else:
        spacing = None
    return spacing


def crop_to_same_size(
    arrays: Iterable[torch.Tensor | np.ndarray],
) -> list[torch.Tensor | np.ndarray]:
    """Given the input tensors, if they are different shapes, will crop them into
    into the shape of the smallest one and return a list of the cropped tensors.

    Args:
        arrays: an iterable containing arrays the shape of which should be standardized.

    Returns:
        An updated list of cropped arrays all of which are the same size.
    """
    smallest_shape = [min(ds) for ds in zip(*[a.shape for a in arrays])]
    slices = tuple([slice(0, d) for d in smallest_shape])
    return [array[slices] for array in arrays]


def transform_coordinate_spacing(
    coor: tuple[int],
    original_spacing: tuple[float],
    desired_spacing: tuple[float],
) -> tuple[int, int, int]:
    """Transforms input coordinate from its original spacing to the desired one.

    Args:
        coor: a tuple of integers indicating a coordinate in an image.
        original_spacing: the spacing of the image that the coordinate belongs to.
        desired_spacing: the spacing that the coordinate should be transformed to.

    Returns:
        The equivalent coordinate in the desired spacing.
    """
    assert len(coor) == len(original_spacing), "Spacing must be provided for all dimensions of input coordinate"
    assert len(coor) == len(desired_spacing), "Desired spacing must be provided for all dimensions of input coordinate"
    new_coor = []
    for c, original_s, new_s in zip(coor, original_spacing, desired_spacing):
        new_c = round(c * original_s / new_s)
        new_coor.append(new_c)
    return tuple(new_coor)


def prepare_inputs_for_inferer(
    desired_spacing: tuple[int, int, int] | None,
    image: MetaTensor,
    clicks: list[tuple[int, int, int]] | None = None,
    bboxes: list[BBox3D] | None = None,
    **kwargs,
) -> tuple[MetaTensor, list[tuple[int, int, int]], list[BBox3D]]:
    """Prepares inputs for inference by transforming image and coordinates to desired spacing.

    Args:
        desired_spacing: Target spacing for the image as (x, y, z) tuple.
        image: Input 5D (1, 1, x, y, z) MetaTensor image to be resampled.
        clicks: Optional list of click coordinates as (x, y, z) tuples.
        bboxes: Optional list of 3D bounding boxes.
        **kwargs: Additional keyword arguments passed to the Spacing transform.

    Returns:
        Tuple containing:
        - Resampled image with desired spacing
        - List of transformed click coordinates (empty if clicks is None)
        - List of transformed bounding boxes (empty if bboxes is None)
    """
    if desired_spacing is None:
        # Don't apply any transformations - assumed inferer can handle any spacing.
        return image, clicks if clicks is not None else [], bboxes if bboxes is not None else []

    assert image.ndim == 5, f"5D MetaTensor expected as input got {image.ndim}"
    spacing_transform = Spacing(pixdim=desired_spacing, **kwargs)
    spaced_image = spacing_transform(image[0]).unsqueeze(0)
    og_spacing = get_spacing_from_metatensor(image)

    # Create coordinate transformation function
    coor_spacing_trans = partial(
        transform_coordinate_spacing, original_spacing=og_spacing, desired_spacing=desired_spacing
    )

    outputs = [spaced_image, [], []]
    if clicks is not None:
        spaced_clicks = [coor_spacing_trans(c) for c in clicks]
        outputs[1] = spaced_clicks

    if bboxes is not None:
        spaced_bboxes = []
        for bbox in bboxes:
            new_min_point = Point3D.from_tuple(
                coor_spacing_trans((bbox.min_point.x, bbox.min_point.y, bbox.min_point.z))
            )
            new_max_point = Point3D.from_tuple(
                coor_spacing_trans((bbox.max_point.x, bbox.max_point.y, bbox.max_point.z))
            )
            if isinstance(bbox, BBox2D):
                spaced_bboxes.append(BBox2D(new_min_point, new_max_point))
            elif isinstance(bbox, BBox3D):
                spaced_bboxes.append(BBox3D(new_min_point, new_max_point))
        outputs[2] = spaced_bboxes

    return outputs[0], outputs[1], outputs[2]
