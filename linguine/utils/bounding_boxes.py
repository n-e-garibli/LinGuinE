# Copyright AstraZeneca 2026
"""Utility functions and classes for working with bounding boxes in 2D and 3D."""

from dataclasses import dataclass

import numpy as np
import torch
from typing_extensions import Self


@dataclass
class Point3D:
    """Represent a 3D coordinate voxel."""

    x: int
    y: int
    z: int

    def __post_init__(self):
        """Validation of the attributes of the instance."""
        # Validating that the coordinates are ints.
        assert isinstance(self.x, int)
        assert isinstance(self.y, int)
        assert isinstance(self.z, int)

        # Validating that the values are positive.
        assert self.x >= 0
        assert self.y >= 0
        assert self.z >= 0

    @classmethod
    def from_tuple(cls, point: tuple[int, int, int]) -> Self:
        """Creates a Point3D from a tuple of coordinates."""
        x, y, z = point
        return Point3D(int(x), int(y), int(z))

    def to_tuple(self) -> tuple[int, int, int]:
        """Creates a tuple representation of the point."""
        return (self.x, self.y, self.z)


@dataclass
class BBox3D:
    """Represents a 3D bounding box with minimum and maximum coordinates.

    This class stores the coordinates of a 3D bounding box, defined by its
    minimum and maximum values along the x, y, and z axes.

    Attributes:
        min_point: Minimum (x, y, z) point of the bound box.
        max_point: Maximum (x, y, z) point of the bound box.
    """

    min_point: Point3D
    max_point: Point3D

    @property
    def x_min(self) -> int:
        """Minimum x coordinate of the bounding box."""
        return self.min_point.x

    @property
    def y_min(self) -> int:
        """Minimum y coordinate of the bounding box."""
        return self.min_point.y

    @property
    def z_min(self) -> int:
        """Minimum z coordinate of the bounding box."""
        return self.min_point.z

    @property
    def x_max(self) -> int:
        """Maximum x coordinate of the bounding box."""
        return self.max_point.x

    @property
    def y_max(self) -> int:
        """Maximum y coordinate of the bounding box."""
        return self.max_point.y

    @property
    def z_max(self) -> int:
        """Maximum z coordinate of the bounding box."""
        return self.max_point.z

    @property
    def center(self) -> Point3D:
        x_center = round((self.x_min + self.x_max) / 2)
        y_center = round((self.y_min + self.y_max) / 2)
        z_center = round((self.z_min + self.z_max) / 2)
        return Point3D(x_center, y_center, z_center)

    def __post_init__(self):
        """Validation of the attributes of the instance."""
        # Validating that min < max.
        if not (self.x_min <= self.x_max):
            raise ValueError(f"x_min ({self.x_min}) must be less than or equal to x_max ({self.x_max})")
        if not (self.y_min <= self.y_max):
            raise ValueError(f"y_min ({self.y_min}) must be less than or equal to y_max ({self.y_max})")
        if not (self.z_min <= self.z_max):
            raise ValueError(f"z_min ({self.z_min}) must be less than or equal to z_max ({self.z_max})")

    def to_bounds(self) -> list[tuple[int, int]]:
        """Converts the bounding box to a representation in the form which clearly
        specifies its bounds in each dimension."""
        return [
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (self.z_min, self.z_max),
        ]

    @classmethod
    def from_bounds(cls, bbox: list[tuple[int, int]]) -> Self:
        """Instantiates the class from a bounding box represented by its bounds.

        Args:
            bbox: A bounding box in the form [(x_min, x_max), (y_min, y_max), (z_min, z_max)].

        Returns:
            An instantiated BBox3D object.
        """
        min_point = Point3D(bbox[0][0], bbox[1][0], bbox[2][0])
        max_point = Point3D(bbox[0][1], bbox[1][1], bbox[2][1])
        return BBox3D(min_point, max_point)


@dataclass
class EmptyBBox:
    """Represents an empty/nonexistent BBox."""

    def to_bounds(self) -> list:
        """Returns an empty list to represent no bounds."""
        return []


@dataclass
class BBox2D(BBox3D):
    """Represents a 2D bounding box, which is a 3D bounding box with one fixed dimension.

    This class enforces that one dimension (x, y, or z) must have the same min and max values,
    effectively making it a 2D bounding box.
    """

    def __post_init__(self):
        """Validation of the attributes of the instance."""
        super().__post_init__()

        # Validate that one of the dimensions (x, y, or z) has the same min and max values.
        if self.x_min == self.x_max:
            self.fixed_dimension = "x"
        elif self.y_min == self.y_max:
            self.fixed_dimension = "y"
        elif self.z_min == self.z_max:
            self.fixed_dimension = "z"
        else:
            raise ValueError("One of the dimensions (x, y, or z) must have the same min and max values.")


def get_bounding_box(
    img: np.ndarray | torch.Tensor,
    as_mask: bool = True,
    bbox_2d: bool = False,
) -> BBox3D | EmptyBBox | torch.Tensor:
    """
    Computes the 3D bounding box from the mask image.

    Args:
        img: A 3D array or tensor containing an image or label.
        as_mask: If this is set to true, the function returns a tensor of the same shape as the input
            with the region within the bounding box set to 1 and rest of the voxels set to 0.
        bbox_2d: Whether to have the z bbox bounds only 1 voxel wide at the mean positive z slice.

    Returns:
        A BBox3D/BBox2D object or a binary torch tensor if as_mask is set to True. If the input image has
        no non-zero elements, returns an initialised EmptyBBox.
    """
    # Convert numpy array to tensor if needed
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    # Get non-zero indices
    non_zero_indices = torch.nonzero(img, as_tuple=True)
    if non_zero_indices[0].numel() == 0:
        return EmptyBBox()

    rmin = non_zero_indices[0].min().item()  # width
    rmax = non_zero_indices[0].max().item()
    cmin = non_zero_indices[1].min().item()  # height
    cmax = non_zero_indices[1].max().item()
    zmin = non_zero_indices[2].min().item()  # depth
    zmax = non_zero_indices[2].max().item()

    # If we want the mask, create it
    if as_mask:
        mask = torch.zeros_like(img)
        mask[rmin:rmax, cmin:cmax, zmin:zmax] = 1
        return mask

    # Otherwise return the appropriate bbox object
    if bbox_2d:
        return BBox2D(
            min_point=Point3D(rmin, cmin, (zmin + zmax) // 2),
            max_point=Point3D(rmax, cmax, (zmin + zmax) // 2),
        )
    return BBox3D(
        min_point=Point3D(rmin, cmin, zmin),
        max_point=Point3D(rmax, cmax, zmax),
    )
