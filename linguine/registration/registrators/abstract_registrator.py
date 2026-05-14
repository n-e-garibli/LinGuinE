# Copyright AstraZeneca 2026
"""An abstract class for extracting a set of 3D landmark coordinates from an input, typically a 3D medical scan.
This is the parent class for rigid registration algorithms that rely on these landmarks as common reference points.
These points are used to calculate the rotation matrix and the translation vector that are then applied to the
source image to transform it into the target image space.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial

import numpy as np
import scipy.spatial
from monai.data.meta_tensor import MetaTensor

from linguine.registration.landmark import LandmarkCoordinate
from linguine.utils.data import get_spacing_from_metatensor


def extract_valid_coors(
    new_coors: np.ndarray,
    target_shape: tuple[int, int, int],
    og_coors: np.ndarray | None = None,
) -> tuple[np.ndarray] | tuple[tuple[np.ndarray], tuple[np.ndarray]]:
    """
    This function filters a set of coordinates based on the shape of the image that they must belong to.
    Any coordinate that is out of bounds will be ignored.
    Args:
        new_coors: an array of mapped coordinates in the space of the target image.
        target_shape: the shape of the target image.
        og_coors: an array of original coordinates in the space of the source image (likely
            the coordinates used to obtain new_coors).
    Returns:
        The valid x, y, and z coordinates respectively in the new coors array that fall within the
        bounds of the target image. If og_coors is provided, an additional tuple with the x, y, and z
        corresponding coordinates in the source image will also be returned.
    """
    x = new_coors[0]
    y = new_coors[1]
    z = new_coors[2]
    valid_mask = (
        (x >= -0.5)
        & (x < target_shape[0] - 0.5)
        & (y >= -0.5)
        & (y < target_shape[1] - 0.5)
        & (z >= -0.5)
        & (z < target_shape[2] - 0.5)
    )
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = z[valid_mask]
    if og_coors is not None:
        x_og_valid = og_coors[0][valid_mask]
        y_og_valid = og_coors[1][valid_mask]
        z_og_valid = og_coors[2][valid_mask]
        return (
            (x_valid, y_valid, z_valid),
            (x_og_valid, y_og_valid, z_og_valid),
        )
    return (x_valid, y_valid, z_valid)


def _map_segmentation_with_mapper(
    mapper,
    target_img_shape: tuple[int, int, int],
    segmentation: np.ndarray,
) -> np.ndarray:
    """
    Maps segmentation from the space of the source image into the space of the target using a provided mapper function.
    Args:
        mapper: a function that transforms coordinates from the space of one image to another.
        target_img_shape: the shape that the mapped segmentation must take.
        segmentation: a 3D binary array containing an arbitrary segmentation.
    Returns:
        A 3D array with the mapped segmentation.
    """
    if not np.any(segmentation):
        print("Segmentation array is empty. Generating an empty segmentation in the target shape...")
        return np.zeros(target_img_shape)

    for idx, lesion_no in enumerate([*np.unique(segmentation)][1:]):
        xs, ys, zs = np.where(segmentation == lesion_no)
        seg_points = np.vstack([xs, ys, zs])
        new_coors = mapper(coors=seg_points.T)

        x_valid, y_valid, z_valid = extract_valid_coors(new_coors=new_coors, target_shape=target_img_shape)
        new_seg = np.zeros(target_img_shape)
        new_seg[
            np.round(x_valid).astype(int),
            np.round(y_valid).astype(int),
            np.round(z_valid).astype(int),
        ] = 1

        if idx == 0:
            out_seg = new_seg.copy() * lesion_no
        else:
            out_seg[new_seg.astype(bool)] = lesion_no

    return out_seg


class ImageRegistrator(ABC):
    """Abstract class for image registration."""

    @abstractmethod
    def map_coordinates(
        self,
        source_image: MetaTensor,
        target_image: MetaTensor,
        coors: list[tuple[int, int, int]],
        round_: bool = True,
        **kwargs,
    ) -> list[tuple[int, int, int]]:
        """Maps coordinates from the space of the source image to the space of the target.

        Args:
            source_image: A 5D MetaTensor (1,1,x,y,z) containing the source image.
            target_image: A 5D MetaTensor (1,1,x,y,z) containing the target image.
            coors: a list of coordinates (x,y,z) to map.
            round_: a bool specifying whether to round the coordinates to integers for easy indexing

        Returns:
            The coordinates in the space of the target image.
        """
        pass

    def map_segmentation(
        self,
        source_image: MetaTensor,
        target_image: MetaTensor,
        target_img_shape: tuple[int, int, int],
        segmentation: np.ndarray,
    ) -> np.ndarray:
        """Maps segmentation from the space of the source image into the space of the target."""
        mapper = partial(
            self.map_coordinates,
            source_image=source_image,
            target_image=target_image,
        )
        return _map_segmentation_with_mapper(mapper, target_img_shape, segmentation)


class PointSetRegistrator(ABC):
    """Abstract class for image registration using point set algorithms.

    Relies on the existence of landmarks in both images with known correspondence.
    Contains methods for loading images, mapping them into the space of other images,
    and visualising registrations.
    """

    def __init__(
        self,
        valid_landmarks: Iterable[str] | None = None,
    ) -> None:
        """Initialises the class.

        Args:
            valid_landmarks: A set of strings representing landmarks that can be used for registration.
                If not provided, all available points will be used.
        """
        # A set of strings representing the landmarks that the registrator can use for
        # registration.
        self.valid_landmarks: Iterable[str] | None = valid_landmarks

    @abstractmethod
    def map_coordinates(
        self,
        source_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float],
        source_landmarks: dict[str, LandmarkCoordinate],
        target_landmarks: dict[str, LandmarkCoordinate],
        coors: list[tuple[int, int, int]] | np.ndarray,
        round_: bool = True,
    ) -> list[tuple[int, int, int]] | np.ndarray:
        """Maps coordinates from the space of the source image to the space of the target.

        Args:
            source_spacing: A tuple of (x,y,z) voxel spacing in the source image.
            target_spacing: A tuple of (x,y,z) voxel spacing in the target image.
            source_landmarks: a dictionary of landmarks available for the source image.
            target_landmarks: a dictionary of landmarks available for the target image.
            coors: a list or numpy array of coordinates (x,y,z) to map.
            round_: a bool specifying whether to round the coordinates to integers for easy indexing

        Returns:
            The coordinates in the space of the target image in the same format as they were inputted.
        """
        pass

    def compute_dist(
        self,
        source_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float],
        source_landmarks: dict[str, LandmarkCoordinate],
        target_landmarks: dict[str, LandmarkCoordinate],
        metric: str = "euclidean",
    ) -> float:
        """To evaluate the registration, this function computes the distance in
        voxel space between the landmark coordinates used for registration in the target image
        and the registered image (as mapped from the source).

        Args:
            source_spacing: A tuple of (x,y,z) voxel spacing in the source image.
            target_spacing: A tuple of (x,y,z) voxel spacing in the target image.
            source_landmarks: a dictionary of landmarks available for the source image.
            target_landmarks: a dictionary of landmarks available for the target image.
            metric: the distance metric to use (as accepted by scipy cdist)
        Returns:
            The average distance between all the landmark coordinates."""

        source_points, target_points = self.get_ordered_points(source_landmarks, target_landmarks)
        # Return 0 if no matching landmarks found
        if len(source_points) == 0:
            return 0.0
        registered_points = self.map_coordinates(
            source_spacing,
            target_spacing,
            source_landmarks,
            target_landmarks,
            source_points,
            round_=False,
        ).T
        assert len(target_points) == len(registered_points), (
            "Must get the same number of registered points as target points."
        )
        # Compute distance betwen each registered point and the actual corresponding point.
        dists = scipy.spatial.distance.cdist(registered_points, target_points, metric=metric)
        return np.mean(np.diag(dists))

    def register(
        self,
        source: MetaTensor,
        target: MetaTensor,
        source_landmarks: dict[str, LandmarkCoordinate],
        target_landmarks: dict[str, LandmarkCoordinate],
    ) -> np.ndarray:
        """
        Registers two images by mapping the source image into the space of the of the target.
        The source and target folder must contain the subdirectory 'image' containing dcm slices for
        the images to be registered.

        Args:
            source: A 3D source image MetaTensor.
            target: A 3D target image MetaTensor.
            source_landmarks: a dictionary of landmarks available for the source image.
            target_landmarks: a dictionary of landmarks available for the target image.

        Returns:
            A 3D array with the source image mapped into the space of the target image.
        """
        assert source.ndim == 3, f"source image must be 3d got {source.ndim}d"
        assert target.ndim == 3, f"target image must be 3d got {target.ndim}d"
        target_img: np.ndarray = target.cpu().numpy()
        source_img: np.ndarray = source.cpu().numpy()
        source_x, source_y, source_z = source_img.shape

        # Generate grid of points for the source image
        x, y, z = np.meshgrid(
            np.arange(source_x),
            np.arange(source_y),
            np.arange(source_z),
            indexing="ij",
        )
        source_points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        # Transform the coordinates into the space of the target image.
        registered_points = self.map_coordinates(
            source_spacing=get_spacing_from_metatensor(source),
            target_spacing=get_spacing_from_metatensor(target),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=source_points.T,
        )

        # Transform the image
        registered_image = self.map_image(
            img=source_img,
            og_coors=source_points,
            new_coors=registered_points,
            target_shape=target_img.shape,
        )
        assert registered_image.shape == target_img.shape, (
            f"Shape mismatch between target image and registered image ({target_img.shape} and {registered_image.shape})"
        )
        return registered_image

    def get_ordered_points(
        self,
        dict1: dict[str, LandmarkCoordinate],
        dict2: dict[str, LandmarkCoordinate],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        From two dictionaries, returns two arrays containing their values for the keys
        that exist in both dictionaries. The arrays will have the same order, meaning that
        the elements with the same key in the original dictionaries will be in the same
        position in the returned arrays.

        This allows the mapping of landmarks between two images, finding and identifying
        instances common to both images.

        Args:
            dict1, dict2: The two dictionaries to consider.

        Returns:
            Two nx3 arrays containing the values in the input dictionaries for the keys
            that exist in both dictionaries.
        """
        P: list[tuple[int, int, int]] = []
        Q: list[tuple[int, int, int]] = []
        for key in dict1.keys():
            if self.valid_landmarks is not None:
                if key not in self.valid_landmarks:
                    continue
            if dict1[key].is_valid and key in dict2 and dict2[key].is_valid:
                P.append((dict1[key].x, dict1[key].y, dict1[key].z))
                Q.append((dict2[key].x, dict2[key].y, dict2[key].z))
        return np.array(P), np.array(Q)

    def map_image(
        self,
        img: np.ndarray,
        og_coors: np.ndarray,
        new_coors: np.ndarray,
        target_shape: tuple[int, int, int],
        pad_value: int = 0,
    ) -> np.ndarray:
        """Maps the input image into the space of the target image using the transformed coordinates.
        This version simply uses rounding to handle mapped coordinates that are not integers. This can
        be improved by using some form of interpolation instead.

        Args:
            img: a 3D numpy array to be mapped.
            og_coors: the coordinates of the original image.
            new_coors: a 3D numpy array containing transformed coordinates of the original image.
            target_shape: the shape that the mapped image must take. Image will be cropped
                and padded with 0s to make it fit if needed.
            pad_value: value to use for padding if image shape and target_shape are different.

        Returns:
            A 3D array with the transformed image.
        """

        new_valid_coors, og_valid_coors = extract_valid_coors(
            new_coors=new_coors, og_coors=og_coors, target_shape=target_shape
        )
        x_valid, y_valid, z_valid = new_valid_coors
        x_og_valid, y_og_valid, z_og_valid = og_valid_coors

        mapped_image = np.ones(target_shape) * pad_value
        mapped_image[
            np.round(x_valid).astype(np.uint16),
            np.round(y_valid).astype(np.uint16),
            np.round(z_valid).astype(np.uint16),
        ] = img[x_og_valid, y_og_valid, z_og_valid]

        return mapped_image

    def map_segmentation(
        self,
        source_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float],
        target_img_shape: tuple[int, int, int],
        segmentation: np.ndarray,
        source_landmarks: dict[str, LandmarkCoordinate],
        target_landmarks: dict[str, LandmarkCoordinate],
    ) -> np.ndarray:
        """Maps segmentation from the space of the source image into the space of the target."""
        mapper = partial(
            self.map_coordinates,
            source_spacing=source_spacing,
            target_spacing=target_spacing,
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
        )
        return _map_segmentation_with_mapper(mapper, target_img_shape, segmentation)
