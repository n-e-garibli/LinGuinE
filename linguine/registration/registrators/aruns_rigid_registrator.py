# Copyright AstraZeneca 2026
"""This file contains the class for performing 3D rigid registration with Aruns
method on two images.
The aim is to map one image (the source image) into the same space as another (the target image)
such that both images appear to have been taken under identical conditions (angle, resolution etc. ),
thus allowing images to be directly compared"""

from collections.abc import Iterable

import numpy as np

from linguine.registration.landmark import LandmarkCoordinate
from linguine.registration.registrators import PointSetRegistrator


class ArunsRigidRegistrator(PointSetRegistrator):
    """A class for performing rigid registration using Aruns method.
    See https://jingnanshi.com/blog/arun_method_for_3d_reg.html for a clear explanation
    and https://bycore.net/usr/uploads/2021/12/2476831585.pdf for Aruns' original paper.
    This allows for rotations and translations based on known point correspondence.
    """

    def __init__(
        self,
        valid_landmarks: Iterable[str] | None = None,
    ) -> None:
        """Initialises the ArunsRigidRegistrator class.

        Args:
            valid_landmarks: A set of strings representing landmarks that can be used for registration.
                If not provided, all available points will be used.
        """
        # A set of strings representing the landmarks that the registrator can use for
        # registration.
        self.valid_landmarks: Iterable[str] | None = valid_landmarks

    def get_rigid_transformation(
        self,
        source_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float],
        source_landmarks: dict[str, LandmarkCoordinate],
        target_landmarks: dict[str, LandmarkCoordinate],
    ) -> tuple[np.ndarray, np.ndarray]:
        """A function for computing the rotation matrix (R) and translation vector (t) to define a transformation
        from the source to the target. Returns a tuple of (R, t).

        Args:
            source_spacing: A tuple of (x,y,z) voxel spacing in the source image.
            target_spacing: A tuple of (x,y,z) voxel spacing in the target image.
            source_landmarks: a dictionary of landmarks available for the source image.
            target_landmarks: a dictionary of landmarks available for the target image.
        """
        P, Q = self.get_ordered_points(source_landmarks, target_landmarks)
        # Transform to real world.
        P = P * np.array(source_spacing).reshape(1, 3)
        Q = Q * np.array(target_spacing).reshape(1, 3)
        P_centroid = P.mean(axis=0)
        Q_centroid = Q.mean(axis=0)
        P_centered = P - P_centroid
        Q_centered = Q - Q_centroid
        R: np.ndarray = self.get_rotation_matrix(P_centered, Q_centered)
        t: np.ndarray = self.get_translation_vector(P_centroid, Q_centroid, R)
        return (R, t)

    def get_rotation_matrix(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Computes and returns the rotation matrix to go from the space of points P to
        the space of points Q.

        Args:
            P: an array containing mean centered coordinates in the source image.
            Q: an array containing mean centered corresponding coordinates in the target image.
        """
        # Computing the covariance matrix
        H: np.ndarray = np.matmul(P.T, Q)
        # Perform singular value decomposition
        U, _, v_t = np.linalg.svd(H)
        # Compute rotation matrix.
        R = np.matmul(v_t.T, U.T)
        if np.linalg.det(R) < 0:  # Need to ensure right hand coordinate system
            v_t[-1, :] *= -1  # Invert the sign of the last row of V^T
            R = np.matmul(v_t.T, U.T)
        return R

    def get_translation_vector(self, p_centroid: np.ndarray, q_centroid: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Computes and returns the translation vector to go from points P to points Q,
        using the centroids of the points and the rotation matrix.

        Args:
            p_centroid: The coordinate representing the centroid of landmarks in the source image.
            q_centroid: The coordinate representing the centroid of landmarks in the target image.
            R: The rotation matrix computed from the landmarks.
        """
        return q_centroid - np.matmul(R, p_centroid)

    def map_coordinates(
        self,
        source_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float],
        source_landmarks: dict[str, LandmarkCoordinate],
        target_landmarks: dict[str, LandmarkCoordinate],
        coors: list[tuple[int, int, int]] | np.ndarray,
        round_: bool = True,
    ) -> list[tuple[int, int, int]] | np.ndarray:
        R, t = self.get_rigid_transformation(source_spacing, target_spacing, source_landmarks, target_landmarks)
        return_array = True
        if isinstance(coors, list):
            return_array = False

        coors_arr = np.array(coors)
        coors_arr = coors_arr * np.array(source_spacing).reshape(1, 3)
        new_coors = np.matmul(R, coors_arr.T).T + t
        new_coors = new_coors / np.array(target_spacing).reshape(1, 3)
        if round_:
            if return_array:
                return np.round(new_coors).astype(np.int16).T
            return [tuple(x) for x in np.round(new_coors).astype(np.int16)]
        else:
            if return_array:
                return new_coors.T
            return [tuple(x) for x in new_coors]
