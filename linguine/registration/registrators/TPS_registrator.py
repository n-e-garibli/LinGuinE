# Copyright AstraZeneca 2026
"""Contains a registrator to perform deformable point set registration using the thin
plate spline (TPS) method.

Algorithm is based on "Principal Warps: Thin-Plate Splines and the Decomposition of Deformation"
by Fred L. Bookstein from IEEE Transactions on Pattern Analysis and Machine Intelligence (June 1989).
See https://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf.
"""

from collections.abc import Iterable

import numpy as np
from scipy.spatial import distance_matrix

from linguine.registration.landmark import LandmarkCoordinate
from linguine.registration.registrators import PointSetRegistrator


class ThinPlateSplineRegistrator(PointSetRegistrator):
    """A point set algorithm for deformable registration.

    Based on minimising the bending energy on a surface as in:
    https://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf
    """

    def __init__(
        self,
        valid_landmarks: Iterable[str] | None = None,
        _lambda: float = 0.0,
    ):
        """Constructs the class.

        Args:
            valid_landmarks: A set of strings representing landmarks that can be used for registration.
                If not provided, all available points will be used.
            _lambda: A hyperparameter to control the amount of allowed deformations. 0.0 implies maximum
                deformation, meaning that the resulting transformation will map every landmark in the source
                perfectly to its corresponding landmark. Higher lambda force greater rigidity.
        """
        super().__init__(valid_landmarks=valid_landmarks)
        self._lambda = _lambda

    def map_coordinates(
        self,
        source_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float],
        source_landmarks: dict[str, LandmarkCoordinate],
        target_landmarks: dict[str, LandmarkCoordinate],
        coors: list[tuple[int, int, int]] | np.ndarray,
        round_: bool = True,
    ) -> list[tuple[int, int, int]] | np.ndarray:
        # Reformat and extract common valid landmarks
        source_points_index, target_points_index = self.get_ordered_points(source_landmarks, target_landmarks)

        # Determine number of common landmarks.
        N = source_points_index.shape[0]
        # Transform landmarks to real world coordinates.
        source_points_rl = source_points_index * np.array(source_spacing).reshape(1, 3)
        target_points_rl = target_points_index * np.array(source_spacing).reshape(1, 3)

        # Define the(N+4)x(N+4) matrix L of the form
        # [ K   |  P ]
        # [ P.T |  O ]
        L = np.zeros((N + 4, N + 4))
        # In the 3D case the radial basis function is euclidean
        # distance (instead of the r^2log(r))
        K = distance_matrix(source_points_rl, source_points_rl)
        L[:N, :N] = K + self._lambda * np.eye(N)
        P = np.hstack([np.ones((N, 1)), source_points_rl])
        L[:N, N:] = P
        L[N:, :N] = P.T

        # Define matrix Y which is effectively just the target landamrks
        # with some 0s that make the shapes compatible in the linear system.
        Y = np.zeros((N + 4, 3))
        Y[:N, :] = target_points_rl

        # Solve the equation L*(W | A) = Y and determine the
        # W and A coefficients for the mapper.
        solution = np.linalg.solve(L, Y)

        A = solution[-4:]
        W = solution[:-4]

        return_array = True
        if isinstance(coors, list):
            return_array = False
        # Transform image indices into real world space using spacing.
        coors = np.array(coors) * np.array(source_spacing).reshape(1, 3)

        # Transform the coordinates to the real world space of the target image.
        # This is a vectorized implementation of the function
        # f(x,y,z) = a0 + a1*x + a2*y + a3*z + sum(w_i * U(|Pi - (x,y,z)|))
        new_coors = A[0] + np.dot(coors, A[1:]) + np.dot(distance_matrix(coors, source_points_rl), W)
        # Transform coordinates into indices in the target image.
        new_coors = new_coors / np.array(target_spacing).reshape(1, 3)

        # Optionally round coordinates to integers for easy indexing.
        if round_:
            if return_array:
                return np.round(new_coors).astype(np.int16).T
            return [tuple(x) for x in np.round(new_coors).astype(np.int16)]
        else:
            if return_array:
                return new_coors.T
            return [tuple(x) for x in new_coors]
