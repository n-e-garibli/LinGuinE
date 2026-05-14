"""Unit tests for registration registrators."""

import unittest
from unittest.mock import patch

import numpy as np
from monai.data.meta_tensor import MetaTensor

from linguine.registration.landmark import LandmarkCoordinate
from linguine.registration.registrators.abstract_registrator import PointSetRegistrator, extract_valid_coors
from linguine.registration.registrators.aruns_rigid_registrator import ArunsRigidRegistrator
from linguine.registration.registrators.TPS_registrator import ThinPlateSplineRegistrator


class TestPointSetRegistrator(unittest.TestCase):
    """Test cases for the abstract PointSetRegistrator class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a concrete implementation for testing abstract methods
        class ConcreteRegistrator(PointSetRegistrator):
            def map_coordinates(
                self, source_spacing, target_spacing, source_landmarks, target_landmarks, coors, round_=True
            ):
                # Simple identity transformation for testing
                if isinstance(coors, list):
                    return coors
                return coors.T if round_ else coors.T

        self.registrator = ConcreteRegistrator()
        self.registrator_with_landmarks = ConcreteRegistrator(valid_landmarks=["heart", "liver"])

    def test_initialization_no_landmarks(self):
        """Test initialization without valid landmarks."""
        # Test using the concrete implementation
        self.assertIsNone(self.registrator.valid_landmarks)

    def test_initialization_with_landmarks(self):
        """Test initialization with valid landmarks."""
        self.assertEqual(self.registrator_with_landmarks.valid_landmarks, ["heart", "liver"])

    def test_get_ordered_points_matching_landmarks(self):
        """Test getting ordered points with matching landmarks."""
        dict1 = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=40, y=50, z=60),
            "kidney": LandmarkCoordinate(x=70, y=80, z=90),
        }
        dict2 = {
            "heart": LandmarkCoordinate(x=15, y=25, z=35),
            "liver": LandmarkCoordinate(x=45, y=55, z=65),
            "spleen": LandmarkCoordinate(x=100, y=110, z=120),
        }

        points1, points2 = self.registrator.get_ordered_points(dict1, dict2)

        expected_points1 = np.array([[10, 20, 30], [40, 50, 60]])
        expected_points2 = np.array([[15, 25, 35], [45, 55, 65]])

        np.testing.assert_array_equal(points1, expected_points1)
        np.testing.assert_array_equal(points2, expected_points2)

    def test_get_ordered_points_with_valid_landmarks_filter(self):
        """Test getting ordered points with valid landmarks filter."""
        dict1 = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=40, y=50, z=60),
            "kidney": LandmarkCoordinate(x=70, y=80, z=90),
        }
        dict2 = {
            "heart": LandmarkCoordinate(x=15, y=25, z=35),
            "liver": LandmarkCoordinate(x=45, y=55, z=65),
            "kidney": LandmarkCoordinate(x=75, y=85, z=95),
        }

        points1, points2 = self.registrator_with_landmarks.get_ordered_points(dict1, dict2)

        # Only heart and liver should be included (kidney is not in valid_landmarks)
        expected_points1 = np.array([[10, 20, 30], [40, 50, 60]])
        expected_points2 = np.array([[15, 25, 35], [45, 55, 65]])

        np.testing.assert_array_equal(points1, expected_points1)
        np.testing.assert_array_equal(points2, expected_points2)

    def test_get_ordered_points_invalid_landmarks_filtered(self):
        """Test that invalid landmarks are filtered out."""
        dict1 = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(),  # Invalid landmark
        }
        dict2 = {
            "heart": LandmarkCoordinate(x=15, y=25, z=35),
            "liver": LandmarkCoordinate(x=45, y=55, z=65),
        }

        points1, points2 = self.registrator.get_ordered_points(dict1, dict2)

        # Only heart should be included (liver is invalid in dict1)
        expected_points1 = np.array([[10, 20, 30]])
        expected_points2 = np.array([[15, 25, 35]])

        np.testing.assert_array_equal(points1, expected_points1)
        np.testing.assert_array_equal(points2, expected_points2)

    def test_compute_dist(self):
        """Test distance computation between landmarks."""
        source_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=40, y=50, z=60),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=12, y=22, z=32),
            "liver": LandmarkCoordinate(x=42, y=52, z=62),
        }

        with patch.object(self.registrator, "map_coordinates") as mock_map:
            # Mock returns transformed coordinates with small offset
            mock_map.return_value = np.array([[11, 21, 31], [41, 51, 61]]).T

            distance = self.registrator.compute_dist(
                source_spacing=(1.0, 1.0, 1.0),
                target_spacing=(1.0, 1.0, 1.0),
                source_landmarks=source_landmarks,
                target_landmarks=target_landmarks,
            )

        # Expected distance should be small (around sqrt(3) for each point)
        self.assertAlmostEqual(distance, np.sqrt(3), places=5)

    def test_register_images_dimension_validation(self):
        """Test that register method validates image dimensions."""
        source_2d = MetaTensor(np.ones((10, 10)))
        target_3d = MetaTensor(np.ones((10, 10, 10)))

        with self.assertRaises(AssertionError):
            self.registrator.register(source_2d, target_3d, {}, {})

    def test_map_image(self):
        """Test image mapping with coordinate transformation."""
        img = np.ones((5, 5, 5))
        og_coors = np.meshgrid(np.arange(5), np.arange(5), np.arange(5), indexing="ij")
        og_coors = np.vstack([coor.ravel() for coor in og_coors])

        # Simple translation transformation
        new_coors = og_coors + 1
        target_shape = (7, 7, 7)

        with patch("linguine.registration.registrators.abstract_registrator.extract_valid_coors") as mock_extract:
            # Mock the extract_valid_coors function
            valid_new = (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]))
            valid_og = (np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 1, 2]))
            mock_extract.return_value = (valid_new, valid_og)

            result = self.registrator.map_image(img, og_coors, new_coors, target_shape)

        self.assertEqual(result.shape, target_shape)
        self.assertIn(1, result)  # Original values should be present in resulting image
        self.assertIn(0, result)  # Result should've been padded due to shape mismatch
        mock_extract.assert_called_once()

    def test_extract_valid_coors_within_bounds(self):
        """Test extraction of valid coordinates within image bounds."""
        # new_coors should be in format: array of shape (3, N) where rows are x, y, z coordinates
        new_coors = np.array([[1, 2, 8], [1, 2, 8], [1, 2, 8]])  # x, y, z coordinates
        target_shape = (10, 10, 10)

        result = extract_valid_coors(new_coors, target_shape)

        x_valid, y_valid, z_valid = result
        np.testing.assert_array_equal(x_valid, [1, 2, 8])
        np.testing.assert_array_equal(y_valid, [1, 2, 8])
        np.testing.assert_array_equal(z_valid, [1, 2, 8])

    def test_extract_valid_coors_out_of_bounds(self):
        """Test extraction filters out coordinates outside image bounds."""
        # Format: array of shape (3, N) where each column is a coordinate
        new_coors = np.array([[-1, 2, 3], [5, 5, 5], [15, 8, 9]])  # Some out of bounds
        target_shape = (10, 10, 10)

        result = extract_valid_coors(new_coors, target_shape)

        x_valid, y_valid, z_valid = result
        # Both second and third coordinates should be valid
        np.testing.assert_array_equal(x_valid, [2, 3])
        np.testing.assert_array_equal(y_valid, [5, 5])
        np.testing.assert_array_equal(z_valid, [8, 9])

    def test_extract_valid_coors_with_original_coordinates(self):
        """Test extraction with both new and original coordinates."""
        new_coors = np.array([[1, 8], [2, 9], [3, 10]])
        og_coors = np.array([[0, 7], [1, 8], [2, 9]])
        target_shape = (10, 10, 10)

        result = extract_valid_coors(new_coors, target_shape, og_coors)

        (x_valid, _, _), (x_og_valid, _, _) = result

        # Only first coordinate should be valid (second is out of bounds)
        np.testing.assert_array_equal(x_valid, [1])
        np.testing.assert_array_equal(x_og_valid, [0])

    def test_compute_dist_empty_landmarks(self):
        """Test distance computation with empty landmarks."""
        distance = self.registrator.compute_dist(
            source_spacing=(1.0, 1.0, 1.0), target_spacing=(1.0, 1.0, 1.0), source_landmarks={}, target_landmarks={}
        )

        # Should return 0 for empty landmarks (no landmarks to compute distance for)
        self.assertEqual(distance, 0.0)

    def test_compute_dist_mismatched_landmarks(self):
        """Test distance computation with mismatched landmarks."""
        source_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
        }
        target_landmarks = {
            "liver": LandmarkCoordinate(x=42, y=52, z=62),
        }

        distance = self.registrator.compute_dist(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
        )

        # Should return 0 for no matching landmarks
        self.assertEqual(distance, 0.0)

    def test_extract_valid_coors_negative_coordinates(self):
        """Test extraction with negative coordinates."""
        # Format: array of shape (3, N) where each column is a coordinate
        new_coors = np.array([[-5, -1, 3], [2, 5, -2], [8, 9, 7]])
        target_shape = (10, 10, 10)

        result = extract_valid_coors(new_coors, target_shape)

        x_valid, y_valid, z_valid = result
        # No coordinates should be valid due to negative values in each coordinate
        self.assertEqual(len(x_valid), 0)
        self.assertEqual(len(y_valid), 0)
        self.assertEqual(len(z_valid), 0)

    def test_extract_valid_coors_float_coordinates(self):
        """Test extraction with float coordinates."""
        # Format: array of shape (3, N) where each column is a coordinate
        new_coors = np.array([[1.5, 5.1], [2.7, 6.3], [8.9, 4.2]])
        target_shape = (10, 10, 10)

        result = extract_valid_coors(new_coors, target_shape)

        x_valid, y_valid, z_valid = result
        # Should accept valid float coordinates
        np.testing.assert_array_equal(x_valid, [1.5, 5.1])
        np.testing.assert_array_equal(y_valid, [2.7, 6.3])
        np.testing.assert_array_equal(z_valid, [8.9, 4.2])

    def test_map_segmentation_empty(self):
        """Test mapping empty segmentation."""
        empty_seg = np.zeros((5, 5, 5))
        target_shape = (10, 10, 10)

        result = self.registrator.map_segmentation(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            target_img_shape=target_shape,
            segmentation=empty_seg,
            source_landmarks={},
            target_landmarks={},
        )

        self.assertEqual(result.shape, target_shape)
        self.assertTrue(np.all(result == 0))

    def test_map_segmentation_with_lesions(self):
        """Test mapping segmentation with multiple lesions."""
        seg = np.zeros((5, 5, 5))
        seg[1:3, 1:3, 1:3] = 1  # First lesion
        seg[3:5, 3:5, 3:5] = 2  # Second lesion

        target_shape = (10, 10, 10)

        with patch.object(self.registrator, "map_coordinates") as mock_map:
            # Mock coordinate mapping - return coordinates shifted by 1
            def side_effect(coors, **kwargs):
                # Convert to array if it's a list and add 1 to all coordinates
                if isinstance(coors, list):
                    return [tuple(np.array(coord) + 1) for coord in coors]
                else:
                    return coors + 1

            mock_map.side_effect = side_effect

            with patch("linguine.registration.registrators.abstract_registrator.extract_valid_coors") as mock_extract:
                # Mock valid coordinate extraction
                mock_extract.return_value = (np.array([2, 3]), np.array([2, 3]), np.array([2, 3]))

                result = self.registrator.map_segmentation(
                    source_spacing=(1.0, 1.0, 1.0),
                    target_spacing=(1.0, 1.0, 1.0),
                    target_img_shape=target_shape,
                    segmentation=seg,
                    source_landmarks={},
                    target_landmarks={},
                )

        self.assertEqual(result.shape, target_shape)
        # Check that lesion 1 was not mapped.
        self.assertFalse(np.any(result == 1))
        # Check that lesion 2 was mappe
        self.assertTrue(np.any(result == 2))

    def test_get_ordered_points_empty_intersection(self):
        """Test getting ordered points with no common landmarks."""
        dict1 = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=40, y=50, z=60),
        }
        dict2 = {
            "kidney": LandmarkCoordinate(x=15, y=25, z=35),
            "spleen": LandmarkCoordinate(x=45, y=55, z=65),
        }

        points1, points2 = self.registrator.get_ordered_points(dict1, dict2)

        # Should return empty arrays
        self.assertEqual(len(points1), 0)
        self.assertEqual(len(points2), 0)

    def test_get_ordered_points_single_landmark(self):
        """Test getting ordered points with single common landmark."""
        dict1 = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=40, y=50, z=60),
        }
        dict2 = {
            "heart": LandmarkCoordinate(x=15, y=25, z=35),
            "kidney": LandmarkCoordinate(x=75, y=85, z=95),
        }

        points1, points2 = self.registrator.get_ordered_points(dict1, dict2)

        expected_points1 = np.array([[10, 20, 30]])
        expected_points2 = np.array([[15, 25, 35]])

        np.testing.assert_array_equal(points1, expected_points1)
        np.testing.assert_array_equal(points2, expected_points2)


class TestArunsRigidRegistrator(unittest.TestCase):
    """Test cases for the ArunsRigidRegistrator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registrator = ArunsRigidRegistrator()
        self.registrator_with_landmarks = ArunsRigidRegistrator(valid_landmarks=["heart", "liver"])

    def test_initialization_no_landmarks(self):
        """Test initialization without valid landmarks."""
        self.assertIsNone(self.registrator.valid_landmarks)

    def test_initialization_with_landmarks(self):
        """Test initialization with valid landmarks."""
        self.assertEqual(self.registrator_with_landmarks.valid_landmarks, ["heart", "liver"])

    def test_get_rotation_matrix_identity(self):
        """Test rotation matrix calculation for identical point sets."""
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        R = self.registrator.get_rotation_matrix(P, Q)

        # Should be identity matrix
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=10)

    def test_get_rotation_matrix_90_degree_rotation(self):
        """Test rotation matrix for 90-degree rotation around z-axis."""
        P = np.array([[1, 0, 0], [0, 1, 0]])  # Points in xy plane
        Q = np.array([[0, 1, 0], [-1, 0, 0]])  # Rotated 90 degrees

        R = self.registrator.get_rotation_matrix(P, Q)

        # Check that rotation matrix is orthogonal and determinant is 1
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)
        np.testing.assert_array_almost_equal(np.dot(R, R.T), np.eye(3), decimal=10)

    def test_get_translation_vector(self):
        """Test translation vector calculation."""
        p_centroid = np.array([1, 2, 3])
        q_centroid = np.array([4, 5, 6])
        R = np.eye(3)  # Identity rotation

        t = self.registrator.get_translation_vector(p_centroid, q_centroid, R)

        expected_t = np.array([3, 3, 3])  # q_centroid - R * p_centroid
        np.testing.assert_array_equal(t, expected_t)

    def test_get_rigid_transformation(self):
        """Test complete rigid transformation calculation."""
        source_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=40, y=50, z=60),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=15, y=25, z=35),
            "liver": LandmarkCoordinate(x=45, y=55, z=65),
        }

        R, t = self.registrator.get_rigid_transformation(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
        )

        # Check that R is a valid rotation matrix
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)
        np.testing.assert_array_almost_equal(np.dot(R, R.T), np.eye(3), decimal=10)

        # Check translation vector shape
        self.assertEqual(t.shape, (3,))

    def test_map_coordinates_list_input(self):
        """Test coordinate mapping with list input."""
        source_landmarks = {
            "heart": LandmarkCoordinate(x=0, y=0, z=0),
            "liver": LandmarkCoordinate(x=10, y=0, z=0),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=5, y=5, z=5),
            "liver": LandmarkCoordinate(x=15, y=5, z=5),
        }

        coordinates = [(0, 0, 0), (5, 5, 5)]

        result = self.registrator.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=True,
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for coord in result:
            self.assertIsInstance(coord, tuple)
            self.assertEqual(len(coord), 3)

    def test_map_coordinates_array_input(self):
        """Test coordinate mapping with numpy array input."""
        source_landmarks = {
            "heart": LandmarkCoordinate(x=0, y=0, z=0),
            "liver": LandmarkCoordinate(x=10, y=0, z=0),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=5, y=5, z=5),
            "liver": LandmarkCoordinate(x=15, y=5, z=5),
        }

        coordinates = np.array([[0, 0, 0], [5, 5, 5]])

        result = self.registrator.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=True,
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 2))  # Transposed output

    def test_map_coordinates_different_spacing(self):
        """Test coordinate mapping with different voxel spacing."""
        source_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=10, z=10),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=10, z=10),
        }

        coordinates = [(10, 10, 10)]

        result = self.registrator.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(2.0, 2.0, 2.0),  # Different target spacing
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=True,
        )

        # With different spacing, coordinates should be scaled
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(coordinates, result)

    def test_map_coordinates_no_rounding(self):
        """Test coordinate mapping without rounding."""
        source_landmarks = {
            "heart": LandmarkCoordinate(x=0, y=0, z=0),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=1, y=1, z=1),
        }

        coordinates = [(0, 0, 0)]

        result = self.registrator.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=False,
        )

        # Without rounding, result should be float coordinates
        self.assertIsInstance(result, list)
        for coord in result:
            self.assertIsInstance(coord, tuple)
            self.assertIsInstance(coord[0], float)
            self.assertAlmostEqual(coord[0], 1.0)


class TestThinPlateSplineRegistrator(unittest.TestCase):
    """Test cases for the ThinPlateSplineRegistrator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registrator = ThinPlateSplineRegistrator()
        self.registrator_with_lambda = ThinPlateSplineRegistrator(_lambda=0.1)
        self.registrator_with_landmarks = ThinPlateSplineRegistrator(valid_landmarks=["heart", "liver"], _lambda=0.05)

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        self.assertIsNone(self.registrator.valid_landmarks)
        self.assertEqual(self.registrator._lambda, 0.0)

    def test_initialization_with_lambda(self):
        """Test initialization with custom lambda parameter."""
        self.assertEqual(self.registrator_with_lambda._lambda, 0.1)

    def test_initialization_with_landmarks_and_lambda(self):
        """Test initialization with landmarks and lambda."""
        landmarks = ["heart", "liver"]
        lambda_value = 0.05
        self.assertEqual(self.registrator_with_landmarks.valid_landmarks, landmarks)
        self.assertEqual(self.registrator_with_landmarks._lambda, lambda_value)

    def test_map_coordinates_sufficient_landmarks(self):
        """Test coordinate mapping with sufficient well-conditioned landmarks."""
        source_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=50, y=60, z=70),
            "kidney": LandmarkCoordinate(x=80, y=10, z=90),
            "spleen": LandmarkCoordinate(x=20, y=80, z=40),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=12, y=22, z=32),
            "liver": LandmarkCoordinate(x=52, y=62, z=72),
            "kidney": LandmarkCoordinate(x=82, y=12, z=92),
            "spleen": LandmarkCoordinate(x=22, y=82, z=42),
        }

        coordinates = [(10, 20, 30)]

        result = self.registrator.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=True,
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], (12, 22, 32))

    def test_map_coordinates_array_input(self):
        """Test coordinate mapping with numpy array input."""
        # Use well-separated landmarks
        source_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=50, y=60, z=70),
            "kidney": LandmarkCoordinate(x=80, y=10, z=90),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=12, y=22, z=32),
            "liver": LandmarkCoordinate(x=52, y=62, z=72),
            "kidney": LandmarkCoordinate(x=82, y=12, z=92),
        }

        coordinates = np.array([[10, 20, 30], [25, 35, 45]])

        result = self.registrator.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=True,
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 2))  # Transposed output

    def test_map_coordinates_with_lambda_regularization(self):
        """Test that lambda parameter affects the transformation."""
        # Use well-separated landmarks
        source_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=50, y=60, z=70),
            "kidney": LandmarkCoordinate(x=80, y=10, z=90),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=15, y=25, z=35),
            "liver": LandmarkCoordinate(x=55, y=65, z=75),
            "kidney": LandmarkCoordinate(x=85, y=15, z=95),
        }

        coordinates = [(25, 35, 45)]

        # Test with lambda = 0 (maximum deformation)
        result_no_lambda = self.registrator.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=False,
        )

        # Test with lambda > 0 (more rigid)
        result_with_lambda = self.registrator_with_lambda.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=False,
        )

        # Results should be different (lambda affects the transformation)
        self.assertNotEqual(result_no_lambda, result_with_lambda)

    def test_map_coordinates_no_rounding(self):
        """Test coordinate mapping without rounding."""
        # Use well-separated landmarks
        source_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=50, y=60, z=70),
            "kidney": LandmarkCoordinate(x=80, y=10, z=90),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=12, y=22, z=32),
            "liver": LandmarkCoordinate(x=52, y=62, z=72),
            "kidney": LandmarkCoordinate(x=82, y=12, z=92),
        }

        coordinates = [(25, 35, 45)]

        result = self.registrator.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=False,
        )

        # Without rounding, result should be float coordinates
        self.assertIsInstance(result, list)
        for coord in result:
            self.assertIsInstance(coord, tuple)

    @patch("linguine.registration.registrators.TPS_registrator.distance_matrix")
    @patch("numpy.linalg.solve")
    def test_distance_matrix_usage(self, mock_solve, mock_distance_matrix):
        """Test that distance_matrix is called correctly."""
        tps_registrator = ThinPlateSplineRegistrator()

        # Setup mocks
        mock_distance_matrix.side_effect = [
            np.array([[0, 10], [10, 0]]),  # K matrix (source landmarks distances)
            np.array([[5, 7]]),  # Distance from coordinates to source landmarks (1x2 for 1 coord, 2 landmarks)
        ]
        mock_solve.return_value = np.zeros((6, 3))  # 2 landmarks + 4 affine parameters

        source_landmarks = {
            "heart": LandmarkCoordinate(x=10, y=20, z=30),
            "liver": LandmarkCoordinate(x=40, y=50, z=60),
        }
        target_landmarks = {
            "heart": LandmarkCoordinate(x=12, y=22, z=32),
            "liver": LandmarkCoordinate(x=42, y=52, z=62),
        }

        coordinates = [(25, 35, 45)]

        tps_registrator.map_coordinates(
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(1.0, 1.0, 1.0),
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            coors=coordinates,
            round_=True,
        )

        # Verify distance_matrix was called (twice: for K matrix and coordinate mapping)
        self.assertEqual(mock_distance_matrix.call_count, 2)


if __name__ == "__main__":
    unittest.main()
