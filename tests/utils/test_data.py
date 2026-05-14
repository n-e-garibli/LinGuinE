import unittest
from pathlib import Path

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor

from linguine.constants import IMAGE
from linguine.utils.bounding_boxes import BBox3D, Point3D
from linguine.utils.data import (
    get_path_from_data_dict_entry,
    get_spacing_from_metatensor,
    prepare_inputs_for_inferer,
    transform_coordinate_spacing,
)


class TestDataUtils(unittest.TestCase):
    def test_get_path_from_data_dict(self):
        """Test extracting paths from data dictionary entries"""
        # Test with string path
        data_dict = {IMAGE: "/path/to/image.nii.gz"}
        path = get_path_from_data_dict_entry(data_dict, IMAGE)
        self.assertIsInstance(path, Path)
        self.assertEqual(str(path), "/path/to/image.nii.gz")

        # Test with Path object
        data_dict = {IMAGE: Path("/path/to/image.nii.gz")}
        path = get_path_from_data_dict_entry(data_dict, IMAGE)
        self.assertIsInstance(path, Path)
        self.assertEqual(str(path), "/path/to/image.nii.gz")

        # Test with tuple
        data_dict = {IMAGE: ("/path/to/image.nii.gz", "extra_info")}
        path = get_path_from_data_dict_entry(data_dict, IMAGE)
        self.assertIsInstance(path, Path)
        self.assertEqual(str(path), "/path/to/image.nii.gz")

        # Test with None
        data_dict = {IMAGE: None}
        path = get_path_from_data_dict_entry(data_dict, IMAGE)
        self.assertIsNone(path)

    def test_get_spacing_from_metatensor(self):
        """Test extracting spacing information from MetaTensor"""
        # Test with tensor without spacing information
        tensor = MetaTensor(np.zeros((1, 1, 64, 64, 64)))
        spacing = get_spacing_from_metatensor(tensor)
        # In this case monai makes an affine with spacing 1.0,1.0,1.0
        self.assertEqual(spacing, (1.0, 1.0, 1.0))

    def test_get_spacing_from_metatensor_with_spacing_key(self):
        """Test extracting spacing from MetaTensor with 'spacing' key"""
        # Test with spacing key (3D)
        meta = {"spacing": [1.5, 2.0, 2.5]}
        tensor = MetaTensor(np.zeros((1, 1, 64, 64, 64)), meta=meta)
        spacing = get_spacing_from_metatensor(tensor)
        self.assertEqual(spacing, (1.5, 2.0, 2.5))

        # Test with spacing key (batch dimension)
        meta = {"spacing": [[1.5, 2.0, 2.5]]}
        tensor = MetaTensor(np.zeros((1, 1, 64, 64, 64)), meta=meta)
        spacing = get_spacing_from_metatensor(tensor)
        self.assertEqual(spacing, (1.5, 2.0, 2.5))

        # Test with torch tensors in spacing
        meta = {"spacing": [torch.tensor(1.5), torch.tensor(2.0), torch.tensor(2.5)]}
        tensor = MetaTensor(np.zeros((1, 1, 64, 64, 64)), meta=meta)
        spacing = get_spacing_from_metatensor(tensor)
        self.assertEqual(spacing, (1.5, 2.0, 2.5))

    def test_get_spacing_from_metatensor_affine_fallback(self):
        """Test extracting spacing from MetaTensor using affine matrix fallback"""
        # Create affine matrix with spacing [1.5, 2.0, 2.5]
        affine = np.array([[1.5, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1]])

        # Create MetaTensor with just the affine
        tensor = MetaTensor(np.zeros((1, 1, 64, 64, 64)), affine=affine)
        spacing = get_spacing_from_metatensor(tensor)
        self.assertEqual(spacing, (1.5, 2.0, 2.5))

        # Test with negative values in affine (should take absolute values via norm)
        affine_negative = np.array([[-1.5, 0, 0, 0], [0, -2.0, 0, 0], [0, 0, -2.5, 0], [0, 0, 0, 1]])
        tensor = MetaTensor(np.zeros((1, 1, 64, 64, 64)), affine=affine_negative)
        spacing = get_spacing_from_metatensor(tensor)
        self.assertEqual(spacing, (1.5, 2.0, 2.5))

        # Test with non-canonical orientation (rotated image)
        # Affine matrix with 45-degree rotation in XY plane and spacing [1.0, 2.0, 3.0]
        cos45 = np.cos(np.pi / 4)
        sin45 = np.sin(np.pi / 4)
        affine_rotated = np.array(
            [
                [1.0 * cos45, -2.0 * sin45, 0, 0],  # X column rotated
                [1.0 * sin45, 2.0 * cos45, 0, 0],  # Y column rotated
                [0, 0, 3.0, 0],  # Z column unchanged
                [0, 0, 0, 1],
            ]
        )
        tensor = MetaTensor(np.zeros((1, 1, 64, 64, 64)), affine=affine_rotated)
        spacing = get_spacing_from_metatensor(tensor)
        # Should extract spacing correctly despite rotation
        expected_spacing = (1.0, 2.0, 3.0)
        self.assertAlmostEqual(spacing[0], expected_spacing[0], places=6)
        self.assertAlmostEqual(spacing[1], expected_spacing[1], places=6)
        self.assertAlmostEqual(spacing[2], expected_spacing[2], places=6)

        # Test with rows that are ordered differently
        affine_rotated2 = np.array(
            [
                [0, -2.0, 0, 0],
                [-1.5, 0, 0, 0],
                [0, 0, -2.5, 0],
                [0, 0, 0, 1],
            ]
        )
        tensor = MetaTensor(np.zeros((1, 1, 64, 64, 64)), affine=affine_rotated2)
        spacing = get_spacing_from_metatensor(tensor)
        self.assertEqual(spacing, (1.5, 2.0, 2.5))

    def test_transform_coordinate_spacing(self):
        """Test coordinate transformation between different spacings"""
        original_spacing = (2.0, 2.0, 2.0)
        desired_spacing = (1.0, 1.0, 1.0)
        coordinate = (10, 10, 10)

        # When going from larger to smaller spacing, coordinates should double
        transformed = transform_coordinate_spacing(coordinate, original_spacing, desired_spacing)
        self.assertEqual(transformed, (20, 20, 20))

        # When going from smaller to larger spacing, coordinates should halve
        transformed = transform_coordinate_spacing(coordinate, desired_spacing, original_spacing)
        self.assertEqual(transformed, (5, 5, 5))

        # Test with different spacings for each dimension
        original_spacing = (2.0, 1.0, 0.5)
        desired_spacing = (1.0, 1.0, 1.0)
        coordinate = (10, 10, 10)
        transformed = transform_coordinate_spacing(coordinate, original_spacing, desired_spacing)
        self.assertEqual(transformed, (20, 10, 5))

    def test_prepare_inputs_for_inferer_image_only(self):
        """Test prepare_inputs_for_inferer with image only"""
        # Create test image with specific spacing
        affine = np.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1]])
        image = MetaTensor(np.ones((1, 1, 32, 32, 32)), affine=affine)

        desired_spacing = (1.0, 1.0, 1.0)

        spaced_image, spaced_clicks, spaced_bboxes = prepare_inputs_for_inferer(
            desired_spacing=desired_spacing, image=image
        )

        # Check that image was resampled
        self.assertIsInstance(spaced_image, MetaTensor)
        self.assertEqual(len(spaced_clicks), 0)
        self.assertEqual(len(spaced_bboxes), 0)

    def test_prepare_inputs_for_inferer_with_clicks(self):
        """Test prepare_inputs_for_inferer with clicks"""
        # Create test image with specific spacing
        affine = np.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1]])
        image = MetaTensor(np.ones((1, 1, 32, 32, 32)), affine=affine)

        desired_spacing = (1.0, 1.0, 1.0)
        clicks = [(10, 10, 10), (20, 20, 20)]

        spaced_image, spaced_clicks, spaced_bboxes = prepare_inputs_for_inferer(
            desired_spacing=desired_spacing, image=image, clicks=clicks
        )

        # Check that image was resampled
        self.assertIsInstance(spaced_image, MetaTensor)
        # Check that clicks were transformed (original spacing 2.0 -> desired spacing 1.0, so coordinates should double)
        self.assertEqual(len(spaced_clicks), 2)
        self.assertEqual(spaced_clicks[0], (20, 20, 20))
        self.assertEqual(spaced_clicks[1], (40, 40, 40))
        self.assertEqual(len(spaced_bboxes), 0)

    def test_prepare_inputs_for_inferer_with_bboxes(self):
        """Test prepare_inputs_for_inferer with bounding boxes"""
        # Create test image with specific spacing
        affine = np.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1]])
        image = MetaTensor(np.ones((1, 1, 32, 32, 32)), affine=affine)

        desired_spacing = (1.0, 1.0, 1.0)
        bbox = BBox3D(Point3D(5, 5, 5), Point3D(15, 15, 15))

        spaced_image, spaced_clicks, spaced_bboxes = prepare_inputs_for_inferer(
            desired_spacing=desired_spacing, image=image, bboxes=[bbox]
        )

        # Check that image was resampled
        self.assertIsInstance(spaced_image, MetaTensor)
        self.assertEqual(len(spaced_clicks), 0)
        # Check that bboxes were transformed (original spacing 2.0 -> desired spacing 1.0, so coordinates should double)
        self.assertEqual(len(spaced_bboxes), 1)
        transformed_bbox = spaced_bboxes[0]
        self.assertIsInstance(transformed_bbox, BBox3D)
        self.assertEqual(transformed_bbox.min_point.x, 10)
        self.assertEqual(transformed_bbox.min_point.y, 10)
        self.assertEqual(transformed_bbox.min_point.z, 10)
        self.assertEqual(transformed_bbox.max_point.x, 30)
        self.assertEqual(transformed_bbox.max_point.y, 30)
        self.assertEqual(transformed_bbox.max_point.z, 30)

    def test_prepare_inputs_for_inferer_with_clicks_and_bboxes(self):
        """Test prepare_inputs_for_inferer with both clicks and bounding boxes"""
        # Create test image with specific spacing
        affine = np.array([[1.5, 0, 0, 0], [0, 1.5, 0, 0], [0, 0, 3.0, 0], [0, 0, 0, 1]])
        image = MetaTensor(np.ones((1, 1, 32, 32, 16)), affine=affine)

        desired_spacing = (1.0, 1.0, 1.0)
        clicks = [(12, 12, 6)]
        bbox = BBox3D(Point3D(3, 3, 2), Point3D(21, 21, 14))

        spaced_image, spaced_clicks, spaced_bboxes = prepare_inputs_for_inferer(
            desired_spacing=desired_spacing, image=image, clicks=clicks, bboxes=[bbox]
        )

        # Check that image was resampled
        self.assertIsInstance(spaced_image, MetaTensor)

        # Check that clicks were transformed
        # Original spacing (1.5, 1.5, 3.0) -> desired spacing (1.0, 1.0, 1.0)
        # x,y: 12 * 1.5 / 1.0 = 18, z: 6 * 3.0 / 1.0 = 18
        self.assertEqual(len(spaced_clicks), 1)
        self.assertEqual(spaced_clicks[0], (18, 18, 18))

        # Check that bboxes were transformed with same scaling
        self.assertEqual(len(spaced_bboxes), 1)
        transformed_bbox = spaced_bboxes[0]
        self.assertIsInstance(transformed_bbox, BBox3D)
        # min_point: (3*1.5/1.0, 3*1.5/1.0, 2*3.0/1.0) = (4.5, 4.5, 6) -> (4, 4, 6) after rounding
        self.assertEqual(transformed_bbox.min_point.x, 4)
        self.assertEqual(transformed_bbox.min_point.y, 4)
        self.assertEqual(transformed_bbox.min_point.z, 6)
        # max_point: (21*1.5/1.0, 21*1.5/1.0, 14*3.0/1.0) = (31.5, 31.5, 42) -> (32, 32, 42) after rounding
        self.assertEqual(transformed_bbox.max_point.x, 32)
        self.assertEqual(transformed_bbox.max_point.y, 32)
        self.assertEqual(transformed_bbox.max_point.z, 42)

    def test_prepare_inputs_for_inferer_none_inputs(self):
        """Test prepare_inputs_for_inferer with None clicks and bboxes"""
        # Create test image
        image = MetaTensor(np.ones((1, 1, 32, 32, 32)))
        desired_spacing = (1.0, 1.0, 1.0)

        spaced_image, spaced_clicks, spaced_bboxes = prepare_inputs_for_inferer(
            desired_spacing=desired_spacing, image=image, clicks=None, bboxes=None
        )

        # Check that image was resampled
        self.assertIsInstance(spaced_image, MetaTensor)
        self.assertEqual(len(spaced_clicks), 0)
        self.assertEqual(len(spaced_bboxes), 0)
