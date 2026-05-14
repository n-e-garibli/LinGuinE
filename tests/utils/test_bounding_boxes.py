import unittest

import numpy as np

from linguine.utils.bounding_boxes import BBox2D, BBox3D, Point3D, get_bounding_box


class TestBBox3D(unittest.TestCase):
    def setUp(self):
        self.min_point = Point3D(1, 2, 3)
        self.max_point = Point3D(4, 5, 6)
        self.bbox = BBox3D(self.min_point, self.max_point)

    def test_bbox3d_properties(self):
        """Test BBox3D coordinate properties"""
        self.assertEqual(self.bbox.x_min, 1)
        self.assertEqual(self.bbox.y_min, 2)
        self.assertEqual(self.bbox.z_min, 3)
        self.assertEqual(self.bbox.x_max, 4)
        self.assertEqual(self.bbox.y_max, 5)
        self.assertEqual(self.bbox.z_max, 6)

    def test_invalid_bbox3d(self):
        """Test that invalid bounding boxes raise ValueError"""
        # min > max in x dimension
        with self.assertRaises(ValueError):
            BBox3D(Point3D(5, 2, 3), Point3D(4, 5, 6))

        # min > max in y dimension
        with self.assertRaises(ValueError):
            BBox3D(Point3D(1, 6, 3), Point3D(4, 5, 6))

        # min > max in z dimension
        with self.assertRaises(ValueError):
            BBox3D(Point3D(1, 2, 7), Point3D(4, 5, 6))

    def test_bbox3d_to_bounds(self):
        """Test converting BBox3D to bounds representation"""
        bounds = self.bbox.to_bounds()
        self.assertEqual(bounds, [(1, 4), (2, 5), (3, 6)])

    def test_bbox3d_from_bounds(self):
        """Test creating BBox3D from bounds"""
        bounds = [(1, 4), (2, 5), (3, 6)]
        bbox = BBox3D.from_bounds(bounds)
        self.assertEqual(bbox.min_point.x, 1)
        self.assertEqual(bbox.min_point.y, 2)
        self.assertEqual(bbox.min_point.z, 3)
        self.assertEqual(bbox.max_point.x, 4)
        self.assertEqual(bbox.max_point.y, 5)
        self.assertEqual(bbox.max_point.z, 6)


class TestBBox2D(unittest.TestCase):
    def test_valid_bbox2d(self):
        """Test creating valid BBox2D instances"""
        # Fixed x dimension
        bbox = BBox2D(Point3D(1, 2, 3), Point3D(1, 5, 6))
        self.assertEqual(bbox.fixed_dimension, "x")

        # Fixed y dimension
        bbox = BBox2D(Point3D(1, 2, 3), Point3D(4, 2, 6))
        self.assertEqual(bbox.fixed_dimension, "y")

        # Fixed z dimension
        bbox = BBox2D(Point3D(1, 2, 3), Point3D(4, 5, 3))
        self.assertEqual(bbox.fixed_dimension, "z")

    def test_invalid_bbox2d(self):
        """Test that invalid 2D bounding boxes raise ValueError"""
        # No fixed dimension
        with self.assertRaises(ValueError):
            BBox2D(Point3D(1, 2, 3), Point3D(4, 5, 6))


class TestGetBoundingBox(unittest.TestCase):
    def test_get_bounding_box(self):
        """Test the get_bounding_box util."""
        data = np.zeros((40, 40, 40))
        # Create a small 10x10x10 lesion in the center
        data[21:36, 27:37, 28:38] = 1.0
        bbox = get_bounding_box(data, as_mask=False)
        self.assertIsInstance(bbox, BBox3D)
        self.assertEqual(bbox.x_min, 21)
        self.assertEqual(bbox.x_max, 35)
        self.assertEqual(bbox.y_min, 27)
        self.assertEqual(bbox.y_max, 36)
        self.assertEqual(bbox.z_min, 28)
        self.assertEqual(bbox.z_max, 37)
