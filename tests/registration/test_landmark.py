import unittest

from linguine.registration.landmark import LandmarkCoordinate
from linguine.utils.bounding_boxes import Point3D


class TestPoint3D(unittest.TestCase):
    def test_valid_point(self):
        """Test creation of valid Point3D"""
        point = Point3D(1, 2, 3)
        self.assertEqual(point.x, 1)
        self.assertEqual(point.y, 2)
        self.assertEqual(point.z, 3)

    def test_negative_coordinates(self):
        """Test that negative coordinates raise assertion error"""
        with self.assertRaises(AssertionError):
            Point3D(-1, 0, 0)
        with self.assertRaises(AssertionError):
            Point3D(0, -1, 0)
        with self.assertRaises(AssertionError):
            Point3D(0, 0, -1)

    def test_non_integer_coordinates(self):
        """Test that non-integer coordinates raise assertion error"""
        with self.assertRaises(AssertionError):
            Point3D(1.5, 0, 0)
        with self.assertRaises(AssertionError):
            Point3D(0, 1.5, 0)
        with self.assertRaises(AssertionError):
            Point3D(0, 0, 1.5)

    def test_from_tuple(self):
        """Test creating Point3D from tuple"""
        point = Point3D.from_tuple((1, 2, 3))
        self.assertEqual(point.x, 1)
        self.assertEqual(point.y, 2)
        self.assertEqual(point.z, 3)


class TestLandmarkCoordinate(unittest.TestCase):
    def test_valid_coordinate(self):
        """Test creation of valid landmark coordinate"""
        coord = LandmarkCoordinate(1, 2, 3)
        self.assertTrue(coord.is_valid)
        self.assertEqual(coord.x, 1)
        self.assertEqual(coord.y, 2)
        self.assertEqual(coord.z, 3)

    def test_invalid_coordinate_none(self):
        """Test that None values make coordinate invalid"""
        coord = LandmarkCoordinate(None, 2, 3)
        self.assertFalse(coord.is_valid)
        coord = LandmarkCoordinate(1, None, 3)
        self.assertFalse(coord.is_valid)
        coord = LandmarkCoordinate(1, 2, None)
        self.assertFalse(coord.is_valid)

    def test_invalid_coordinate_negative(self):
        """Test that negative values make coordinate invalid"""
        coord = LandmarkCoordinate(-1, 2, 3)
        self.assertFalse(coord.is_valid)
        coord = LandmarkCoordinate(1, -2, 3)
        self.assertFalse(coord.is_valid)
        coord = LandmarkCoordinate(1, 2, -3)
        self.assertFalse(coord.is_valid)

    def test_all_none_coordinate(self):
        """Test coordinate with all None values"""
        coord = LandmarkCoordinate(None, None, None)
        self.assertFalse(coord.is_valid)
