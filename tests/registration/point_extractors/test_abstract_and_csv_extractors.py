"""Unit tests for the abstract point extractor and CSV point extractor."""

import os
import tempfile
import unittest
from typing import Any

import pandas as pd

from linguine.constants import FILE_ID
from linguine.registration.landmark import LandmarkCoordinate
from linguine.registration.point_extractors.abstract_extractor import PointExtractor
from linguine.registration.point_extractors.csv_extractor import CSVPointExtractor


class ConcretePointExtractor(PointExtractor):
    """A concrete implementation of PointExtractor for testing purposes."""

    @property
    def valid_ids(self) -> set[str]:
        return {"landmark1", "landmark2", "landmark3"}

    def extract_points(
        self,
        input: dict[str, Any],
        include_ids: list[str] | None = None,
        *args,
        **kwargs,
    ) -> dict[str, LandmarkCoordinate]:
        """Mock implementation that returns test landmarks."""
        all_landmarks = {
            "landmark1": LandmarkCoordinate(x=10, y=20, z=30),
            "landmark2": LandmarkCoordinate(x=40, y=50, z=60),
            "landmark3": LandmarkCoordinate(),  # Invalid landmark
        }

        if include_ids is not None:
            return {k: v for k, v in all_landmarks.items() if k in include_ids}
        return all_landmarks


class TestAbstractPointExtractor(unittest.TestCase):
    """Test cases for the abstract PointExtractor class."""

    def setUp(self):
        self.extractor = ConcretePointExtractor()

    def test_extract_all_points(self):
        """Test extracting all available points."""
        input_data = {FILE_ID: "test_file"}
        result = self.extractor.extract_points(input_data)

        self.assertEqual(len(result), 3)
        self.assertIn("landmark1", result)
        self.assertIn("landmark2", result)
        self.assertIn("landmark3", result)

        # Check valid landmark
        self.assertTrue(result["landmark1"].is_valid)
        self.assertEqual(result["landmark1"].x, 10)
        self.assertEqual(result["landmark1"].y, 20)
        self.assertEqual(result["landmark1"].z, 30)

        # Check invalid landmark
        self.assertFalse(result["landmark3"].is_valid)

    def test_extract_specific_points(self):
        """Test extracting only specific landmarks."""
        input_data = {FILE_ID: "test_file"}
        include_ids = ["landmark1", "landmark2"]
        result = self.extractor.extract_points(input_data, include_ids=include_ids)

        self.assertEqual(len(result), 2)
        self.assertIn("landmark1", result)
        self.assertIn("landmark2", result)
        self.assertNotIn("landmark3", result)


class TestCSVPointExtractor(unittest.TestCase):
    """Test cases for the CSVPointExtractor class."""

    def setUp(self):
        # Create a temporary CSV file for testing
        self.test_data = pd.DataFrame(
            {
                FILE_ID: ["img1", "img1", "img1", "img2", "img2"],
                "name": ["landmark1", "landmark2", "landmark3", "landmark1", "landmark2"],
                "x": [10, 20, 30, 40, 50],
                "y": [15, 25, 35, 45, 55],
                "z": [5, 10, 15, 20, 25],
            }
        )

        self.temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

        self.extractor = CSVPointExtractor(landmark_csv_path=self.temp_file.name)

    def tearDown(self):
        # Clean up temporary file
        os.unlink(self.temp_file.name)

    def test_initialization_valid_csv(self):
        """Test successful initialization with valid CSV."""
        self.assertIsInstance(self.extractor, CSVPointExtractor)
        self.assertEqual(self.extractor.img_identifier, FILE_ID)

    def test_initialization_missing_column(self):
        """Test initialization fails with missing required column."""
        with self.assertRaises(AssertionError):
            CSVPointExtractor(landmark_csv_path=self.temp_file.name, img_identifier_colname="missing_column")

    def test_valid_ids_property(self):
        """Test that valid_ids returns all unique landmark names."""
        expected_ids = {"landmark1", "landmark2", "landmark3"}
        self.assertEqual(self.extractor.valid_ids, expected_ids)

    def test_extract_points_existing_file(self):
        """Test extracting points for an existing file ID."""
        input_data = {FILE_ID: "img1"}
        result = self.extractor.extract_points(input_data)

        self.assertEqual(len(result), 3)

        # Check landmark1
        self.assertTrue(result["landmark1"].is_valid)
        self.assertEqual(result["landmark1"].x, 10)
        self.assertEqual(result["landmark1"].y, 15)
        self.assertEqual(result["landmark1"].z, 5)

        # Check landmark2
        self.assertTrue(result["landmark2"].is_valid)
        self.assertEqual(result["landmark2"].x, 20)
        self.assertEqual(result["landmark2"].y, 25)
        self.assertEqual(result["landmark2"].z, 10)

    def test_extract_points_nonexistent_file(self):
        """Test extracting points for a non-existent file ID."""
        input_data = {FILE_ID: "nonexistent"}
        result = self.extractor.extract_points(input_data)

        # Should return empty landmarks for all valid IDs
        self.assertEqual(len(result), 3)
        for landmark in result.values():
            self.assertFalse(landmark.is_valid)

    def test_extract_points_with_include_ids(self):
        """Test extracting only specific landmark IDs."""
        input_data = {FILE_ID: "img1"}
        include_ids = ["landmark1", "landmark3"]
        result = self.extractor.extract_points(input_data, include_ids=include_ids)

        self.assertEqual(len(result), 2)
        self.assertIn("landmark1", result)
        self.assertIn("landmark3", result)
        self.assertNotIn("landmark2", result)

        # landmark1 should be valid
        self.assertTrue(result["landmark1"].is_valid)
        # landmark3 should be valid
        self.assertTrue(result["landmark3"].is_valid)

    def test_extract_points_partial_data(self):
        """Test extracting points for file with only some landmarks."""
        input_data = {FILE_ID: "img2"}
        result = self.extractor.extract_points(input_data)

        self.assertEqual(len(result), 3)

        # img2 only has landmark1 and landmark2
        self.assertTrue(result["landmark1"].is_valid)
        self.assertTrue(result["landmark2"].is_valid)
        self.assertFalse(result["landmark3"].is_valid)  # Missing from img2

    def test_custom_column_names(self):
        """Test CSV extractor with custom column names."""
        # Create CSV with different column names
        custom_data = pd.DataFrame(
            {"image_id": ["test1"], "point_name": ["point1"], "coord_x": [100], "coord_y": [200], "coord_z": [300]}
        )

        custom_temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        custom_data.to_csv(custom_temp_file.name, index=False)
        custom_temp_file.close()

        try:
            custom_extractor = CSVPointExtractor(
                landmark_csv_path=custom_temp_file.name,
                img_identifier_colname="image_id",
                landmark_id_colname="point_name",
                x_colname="coord_x",
                y_colname="coord_y",
                z_colname="coord_z",
            )

            input_data = {FILE_ID: "test1"}
            result = custom_extractor.extract_points(input_data)

            self.assertEqual(len(result), 1)
            self.assertIn("point1", result)
            self.assertTrue(result["point1"].is_valid)
            self.assertEqual(result["point1"].x, 100)
            self.assertEqual(result["point1"].y, 200)
            self.assertEqual(result["point1"].z, 300)

        finally:
            os.unlink(custom_temp_file.name)


if __name__ == "__main__":
    unittest.main()
