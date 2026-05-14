import unittest
from unittest.mock import Mock

from linguine.constants import CLICKS, FILE_ID, IMAGE, LABEL, PATIENT_ID, USE_AS_SOURCE
from linguine.dataset_processor import LinguineDatasetProcessor


class TestDatasetProcessorValidation(unittest.TestCase):
    """Test the dataset-level validation for use_as_source flags."""

    def test_dataset_validation_with_multiple_source_flags(self):
        """Test that dataset validation catches multiple use_as_source flags per patient."""
        data_dicts = [
            {PATIENT_ID: "patient1", FILE_ID: "scan1", IMAGE: "path1", USE_AS_SOURCE: True},
            {PATIENT_ID: "patient1", FILE_ID: "scan2", IMAGE: "path2", USE_AS_SOURCE: True},
            {PATIENT_ID: "patient2", FILE_ID: "scan3", IMAGE: "path3", USE_AS_SOURCE: True},
        ]

        # Create a minimal processor instance just for testing validation
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.data_dicts = data_dicts
        processor.cfg = Mock()
        processor.cfg.patient_ids = None
        grouped_scans = processor._group_datadicts_by_patient()

        with self.assertRaises(ValueError) as context:
            processor._validate_source_scan_flags(grouped_scans)
        self.assertIn("Multiple scans have 'use_as_source=True' for patient patient1", str(context.exception))

    def test_dataset_validation_passes_with_valid_flags(self):
        """Test that validation passes when each patient has at most one source flag."""
        data_dicts = [
            {PATIENT_ID: "patient1", FILE_ID: "scan1", IMAGE: "path1", LABEL: "path1", USE_AS_SOURCE: True},
            {PATIENT_ID: "patient1", FILE_ID: "scan2", IMAGE: "path2"},
            {
                PATIENT_ID: "patient2",
                FILE_ID: "scan3",
                IMAGE: "path3",
                CLICKS: [(1.0, 1.0, 1.0)],
                USE_AS_SOURCE: True,
            },
            {PATIENT_ID: "patient3", FILE_ID: "scan4", IMAGE: "path4"},  # No source flag
        ]

        # Create a minimal processor instance just for testing validation
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.data_dicts = data_dicts
        processor.cfg = Mock()
        processor.cfg.patient_ids = None
        grouped_scans = processor._group_datadicts_by_patient()
        # Should not raise any exception since one patient has a label and another has clicks
        processor._validate_source_scan_flags(grouped_scans)
