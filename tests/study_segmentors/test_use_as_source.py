from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from linguine.constants import FILE_ID, IMAGE, LABEL, USE_AS_SOURCE
from linguine.study_segmentors.base_segmentor import NoSourceScan
from linguine.study_segmentors.from_one_tp import FromOneTimepointSegmentor


class TestSourceScanSelection:
    """Test the use_as_source flag functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_propagator = Mock()
        self.mock_propagator.cfg.device = "cuda:0"  # Fix device mismatch
        self.mock_image_loader = Mock()
        self.mock_image_loader.return_value = (Mock(), torch.ones(10, 10, 10))  # Mock image, valid label

    @patch("linguine.study_segmentors.base_segmentor.get_path_from_data_dict_entry")
    def test_use_as_source_flag_selection(self, mock_get_path):
        """Test that scan with use_as_source=True is selected as source."""
        # Mock path validation to return valid paths
        mock_get_path.return_value = Path("/valid/path.nii.gz")

        patient_scans = [
            {FILE_ID: "scan1", IMAGE: "path1", LABEL: "label1"},
            {FILE_ID: "scan2", IMAGE: "path2", LABEL: "label2", USE_AS_SOURCE: True},
            {FILE_ID: "scan3", IMAGE: "path3", LABEL: "label3"},
        ]

        with patch.object(FromOneTimepointSegmentor, "_sort_scans_by_timepoint", return_value=patient_scans):
            with patch("pathlib.Path.exists", return_value=True):
                segmentor = FromOneTimepointSegmentor(
                    patient_id="test_patient",
                    patient_scans=patient_scans,
                    propagator=self.mock_propagator,
                    image_loader=self.mock_image_loader,
                )

        assert segmentor.source_scan[FILE_ID] == "scan2"

    def test_multiple_use_as_source_raises_error(self):
        """Test that multiple use_as_source=True flags raise ValueError."""
        patient_scans = [
            {FILE_ID: "scan1", IMAGE: "path1", LABEL: "label1", USE_AS_SOURCE: True},
            {FILE_ID: "scan2", IMAGE: "path2", LABEL: "label2", USE_AS_SOURCE: True},
            {FILE_ID: "scan3", IMAGE: "path3", LABEL: "label3"},
        ]

        with patch.object(FromOneTimepointSegmentor, "_sort_scans_by_timepoint", return_value=patient_scans):
            with pytest.raises(ValueError, match="Multiple scans have 'use_as_source=True'"):
                FromOneTimepointSegmentor(
                    patient_id="test_patient",
                    patient_scans=patient_scans,
                    propagator=self.mock_propagator,
                    image_loader=self.mock_image_loader,
                )

    def test_fallback_to_automatic_selection(self):
        """Test fallback to automatic selection when no use_as_source flag is present."""

        patient_scans = [
            {FILE_ID: "scan1", IMAGE: "path1", LABEL: Path("/valid/path.nii.gz")},
            {FILE_ID: "scan2", IMAGE: "path2", LABEL: "label2"},
            {FILE_ID: "scan3", IMAGE: "path3", LABEL: "label3"},
        ]

        with patch.object(FromOneTimepointSegmentor, "_sort_scans_by_timepoint", return_value=patient_scans):
            with patch("pathlib.Path.exists", return_value=True):
                segmentor = FromOneTimepointSegmentor(
                    patient_id="test_patient",
                    patient_scans=patient_scans,
                    propagator=self.mock_propagator,
                    image_loader=self.mock_image_loader,
                )

        # Should select first scan (automatic selection)
        assert segmentor.source_scan[FILE_ID] == "scan1"

    def test_invalid_specified_source_scan(self):
        """Test that invalid specified source scan raises NoSourceScan."""

        patient_scans = [
            {FILE_ID: "scan1", IMAGE: "path1", LABEL: "label1"},
            {FILE_ID: "scan2", IMAGE: "path2", LABEL: Path("/valid/path.nii.gz"), USE_AS_SOURCE: True},
        ]

        # Mock empty label for the specified source scan
        def mock_loader_with_empty_label(scan_dict):
            if scan_dict[FILE_ID] == "scan2":
                return Mock(), torch.zeros(10, 10, 10)  # Empty label
            return Mock(), torch.ones(10, 10, 10)  # Valid label

        self.mock_image_loader.side_effect = mock_loader_with_empty_label

        with patch.object(FromOneTimepointSegmentor, "_sort_scans_by_timepoint", return_value=patient_scans):
            with patch("pathlib.Path.exists", return_value=True):
                with pytest.raises(NoSourceScan):
                    FromOneTimepointSegmentor(
                        patient_id="test_patient",
                        patient_scans=patient_scans,
                        propagator=self.mock_propagator,
                        image_loader=self.mock_image_loader,
                    )
