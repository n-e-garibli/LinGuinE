import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor

from linguine.constants import CLICKS, FILE_ID, IMAGE, LABEL, PATIENT_ID, TIMEPOINT, USE_AS_SOURCE
from linguine.metrics import MetricsBundle
from linguine.study_segmentors.base_segmentor import LongitudinalStudySegmentor, NoSourceScan


class MockSegmentor(LongitudinalStudySegmentor):
    """Concrete implementation of LongitudinalStudySegmentor for testing."""

    def segment_target_scan(
        self, target_scan: dict[str, Any], predict_only: bool
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Mock implementation that returns dummy prediction data."""
        if target_scan[FILE_ID] == "failing_scan":
            return None

        # Create dummy prediction and affine
        pred = np.ones((64, 64, 64), dtype=np.uint8)
        affine = np.eye(4)

        # Update results for testing
        if not predict_only and hasattr(self, "source_label"):
            for lesion_id in self.lesion_ids:
                self._update_results(
                    target_file_id=target_scan[FILE_ID],
                    target_prompts=[(32, 32, 32)],
                    target_metrics=MetricsBundle(dice=0.85),
                    lesion_id=lesion_id,
                )

        return pred, affine


class TestLongitudinalStudySegmentor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock components
        self.mock_propagator = Mock()
        self.mock_propagator.cfg.device = "cpu"

        self.mock_image_loader = Mock()

        # Create sample patient scans with various timepoint patterns
        self.patient_scans = [
            {FILE_ID: "patient1_baseline", IMAGE: "/path/baseline.nii.gz", LABEL: "/path/baseline_label.nii.gz"},
            {FILE_ID: "patient1_week_4", IMAGE: "/path/week4.nii.gz"},
            {FILE_ID: "patient1_week_8", IMAGE: "/path/week8.nii.gz"},
            {FILE_ID: "patient1_end_of_treatment", IMAGE: "/path/eot.nii.gz"},
        ]

        # Create sample label with multiple lesions
        self.sample_label = torch.tensor(
            [
                [[[0, 0, 0], [1, 1, 0], [0, 2, 2]]],
            ],
            dtype=torch.float32,
        )

        # Mock image loader to return image and label
        self.mock_image_loader.return_value = (
            Mock(spec=MetaTensor),  # Mock image
            self.sample_label,  # Sample label
        )

    @patch.object(Path, "exists", return_value=True)
    def test_initialization_success(self, mock_exists):
        """Test successful initialization of LongitudinalStudySegmentor."""
        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        self.assertEqual(segmentor.patient, "patient1")
        self.assertEqual(segmentor.device, "cpu")
        self.assertEqual(len(segmentor.lesion_ids), 2)  # lesions 1 and 2
        self.assertIn(1, segmentor.lesion_ids)
        self.assertIn(2, segmentor.lesion_ids)
        self.assertEqual(len(segmentor.target_scans), 3)  # All scans after source

    def test_device_mismatch_error(self):
        """Test that device mismatch raises assertion error."""
        self.mock_propagator.cfg.device = "cpu"

        with self.assertRaises(AssertionError):
            MockSegmentor(
                patient_id="patient1",
                patient_scans=self.patient_scans,
                propagator=self.mock_propagator,
                image_loader=self.mock_image_loader,
                device="cuda:0",
            )

    def test_sort_scans_by_timepoint_basic(self):
        """Test basic timepoint sorting functionality."""
        scans = [
            {FILE_ID: "patient1_week_8", IMAGE: "/path/week8.nii.gz"},
            {FILE_ID: "patient1_baseline", IMAGE: "/path/baseline.nii.gz"},
            {FILE_ID: "patient1_week_4", IMAGE: "/path/week4.nii.gz"},
            {FILE_ID: "patient1_end_of_treatment", IMAGE: "/path/eot.nii.gz"},
        ]

        sorted_scans = LongitudinalStudySegmentor._sort_scans_by_timepoint(scans)

        # Check that baseline comes first
        self.assertEqual(sorted_scans[0][FILE_ID], "patient1_baseline")
        self.assertEqual(sorted_scans[1][FILE_ID], "patient1_week_4")
        self.assertEqual(sorted_scans[2][FILE_ID], "patient1_week_8")
        # End of treatment should be last
        self.assertEqual(sorted_scans[3][FILE_ID], "patient1_end_of_treatment")

    def test_sort_scans_with_dedicated_timepoint_field(self):
        """Test sorting when scans have dedicated timepoint field."""
        scans = [
            {FILE_ID: "scan3", TIMEPOINT: "week_8", IMAGE: "/path/scan3.nii.gz"},
            {FILE_ID: "scan1", TIMEPOINT: "baseline", IMAGE: "/path/scan1.nii.gz"},
            {FILE_ID: "scan2", TIMEPOINT: "week_4", IMAGE: "/path/scan2.nii.gz"},
        ]

        sorted_scans = LongitudinalStudySegmentor._sort_scans_by_timepoint(scans)

        self.assertEqual(sorted_scans[0][FILE_ID], "scan1")  # baseline
        self.assertEqual(sorted_scans[1][FILE_ID], "scan2")  # week_4
        self.assertEqual(sorted_scans[2][FILE_ID], "scan3")  # week_8

    def test_sort_scans_with_retreatment(self):
        """Test sorting with retreatment scans."""
        scans = [
            {FILE_ID: "patient1_week_4", IMAGE: "/path/week4.nii.gz"},
            {FILE_ID: "patient1_rebaseline", IMAGE: "/path/rebaseline.nii.gz"},
            {FILE_ID: "patient1_baseline", IMAGE: "/path/baseline.nii.gz"},
            {FILE_ID: "patient1_retreatment_week_2", IMAGE: "/path/retreatment.nii.gz"},
        ]

        sorted_scans = LongitudinalStudySegmentor._sort_scans_by_timepoint(scans)

        # Regular timepoints should come first, then retreatment timepoints
        file_ids = [scan[FILE_ID] for scan in sorted_scans]
        baseline_idx = file_ids.index("patient1_baseline")
        week4_idx = file_ids.index("patient1_week_4")
        rebaseline_idx = file_ids.index("patient1_rebaseline")
        retreatment_idx = file_ids.index("patient1_retreatment_week_2")

        self.assertLess(baseline_idx, week4_idx)
        self.assertLess(week4_idx, rebaseline_idx)
        self.assertLess(rebaseline_idx, retreatment_idx)

    def test_sort_scans_with_unscheduled(self):
        """Test that unscheduled scans are placed at the end."""
        scans = [
            {FILE_ID: "patient1_week_4", IMAGE: "/path/week4.nii.gz"},
            {FILE_ID: "patient1_unscheduled", IMAGE: "/path/unscheduled.nii.gz"},
            {FILE_ID: "patient1_baseline", IMAGE: "/path/baseline.nii.gz"},
        ]
        sorted_scans = LongitudinalStudySegmentor._sort_scans_by_timepoint(scans)
        # Verify correct order: baseline, week_4, unscheduled
        self.assertEqual(len(sorted_scans), 3)
        self.assertEqual(sorted_scans[0][FILE_ID], "patient1_baseline")
        self.assertEqual(sorted_scans[1][FILE_ID], "patient1_week_4")
        self.assertEqual(sorted_scans[2][FILE_ID], "patient1_unscheduled")

    def test_sort_scans_with_custom_timepoint_flags(self):
        """Test sorting with custom timepoint identifier patterns."""
        scans = [
            {FILE_ID: "patient1_visit_3", IMAGE: "/path/visit3.nii.gz"},
            {FILE_ID: "patient1_baseline", IMAGE: "/path/baseline.nii.gz"},
            {FILE_ID: "patient1_visit_1", IMAGE: "/path/visit1.nii.gz"},
            {FILE_ID: "patient1_visit_2", IMAGE: "/path/visit2.nii.gz"},
        ]

        # Use custom timepoint flags
        custom_flags = ["visit_"]
        sorted_scans = LongitudinalStudySegmentor._sort_scans_by_timepoint(scans, custom_flags)

        # Check that scans are properly sorted
        self.assertEqual(sorted_scans[0][FILE_ID], "patient1_baseline")
        self.assertEqual(sorted_scans[1][FILE_ID], "patient1_visit_1")
        self.assertEqual(sorted_scans[2][FILE_ID], "patient1_visit_2")
        self.assertEqual(sorted_scans[3][FILE_ID], "patient1_visit_3")

    def test_sort_scans_with_custom_timepoint_flags_filters_non_matching(self):
        """Test filtering behavior when custom flags don't match scan naming patterns.

        This test verifies that when custom timepoint flags are provided that deliberately
        don't match the scan naming patterns (e.g., custom flags "visit_", "appointment_"
        vs scans named "week_1", "week_2"), the non-matching scans are excluded from the
        sorted result. The filtering behavior itself is existing functionality, but this
        test ensures it works correctly with the new custom timepoint flags feature.
        """
        scans = [
            {FILE_ID: "patient1_week_2", IMAGE: "/path/week2.nii.gz"},
            {FILE_ID: "patient1_baseline", IMAGE: "/path/baseline.nii.gz"},
            {FILE_ID: "patient1_week_1", IMAGE: "/path/week1.nii.gz"},
        ]

        # Use custom flags that don't match the "week_" pattern
        custom_flags = ["visit_", "appointment_"]
        sorted_scans = LongitudinalStudySegmentor._sort_scans_by_timepoint(scans, custom_flags)

        # Only baseline should be returned since week_X doesn't match custom patterns
        self.assertEqual(len(sorted_scans), 1)
        self.assertEqual(sorted_scans[0][FILE_ID], "patient1_baseline")

    @patch.object(Path, "exists", return_value=True)
    def test_pick_source_scan_success(self, mock_exists):
        """Test successful source scan selection."""
        # Add label field to first scan
        scans_with_labels = self.patient_scans.copy()

        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=scans_with_labels,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        self.assertEqual(segmentor.source_scan[FILE_ID], "patient1_baseline")

    @patch.object(Path, "exists", return_value=True)
    def test_pick_source_scan_no_suitable_scan(self, mock_exists):
        """Test that NoSourceScan exception is raised when no suitable source scan exists."""
        # Mock image loader to return empty label
        self.mock_image_loader.return_value = (
            Mock(spec=MetaTensor),
            torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32),
        )

        with self.assertRaises(NoSourceScan):
            MockSegmentor(
                patient_id="patient1",
                patient_scans=self.patient_scans,
                propagator=self.mock_propagator,
                image_loader=self.mock_image_loader,
                device="cpu",
            )

    @patch("linguine.study_segmentors.base_segmentor.find_mask_center")
    @patch.object(Path, "exists", return_value=True)
    def test_get_prompts_per_lesion(self, mock_exists, mock_find_center):
        """Test click generation for each lesion."""
        mock_find_center.side_effect = [(10, 10, 10), (20, 20, 20)]

        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        self.assertEqual(len(segmentor.source_prompts_per_lesion), 2)
        self.assertEqual(segmentor.source_prompts_per_lesion[0], [(10, 10, 10)])
        self.assertEqual(segmentor.source_prompts_per_lesion[1], [(20, 20, 20)])

    @patch.object(Path, "exists", return_value=True)
    def test_get_clicks_per_lesion_with_user_clicks(self, mock_exists):
        """Test click handling when user provides clicks in source scan."""
        # Create patient scans with clicks
        scans_with_clicks = self.patient_scans.copy()
        scans_with_clicks[0][CLICKS] = [(32, 32, 32), (40, 40, 40)]

        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=scans_with_clicks,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        clicks_per_lesion = segmentor._get_clicks_per_lesion()

        # There should only be one lesion with two clicks provided.
        self.assertEqual(len(clicks_per_lesion), 1)
        self.assertEqual(len(clicks_per_lesion[0]), 2)

    @patch("os.makedirs")
    @patch("nibabel.save")
    @patch("nibabel.nifti1.Nifti1Image")
    @patch.object(Path, "exists", return_value=True)
    def test_call_with_save_dir(self, mock_exists, mock_nifti, mock_save, mock_makedirs):
        """Test the main segmentation workflow with save directory."""
        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        results = segmentor(predict_only=False, save_dir="/tmp/test")

        # Check that directories were created
        mock_makedirs.assert_called_once_with(name="/tmp/test", exist_ok=True)

        # Check that predictions were saved (3 target scans)
        self.assertEqual(mock_save.call_count, 3)

        # Check results structure
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    @patch.object(Path, "exists", return_value=True)
    def test_get_clicks_per_lesion_invalid_format(self, mock_exists):
        """Test error handling for invalid click formats."""
        scans_with_clicks = self.patient_scans.copy()

        # Test non-list clicks
        scans_with_clicks[0][CLICKS] = (32, 32, 32)  # tuple instead of list
        with self.assertRaises(ValueError) as context:
            MockSegmentor(
                patient_id="patient1",
                patient_scans=scans_with_clicks,
                propagator=self.mock_propagator,
                image_loader=self.mock_image_loader,
                device="cpu",
            )
        self.assertIn("Clicks must be a list", str(context.exception))

        # Test invalid click dimensions
        scans_with_clicks[0][CLICKS] = [(32, 32)]  # 2D instead of 3D
        with self.assertRaises(ValueError) as context:
            MockSegmentor(
                patient_id="patient1",
                patient_scans=scans_with_clicks,
                propagator=self.mock_propagator,
                image_loader=self.mock_image_loader,
                device="cpu",
            )
        self.assertIn("Each click must be a 3-element tuple/list", str(context.exception))

        # Test non-numeric coordinates
        scans_with_clicks[0][CLICKS] = [("a", "b", "c")]
        with self.assertRaises(ValueError) as context:
            MockSegmentor(
                patient_id="patient1",
                patient_scans=scans_with_clicks,
                propagator=self.mock_propagator,
                image_loader=self.mock_image_loader,
                device="cpu",
            )

    @patch.object(Path, "exists", return_value=True)
    def test_pick_source_scan_with_clicks_no_label(self, mock_exists):
        """Test source scan selection when scan has clicks but no valid label."""
        # Mock image loader to return empty label (simulate no ground truth)
        empty_label = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32)
        self.mock_image_loader.return_value = (Mock(spec=MetaTensor), empty_label)

        scans_with_clicks = self.patient_scans.copy()
        scans_with_clicks[0][CLICKS] = [(32, 32, 32)]
        scans_with_clicks[0][USE_AS_SOURCE] = True
        # Remove label to simulate clicks-only scenario
        if LABEL in scans_with_clicks[0]:
            del scans_with_clicks[0][LABEL]

        # Mock the propagator's get_metrics_and_prediction method
        mock_prediction = torch.ones((1, 1, 64, 64, 64), dtype=torch.float32)
        self.mock_propagator.get_metrics_and_prediction.return_value = (MetricsBundle(dice=0.0), mock_prediction)

        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=scans_with_clicks,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )
        # Should successfully use clicks to generate a label
        self.assertEqual(segmentor.source_scan[FILE_ID], "patient1_baseline")
        self.mock_propagator.get_metrics_and_prediction.assert_called_once()

    @patch.object(Path, "exists", return_value=True)
    def test_automatic_source_scan_selection_with_clicks(self, mock_exists):
        """Test automatic source scan selection falls back to clicks when no labels."""
        # Mock all scans to have no valid labels
        empty_label = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32)
        img = torch.ones((1, 1, 64, 64, 64), dtype=torch.float32)
        self.mock_image_loader.return_value = (img, empty_label)

        scans_with_clicks = self.patient_scans.copy()
        scans_with_clicks[1][CLICKS] = [(32, 32, 32)]  # Second scan has clicks

        # Mock the propagator's get_metrics_and_prediction method
        mock_prediction = torch.ones((1, 1, 64, 64, 64), dtype=torch.float32)
        self.mock_propagator.get_metrics_and_prediction.return_value = (MetricsBundle(dice=0.0), mock_prediction)

        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=scans_with_clicks,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        # Should use the scan with clicks as source
        self.assertEqual(segmentor.source_scan[FILE_ID], "patient1_week_4")
        self.mock_propagator.get_metrics_and_prediction.assert_called_once()

    @patch.object(Path, "exists", return_value=True)
    def test_call_predict_only(self, mock_exists):
        """Test the main segmentation workflow in predict-only mode."""
        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        results = segmentor(predict_only=True)
        self.assertIsInstance(results, list)
        for result in results:
            # No dices in predict only mode
            self.assertEqual(result["target_dice"], np.nan)

    @patch.object(Path, "exists", return_value=True)
    def test_call_with_failing_segment_target_scan(self, mock_exists):
        """Test workflow when segment_target_scan returns None."""
        # Create a segmentor with a scan that will fail
        scans_with_failing = self.patient_scans + [{FILE_ID: "failing_scan", IMAGE: "/path/failing.nii.gz"}]

        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=scans_with_failing,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        # Should handle None return gracefully
        results = segmentor(predict_only=False)
        self.assertIsInstance(results, list)

    @patch.object(Path, "exists", return_value=True)
    def test_target_scans_selection(self, mock_exists):
        """Test that target scans are correctly selected as scans after source."""
        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        # Source scan should be the first (baseline)
        self.assertEqual(segmentor.source_scan[FILE_ID], "patient1_baseline")

        # Target scans should be the remaining 3
        target_file_ids = [scan[FILE_ID] for scan in segmentor.target_scans]
        expected_ids = ["patient1_week_4", "patient1_week_8", "patient1_end_of_treatment"]
        self.assertEqual(target_file_ids, expected_ids)

    @patch.object(Path, "exists", return_value=True)
    def test_empty_patient_scans(self, mock_exists):
        """Test handling of empty patient scans list."""
        with self.assertRaises(NoSourceScan):
            MockSegmentor(
                patient_id="patient1",
                patient_scans=[],
                propagator=self.mock_propagator,
                image_loader=self.mock_image_loader,
                device="cpu",
            )

    @patch.object(Path, "exists", return_value=True)
    def test_update_results(self, mock_exists):
        """Test results update functionality."""
        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        # Manually update results
        segmentor._update_results(
            target_file_id="test_scan",
            target_prompts=[(32, 32, 32)],
            target_metrics=MetricsBundle(dice=0.9, hd95=3.2),
            lesion_id=1,
        )

        self.assertEqual(len(segmentor.results), 1)
        result = segmentor.results[0]
        self.assertEqual(result[PATIENT_ID], "patient1")
        self.assertEqual(result["target_scan"], "test_scan")
        self.assertEqual(result["target_dice"], 0.9)
        self.assertEqual(result["target_hd95"], 3.2)
        self.assertEqual(result["lesion_id"], 1)

    @patch.object(Path, "exists", return_value=True)
    def test_single_scan_patient(self, mock_exists):
        """Test handling of patient with only one scan."""
        single_scan = [self.patient_scans[0]]

        segmentor = MockSegmentor(
            patient_id="patient1",
            patient_scans=single_scan,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        # Should have no target scans to process
        self.assertEqual(len(segmentor.target_scans), 0)

        results = segmentor(predict_only=False)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
