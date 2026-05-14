import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor

from linguine.constants import FILE_ID, IMAGE, LABEL
from linguine.metrics import MetricsBundle
from linguine.study_segmentors.chain import ChainSegmentor


class TestChainSegmentor(unittest.TestCase):
    """Test cases for the ChainSegmentor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock components
        self.mock_propagator = Mock()
        self.mock_propagator.cfg.device = "cpu"
        self.mock_propagator.inferer.spacing = (1.0, 1.0, 1.0)

        # Set up propagator to return new random prediction each time
        def mock_propagator_side_effect(*args, **kwargs):
            return (MetricsBundle(0.85), [(32, 32, 32)], self._create_mock_prediction())

        self.mock_propagator.side_effect = mock_propagator_side_effect

        self.mock_image_loader = Mock()

        # Create sample patient scans
        self.patient_scans = [
            {FILE_ID: "patient1_baseline", IMAGE: "/path/baseline.nii.gz", LABEL: "/path/baseline_label.nii.gz"},
            {FILE_ID: "patient1_week_4", IMAGE: "/path/week4.nii.gz"},
            {FILE_ID: "patient1_week_8", IMAGE: "/path/week8.nii.gz"},
            {FILE_ID: "patient1_week_12", IMAGE: "/path/week12.nii.gz"},
        ]

        # Create sample label with multiple lesions
        self.sample_label = MetaTensor(
            [
                [[[0, 0, 0], [1, 1, 0], [0, 2, 2]]],
            ],
            affine=np.eye(4),
            dtype=torch.float32,
        )

        # Mock image loader to return image and label
        self.mock_image = Mock(spec=MetaTensor)
        self.mock_image.affine = np.eye(4)
        self.mock_image.meta = {"spacing": (1.0, 1.0, 1.0)}
        self.mock_image_loader.return_value = (self.mock_image, self.sample_label)

    def _create_mock_prediction(self, shape=(1, 1, 64, 64, 64)):
        """Helper method to create a properly structured mock prediction."""
        mock_pred = Mock()
        mock_pred.affine = np.eye(4)
        # Create the array with zeros and place ones in random locations
        pred_array = np.zeros(shape)
        # Generate random indices for placing ones (about 10% of the array)
        num_ones = max(1, int(np.prod(shape) * 0.1))
        flat_indices = np.random.choice(np.prod(shape), size=num_ones, replace=False)
        pred_array.flat[flat_indices] = 1
        mock_pred.__getitem__ = Mock(return_value=pred_array)
        mock_pred.__eq__ = Mock(return_value=pred_array)  # For np.where comparison
        return mock_pred

    @patch.object(Path, "exists", return_value=True)
    def test_initialization_with_resampling(self, mock_exists):
        """Test ChainSegmentor initialization with resampling enabled."""
        segmentor = ChainSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
            with_resampling=True,
        )
        self.assertTrue(segmentor.with_resampling)

    @patch.object(Path, "exists", return_value=True)
    def test_initialization_without_resampling(self, mock_exists):
        """Test ChainSegmentor initialization without resampling enabled."""
        segmentor = ChainSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
            with_resampling=False,
        )
        self.assertFalse(segmentor.with_resampling)

    @patch("linguine.study_segmentors.chain.find_mask_center")
    @patch.object(Path, "exists", return_value=True)
    def test_segment_target_scan_with_resampling(self, mock_exists, mock_find_center):
        """Test target scan segmentation with resampling enabled - unique to ChainSegmentor."""
        mock_find_center.return_value = (42, 42, 42)
        # Mock propagator will return random predictions via side_effect

        segmentor = ChainSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
            with_resampling=True,
        )
        target_scan = {FILE_ID: "patient1_week_4", IMAGE: "/path/week4.nii.gz"}
        with patch("linguine.study_segmentors.chain.LOGGER") as mock_logger:
            result = segmentor.segment_target_scan(target_scan, predict_only=False)
            # Should log the resampled click
            mock_logger.info.assert_any_call("Sampled new click from prediction: (42, 42, 42)")
            # Should update source clicks with resampled click
            self.assertEqual(segmentor.source_prompts_per_lesion[0], [(42, 42, 42)])
            self.assertIsNotNone(result)

    @patch.object(Path, "exists", return_value=True)
    def test_segment_target_scan_without_resampling(self, mock_exists):
        """Test target scan segmentation without resampling - unique chain behavior."""
        # Mock propagator will return random predictions via side_effect
        segmentor = ChainSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
            with_resampling=False,
        )

        target_scan = {FILE_ID: "patient1_week_4", IMAGE: "/path/week4.nii.gz"}
        segmentor.segment_target_scan(target_scan, predict_only=False)

        # Should update source clicks with target clicks (no resampling)
        self.assertEqual(segmentor.source_prompts_per_lesion[0], [(32, 32, 32)])

    @patch.object(Path, "exists", return_value=True)
    def test_segment_target_scan_updates_source_scan(self, mock_exists):
        """Test that target scan becomes the new source scan after segmentation - chain-specific."""
        # Mock propagator will return random predictions via side_effect

        segmentor = ChainSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        initial_source = segmentor.source_scan[FILE_ID]
        target_scan = {FILE_ID: "patient1_week_4", IMAGE: "/path/week4.nii.gz"}

        segmentor.segment_target_scan(target_scan, predict_only=False)

        # Source scan should be updated to the target scan - this is chain-specific behavior
        self.assertNotEqual(segmentor.source_scan[FILE_ID], initial_source)
        self.assertEqual(segmentor.source_scan[FILE_ID], "patient1_week_4")

    @patch.object(Path, "exists", return_value=True)
    def test_segment_target_scan_lesion_dropout(self, mock_exists):
        """Test that lesions can drop out during chain propagation - chain-specific behavior."""

        # Mock propagator to fail for one lesion and succeed for another
        def mock_propagator_side_effect(*args, **kwargs):
            lesion_label = kwargs.get("source_label")
            if torch.sum(lesion_label) == 2:  # First lesion
                return (None, None, None)  # Failed propagation
            else:  # Second lesion
                mock_pred = self._create_mock_prediction()
                return (0.75, [(35, 35, 35)], mock_pred)

        self.mock_propagator.side_effect = mock_propagator_side_effect

        segmentor = ChainSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        initial_lesion_count = len(segmentor.lesion_ids)
        target_scan = {FILE_ID: "patient1_week_4", IMAGE: "/path/week4.nii.gz"}

        with patch("linguine.study_segmentors.chain.LOGGER") as mock_logger:
            segmentor.segment_target_scan(target_scan, predict_only=False)

            # Should log warning about lesion dropout
            mock_logger.warning.assert_called()

            # Should have fewer lesions for next iteration - chain-specific
            self.assertLess(len(segmentor.lesion_ids), initial_lesion_count)

    @patch.object(Path, "exists", return_value=True)
    def test_chain_propagation_workflow(self, mock_exists):
        """Test the complete chain propagation workflow across multiple timepoints."""

        # Mock propagator will return random predictions via side_effect

        segmentor = ChainSegmentor(
            patient_id="patient1",
            patient_scans=self.patient_scans,
            propagator=self.mock_propagator,
            image_loader=self.mock_image_loader,
            device="cpu",
        )

        # Track how source scan changes during chain propagation
        initial_source = segmentor.source_scan[FILE_ID]
        initial_n_lesions = len(segmentor.lesion_ids)
        self.assertEqual(initial_n_lesions, 2)

        # Simulate the chain propagation by calling the main workflow
        results = segmentor(predict_only=False)

        # Expect first two propagations to have been from the initial source scan
        # One per lesion.
        self.assertEqual(results[0]["source_scan"], initial_source)
        self.assertEqual(results[0]["lesion_id"], 1.0)
        self.assertEqual(results[1]["source_scan"], initial_source)
        self.assertEqual(results[1]["lesion_id"], 2.0)

        # Should have a results for each lesion per scan
        self.assertEqual(len(results), len(segmentor.target_scans) * len(segmentor.lesion_ids))

        # Source scan should have been updated through the chain
        self.assertNotEqual(segmentor.source_scan[FILE_ID], initial_source)


if __name__ == "__main__":
    unittest.main()
