import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import SimpleITK as sitk

from linguine.config import LinguineConfig
from linguine.constants import FILE_ID, IMAGE, PATIENT_ID
from linguine.dataset_processor import LinguineDatasetProcessor
from linguine.inferers import (
    AbstractInferer,
    BasicBoostedInferer,
    MergeProbabilitiesBoostedInferer,
    OrientationEnsembleInferer,
    ResampleAdditiveBoostedInferer,
)
from linguine.prompt_selectors import ThresholdPS, UnguidedModelPS
from linguine.registration.registrators import ArunsRigidRegistrator, ThinPlateSplineRegistrator


class TestLinguineDatasetProcessor(unittest.TestCase):
    """Test cases for LinguineDatasetProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.save_folder = os.path.join(self.temp_dir, "test_output")
        os.makedirs(self.save_folder, exist_ok=True)

        # Create mock configuration
        self.mock_cfg = Mock(spec=LinguineConfig)
        self.mock_cfg.device = "cpu"
        self.mock_cfg.patient_ids = None
        self.mock_cfg.predict_only = False
        self.mock_cfg.save_predictions = False

        # Mock prompt selection config
        self.mock_cfg.prompt_selection = Mock()
        self.mock_cfg.prompt_selection.type = None
        self.mock_cfg.prompt_selection.u_threshold = 0.8
        self.mock_cfg.prompt_selection.l_threshold = 0.2

        # Mock registration config
        self.mock_cfg.registration = Mock()
        self.mock_cfg.registration.point_extractor_csv = None
        self.mock_cfg.registration.registrator = "aruns"
        self.mock_cfg.registration.tps_lambda = 0.1

        # Mock analysis config
        self.mock_cfg.analysis = Mock()
        self.mock_cfg.analysis.boosting = None
        self.mock_cfg.analysis.iteration_mode = "FROM_ONE_TIMEPOINT"
        self.mock_cfg.analysis.min_pred_size_mm = 10.0

        # Create mock inferer
        self.mock_inferer = Mock(spec=AbstractInferer)

        # Sample data dicts
        self.data_dicts = [
            {PATIENT_ID: "patient1", FILE_ID: "scan1", IMAGE: "path1"},
            {PATIENT_ID: "patient1", FILE_ID: "scan2", IMAGE: "path2"},
            {PATIENT_ID: "patient2", FILE_ID: "scan3", IMAGE: "path3"},
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_boost_inferer_basic(self):
        """Test _boost_inferer with basic boosting."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        result = processor._boost_inferer(self.mock_inferer, "basic")

        self.assertIsInstance(result, BasicBoostedInferer)

    def test_boost_inferer_resample_additive(self):
        """Test _boost_inferer with resample additive boosting."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        result = processor._boost_inferer(self.mock_inferer, "resample_additive")

        self.assertIsInstance(result, ResampleAdditiveBoostedInferer)

    def test_boost_inferer_resample_merge_probs(self):
        """Test _boost_inferer with resample merge probabilities boosting."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        result = processor._boost_inferer(self.mock_inferer, "resample_merge_probs")

        self.assertIsInstance(result, MergeProbabilitiesBoostedInferer)

    def test_boost_inferer_orientation_ensemble(self):
        """Test _boost_inferer with orientation ensemble boosting."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        result = processor._boost_inferer(self.mock_inferer, "orientation_ensemble")

        self.assertIsInstance(result, OrientationEnsembleInferer)

    def test_boost_inferer_unknown_technique(self):
        """Test _boost_inferer raises error for unknown boosting technique."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        with self.assertRaises(ValueError) as context:
            processor._boost_inferer(self.mock_inferer, "unknown_boosting")

        self.assertIn("Unknown boosting technique", str(context.exception))

    def test_get_ps_none_type(self):
        """Test _get_ps with None type returns None."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        processor.image_loader = Mock()
        self.mock_cfg.prompt_selection.type = None

        result = processor._get_ps(self.mock_inferer)

        self.assertIsNone(result)

    def test_get_ps_unguided_type(self):
        """Test _get_ps with unguided type returns UnguidedModelPS."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        processor.image_loader = Mock()
        self.mock_cfg.prompt_selection.type = "unguided"

        result = processor._get_ps(self.mock_inferer)

        self.assertIsInstance(result, UnguidedModelPS)

    def test_get_ps_threshold_type(self):
        """Test _get_ps with threshold type returns ThresholdPS."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        processor.image_loader = Mock()
        self.mock_cfg.prompt_selection.type = "threshold"
        self.mock_cfg.analysis.seed = 42

        result = processor._get_ps(self.mock_inferer)

        self.assertIsInstance(result, ThresholdPS)

    def test_get_registrator_aruns(self):
        """Test _get_registrator with Arun's method."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        self.mock_cfg.registration.registrator = "aruns"

        result = processor._get_registrator()

        self.assertIsInstance(result, ArunsRigidRegistrator)

    def test_get_registrator_tps(self):
        """Test _get_registrator with Thin Plate Spline method."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        self.mock_cfg.registration.registrator = "tps"

        result = processor._get_registrator()

        self.assertIsInstance(result, ThinPlateSplineRegistrator)

    def test_get_registrator_perfect(self):
        """Test _get_registrator with perfect mode returns None."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        self.mock_cfg.registration.registrator = "perfect"

        result = processor._get_registrator()

        self.assertIsNone(result)

    def test_get_registrator_invalid(self):
        """Test _get_registrator raises KeyError for invalid registrator."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        self.mock_cfg.registration.registrator = "invalid_registrator"

        with self.assertRaises(KeyError):
            processor._get_registrator()

    def test_get_study_segmentor_invalid_mode(self):
        """Test get_study_segmentor raises KeyError for invalid mode."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        processor.propagator = Mock()
        processor.image_loader = Mock()

        with self.assertRaises(KeyError):
            processor.get_study_segmentor("INVALID_MODE", "patient1", self.data_dicts[:2])

    def test_group_datadicts_by_patient(self):
        """Test _group_datadicts_by_patient groups correctly."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.data_dicts = self.data_dicts
        processor.cfg = self.mock_cfg

        result = processor._group_datadicts_by_patient()

        self.assertEqual(len(result), 2)  # Two patients
        self.assertEqual(len(result["patient1"]), 2)  # Patient1 has 2 scans
        self.assertEqual(len(result["patient2"]), 1)  # Patient2 has 1 scan

    def test_group_datadicts_by_patient_with_filter(self):
        """Test _group_datadicts_by_patient with patient ID filter."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.data_dicts = self.data_dicts
        processor.cfg = self.mock_cfg
        processor.cfg.patient_ids = ["patient1"]  # Only include patient1

        result = processor._group_datadicts_by_patient()

        self.assertEqual(len(result), 1)  # Only one patient
        self.assertIn("patient1", result)
        self.assertNotIn("patient2", result)

    def test_group_datadicts_missing_patient_id(self):
        """Test _group_datadicts_by_patient raises error for missing patient_id."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.data_dicts = [{FILE_ID: "scan1", IMAGE: "path1"}]  # Missing patient_id
        processor.cfg = self.mock_cfg

        with self.assertRaises(ValueError) as context:
            processor._group_datadicts_by_patient()

        self.assertIn("patient_id key not found", str(context.exception))

    @patch("linguine.dataset_processor.CSVPointExtractor")
    def test_get_point_extractor_with_csv(self, mock_csv_extractor):
        """Test _get_point_extractor when CSV path is provided."""
        # Create a temporary CSV file
        csv_path = os.path.join(self.temp_dir, "test_landmarks.csv")
        with open(csv_path, "w") as f:
            f.write("file_id,name,x,y,z\n")
            f.write("scan1,landmark1,10,20,30\n")

        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        processor.cfg.registration.point_extractor_csv = csv_path

        # Mock the extractor instance
        mock_extractor_instance = Mock()
        mock_extractor_instance.valid_ids = {"landmark1", "landmark2"}
        mock_csv_extractor.return_value = mock_extractor_instance

        result = processor._get_point_extractor()

        mock_csv_extractor.assert_called_once_with(landmark_csv_path=csv_path)
        self.assertEqual(result, mock_extractor_instance)

    def test_get_point_extractor_csv_not_exists(self):
        """Test _get_point_extractor raises error when CSV doesn't exist."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        processor.cfg.registration.point_extractor_csv = "/nonexistent/path.csv"

        with self.assertRaises(ValueError) as context:
            processor._get_point_extractor()

        self.assertIn("not found", str(context.exception))

    @patch("linguine.dataset_processor.LinguineClickPropagator")
    def test_setup_propagator(self, mock_propagator_class):
        """Test _setup_propagator creates propagator correctly."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        processor.registrator = Mock()
        processor.point_extractor = Mock()
        processor.prompt_selector = Mock()
        processor.dd = Mock()

        mock_propagator_instance = Mock()
        mock_propagator_class.return_value = mock_propagator_instance

        result = processor._setup_propagator(self.mock_inferer)

        mock_propagator_class.assert_called_once_with(
            cfg=self.mock_cfg,
            inferer=self.mock_inferer,
            registrator=processor.registrator,
            point_extractor=processor.point_extractor,
            prompt_selector=processor.prompt_selector,
            disappearance_detector=processor.dd,
        )
        self.assertEqual(result, mock_propagator_instance)

    @patch("linguine.dataset_processor.sitk.WriteImage")
    def test_postprocess_predictions_success(self, mock_write_image):
        """Test postprocess_predictions successfully resamples existing predictions."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.save_folder = self.save_folder
        processor.data_dicts = [
            {FILE_ID: "scan1", IMAGE: "/path/to/image1.nii.gz"},
            {FILE_ID: "scan2", IMAGE: "/path/to/image2.nii.gz"},
        ]

        # Create mock prediction files
        pred_path1 = Path(self.save_folder) / "pred_scan1.nii.gz"
        pred_path2 = Path(self.save_folder) / "pred_scan2.nii.gz"
        pred_path1.touch()
        pred_path2.touch()

        # Mock the resampling method
        mock_resampled_image = Mock()
        processor._resample_pred_to_match_image = Mock(return_value=mock_resampled_image)

        processor.postprocess_predictions()

        # Verify _resample_pred_to_match_image was called for each existing prediction
        self.assertEqual(processor._resample_pred_to_match_image.call_count, 2)
        processor._resample_pred_to_match_image.assert_any_call(pred_path1, Path("/path/to/image1.nii.gz"))
        processor._resample_pred_to_match_image.assert_any_call(pred_path2, Path("/path/to/image2.nii.gz"))

        # Verify WriteImage was called for each resampled prediction
        self.assertEqual(mock_write_image.call_count, 2)
        mock_write_image.assert_any_call(mock_resampled_image, str(pred_path1))
        mock_write_image.assert_any_call(mock_resampled_image, str(pred_path2))

    @patch("linguine.dataset_processor.sitk.WriteImage")
    @patch("linguine.dataset_processor.LOGGER")
    def test_postprocess_predictions_handles_errors(self, mock_logger, mock_write_image):
        """Test postprocess_predictions handles and logs errors during resampling."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.save_folder = self.save_folder
        processor.data_dicts = [
            {FILE_ID: "scan1", IMAGE: "/path/to/image1.nii.gz"},
            {FILE_ID: "scan2", IMAGE: "/path/to/image2.nii.gz"},
        ]

        # Create prediction files
        pred_path1 = Path(self.save_folder) / "pred_scan1.nii.gz"
        pred_path2 = Path(self.save_folder) / "pred_scan2.nii.gz"
        pred_path1.touch()
        pred_path2.touch()

        # Mock resampling to fail for first file, succeed for second
        mock_resampled_image = Mock()
        processor._resample_pred_to_match_image = Mock(
            side_effect=[RuntimeError("Resampling failed"), mock_resampled_image]
        )

        processor.postprocess_predictions()

        # Verify error was logged for the failed file
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        self.assertIn("Failed to resample prediction", error_call)
        self.assertIn("pred_scan1.nii.gz", error_call)

        # Verify successful file was still processed
        mock_write_image.assert_called_once_with(mock_resampled_image, str(pred_path2))

    @patch("linguine.dataset_processor.sitk.ReadImage")
    @patch("linguine.dataset_processor.sitk.ResampleImageFilter")
    def test_resample_pred_to_match_image_nifti_file(self, mock_resample_filter, mock_read_image):
        """Test _resample_pred_to_match_image with NIfTI file."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        # Create mock images
        mock_reference_image = Mock()
        mock_prediction_image = Mock()
        mock_resampled_image = Mock()

        mock_read_image.side_effect = [mock_reference_image, mock_prediction_image]

        # Mock resampler
        mock_resampler = Mock()
        mock_resampler.Execute.return_value = mock_resampled_image
        mock_resample_filter.return_value = mock_resampler

        pred_path = Path("/path/to/pred.nii.gz")
        image_path = Path("/path/to/image.nii.gz")

        result = processor._resample_pred_to_match_image(pred_path, image_path)

        # Verify correct image loading
        expected_calls = [unittest.mock.call(str(image_path)), unittest.mock.call(str(pred_path))]
        mock_read_image.assert_has_calls(expected_calls)

        # Verify resampler configuration
        mock_resampler.SetReferenceImage.assert_called_once_with(mock_reference_image)
        mock_resampler.SetInterpolator.assert_called_once_with(sitk.sitkNearestNeighbor)
        mock_resampler.SetDefaultPixelValue.assert_called_once_with(0)

        # Verify execution
        mock_resampler.Execute.assert_called_once_with(mock_prediction_image)
        self.assertEqual(result, mock_resampled_image)

    @patch("linguine.dataset_processor.sitk.ReadImage")
    @patch("linguine.dataset_processor.sitk.ImageSeriesReader")
    @patch("linguine.dataset_processor.sitk.ResampleImageFilter")
    def test_resample_pred_to_match_image_dicom_directory(
        self, mock_resample_filter, mock_series_reader, mock_read_image
    ):
        """Test _resample_pred_to_match_image with DICOM directory."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        # Create mock images
        mock_reference_image = Mock()
        mock_prediction_image = Mock()
        mock_resampled_image = Mock()

        # Mock DICOM series reader
        mock_reader_instance = Mock()
        mock_reader_instance.Execute.return_value = mock_reference_image
        mock_reader_instance.GetGDCMSeriesFileNames.return_value = ["file1.dcm", "file2.dcm"]
        mock_series_reader.return_value = mock_reader_instance

        mock_read_image.return_value = mock_prediction_image

        # Mock resampler
        mock_resampler = Mock()
        mock_resampler.Execute.return_value = mock_resampled_image
        mock_resample_filter.return_value = mock_resampler

        pred_path = Path("/path/to/pred.nii.gz")
        image_path = Path("/path/to/dicom_dir")

        # Mock Path.is_dir() to return True
        with patch.object(Path, "is_dir", return_value=True):
            result = processor._resample_pred_to_match_image(pred_path, image_path)

        # Verify DICOM series reading
        mock_reader_instance.GetGDCMSeriesFileNames.assert_called_once_with(str(image_path))
        mock_reader_instance.SetFileNames.assert_called_once_with(["file1.dcm", "file2.dcm"])
        mock_reader_instance.Execute.assert_called_once()

        # Verify prediction loading
        mock_read_image.assert_called_once_with(str(pred_path))

        # Verify resampler was configured and executed
        mock_resampler.SetReferenceImage.assert_called_once_with(mock_reference_image)
        mock_resampler.Execute.assert_called_once_with(mock_prediction_image)
        self.assertEqual(result, mock_resampled_image)

    @patch("linguine.dataset_processor.sitk.ImageSeriesReader")
    def test_resample_pred_to_match_image_dicom_no_files(self, mock_series_reader):
        """Test _resample_pred_to_match_image raises error when DICOM directory has no files."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        # Mock DICOM series reader with empty file list
        mock_reader_instance = Mock()
        mock_reader_instance.GetGDCMSeriesFileNames.return_value = []
        mock_series_reader.return_value = mock_reader_instance

        pred_path = Path("/path/to/pred.nii.gz")
        image_path = Path("/path/to/empty_dicom_dir")

        with patch.object(Path, "is_dir", return_value=True):
            with self.assertRaises(RuntimeError) as context:
                processor._resample_pred_to_match_image(pred_path, image_path)

        self.assertIn("No DICOM files found", str(context.exception))

    @patch("linguine.dataset_processor.sitk.ReadImage")
    def test_resample_pred_to_match_image_image_load_failure(self, mock_read_image):
        """Test _resample_pred_to_match_image handles image loading failures."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        # Mock image loading failure
        mock_read_image.side_effect = Exception("Failed to load image")

        pred_path = Path("/path/to/pred.nii.gz")
        image_path = Path("/path/to/image.nii.gz")

        with patch.object(Path, "is_dir", return_value=False):
            with self.assertRaises(RuntimeError) as context:
                processor._resample_pred_to_match_image(pred_path, image_path)

        self.assertIn("Failed to load image", str(context.exception))

    @patch("linguine.dataset_processor.sitk.ReadImage")
    @patch("linguine.dataset_processor.sitk.ResampleImageFilter")
    def test_resample_pred_to_match_image_resampling_failure(self, mock_resample_filter, mock_read_image):
        """Test _resample_pred_to_match_image handles resampling failures."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        # Create mock images
        mock_reference_image = Mock()
        mock_prediction_image = Mock()

        mock_read_image.side_effect = [mock_reference_image, mock_prediction_image]

        # Mock resampler to fail during execution
        mock_resampler = Mock()
        mock_resampler.Execute.side_effect = Exception("Resampling failed")
        mock_resample_filter.return_value = mock_resampler

        pred_path = Path("/path/to/pred.nii.gz")
        image_path = Path("/path/to/image.nii.gz")

        with patch.object(Path, "is_dir", return_value=False):
            with self.assertRaises(RuntimeError) as context:
                processor._resample_pred_to_match_image(pred_path, image_path)

        self.assertIn("Failed to resample prediction", str(context.exception))

    @patch("linguine.dataset_processor.LOGGER")
    def test_postprocess_predictions_empty_data_dicts(self, mock_logger):
        """Test postprocess_predictions handles empty data_dicts list."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.save_folder = self.save_folder
        processor.data_dicts = []

        processor._resample_pred_to_match_image = Mock()

        processor.postprocess_predictions()

        # Should not call resampling method
        processor._resample_pred_to_match_image.assert_not_called()

    @patch("linguine.dataset_processor.sitk.ReadImage")
    @patch("linguine.dataset_processor.sitk.ResampleImageFilter")
    def test_resample_pred_to_match_image_with_different_spacings(self, mock_resample_filter, mock_read_image):
        """Test _resample_pred_to_match_image correctly handles images with different spacings."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)

        # Create mock images with different properties
        mock_reference_image = Mock()
        mock_reference_image.GetSpacing.return_value = (1.0, 1.0, 2.0)
        mock_reference_image.GetSize.return_value = (256, 256, 128)

        mock_prediction_image = Mock()
        mock_prediction_image.GetSpacing.return_value = (2.0, 2.0, 4.0)
        mock_prediction_image.GetSize.return_value = (128, 128, 64)

        mock_resampled_image = Mock()

        mock_read_image.side_effect = [mock_reference_image, mock_prediction_image]

        # Mock resampler
        mock_resampler = Mock()
        mock_resampler.Execute.return_value = mock_resampled_image
        mock_resample_filter.return_value = mock_resampler

        pred_path = Path("/path/to/pred.nii.gz")
        image_path = Path("/path/to/image.nii.gz")

        with patch.object(Path, "is_dir", return_value=False):
            result = processor._resample_pred_to_match_image(pred_path, image_path)

        # Verify that the reference image was used for resampling configuration
        mock_resampler.SetReferenceImage.assert_called_once_with(mock_reference_image)
        self.assertEqual(result, mock_resampled_image)

    def test_get_registrator_with_all_in_csv_config(self):
        """Test _get_registrator with 'all_in_csv' string configuration."""
        processor = LinguineDatasetProcessor.__new__(LinguineDatasetProcessor)
        processor.cfg = self.mock_cfg
        processor.cfg.registration.registrator = "tps"

        result = processor._get_registrator()

        self.assertIsInstance(result, ThinPlateSplineRegistrator)
        # Should use all landmarks from CSV, so valid_landmarks should be None
        self.assertIsNone(result.valid_landmarks)


if __name__ == "__main__":
    unittest.main()
