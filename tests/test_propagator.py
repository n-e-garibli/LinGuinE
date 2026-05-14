# Copyright AstraZeneca 2025
"""Comprehensive unit tests for the LinguineClickPropagator class."""

import unittest
from unittest.mock import Mock, call, patch

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor

from linguine.config import (
    AnalysisConfig,
    LinguineConfig,
    MaskSamplingConfig,
    PromptSelectionConfig,
    RegistrationConfig,
)
from linguine.constants import IMAGE, LABEL
from linguine.inferers.base_inferer import AbstractInferer
from linguine.metrics import MetricsBundle
from linguine.prompt_selectors.base_ps import PromptSelector
from linguine.propagator import LinguineBboxPropagator, LinguineClickPropagator
from linguine.registration.point_extractors import PointExtractor
from linguine.registration.registrators import PointSetRegistrator
from linguine.utils.bounding_boxes import BBox2D, BBox3D, Point3D


class TestLinguineClickPropagator(unittest.TestCase):
    """Test suite for LinguineClickPropagator class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create test configuration
        self.cfg = LinguineConfig()

        # Create mock dependencies
        self.mock_inferer = Mock(spec=AbstractInferer)
        self.mock_inferer.spacing = (1.0, 1.0, 1.0)
        self.mock_registrator = Mock(spec=PointSetRegistrator)
        self.mock_point_extractor = Mock(spec=PointExtractor)
        self.mock_ps = Mock(spec=PromptSelector)

        # Create propagator instance
        self.propagator = LinguineClickPropagator(
            cfg=self.cfg,
            inferer=self.mock_inferer,
            registrator=self.mock_registrator,
            point_extractor=self.mock_point_extractor,
            prompt_selector=self.mock_ps,
        )

        # Create test data
        self.test_img_shape = (1, 1, 64, 64, 64)
        self.test_spacing = (1.0, 1.0, 1.0)

        # Create mock MetaTensor objects
        self.source_img = self._create_mock_metatensor(self.test_img_shape)
        self.target_img = self._create_mock_metatensor(self.test_img_shape)
        self.source_label = self._create_mock_metatensor(self.test_img_shape)
        self.target_label = self._create_mock_metatensor(self.test_img_shape)

        # Create test data dictionaries
        self.source_dict = {IMAGE: "source.nii", LABEL: "source_label.nii"}
        self.target_dict = {IMAGE: "target.nii", LABEL: "target_label.nii"}

        # Test clicks
        self.source_clicks = [(32, 32, 32)]

    def _create_mock_metatensor(self, shape, has_lesion=True):
        """Create a mock MetaTensor with proper shape and metadata."""
        if has_lesion:
            data = torch.ones(shape, dtype=torch.float32)
        else:
            data = torch.zeros(shape, dtype=torch.float32)

        meta_tensor = MetaTensor(data)
        # Mock spacing metadata
        meta_tensor.meta = {"spacing": torch.tensor(self.test_spacing), "affine": torch.eye(4)}
        return meta_tensor

    def test_init(self):
        """Test proper initialization of LinguineClickPropagator."""
        self.assertEqual(self.propagator.cfg, self.cfg)
        self.assertEqual(self.propagator.inferer, self.mock_inferer)
        self.assertEqual(self.propagator.registrator, self.mock_registrator)
        self.assertEqual(self.propagator.point_extractor, self.mock_point_extractor)
        self.assertIsNotNone(self.propagator.cv_metrics)

    def test_call_successful_propagation(self):
        """Test successful click propagation and inference."""

        with (
            patch.object(self.propagator, "_prepare_target_clicks") as mock_prepare,
            patch.object(self.propagator, "get_metrics_and_prediction") as mock_metrics,
        ):
            mock_prepare.return_value = [(30, 30, 30)]
            mock_metrics.return_value = (MetricsBundle(dice=0.85), self.target_img)

            metrics, clicks, pred = self.propagator(
                source_dict=self.source_dict,
                target_dict=self.target_dict,
                target_img=self.target_img,
                target_label=self.target_label,
                source_prompts=self.source_clicks,
                source_label=self.source_label,
            )

            self.assertEqual(metrics.dice, 0.85)
            self.assertEqual(clicks, [(30, 30, 30)])
            # Check that pred is a MetaTensor with same shape
            self.assertIsInstance(pred, MetaTensor)
            self.assertEqual(pred.shape, self.target_img.shape)

    def test_call_no_target_lesion_predict_only(self):
        """Test behavior when target has no lesion in predict_only mode."""
        self.cfg.predict_only = True
        empty_label = self._create_mock_metatensor(self.test_img_shape, has_lesion=False)

        with patch.object(self.propagator, "_prepare_target_clicks") as mock_prepare:
            mock_prepare.return_value = [(30, 30, 30)]

            with patch.object(self.propagator, "get_metrics_and_prediction") as mock_metrics:
                mock_metrics.return_value = (None, self.target_img)

                metrics, clicks, _ = self.propagator(
                    source_dict=self.source_dict,
                    target_dict=self.target_dict,
                    target_img=self.target_img,
                    target_label=empty_label,
                    source_prompts=self.source_clicks,
                    source_label=self.source_label,
                )

                self.assertIsNone(metrics)
                self.assertEqual(clicks, [(30, 30, 30)])

    def test_call_no_target_clicks(self):
        """Test behavior when no target clicks are generated."""
        with patch.object(self.propagator, "_prepare_target_clicks") as mock_prepare:
            mock_prepare.return_value = []

            metrics, clicks, pred = self.propagator(
                source_dict=self.source_dict,
                target_dict=self.target_dict,
                target_img=self.target_img,
                target_label=self.target_label,
                source_prompts=self.source_clicks,
                source_label=self.source_label,
            )

            self.assertIsNone(metrics)
            self.assertEqual(clicks, [])
            self.assertIsNone(pred)

    @patch("linguine.propagator.find_mask_center")
    def test_prepare_target_clicks_perfect_registration(self, mock_find_center):
        """Test click preparation with perfect registration."""
        self.cfg.registration.registrator = "perfect"
        mock_find_center.return_value = (32, 32, 32)

        result = self.propagator._prepare_target_clicks(
            source_dict=self.source_dict,
            source_clicks=self.source_clicks,
            source_label=self.source_label,
            target_dict=self.target_dict,
            target_img=self.target_img,
            target_label=self.target_label,
            source_img=self.source_img,
        )

        self.assertEqual(result, [(32, 32, 32)])
        mock_find_center.assert_called_once_with(self.target_label)

    @patch("linguine.propagator.sample_clicks")
    def test_find_best_clicks(self, mock_sample_clicks):
        """Test finding best clicks through sampling."""
        mock_sample_clicks.return_value = [(20, 20, 20), (25, 25, 25)]

        with patch.object(self.propagator, "_propagate_clicks") as mock_propagate:
            mock_propagate.return_value = [(22, 22, 22), (27, 27, 27)]
            self.mock_ps.get_best_clicks.return_value = [(22, 22, 22)]

            result = self.propagator._find_best_clicks(
                source_dict=self.source_dict,
                target_dict=self.target_dict,
                source_label=self.source_label,
                target_img=torch.zeros(size=(1, 1, 30, 30, 30)),
                target_spacing=self.test_spacing,
                n_clicks=1,
                source_img=None,
            )

            self.assertEqual(result, [(22, 22, 22)])
            self.mock_ps.get_best_clicks.assert_called_once()

    @patch("linguine.propagator.crop_to_same_size")
    @patch("linguine.propagator.prepare_inputs_for_inferer")
    def test_get_metrics_and_prediction_with_label(self, mock_prepare_inputs, mock_crop):
        """Test metrics calculation and prediction when label is provided."""
        # Mock prepare_inputs_for_inferer to return transformed inputs
        # Note: This function is called twice - once for image, once for label
        mock_prepare_inputs.side_effect = [
            (self.target_img, [(30, 30, 30)], []),  # For image transformation
            (self.target_label, [], []),  # For label transformation
        ]

        # Setup inference result
        pred_array = np.ones((64, 64, 64), dtype=np.float32)
        self.mock_inferer.infer.return_value = pred_array

        # Setup dice calculation
        mock_crop.return_value = [pred_array, np.ones((64, 64, 64))]

        metrics, pred = self.propagator.get_metrics_and_prediction(
            img=self.target_img, label=self.target_label, fg_clicks=[(32, 32, 32)]
        )

        self.assertEqual(metrics.dice, 1.0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.hd95, 0.0)
        self.assertEqual(metrics.assd, 0.0)
        self.assertIsInstance(pred, MetaTensor)
        self.mock_inferer.infer.assert_called_once()
        # Verify the function was called once (for the image)
        self.assertEqual(mock_prepare_inputs.call_count, 1)

    @patch("linguine.propagator.prepare_inputs_for_inferer")
    def test_get_metrics_and_prediction_without_label(self, mock_prepare_inputs):
        """Test prediction when no label is provided."""
        # Mock prepare_inputs_for_inferer to return transformed inputs
        mock_prepare_inputs.return_value = (self.target_img, [(30, 30, 30)], [])

        pred_array = np.ones((64, 64, 64), dtype=np.float32)
        self.mock_inferer.infer.return_value = pred_array

        metrics, pred = self.propagator.get_metrics_and_prediction(
            img=self.target_img, label=None, fg_clicks=[(32, 32, 32)]
        )

        self.assertTrue(np.isnan(metrics.dice))
        self.assertIsInstance(pred, MetaTensor)
        # Verify the function was called once - image only
        self.assertEqual(mock_prepare_inputs.call_count, 1)

    def test_propagate_clicks_empty_list(self):
        """Test propagation with empty click list."""
        result = self.propagator._propagate_clicks(
            source_dict=self.source_dict,
            target_dict=self.target_dict,
            source_spacing=self.test_spacing,
            target_spacing=self.test_spacing,
            source_clicks=[],
            source_image=None,
            target_image=None,
        )

        self.assertEqual(result, [])

    def test_propagate_clicks_successful(self):
        """Test successful click propagation."""
        source_landmarks = [(10, 10, 10), (50, 50, 50)]
        target_landmarks = [(12, 12, 12), (48, 48, 48)]

        self.mock_point_extractor.extract_points.side_effect = [source_landmarks, target_landmarks]
        self.mock_registrator.map_coordinates.return_value = [(30, 30, 30)]

        result = self.propagator._propagate_clicks(
            source_dict=self.source_dict,
            target_dict=self.target_dict,
            source_spacing=self.test_spacing,
            target_spacing=self.test_spacing,
            source_clicks=[(32, 32, 32)],
            source_image=None,
            target_image=None,
        )

        self.assertEqual(result, [(30, 30, 30)])
        self.mock_registrator.map_coordinates.assert_called_once()

    @patch("linguine.propagator.prepare_inputs_for_inferer")
    def test_inference_with_bboxes(self, mock_prepare_inputs):
        """Test inference when bounding boxes are provided."""
        bbox = BBox3D(Point3D(10, 10, 10), Point3D(50, 50, 50))

        mock_prepare_inputs.return_value = (self.target_img, [], [bbox])

        pred_array = np.ones((64, 64, 64), dtype=np.float32)
        self.mock_inferer.infer.return_value = pred_array

        metrics, _ = self.propagator.get_metrics_and_prediction(
            img=self.target_img, label=None, fg_clicks=[], bboxes=[bbox]
        )

        self.assertTrue(np.isnan(metrics.dice))
        self.mock_inferer.infer.assert_called_once()
        # Verify that bboxes were passed to inferer
        call_args = self.mock_inferer.infer.call_args
        self.assertEqual(call_args[1]["bboxes"], [bbox])


class TestLinguineClickPropagatorIntegration(unittest.TestCase):
    """Integration tests for LinguineClickPropagator with more realistic scenarios."""

    def setUp(self):
        """Set up integration test fixtures."""

        # Realistic configuration
        self.cfg = LinguineConfig(
            save_predictions=True,
            predict_only=False,
            device="cpu",
            registration=RegistrationConfig(registrator="aruns", tps_lambda=0.1),
            analysis=AnalysisConfig(
                iteration_mode="FROM_ONE_TIMEPOINT",
            ),
            prompt_selection=PromptSelectionConfig(
                type="threshold",
                n_clicks=2,
                l_threshold=-500,
                u_threshold=500,
                mask_sampling=MaskSamplingConfig(method="normal", num_samples=10),
            ),
        )

        # Create realistic mock objects
        self.mock_inferer = Mock(spec=AbstractInferer)
        self.mock_inferer.spacing = (1.5, 1.5, 2.0)

        self.mock_registrator = Mock(spec=PointSetRegistrator)
        self.mock_point_extractor = Mock(spec=PointExtractor)
        self.mock_ps = Mock(spec=PromptSelector)

        self.propagator = LinguineClickPropagator(
            cfg=self.cfg,
            inferer=self.mock_inferer,
            registrator=self.mock_registrator,
            point_extractor=self.mock_point_extractor,
            prompt_selector=self.mock_ps,
        )

    @patch("linguine.propagator.get_spacing_from_metatensor")
    def test_end_to_end_propagation_workflow(self, mock_get_spacing):
        """Test complete end-to-end propagation workflow."""
        # Setup realistic data
        mock_get_spacing.return_value = (1.0, 1.0, 1.0)
        target_img = MetaTensor(torch.ones(1, 1, 32, 32, 32))
        source_label = MetaTensor(torch.zeros(1, 1, 32, 32, 32))
        target_label = MetaTensor(torch.zeros(1, 1, 32, 32, 32))

        # Add some lesion data
        source_label[0, 0, 10:20, 10:20, 10:20] = 1
        target_label[0, 0, 12:22, 12:22, 12:22] = 1
        source_clicks = [(15, 15, 15), (16, 16, 16)]

        # Setup mock responses
        self.mock_point_extractor.extract_points.side_effect = [
            [(5, 5, 5), (25, 25, 25)],  # source landmarks
            [(7, 7, 7), (23, 23, 23)],  # target landmarks
        ]

        self.mock_registrator.map_coordinates.return_value = [(17, 17, 17), (18, 18, 18)]
        self.mock_ps.get_best_clicks.return_value = [(17, 17, 17)]

        # Setup inferer
        pred_result = np.zeros((32, 32, 32))
        pred_result[12:22, 12:22, 12:22] = 1
        self.mock_inferer.infer.return_value = pred_result

        with (
            patch("linguine.propagator.prepare_inputs_for_inferer") as mock_prepare_inputs,
            patch("linguine.propagator.crop_to_same_size") as mock_crop,
        ):
            # Mock prepare_inputs_for_inferer to return the inputs unchanged for simplicity
            mock_prepare_inputs.side_effect = lambda **kwargs: (
                kwargs.get("image", target_img),
                kwargs.get("clicks", []),
                kwargs.get("bboxes", []),
            )
            mock_crop.return_value = [pred_result, target_label[0, 0].numpy()]

            metrics, clicks, pred = self.propagator(
                source_dict={IMAGE: "source.nii", LABEL: "source_label.nii"},
                target_dict={IMAGE: "target.nii", LABEL: "target_label.nii"},
                target_img=target_img,
                target_label=target_label,
                source_prompts=source_clicks,
                source_label=source_label,
            )

            # Verify results
            self.assertEqual(metrics.dice, 1.0)
            self.assertEqual(metrics.hd95, 0.0)
            self.assertEqual(metrics.assd, 0.0)
            self.assertEqual(clicks, [(17, 17, 17)])
            self.assertIsInstance(pred, MetaTensor)

            # Verify method calls
            self.mock_point_extractor.extract_points.assert_has_calls(
                [
                    call({IMAGE: "source.nii", LABEL: "source_label.nii"}),
                    call({IMAGE: "target.nii", LABEL: "target_label.nii"}),
                ]
            )
            self.mock_registrator.map_coordinates.assert_called_once()
            self.mock_inferer.infer.assert_called_once()

    def test_multiple_failure_recovery_scenarios(self):
        """Test various failure scenarios and recovery mechanisms."""
        # Reset the mock PS for each scenario to avoid state leakage
        self.mock_ps.reset_mock()

        # Test registration failure scenario
        with (
            patch.object(self.propagator, "_propagate_clicks") as mock_propagate,
            patch.object(self.propagator, "_find_best_clicks") as mock_find_valid,
        ):
            mock_propagate.return_value = []  # Simulate registration failure
            mock_find_valid.return_value = []

            result = self.propagator._prepare_target_clicks(
                source_dict={IMAGE: "source.nii"},
                source_clicks=[(32, 32, 32)],
                source_label=MetaTensor(torch.ones(1, 1, 32, 32, 32)),
                target_dict={IMAGE: "target.nii"},
                target_img=MetaTensor(torch.ones(1, 1, 32, 32, 32)),
                target_label=MetaTensor(torch.ones(1, 1, 32, 32, 32)),
                source_img=None,
            )

            self.assertEqual(result, [])

        # Reset mock again
        self.mock_ps.reset_mock()

        # Test successful resampling scenario
        with (
            patch.object(self.propagator, "_propagate_clicks") as mock_propagate,
            patch.object(self.propagator, "_find_best_clicks") as mock_find_valid,
        ):
            mock_propagate.return_value = [(30, 30, 30)]
            mock_find_valid.return_value = [(20, 20, 20)]  # Successful resampling

            result = self.propagator._prepare_target_clicks(
                source_dict={IMAGE: "source.nii"},
                source_clicks=[(32, 32, 32)],
                source_label=MetaTensor(torch.ones(1, 1, 32, 32, 32)),
                target_dict={IMAGE: "target.nii"},
                target_img=MetaTensor(torch.ones(1, 1, 32, 32, 32)),
                target_label=MetaTensor(torch.ones(1, 1, 32, 32, 32)),
                source_img=None,
            )

            self.assertEqual(result, [(20, 20, 20)])


class TestLinguineBboxPropagator(unittest.TestCase):
    """Test suite for LinguineBboxPropagator class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create test configuration with bbox propagation mode
        self.cfg = LinguineConfig(
            analysis=AnalysisConfig(prompt_to_propagate="bbox"),
            prompt_selection=PromptSelectionConfig(type=None),
        )

        # Create mock dependencies
        self.mock_inferer = Mock(spec=AbstractInferer)
        self.mock_inferer.spacing = (1.0, 1.0, 1.0)
        self.mock_registrator = Mock(spec=PointSetRegistrator)
        self.mock_point_extractor = Mock(spec=PointExtractor)
        self.mock_ps = Mock(spec=PromptSelector)

        # Import the new class
        from linguine.propagator import LinguineBboxPropagator

        # Create propagator instance
        self.propagator = LinguineBboxPropagator(
            cfg=self.cfg,
            inferer=self.mock_inferer,
            registrator=self.mock_registrator,
            point_extractor=self.mock_point_extractor,
            prompt_selector=self.mock_ps,
        )

        # Create test data
        self.test_img_shape = (1, 1, 64, 64, 64)
        self.test_spacing = (1.0, 1.0, 1.0)

        # Create mock MetaTensor objects with a lesion
        self.source_label = self._create_label_with_lesion()
        self.target_img = self._create_mock_metatensor(self.test_img_shape)
        self.target_label = self._create_label_with_lesion()

        # Create test data dictionaries
        self.source_dict = {IMAGE: "source.nii", LABEL: "source_label.nii"}
        self.target_dict = {IMAGE: "target.nii", LABEL: "target_label.nii"}

    def _create_mock_metatensor(self, shape):
        """Create a mock MetaTensor with proper shape and metadata."""
        data = torch.ones(shape, dtype=torch.float32)
        meta_tensor = MetaTensor(data)
        meta_tensor.meta = {"spacing": torch.tensor(self.test_spacing), "affine": torch.eye(4)}
        return meta_tensor

    def _create_label_with_lesion(self):
        """Create a label with a small lesion in the center."""
        data = torch.zeros(self.test_img_shape, dtype=torch.float32)
        # Create a small 10x10x10 lesion in the center
        data[0, 0, 27:37, 27:37, 27:37] = 1.0
        meta_tensor = MetaTensor(data)
        meta_tensor.meta = {"spacing": torch.tensor(self.test_spacing), "affine": torch.eye(4)}
        return meta_tensor

    def test_init(self):
        """Test proper initialization of LinguineBboxPropagator."""
        self.assertEqual(self.propagator.cfg, self.cfg)
        self.assertEqual(self.propagator.inferer, self.mock_inferer)
        self.assertEqual(self.propagator.registrator, self.mock_registrator)

    @patch.object(LinguineBboxPropagator, "_propagate_clicks")
    def test_propagate_bbox(self, mock_propagate_clicks):
        """Test propagating a bounding box."""
        source_bbox = BBox3D(Point3D(10, 10, 10), Point3D(20, 20, 20))

        # Mock the propagate_clicks to return transformed corners
        mock_propagate_clicks.return_value = [
            (15, 15, 15),
            (15, 15, 25),
            (15, 25, 15),
            (15, 25, 25),
            (25, 15, 15),
            (25, 15, 25),
            (25, 25, 15),
            (25, 25, 25),
        ]

        target_bbox = self.propagator._propagate_bbox(
            source_dict=self.source_dict,
            target_dict=self.target_dict,
            source_bbox=source_bbox,
            source_spacing=self.test_spacing,
            target_spacing=self.test_spacing,
            source_image=None,
            target_image=None,
        )

        self.assertIsInstance(target_bbox, BBox3D)
        self.assertEqual(target_bbox.x_min, 15)
        self.assertEqual(target_bbox.x_max, 25)
        self.assertEqual(target_bbox.y_min, 15)
        self.assertEqual(target_bbox.y_max, 25)
        self.assertEqual(target_bbox.z_min, 15)
        self.assertEqual(target_bbox.z_max, 25)

    @patch.object(LinguineBboxPropagator, "_propagate_clicks")
    def test_propagate_bbox_returns_none_on_failure(self, mock_propagate_clicks):
        """Test that bbox propagation returns None when clicks fail to propagate."""
        source_bbox = BBox3D(Point3D(10, 10, 10), Point3D(20, 20, 20))
        mock_propagate_clicks.return_value = []

        target_bbox = self.propagator._propagate_bbox(
            source_dict=self.source_dict,
            target_dict=self.target_dict,
            source_bbox=source_bbox,
            source_spacing=self.test_spacing,
            target_spacing=self.test_spacing,
            source_image=None,
            target_image=None,
        )

        self.assertIsNone(target_bbox)

    @patch("linguine.propagator.prepare_inputs_for_inferer")
    def test_call_successful_propagation(self, mock_prepare_inputs):
        """Test successful bbox propagation and inference."""
        # Setup mock
        target_bbox = BBox3D(Point3D(15, 15, 15), Point3D(25, 25, 25))
        mock_prepare_inputs.return_value = (self.target_img, [], [target_bbox])

        pred_array = np.ones((64, 64, 64), dtype=np.float32)
        self.mock_inferer.infer.return_value = pred_array

        with patch.object(self.propagator, "_propagate_bbox") as mock_propagate_bbox:
            mock_propagate_bbox.return_value = target_bbox

            metrics, bboxes, pred = self.propagator(
                source_dict=self.source_dict,
                target_dict=self.target_dict,
                target_img=self.target_img,
                target_label=self.target_label,
                source_prompts=[BBox3D(Point3D(10, 10, 10), Point3D(15, 15, 15))],
                source_label=self.source_label,
            )

        # Verify bbox was propagated and inference was performed
        self.assertIsNotNone(metrics)
        self.assertEqual(bboxes, [target_bbox])  # No clicks for bbox propagation
        self.assertIsNotNone(pred)
        self.mock_inferer.infer.assert_called_once()

        # Verify bbox was passed to inferer
        call_args = self.mock_inferer.infer.call_args
        self.assertEqual(len(call_args[1]["bboxes"]), 1)

    def test_bbox_2d_initialization(self):
        """Test that LinguineBboxPropagator can be initialized with bbox_2d flag."""
        propagator_2d = LinguineBboxPropagator(
            cfg=self.cfg,
            inferer=self.mock_inferer,
            registrator=self.mock_registrator,
            point_extractor=self.mock_point_extractor,
            prompt_selector=self.mock_ps,
            bbox_2d=True,
        )
        self.assertTrue(propagator_2d.bbox_2d)

        propagator_3d = LinguineBboxPropagator(
            cfg=self.cfg,
            inferer=self.mock_inferer,
            registrator=self.mock_registrator,
            point_extractor=self.mock_point_extractor,
            prompt_selector=self.mock_ps,
            bbox_2d=False,
        )
        self.assertFalse(propagator_3d.bbox_2d)

    @patch.object(LinguineBboxPropagator, "_propagate_clicks")
    def test_propagate_bbox_2d(self, mock_propagate_clicks):
        """Test that _propagate_bbox returns BBox2D when bbox_2d is True."""
        # Create a 2D propagator
        propagator_2d = LinguineBboxPropagator(
            cfg=self.cfg,
            inferer=self.mock_inferer,
            registrator=self.mock_registrator,
            point_extractor=self.mock_point_extractor,
            prompt_selector=self.mock_ps,
            bbox_2d=True,
        )

        # Create a source 2D bbox (z dimension is fixed)
        source_bbox = BBox2D(Point3D(10, 10, 15), Point3D(20, 20, 15))

        # Mock the propagate_clicks to return transformed corners with fixed z
        mock_propagate_clicks.return_value = [
            (15, 15, 20),
            (15, 15, 20),
            (15, 25, 20),
            (15, 25, 20),
            (25, 15, 20),
            (25, 15, 20),
            (25, 25, 20),
            (25, 25, 20),
        ]

        target_bbox = propagator_2d._propagate_bbox(
            source_dict=self.source_dict,
            target_dict=self.target_dict,
            source_bbox=source_bbox,
            source_spacing=self.test_spacing,
            target_spacing=self.test_spacing,
            source_image=None,
            target_image=None,
        )

        self.assertIsInstance(target_bbox, BBox2D)
        self.assertEqual(target_bbox.x_min, 15)
        self.assertEqual(target_bbox.x_max, 25)
        self.assertEqual(target_bbox.y_min, 15)
        self.assertEqual(target_bbox.y_max, 25)
        self.assertEqual(target_bbox.z_min, 20)
        self.assertEqual(target_bbox.z_max, 20)

    @patch.object(LinguineBboxPropagator, "_propagate_clicks")
    def test_propagate_bbox_2d_with_varying_z_coords(self, mock_propagate_clicks):
        """Test that _propagate_bbox handles varying z coordinates by fixing them to the mean."""
        # Create a 2D propagator
        propagator_2d = LinguineBboxPropagator(
            cfg=self.cfg,
            inferer=self.mock_inferer,
            registrator=self.mock_registrator,
            point_extractor=self.mock_point_extractor,
            prompt_selector=self.mock_ps,
            bbox_2d=True,
        )

        # Create a source 2D bbox (z dimension is fixed at 15)
        source_bbox = BBox2D(Point3D(10, 10, 15), Point3D(20, 20, 15))

        # Mock the propagate_clicks to return transformed corners with VARYING z values
        # This simulates the scenario where registration doesn't preserve the fixed dimension
        mock_propagate_clicks.return_value = [
            (15, 15, 18),  # z varies from 18 to 22
            (15, 15, 19),
            (15, 25, 20),
            (15, 25, 21),
            (25, 15, 20),
            (25, 15, 21),
            (25, 25, 22),
            (25, 25, 22),
        ]

        target_bbox = propagator_2d._propagate_bbox(
            source_dict=self.source_dict,
            target_dict=self.target_dict,
            source_bbox=source_bbox,
            source_spacing=self.test_spacing,
            target_spacing=self.test_spacing,
            source_image=None,
            target_image=None,
        )

        # The bbox should still be a valid BBox2D with fixed z dimension
        self.assertIsInstance(target_bbox, BBox2D)
        self.assertEqual(target_bbox.fixed_dimension, "z")

        # x and y should span the full range
        self.assertEqual(target_bbox.x_min, 15)
        self.assertEqual(target_bbox.x_max, 25)
        self.assertEqual(target_bbox.y_min, 15)
        self.assertEqual(target_bbox.y_max, 25)

        # z should be fixed to the mean: (18+19+20+21+20+21+22+22)/8 = 20.375 -> 20
        expected_z = int((18 + 19 + 20 + 21 + 20 + 21 + 22 + 22) / 8)
        self.assertEqual(target_bbox.z_min, expected_z)
        self.assertEqual(target_bbox.z_max, expected_z)


if __name__ == "__main__":
    unittest.main()
