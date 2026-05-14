"""Unit tests for configuration classes."""

import unittest

from linguine.config import (
    AnalysisConfig,
    LinguineConfig,
    MaskSamplingConfig,
    PromptSelectionConfig,
    RegistrationConfig,
)


class TestConfigClasses(unittest.TestCase):
    """Test cases for configuration classes."""

    def test_registration_config_default_initialization(self):
        """Test RegistrationConfig default initialization."""
        config = RegistrationConfig()

        self.assertIsNone(config.point_extractor_csv)
        self.assertEqual(config.registrator, "aruns")
        self.assertEqual(config.tps_lambda, 0.0)

    def test_analysis_config_defaults(self):
        """Test AnalysisConfig default initialization."""
        config = AnalysisConfig()

        self.assertEqual(config.iteration_mode, "FROM_ONE_TIMEPOINT")
        self.assertIsNone(config.boosting)
        self.assertEqual(config.min_pred_size_mm, 0.0)
        self.assertEqual(config.seed, 42)
        self.assertFalse(config.forward_in_time_only)

    def test_analysis_config_validation(self):
        """Test AnalysisConfig validation."""
        # Valid configurations should not raise errors
        AnalysisConfig(iteration_mode="CHAIN")
        AnalysisConfig(iteration_mode="CHAIN_WITH_RESAMPLING")
        AnalysisConfig(boosting="basic")
        AnalysisConfig(boosting="resample_additive")
        AnalysisConfig(boosting="resample_merge_probs")
        AnalysisConfig(boosting="orientation_ensemble")

        # Invalid iteration_mode should raise error
        with self.assertRaises(AssertionError):
            AnalysisConfig(iteration_mode="INVALID_MODE")

        # Invalid boosting should raise error
        with self.assertRaises(AssertionError):
            AnalysisConfig(boosting="invalid_boost")

    def test_mask_sampling_config_defaults(self):
        """Test MaskSamplingConfig default initialization."""
        config = MaskSamplingConfig()

        self.assertEqual(config.method, "uniform")
        self.assertEqual(config.num_samples, 27)
        self.assertEqual(config.num_voxels_per_click_per_dimension, (8, 8, 8))
        self.assertEqual(config.num_clicks_per_dimension, (3, 3, 3))
        self.assertTrue(config.replacement)

    def test_mask_sampling_config_validation(self):
        """Test MaskSamplingConfig validation."""
        # Valid methods should not raise errors
        for method in ["quadratic", "normal", "fixed_number_clicks", "fixed_click_distance", "uniform"]:
            MaskSamplingConfig(method=method)

        # Invalid method should raise error
        with self.assertRaises(AssertionError):
            MaskSamplingConfig(method="invalid_method")

        # Negative num_samples should raise error
        with self.assertRaises(AssertionError):
            MaskSamplingConfig(num_samples=-1)

        # Non-integer num_samples should raise error
        with self.assertRaises(AssertionError):
            MaskSamplingConfig(num_samples=27.5)

    def test_ps_config_validation(self):
        """Test PromptSelectionConfig validation."""
        # Valid types should not raise errors
        PromptSelectionConfig(type=None)
        PromptSelectionConfig(type="unguided", mask_sampling=MaskSamplingConfig(num_samples=20))
        PromptSelectionConfig(type="threshold", mask_sampling=MaskSamplingConfig(num_samples=20))

        # Invalid type should raise error
        with self.assertRaises(AssertionError):
            PromptSelectionConfig(type="invalid_type")

        # A not null prompt selector with num_samples=0 should raise error
        with self.assertRaises(ValueError):
            PromptSelectionConfig(type="threshold", mask_sampling=MaskSamplingConfig(num_samples=0))

        # Negative n_clicks should raise error
        with self.assertRaises(AssertionError):
            PromptSelectionConfig(n_clicks=-1)

        # Non-integer n_clicks should raise error
        with self.assertRaises(AssertionError):
            PromptSelectionConfig(n_clicks=1.5)

    def test_linguine_config_defaults(self):
        """Test LinguineConfig default initialization."""
        config = LinguineConfig()

        self.assertFalse(config.save_predictions)
        self.assertFalse(config.save_with_original_affine)
        self.assertFalse(config.predict_only)
        self.assertIsInstance(config.registration, RegistrationConfig)
        self.assertIsInstance(config.analysis, AnalysisConfig)
        self.assertIsInstance(config.prompt_selection, PromptSelectionConfig)
        self.assertEqual(config.device, "cpu")
        self.assertIsNone(config.patient_ids)

    def test_linguine_config_validation(self):
        """Test LinguineConfig validation."""
        # predict_only with perfect registration should raise error
        with self.assertRaises(AssertionError):
            LinguineConfig(predict_only=True, registration=RegistrationConfig(registrator="perfect"))

        # save_with_original_affine without save_predictions should raise error
        with self.assertRaises(AssertionError):
            LinguineConfig(save_with_original_affine=True, save_predictions=False)

    def test_analysis_config_prompt_to_propagate_default(self):
        """Test that AnalysisConfig defaults to click prompt."""
        config = AnalysisConfig()
        self.assertEqual(config.prompt_to_propagate, "click")

    def test_analysis_config_prompt_to_propagate_bbox(self):
        """Test that AnalysisConfig accepts bbox prompt."""
        config = AnalysisConfig(prompt_to_propagate="bbox")
        self.assertEqual(config.prompt_to_propagate, "bbox")

    def test_analysis_config_prompt_to_propagate_invalid(self):
        """Test that AnalysisConfig rejects invalid prompt_to_propagate values."""
        with self.assertRaises(AssertionError) as context:
            AnalysisConfig(prompt_to_propagate="invalid")
        self.assertIn("invalid prompt_to_propagate", str(context.exception).lower())

    def test_analysis_config_prompt_to_propagate_click(self):
        """Test that AnalysisConfig accepts click prompt explicitly."""
        config = AnalysisConfig(prompt_to_propagate="click")
        self.assertEqual(config.prompt_to_propagate, "click")

    def test_analysis_config_prompt_to_propagate_bbox_2d(self):
        """Test that AnalysisConfig accepts bbox_2d prompt."""
        config = AnalysisConfig(prompt_to_propagate="bbox_2d")
        self.assertEqual(config.prompt_to_propagate, "bbox_2d")


if __name__ == "__main__":
    unittest.main()
