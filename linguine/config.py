# Copyright AstraZeneca 2026
"""This file contains a config for the click propagation experiment."""

import importlib.util
from dataclasses import dataclass, field

from omegaconf import OmegaConf


@dataclass
class RegistrationConfig:
    """Config for the registration used for click propagation."""

    # The path to a csv containing landmarks to use in the experiment.
    point_extractor_csv: str | None = None
    # The registration algorithm to use in the analysis
    registrator: str = "aruns"
    # If using tps, the value of the regularisation lambda parameter.
    tps_lambda: float = 0.0
    # Whether to crop to foreground region for ICON-based registrators
    # For registrators where a masked preprocessing is applied, this will crop to the
    # bounding box around the mask.
    crop_foreground: bool = False

    def __post_init__(self):
        if self.registrator in ["lung_grad_icon", "uni_grad_icon", "uni_grad_icon_roi_refined"]:
            spec = importlib.util.find_spec("icon_registration")
            if spec is None:
                raise ValueError(
                    f"Cannot use registrator of type {self.registrator}. Please install icon_registration first."
                )


@dataclass
class AnalysisConfig:
    """Config for the analysis performed in the click propagation experiment."""

    # Specify the analysis method to use to evaluate click propagation.
    # "FROM_ONE_TIMEPOINT" will map all clicks from the first timepoint scan to all subsequent scans
    # "CHAIN" will map the screening scan to week 6, then week 6 to week 12, etc etc like a Markovian single step chain.
    # "CHAIN_WITH_RESAMPLING" does the same thing as "CHAIN" however at every step it will resample the guidance click
    # from the obtained model prediction.
    iteration_mode: str = "FROM_ONE_TIMEPOINT"
    # Whether/how to boost the guided model. Must be one of [None, "basic", "resample_additive", "resample_merge_probs" ]
    # More descriptions on how each method works can be found in boosted_inferers.py
    boosting: str | None = None
    # Minimum size of the predicted lesion to keep. Anything smaller will be filtered out.
    min_pred_size_mm: float = 0.0
    # Random seed to use
    seed: int = 42
    # Whether to only propagate to timepoints that come after the source scan in time
    forward_in_time_only: bool = False
    # Prompt to propagate: "click" for click-based propagation, "bbox" for 3D bounding box propagation, "bbox_2d" for 2D bounding box propagation
    prompt_to_propagate: str = "click"
    # Whether to perturb the click a set number of times and return the average dice score
    # relative to the 'actual' prediction, to be used for confidence estimation
    num_perturbations: int = 0
    # List of timepoint identifier patterns to use when sorting scans by timepoint. Each pattern
    # should be a string like "week_", "cycle_", etc. used to extract numeric timepoint values
    # from scan identifiers (e.g., "week_4" -> 4). If not provided, defaults to common patterns.
    timepoint_flags: list[str] | None = None

    def __post_init__(self):
        assert self.iteration_mode in [
            "CHAIN",
            "FROM_ONE_TIMEPOINT",
            "CHAIN_WITH_RESAMPLING",
        ], f"Got invalid analysis method {self.iteration_mode}"
        assert self.boosting in [
            None,
            "basic",
            "resample_additive",
            "resample_merge_probs",
            "perturb_ensemble",
            "click_ensemble",
            "orientation_ensemble",
        ], f"Got invalid bootstrap mode: {self.boosting}"
        assert self.prompt_to_propagate in [
            "click",
            "bbox",
            "bbox_2d",
        ], f"Got invalid prompt_to_propagate: {self.prompt_to_propagate}. Must be 'click', 'bbox', or 'bbox_2d'."
        assert isinstance(self.num_perturbations, int) and (self.num_perturbations >= 0), (
            "num_perturbations must be an integer"
        )
        if self.num_perturbations == 0:
            print("No perturbations will be used in order to estimate confidence.")


@dataclass
class MaskSamplingConfig:
    """Config for sampling from the source mask"""

    # Sampling method to use: must be one of 'quadratic', 'normal', 'fixed_number_clicks', 'fixed_click_distance', 'uniform'.
    method: str = "uniform"
    # If method is quadratic or normal, number of clicks to sample
    num_samples: int = 27
    # If method is fixed_number_clicks, number of voxels between each click per dimension
    num_voxels_per_click_per_dimension: tuple[int, int, int] = (8, 8, 8)
    # If method is fixed_number_clicks, number of clicks to sample per dimension
    num_clicks_per_dimension: tuple[int, int, int] = (3, 3, 3)
    # If sampling method returns a background click, replace with nearest foreground click in a search space consisting of every voxel that is closer to the given click than any other.
    replacement: bool = True

    def __post_init__(self):
        assert self.method in [
            "quadratic",
            "normal",
            "fixed_number_clicks",
            "fixed_click_distance",
            "uniform",
        ], f"Invalid method of{self.method} provided."
        assert self.num_samples >= 0, f"Number of resamples cannot be negative, got {self.num_samples}"
        assert isinstance(self.num_samples, int), (
            f"num_samples argument must be an integer, got: {type(self.num_samples)}"
        )


@dataclass
class PromptSelectionConfig:
    """Config for propagated prompt selection."""

    # The type of prompt selector to use.
    type: str | None = None
    # Maximum number of propagated clicks to allow when resampling.
    n_clicks: int = 1
    # Lower attenutation threshold if using threshold selector.
    l_threshold: float | int = -1000
    # Upper attenutation threshold if using threshold selector.
    u_threshold: float | int = 1000
    mask_sampling: MaskSamplingConfig = field(default_factory=MaskSamplingConfig)

    def __post_init__(self):
        assert self.type in [
            None,
            "unguided",
            "threshold",
        ], f"Prompt selector of type {self.type} is not supported."
        if self.type is not None:
            if self.mask_sampling.num_samples <= 0:
                raise ValueError(
                    f"If using a prompt selector number of samples must be greater than 0, got {self.mask_sampling.num_samples}"
                )
        assert self.n_clicks > 0, f"Number of clicks cannot be negative, got: {self.n_clicks}"
        assert isinstance(self.n_clicks, int), f"n_clicks must be an integer, got: {type(self.n_clicks)}"


@dataclass
class LinguineConfig:
    """Config for the click propagation experiment."""

    # If set to true, will save all predictions on a per lesion basis as nifti images.
    save_predictions: bool = False
    # Whether to postprocess the predictions to match the original image affine.
    save_with_original_affine: bool = False
    # If set to true, will not compute any dice scores or attempt to load labels for scans
    # that guidance is being propagated to. Useful for segmenting scans without ground truth.
    predict_only: bool = False
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    prompt_selection: PromptSelectionConfig = field(default_factory=PromptSelectionConfig)
    # Name of the device to do the computations on.
    device: str = "cpu"
    patient_ids: list[str] | None = None

    def __post_init__(self):
        if self.predict_only:
            assert self.registration.registrator != "perfect", (
                "Perfect registration not supported with predict only mode."
            )
        if self.save_with_original_affine:
            assert self.save_predictions, "Cannot save with original affine if save_predictions is False."

    def to_yaml(self, path: str) -> None:
        """Saves the initialised config to a yaml format at the specified path.

        Args:
            path: a string representing the absolute path to save the yaml to.
        """
        # Using OmegaConf to create YAML string
        yaml_string: str = OmegaConf.to_yaml(OmegaConf.structured(self))
        with open(path, "w") as stream:
            stream.write(yaml_string)
