# Copyright AstraZeneca 2026
"""Dataset processing module for longitudinal click propagation experiments.
This module provides the `LinguineDatasetProcessor` class, which is the class to use
for running the LinGuinE algorithm on a full dataset."""

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.cuda import OutOfMemoryError, empty_cache
from tqdm import tqdm

from linguine.config import LinguineConfig
from linguine.constants import (
    CLICKS,
    FILE_ID,
    IMAGE,
    LABEL,
    PATIENT_ID,
    RESULTS_CSV,
    USE_AS_SOURCE,
)
from linguine.disappearance_detectors import DisappearanceDetector, SizeFilter
from linguine.image_loaders.base_image_loader import BaseImageLoader
from linguine.inferers import (
    AbstractInferer,
    BasicBoostedInferer,
    ClickEnsembleInferer,
    MergeProbabilitiesBoostedInferer,
    OrientationEnsembleInferer,
    PerturbationEnsembleInferer,
    ResampleAdditiveBoostedInferer,
)
from linguine.prompt_selectors import PromptSelector, ThresholdPS, UnguidedModelPS
from linguine.propagator import LinguineBboxPropagator, LinguineClickPropagator
from linguine.registration.point_extractors import (
    CSVPointExtractor,
    PointExtractor,
)
from linguine.registration.registrators import (
    ArunsRigidRegistrator,
    ImageRegistrator,
    PointSetRegistrator,
    ThinPlateSplineRegistrator,
)
from linguine.study_segmentors import (
    ChainSegmentor,
    FromOneTimepointSegmentor,
    LongitudinalStudySegmentor,
    NoSourceScan,
)

LOGGER = logging.getLogger(__name__)


class LinguineDatasetProcessor:
    """Runs a longitudinal click propagation experiment on an entire dataset.

    This class handles the end-to-end processing of medical imaging datasets for
    longitudinal click propagation experiments. It manages:
    - Image loading and preprocessing
    - Click validity classification
    - Registration between timepoints
    - Segmentation propagation
    - Results collection and saving

    The processor can be configured to use custom components for each stage of the
    pipeline through the constructor arguments.
    """

    def __init__(
        self,
        cfg: LinguineConfig,
        inferer: AbstractInferer,
        save_folder: str,
        data_dicts: list[dict[str, Any]],
        custom_image_loader: BaseImageLoader | None = None,
        custom_PS: PromptSelector | None = None,
        custom_registrator: PointSetRegistrator | ImageRegistrator | None = None,
        custom_point_extractor: PointExtractor | None = None,
        custom_DD: DisappearanceDetector | None = None,
    ):
        """Initialize the dataset processor with configuration and components.

        Args:
            cfg: Configuration object containing all experiment parameters.
            inferer: An inferer object for performing segmentation inference.
            save_folder: Directory path where results, predictions and other assets will be saved.
            data_dicts: List of dictionaries containing dataset samples, each with:
                - patient_id: Unique patient identifier
                - file_id: Unique scan identifier
                - image: image path or data
                - label: Optional ground truth annotations
            custom_image_loader: Optional custom image loader implementation.
            custom_PS: Optional custom prompt selector.
            custom_registrator: Optional custom point set registration implementation.
            custom_point_extractor: Optional custom landmark point extraction implementation.
            custom_DD: Optional custom tumour disappearance detector.

        Note:
            If any custom component is None, a default implementation will be created
            based on the configuration in cfg.
        """
        self.cfg: LinguineConfig = cfg
        self.save_folder: str = save_folder
        self.data_dicts = data_dicts

        if custom_image_loader is not None:
            self.image_loader = custom_image_loader
        else:
            self.image_loader = BaseImageLoader(device=cfg.device)

        if custom_PS is not None:
            self.prompt_selector = custom_PS
        else:
            self.prompt_selector = self._get_ps(inferer=inferer)

        if custom_registrator is not None:
            self.registrator = custom_registrator
        else:
            self.registrator = self._get_registrator()

        if custom_point_extractor is not None:
            self.point_extractor = custom_point_extractor
        else:
            self.point_extractor = self._get_point_extractor()

        if custom_DD is not None:
            self.dd = custom_DD
        else:
            self.dd = self.get_dd()

        if cfg.analysis.boosting is not None:
            inferer = self._boost_inferer(inferer, cfg.analysis.boosting)

        self.propagator = self._setup_propagator(inferer=inferer)
        # Will store a dictionary of results for each propagation in the experiment.
        self.results: list[dict] = []

        # Try to set some seeds
        torch.manual_seed(self.cfg.analysis.seed)
        # Yes yes this is not the optimal way, but gonna do it just in case.
        np.random.seed(self.cfg.analysis.seed)
        random.seed(self.cfg.analysis.seed)

    def _boost_inferer(self, inferer: AbstractInferer, boosting: str) -> AbstractInferer:
        """Creates a boosted version of the base inferer.

        Args:
            inferer: The base inferer to boost.
            boosting: The boosting technique to apply. Must be one of:
                'basic' - Simple resampling of the center of the prediction.
                'resample_additive' - Resampling with additive combination
                'resample_merge_probs' - Resampling with probability merging
                'perturb_ensemble' - Perturbation of the original click & ensemble
                'click_ensemble' - Ensemble of predictions from each individual click

        Returns:
            AbstractInferer: A boosted version of the input inferer.

        Raises:
            ValueError: If an unsupported boosting technique is specified.
        """
        if boosting == "basic":
            return BasicBoostedInferer(inferer)
        elif boosting == "resample_additive":
            return ResampleAdditiveBoostedInferer(inferer)
        elif boosting == "resample_merge_probs":
            return MergeProbabilitiesBoostedInferer(inferer)
        elif boosting == "perturb_ensemble":
            return PerturbationEnsembleInferer(inferer)
        elif boosting == "click_ensemble":
            return ClickEnsembleInferer(inferer)
        elif boosting == "orientation_ensemble":
            return OrientationEnsembleInferer(inferer)
        else:
            raise ValueError(f"Unknown boosting technique: {boosting}")

    def _setup_propagator(self, inferer: AbstractInferer) -> LinguineClickPropagator:
        """Initializes and configures the propagator for the experiment.

        Creates a LinguineClickPropagator or LinguineBboxPropagator instance based on
        the prompt_to_propagate configuration. Includes the inferer, registrator, point
        extractor, and click classifier.

        Args:
            inferer: The inferer object to use for segmentation prediction.

        Returns:
            LinguineClickPropagator or LinguineBboxPropagator: A configured propagator ready for use.
        """
        # Choose the appropriate propagator class based on prompt_to_propagate
        if self.cfg.analysis.prompt_to_propagate in ["bbox", "bbox_2d"]:
            bbox_2d = self.cfg.analysis.prompt_to_propagate == "bbox_2d"
            bbox_type = "2D" if bbox_2d else "3D"
            LOGGER.info(f"Using LinguineBboxPropagator for {bbox_type} bounding box propagation.")
            propagator = LinguineBboxPropagator(
                cfg=self.cfg,
                inferer=inferer,
                registrator=self.registrator,
                point_extractor=self.point_extractor,
                prompt_selector=self.prompt_selector,
                bbox_2d=bbox_2d,
                disappearance_detector=self.dd,
            )
        else:
            LOGGER.info("Using LinguineClickPropagator for click propagation.")
            propagator = LinguineClickPropagator(
                cfg=self.cfg,
                inferer=inferer,
                registrator=self.registrator,
                point_extractor=self.point_extractor,
                prompt_selector=self.prompt_selector,
                disappearance_detector=self.dd,
            )
        return propagator

    def get_dd(self) -> DisappearanceDetector:
        if self.cfg.analysis.min_pred_size_mm > 0:
            dd = SizeFilter(minimum_axial_length_mm=self.cfg.analysis.min_pred_size_mm)
        else:
            dd = None
        return dd

    def _get_ps(self, inferer) -> PromptSelector | None:
        """Creates a prompt selector based on configuration.

        Args:
            inferer: The inferer object to use (required for unguided PS).

        Returns:
            A prompt selector
        """
        ps_type = self.cfg.prompt_selection.type
        if ps_type is None:
            return None
        elif ps_type == "unguided":
            return UnguidedModelPS(inferer=inferer)
        elif ps_type == "threshold":
            return ThresholdPS(
                u_threshold=self.cfg.prompt_selection.u_threshold,
                l_threshold=self.cfg.prompt_selection.l_threshold,
                seed=self.cfg.analysis.seed,
            )

    def _get_point_extractor(self) -> CSVPointExtractor:
        """Creates a point extractor for landmark generation.

        Returns:
            CSVPointExtractor: Point extractor that yields anatomical landmarks
                from the provided CSV .

        Raises:
            ValueError: If the specified CSV file does not exist
        """
        csv_path = self.cfg.registration.point_extractor_csv
        if csv_path:
            if not os.path.exists(csv_path):
                raise ValueError(f"CSV in {self.cfg.registration.point_extractor_csv} not found.")
            LOGGER.info(f"Will use csv in {self.cfg.registration.point_extractor_csv}")
            extractor = CSVPointExtractor(landmark_csv_path=csv_path)
        else:
            raise ValueError("No landmark csv provided.")

        return extractor

    def _get_registrator(self) -> PointSetRegistrator | ImageRegistrator | None:
        """Creates a registrator based on configuration."""
        # Registrators with optional dependencies
        if self.cfg.registration.registrator == "lung_grad_icon":
            from linguine.registration.registrators.icon_registrator import GradIconLungRegistrator

            return GradIconLungRegistrator(crop_foreground=self.cfg.registration.crop_foreground)
        elif self.cfg.registration.registrator == "uni_grad_icon":
            from linguine.registration.registrators.icon_registrator import UniGradIconRegistrator

            return UniGradIconRegistrator(crop_foreground=self.cfg.registration.crop_foreground)
        elif self.cfg.registration.registrator == "uni_grad_icon_roi_refined":
            from linguine.registration.registrators.icon_registrator import UniGradIconWithROIRefinement

            # crop_foreground not provided because this class utilises it in a special way.
            return UniGradIconWithROIRefinement()
        LOGGER.info("Will perform registration with all landmarks from CSV.")
        registrators = {
            "aruns": ArunsRigidRegistrator(valid_landmarks=None),
            "tps": ThinPlateSplineRegistrator(
                valid_landmarks=None,
                _lambda=self.cfg.registration.tps_lambda,
            ),
            # This mode will output the perfect center click for
            # each target tumour - no actual registration performed.
            "perfect": None,
        }
        return registrators[self.cfg.registration.registrator]

    def get_study_segmentor(
        self, mode: str, patient_id: str, patient_scans: list[dict[str, Any]]
    ) -> LongitudinalStudySegmentor:
        """Creates and returns a study segmentor instance based on the specified mode.

        Args:
            mode: The segmentation mode to use. Must be one of: 'CHAIN',
                'CHAIN_WITH_RESAMPLING', or "FROM_ONE_TIMEPOINT".
            patient_id: The unique identifier for the patient.
            patient_scans: List of dictionaries containing scan data for the patient.
                Each dict should contain image and metadata for one timepoint.

        Returns:
            LongitudinalStudySegmentor: An instance of the appropriate segmentor class
                configured for the specified mode.

        Raises:
            KeyError: If an invalid mode is provided.
        """
        # Map modes to segmentor classes
        segmentor_mapping = {
            "CHAIN": ChainSegmentor,
            "CHAIN_WITH_RESAMPLING": ChainSegmentor,
            "FROM_ONE_TIMEPOINT": FromOneTimepointSegmentor,
        }
        # Get the segmentor class
        segmentor_class = segmentor_mapping[mode]

        # Handle additional arguments for specific modes due to non-uniform interface.
        extra_args = {}
        if mode == "CHAIN_WITH_RESAMPLING":
            extra_args["with_resampling"] = True
        elif mode == "CHAIN":
            extra_args["with_resampling"] = False

        # Instantiate and return the segmentor
        return segmentor_class(
            patient_id=patient_id,
            patient_scans=patient_scans,
            propagator=self.propagator,
            image_loader=self.image_loader,
            device=self.cfg.device,
            forward_in_time_only=self.cfg.analysis.forward_in_time_only,
            timepoint_flags=self.cfg.analysis.timepoint_flags,
            **extra_args,
        )

    def _group_datadicts_by_patient(self) -> dict[str, list[dict[str, Any]]]:
        """Groups the data dictionaries by patient ID.

        Returns:
            A dictionary mapping patient IDs to lists of their corresponding data
            dictionaries. If self.cfg.patient_ids is set, only includes data for the
            specified patient IDs.

        Raises:
            ValueError: If any data dictionary is missing the 'patient_id' key.
        """
        grouped_data_dicts: dict[str, list[dict[str, Any]]] = {}
        for data_dict in self.data_dicts:
            if PATIENT_ID not in data_dict:
                raise ValueError(
                    PATIENT_ID in data_dict, f"patient_id key not found in data dict for file {data_dict[FILE_ID]}"
                )
            patient_id = data_dict[PATIENT_ID]
            if self.cfg.patient_ids is not None:
                if patient_id not in self.cfg.patient_ids:
                    # Running on a subset of patients - ignore this one.
                    continue
            if patient_id in grouped_data_dicts:
                grouped_data_dicts[patient_id].append(data_dict)
            else:
                grouped_data_dicts[patient_id] = [data_dict]
        return grouped_data_dicts

    def _validate_source_scan_flags(self, grouped_data_dicts: dict[str, list[dict[str, Any]]]) -> None:
        """Validate that only one scan per patient has use_as_source=True.

        Args:
            grouped_data_dicts: A dictionary mapping patient IDs to lists of their
                corresponding data dictionaries. Output of the _group_datadicts_by_patient() method.

        Raises:
            ValueError: If multiple scans have use_as_source=True for the same patient.
            ValueError: If the source scan does not have a label or clicks for guidance.
        """
        for patient_id, patient_scans in grouped_data_dicts.items():
            source_flagged_scans = [scan for scan in patient_scans if scan.get(USE_AS_SOURCE, False)]
            if len(source_flagged_scans) > 1:
                flagged_file_ids = [scan[FILE_ID] for scan in source_flagged_scans]
                raise ValueError(
                    f"Multiple scans have 'use_as_source=True' for patient {patient_id}: {flagged_file_ids}. "
                    "Only one scan per patient can be marked as the source scan."
                )
            elif len(source_flagged_scans) == 1:
                source_scan = source_flagged_scans[0]
                if not (LABEL in source_scan or CLICKS in source_scan):
                    raise ValueError(
                        f"No guidance to propagate from specified source scan {source_scan[FILE_ID]} for patient {patient_id}."
                    )

    def process_dataset(self) -> None:
        """Run the click propagation experiment on the dataset.

        Processes each patient's scans according to the configured mode and parameters.
        Skips patients with fewer than 2 scans. Results are stored in self.results.
        """
        grouped_data_dicts = self._group_datadicts_by_patient()
        # Validate source scan flags before processing
        self._validate_source_scan_flags(grouped_data_dicts)
        n_patients = len(grouped_data_dicts)
        LOGGER.info(f"Segmenting {n_patients} studies...")
        # Iterate through all the patients and their relevant data dicts.
        for patient, patient_scans in tqdm(grouped_data_dicts.items(), total=n_patients):
            LOGGER.info(f"Considering patient {patient}...")
            if len(patient_scans) < 2:
                LOGGER.info(f"Not enough valid scans for patient {patient}, skipping...")
                continue
            # Segment the longitudinal study for this patient with correct mode.
            try:
                study_segmentor = self.get_study_segmentor(self.cfg.analysis.iteration_mode, patient, patient_scans)
            except NoSourceScan:
                LOGGER.warning(f"Not segmenting for patient {patient} due to lack of usable source scan.")
                continue
            try:
                patient_results = study_segmentor(
                    predict_only=self.cfg.predict_only,
                    save_dir=self.save_folder if self.cfg.save_predictions else None,
                )
                self.results.extend(patient_results)
            except OutOfMemoryError as e:
                LOGGER.error(f"CUDA out of memory while processing patient {patient}: {e}")
                LOGGER.error("Skipping this patient and continuing with the next one.")
                empty_cache()
                continue
            self.save_results()
        # Log prompt validity specific metrics.
        if not self.cfg.predict_only:
            self.propagator.cv_metrics.log_metrics()
        if self.cfg.save_with_original_affine:
            LOGGER.info("Resampling saved predictions to match original images...")
            self.postprocess_predictions()

    def postprocess_predictions(self) -> None:
        """Resample all saved predictions to match their original image geometry.

        This method processes all predictions saved during the experiment and resamples
        them via SimpleITK to match the spatial properties (spacing, direction, origin) of their
        corresponding original images.

        Raises:
            RuntimeError: If image loading or resampling fails for any prediction
        """
        for data_dict in tqdm(self.data_dicts):
            img_path = Path(data_dict[IMAGE])
            pred_path = Path(self.save_folder).joinpath("pred_" + data_dict[FILE_ID] + ".nii.gz")
            if pred_path.exists():
                try:
                    itk_pred = self._resample_pred_to_match_image(pred_path, img_path)
                    # Save the prediction with the original affine.
                    sitk.WriteImage(itk_pred, str(pred_path))
                except Exception as e:
                    LOGGER.error(f"Failed to resample prediction {pred_path} for image {img_path}: {e}")

    def _resample_pred_to_match_image(self, pred_path: Path, image_path: Path) -> sitk.Image:
        """Transform a LinGuinE output prediction to match the original image geometry.

        This method resamples a prediction to match the spatial properties (spacing,
        direction, origin, and size) of the original image.

        Args:
            pred_path: Path to a LinGuinE output prediction file.
            image_path: Path to the original image that the prediction corresponds to.
                Can be either a DICOM directory or a single image file.

        Returns:
            sitk.Image: The resampled prediction matching the original image geometry.

        Raises:
            RuntimeError: If the image cannot be loaded or resampling fails.
        """
        try:
            # Try to load as DICOM series first (if image_path is a directory)
            if image_path.is_dir():
                reader = sitk.ImageSeriesReader()
                dicom_files = reader.GetGDCMSeriesFileNames(str(image_path))
                if dicom_files:
                    reader.SetFileNames(dicom_files)
                    image = reader.Execute()
                else:
                    raise RuntimeError(f"No DICOM files found in directory: {image_path}")
            else:
                # Load as single image file (NIfTI, etc.)
                image = sitk.ReadImage(str(image_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {image_path}: {e}") from e

        try:
            # Set up a resampler object to match the reference image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)

            # Load and transform the prediction to match the reference image
            pred = sitk.ReadImage(str(pred_path))
            new_pred = resampler.Execute(pred)
            return new_pred
        except Exception as e:
            raise RuntimeError(f"Failed to resample prediction {pred_path}: {e}") from e

    def save_results(self) -> None:
        """Saves experiment results and metrics to CSV files.

        Creates two output files in the configured save directory:
        1. results.csv: Contains per-case results including:
            - Patient and timepoint identifiers
            - Click coordinates for each tumor
            - Segmentation metrics (if ground truth available)
            - Click validity counts per lesion
            - Distances between clicks and ground truth centers (if ground truth available)

        2. cv_metrics.csv: Contains aggregated prompt validity metrics
            (only created if not in predict_only mode)
        """
        # Turn list of dictionaries into a dataframe
        results_df = pd.DataFrame(self.results)
        if not self.cfg.predict_only:
            # Save and add additional click validity metrics.
            self.propagator.cv_metrics.save_metrics_csv(os.path.join(self.save_folder, "cv_metrics.csv"))
            # Add additional click validity metrics into the results.
            results_df["valid"] = self.propagator.cv_metrics.valid_click_per_lesion_log
            results_df["invalid"] = self.propagator.cv_metrics.invalid_click_per_lesion_log
            results_df["label_dist"] = self.propagator.cv_metrics.distances
        # Save results df to csv.
        csv_path = os.path.join(self.save_folder, RESULTS_CSV)
        results_df.to_csv(csv_path, index=False)
        LOGGER.debug(f"RESULTS SAVED TO {csv_path}")
