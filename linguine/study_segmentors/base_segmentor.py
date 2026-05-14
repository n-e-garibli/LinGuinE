# Copyright AstraZeneca 2026
"""A base class for segmenting a single longitudinal study."""

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from linguine.constants import CLICKS, FILE_ID, LABEL, PATIENT_ID, TIMEPOINT, USE_AS_SOURCE
from linguine.image_loaders.base_image_loader import BaseImageLoader
from linguine.metrics import MetricsBundle
from linguine.propagator import LinguineBboxPropagator, LinguineClickPropagator
from linguine.utils.bounding_boxes import BBox2D, BBox3D, get_bounding_box
from linguine.utils.data import get_path_from_data_dict_entry
from linguine.utils.sampling import find_mask_center

LOGGER = logging.getLogger(__name__)


class NoSourceScan(Exception):
    """Exception for when a patient has no usable source scan."""

    pass


class LongitudinalStudySegmentor(ABC):
    """A base class to segment a longitudinal study using guidance propagation.

    A longitudinal study is a collection of scans from a single patient.

    This base class requires only an implementation of the segment_target_scan()
    method which will segment one given scan belonging to the patient."""

    def __init__(
        self,
        patient_id: str,
        patient_scans: list[dict[str, Any]],
        propagator: LinguineClickPropagator | LinguineBboxPropagator,
        image_loader: BaseImageLoader,
        device: str = "cuda:0",
        forward_in_time_only: bool = False,
        timepoint_flags: list[str] | None = None,
    ):
        """Construct the class.

        Args:
            patient_id: The id of the patient whose study must be segmented.
            patient_scans: A list of data dicts for the patient. Each data dict represents an image in the study
                and must contain the IMAGE, TIMEPOINT, and FILE_ID fields.
            propagator: An initialised propagator object to perform the prompt propagation and segmentation with.
            image_loader: An object, which, when called will load the image and (optionally) label from the data dict.
            device: The name of the device to perform the segmentations on.
            forward_in_time_only: Whether to only propagate to timepoints that come after the source scan in time.
                If False, will propagate to all remaining timepoints (both before and after source scan).
            timepoint_flags: Optional list of timepoint identifier patterns (e.g., ["week_", "cycle_"]) to use
                when sorting scans. If None, a default set of patterns will be used.
        """
        self.patient: str = patient_id
        # Get patient scans in ascending order (from earliest timepoint to latest)
        self.patient_scans = self._sort_scans_by_timepoint(patient_scans, timepoint_flags)
        assert propagator.cfg.device == device, (
            f"Device mismatch between propagator {propagator.cfg.device} and segmentor {device}."
        )
        self.image_loader = image_loader
        self.propagator: LinguineClickPropagator = propagator
        self.device: str = device
        self.source_image = None  # Will be assigned as a side effect of self._pick_source_scan()
        self.source_scan, self.source_label = self._pick_source_scan()
        # Obtain lesion_ids loaded
        self.lesion_ids = [j.item() for j in self.source_label.unique()[1:]]
        LOGGER.info(f"Lesions being segmented have values {self.lesion_ids} in the label array.")
        if isinstance(propagator, LinguineBboxPropagator):
            self.source_prompts_per_lesion = self._get_bboxes_per_lesion()
        else:
            self.source_prompts_per_lesion = self._get_clicks_per_lesion()

        # Determine target scans based on forward_in_time_only setting
        source_scan_i = self.patient_scans.index(self.source_scan)
        if forward_in_time_only:
            # Will propagate to all remaining scans going forward in time only
            self.target_scans = self.patient_scans[source_scan_i + 1 :]
        else:
            # Will propagate to all remaining scans (both before and after source scan)
            self.target_scans = self.patient_scans[:source_scan_i] + self.patient_scans[source_scan_i + 1 :]
        # Will store propagation results
        self.results: list[dict] = []

    @staticmethod
    def _sort_scans_by_timepoint(
        scans: list[dict[str, Any]], timepoint_flags: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Sorts scans by timepoint from earliest to latest.

        Args:
            scans: list of scan names to sort. The timepoint identifier
                must be the first part of the scan name string.
            timepoint_flags: Optional list of timepoint identifier patterns (e.g., ["week_", "cycle_"])
                to use when extracting numeric timepoint values. If None, uses a default set of patterns.

        Returns:
            A sorted version of the input list.
        """
        # Use provided timepoint flags or default patterns
        if timepoint_flags is None:
            timepoint_flags = [
                "week_",
                "end_of_cycle_",
                "disease_assessment_",
                "cycle_",
                "fu_",
                "follow_up_",
                "exam_",
                "timepoint_",
            ]

        # Mapping of file_id to a numerical timepoint identifier
        scan_to_timepoint: dict[str, int] = {}
        # Mapping of a retreatment scan file_id to a numerical timepoint identifier
        re_treatment_scan_to_timepoint: dict[str, int] = {}

        # Strings to help identify retreatement timepoints
        retreatment_identifiers = [
            "rebaseline",
            "re_baseline",
            "rescreening",
            "re_screening",
            "retreatment",
            "re_treatment",
        ]

        # Scans done at the end of the trial
        final_scans = []
        non_final_scans = []
        for scan in scans:
            if TIMEPOINT in scan:
                # Use dedicated timepoint field in the datadict as the timepoint identifier
                tp_id = scan[TIMEPOINT]
            else:
                # Use the file id as the tinpeoint identifier hoping it contains the necessary info.
                tp_id = scan[FILE_ID]

            lc_scan = tp_id.lower()

            dict_to_update = scan_to_timepoint
            if any(i in lc_scan for i in retreatment_identifiers):
                dict_to_update = re_treatment_scan_to_timepoint

            if any(i in lc_scan for i in ("screening", "baseline")):
                dict_to_update[scan[FILE_ID]] = 0
                non_final_scans.append(scan)
            elif "end_of_treatment" in lc_scan:
                final_scans.append(scan)
            elif "unscheduled" in lc_scan:
                # Unscheduled scans are assumed to be done at the end.
                final_scans.append(scan)
            else:
                # Iterate through possible timepoint identifying patterns
                for tp_identifier in timepoint_flags:
                    if tp_identifier in lc_scan:
                        pattern = rf"{tp_identifier}(\d+)"
                        match = re.search(pattern, lc_scan)
                        dict_to_update[scan[FILE_ID]] = int(match.group(1))
                        non_final_scans.append(scan)
                        break

        sorted_file_ids: list[str] = []
        for dic in [scan_to_timepoint, re_treatment_scan_to_timepoint]:
            # Sort dictionary keys by the corresponding value
            sorted_items = sorted(dic.items(), key=lambda x: x[1])
            sorted_file_ids.extend([i[0] for i in sorted_items])
        # Create a mapping of file_id to its position in the order list
        order_map = {file_id: index for index, file_id in enumerate(sorted_file_ids)}
        sorted_scans = sorted(non_final_scans, key=lambda x: order_map[x[FILE_ID]])

        return sorted_scans + final_scans

    def _pick_source_scan(self) -> tuple[dict[str, Any], torch.Tensor]:
        """Pick the scan that should be used as a source scan.

        First checks if any scan has the USE_AS_SOURCE flag set to True. If found,
        validates that only one scan per patient has this flag and uses that scan.
        Otherwise, falls back to selecting the earliest timepoint scan that can be used as a source scan.

        Returns:
            A tuple containing information about the scan to use as the source. Specifically,
            its (scan name, label metatensor).

        Raises:
            ValueError: If multiple scans have use_as_source=True for the same patient.
            NoSourceScan: If no suitable source scan can be found.
        """
        # First check if any scan has the USE_AS_SOURCE flag
        source_flagged_scans = [scan for scan in self.patient_scans if scan.get(USE_AS_SOURCE, False)]
        if len(source_flagged_scans) > 1:
            flagged_file_ids = [scan[FILE_ID] for scan in source_flagged_scans]
            raise ValueError(
                f"Multiple scans have 'use_as_source=True' for patient {self.patient}: {flagged_file_ids}."
                "Only one scan per patient can be marked as the source scan."
            )
        if len(source_flagged_scans) == 1:
            data_dict = source_flagged_scans[0]
            label = self._validate_and_load_source_scan(data_dict)
            if label is None:
                if CLICKS not in data_dict or data_dict[CLICKS] is None:
                    raise NoSourceScan(f"User-specified source scan {data_dict[FILE_ID]} is not valid.")
                else:
                    # Obtain guided prediction on the source image using the provided clicks and treat
                    # this as the source label.
                    if len(data_dict[CLICKS]) != 1:
                        LOGGER.warning(
                            "Found multiple clicks for the source scan - they will be assumed to belong to the same tumour."
                            "Propagation from click guidance is not supported for multi-lesion segmentation yet."
                        )
                    LOGGER.info(
                        f"Using clicks {data_dict[CLICKS]} from the source scan {data_dict[FILE_ID]} to obtain a label."
                    )
                    _, label = self.propagator.get_metrics_and_prediction(
                        img=self.image_loader(data_dict=data_dict)[0], fg_clicks=data_dict[CLICKS]
                    )
            LOGGER.info(f"Will treat {data_dict[FILE_ID]} as the source scan (user-specified).")
            return data_dict, label
        return self._pick_automatic_source_scan()

    def _pick_automatic_source_scan(self) -> tuple[dict[str, Any], torch.Tensor]:
        """Automatically select the earliest suitable timepoint scan as source.

        Returns:
            A tuple containing the scan dictionary and its label tensor.

        Raises:
            NoSourceScan: If no suitable source scan can be found.
        """
        # Iterate through the patient scans, evaluating each one as a candidate
        for source_scan_candidate in self.patient_scans:
            label = self._validate_and_load_source_scan(source_scan_candidate)
            if label is not None:
                # If a valid label is found, this scan can be used as the source scan
                LOGGER.info(f"Will treat {source_scan_candidate[FILE_ID]} as the source scan (auto-selected).")
                return source_scan_candidate, label
        # Iterate again, this time checking for scans with clicks
        for source_scan_candidate in self.patient_scans:
            if CLICKS in source_scan_candidate and source_scan_candidate[CLICKS] is not None:
                # If clicks are provided, use them directly
                LOGGER.info(
                    f"Using clicks {source_scan_candidate[CLICKS]} from the source scan {source_scan_candidate[FILE_ID]} to obtain a label."
                )
                _, label = self.propagator.get_metrics_and_prediction(
                    img=self.image_loader(data_dict=source_scan_candidate)[0], fg_clicks=source_scan_candidate[CLICKS]
                )
                return source_scan_candidate, label
        raise NoSourceScan("Unable to select suitable source scan.")

    def _validate_and_load_source_scan(self, scan_dict: dict[str, Any]) -> torch.Tensor | None:
        """Validate a source scan candidate and load its label.

        Args:
            scan_dict: The scan dictionary to validate.

        Returns:
            The loaded label tensor or None if scan is not suitable
        """
        # Check for label presence and path validity
        if LABEL not in scan_dict:
            return None
        label_path: Path = get_path_from_data_dict_entry(scan_dict, LABEL)
        if label_path is None or not label_path.exists():
            return None
        # Load and validate the label
        self.source_image, label = self.image_loader(scan_dict)
        if not label.any():
            return None

        return label

    def _get_clicks_per_lesion(self) -> list[list[tuple[int, int, int]]]:
        """Sample clicks for every lesion in the source label.

        Returns:
            A list of clicks sampled per lesion."""
        if CLICKS in self.source_scan and self.source_scan[CLICKS] is not None:
            # If clicks are provided in the source scan, validate and distribute them
            clicks = self.source_scan[CLICKS]
            # Validate click format
            if not isinstance(clicks, list):
                raise ValueError(f"Clicks must be a list, got {type(clicks)}")
            for click in clicks:
                if not isinstance(click, list | tuple) or len(click) != 3:
                    raise ValueError(f"Each click must be a 3-element tuple/list (x, y, z), got {click}")
                if not all(isinstance(coord, int) for coord in click):
                    raise ValueError(f"Click coordinates must be integers, got {click}")
            # Convert to proper format
            validated_clicks = [tuple(int(coord) for coord in click) for click in clicks]
            return [validated_clicks]

        # Will store all guidance clicks sampled for each lesion.
        clicks_per_lesion: list = []
        for lesion_id in self.lesion_ids:
            # Binarize the label to only contain lesion of interest.
            lesion_label = (self.source_label == lesion_id).float()
            # Generate a single fg center click for this lesion.
            lesion_clicks = [find_mask_center(lesion_label)]
            clicks_per_lesion.append(lesion_clicks)
        return clicks_per_lesion

    def _get_bboxes_per_lesion(self) -> list[list[BBox3D]] | list[list[BBox2D]]:
        """Spawn bounding boxes for every lesion in the source label.

        Returns:
            A list of bboxes sampled per lesion."""

        # Will store bounding boxes sampled for each lesion.
        bboxes_per_lesion: list = []
        # Check if we should use 2D bboxes from the config
        use_bbox_2d = self.propagator.cfg.analysis.prompt_to_propagate == "bbox_2d"
        for lesion_id in self.lesion_ids:
            # Binarize the label to only contain lesion of interest.
            lesion_label = (self.source_label == lesion_id).float()
            label_3d = lesion_label[0, 0].cpu()
            bbox = get_bounding_box(label_3d, as_mask=False, bbox_2d=use_bbox_2d)
            bboxes_per_lesion.append([bbox])
        return bboxes_per_lesion

    def _update_results(
        self,
        target_file_id: str,
        target_prompts: list[tuple[int, int, int] | BBox3D],
        target_metrics: MetricsBundle,
        lesion_id: int,
    ) -> None:
        """Update the results attribute with information about the latest propagation.

        Args:
            target_file_id: id of the scan used as the target.
            target_prompts: List of fg clicks or bounding boxes provided in the target scan.
            target_metrics: Metric bundle obtained on the scan.
            lesion_id: The id of the lesion being considered in the propagation.
        """
        i = self.lesion_ids.index(lesion_id)
        source_prompts = self.source_prompts_per_lesion[i]
        results_row = {
            PATIENT_ID: self.patient,
            "source_scan": self.source_scan[FILE_ID],
            "target_scan": target_file_id,
            "source_fg_prompts": str(source_prompts),
            "target_fg_prompts": str(target_prompts),
            "target_dice": target_metrics.dice,
            "target_hd95": target_metrics.hd95,
            "target_precision": target_metrics.precision,
            "target_recall": target_metrics.recall,
            "target_assd": target_metrics.assd,
            "target_nsd": target_metrics.nsd,
            "target_med": target_metrics.med,
            "lesion_id": lesion_id,
            "confidence": target_metrics.confidence,
            "detected_disappearance": target_metrics.detected_disappearance,
        }
        self.results.append(results_row)

    def __call__(self, predict_only: bool = False, save_dir: str | None = None) -> list[dict]:
        """Segment the longitudinal study.

        Args:
            predict_only: If true, no metrics will be computed and it will be assumed that labels do not
                exist for any of the target scans.
            save_dir: A directory to save predictions as nifti images. If not provided, predictions will not be saved.

        Returns:
            A list of dictionaries. Each dictionary contains information about a click propagation on a per lesion basis.
        """
        if save_dir is not None:
            os.makedirs(name=save_dir, exist_ok=True)
        for target_scan in self.target_scans:
            LOGGER.info(f"Treating {self.source_scan[FILE_ID]} as the source and {target_scan[FILE_ID]} as the target")
            seg_data = self.segment_target_scan(target_scan=target_scan, predict_only=predict_only)
            if save_dir is not None and seg_data is not None:
                pred, affine = seg_data
                nifti = nib.nifti1.Nifti1Image(pred, affine)
                f_id = target_scan[FILE_ID]
                filename = os.path.join(save_dir, f"pred_{f_id}.nii.gz")
                nib.save(nifti, filename=filename)
                LOGGER.info(f"Saved prediction to {filename}")
        return self.results

    @abstractmethod
    def segment_target_scan(
        self, target_scan: dict[str, Any], predict_only: bool
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Segment a target scan and updates results attribute.

        Args:
            target_scan: A data dictionary for the scan to segment.
            predict_only: If True, no metrics will be computed. It will be assumed that the label
                does not exist for this target scan.

        Returns:
            A 3D numpy array containing the prediction and an array containing the affine for this prediction.
            The labels will be consistent with the labels of the source scan. Alternatively returns None if no
            propagations were successful for any of the lesions in the image.
        """
        pass
