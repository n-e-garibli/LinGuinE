# Copyright AstraZeneca 2026
"""A non autoregressive class for segmenting a single longitudinal study."""

import logging
from typing import Any

import numpy as np

from linguine.constants import FILE_ID
from linguine.study_segmentors.base_segmentor import LongitudinalStudySegmentor

LOGGER = logging.getLogger(__name__)


class FromOneTimepointSegmentor(LongitudinalStudySegmentor):
    """Segment a longitudinal study by propagating information from one timepoint every segmentation.

    By default, this will be the earliest possible timepoint in the study, but this can be configured using the USE_AS_SOURCE flag in the input data dictionaries."""

    def segment_target_scan(
        self, target_scan: dict[str, Any], predict_only: bool
    ) -> tuple[np.ndarray, np.ndarray] | None:
        assert target_scan[FILE_ID] != self.source_scan[FILE_ID], "Source and target scans cannot be the same."
        target_img, target_label = self.image_loader(target_scan)
        # Will store individual lesion predictions.
        lesion_preds = []
        for i, lesion_id in enumerate(self.lesion_ids):
            if target_label is not None:
                lesion_label = (target_label == lesion_id).int()
                lesion_label.meta["lesion_id"] = lesion_id
            else:
                lesion_label = None
            source_label = (self.source_label == lesion_id).int()
            source_label.meta["lesion_id"] = lesion_id
            target_metrics, target_prompts, pred = self.propagator(
                target_dict=target_scan,
                source_dict=self.source_scan,
                target_img=target_img,
                target_label=lesion_label,
                source_prompts=self.source_prompts_per_lesion[i],
                source_label=source_label,
                source_img=self.source_image,
            )
            if not predict_only:
                if target_metrics is None:
                    # Failed to propagate prompt likely due to
                    # transformed prompt falling outside of FOV of the target scan.
                    # No prediction is available for this lesion.
                    continue
                else:
                    LOGGER.info(
                        f"DICE SCORE FOR LESION {lesion_id} ON SCAN {target_scan[FILE_ID]}: {target_metrics.dice}"
                    )
            self._update_results(
                target_file_id=target_scan[FILE_ID],
                target_prompts=target_prompts,
                target_metrics=target_metrics,
                lesion_id=lesion_id,
            )
            if pred is not None:
                # Extract the affine of the prediction so that the nifti can
                # be saved with a proper affine.
                pred_affine = pred.affine
                # Turn the lesion prediction into 3D numpy array with the
                # same lesion label as in the source image.
                # Must use uint16 because lesion ids can be > 255
                pred = np.where(pred == 1, lesion_id, 0).astype(np.uint16)[0, 0]
                lesion_preds.append(pred)

        if len(lesion_preds) > 0:
            # Combines all lesion predictions into one array, preserving
            # the lesion label value. In the case of two lesions being predicted
            # at the same voxel, the larger lesion label will be kept.
            full_pred = np.maximum.reduce(lesion_preds)
            return full_pred, pred_affine
        return None
