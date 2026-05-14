# Copyright AstraZeneca 2026
"""Autoregressive classes for segmenting a single longitudinal study."""

import logging
from typing import Any

import numpy as np
from monai.data.meta_tensor import MetaTensor

from linguine.constants import FILE_ID
from linguine.image_loaders.base_image_loader import BaseImageLoader
from linguine.propagator import LinguineBboxPropagator, LinguineClickPropagator
from linguine.study_segmentors.base_segmentor import LongitudinalStudySegmentor
from linguine.utils.bounding_boxes import get_bounding_box
from linguine.utils.sampling import find_mask_center

LOGGER = logging.getLogger(__name__)


class ChainSegmentor(LongitudinalStudySegmentor):
    """Segment a longitudinal study by propagating infromation from the earliest available timepoint
    in a markovian chain."""

    def __init__(
        self,
        patient_id: str,
        patient_scans: list[dict[str, Any]],
        propagator: LinguineClickPropagator,
        image_loader: BaseImageLoader,
        device: str = "cuda:0",
        with_resampling: bool = False,
        forward_in_time_only: bool = True,
        timepoint_flags: list[str] | None = None,
    ):
        if not forward_in_time_only:
            raise NotImplementedError("Backward propagation is not supported in ChainSegmentor.")
        super().__init__(
            patient_id=patient_id,
            patient_scans=patient_scans,
            propagator=propagator,
            image_loader=image_loader,
            device=device,
            forward_in_time_only=forward_in_time_only,
            timepoint_flags=timepoint_flags,
        )
        self.with_resampling: bool = with_resampling

    def segment_target_scan(
        self, target_scan: dict[str, Any], predict_only: bool
    ) -> tuple[np.ndarray, np.ndarray] | None:
        assert target_scan[FILE_ID] != self.source_scan[FILE_ID], "Source and target scans cannot be the same."
        # Will store prompts to be considered in the next iteration
        new_source_prompts = []
        # Will store lesion preds to be considered as labels in the next iteration.
        lesion_preds = []
        target_img, target_label = self.image_loader(target_scan)
        for i, lesion_id in enumerate(self.lesion_ids):
            if target_label is not None:
                lesion_label: MetaTensor = (target_label == lesion_id).int()
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
                if (target_metrics is None) and (pred is None):
                    # Failed to propagate click likely due to transformed click falling outside
                    # of FOV of the target scan. Will skip using this scan as the target.
                    LOGGER.warning(
                        f"Can no longer propagate for lesion {lesion_id}. It will be ignored in further propagations."
                    )
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
            if pred is not None and pred.any():
                # Extract the affine of the prediction so that the nifti can
                # be saved with a proper affine.
                pred_affine = pred.affine
                # Turn the lesion prediction into 3D numpy array with the
                # same lesion label as in the source image.
                # Must use int16 because lesion ids can be > 255
                # Can't use uint16 since in chain mode this is later converted to a tensor.
                pred = np.where(pred == 1, lesion_id, 0).astype(np.int16)[0, 0]
                lesion_preds.append(pred)

                # Update source prompts for next iteration.
                if self.with_resampling:
                    # Generate new guidance prompts using segmentation obtained
                    if isinstance(self.propagator, LinguineBboxPropagator):
                        resampled_bbox = get_bounding_box(pred, as_mask=False)
                        new_source_prompts.append(resampled_bbox)
                    else:
                        resampled_click = find_mask_center(pred)
                        LOGGER.info(f"Sampled new click from prediction: {resampled_click}")
                        new_source_prompts.append([resampled_click])
                else:
                    new_source_prompts.append(target_prompts)

        # Update source scan information for next iteration.
        self.source_prompts_per_lesion = new_source_prompts
        self.source_scan = target_scan
        self.source_image = target_img

        if len(lesion_preds) > 0:
            # Combines all lesion predictions into one array, preserving
            # the lesion label value. In the case of two lesions being predicted
            # at the same voxel, the larger lesion label will be kept because
            # idk how to nicely handle this situation.
            full_pred = np.maximum.reduce(lesion_preds)
            self.source_label = (
                MetaTensor(x=full_pred, affine=pred_affine, meta=target_img.meta, device=self.source_label.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.lesion_ids = [float(x) for x in self.source_label.unique()[1:]]
            # This can fail if you have two predictions that PERFECTLY overlap (so we got a prompt
            # for each tumour but its not in the unique label array above.)
            assert len(self.lesion_ids) == len(self.source_prompts_per_lesion)
            return full_pred, pred_affine

        self.source_label = target_img.clone().zero_()
        self.lesion_ids = []
        assert len(self.lesion_ids) == len(self.source_prompts_per_lesion)
        return None
