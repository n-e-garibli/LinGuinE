# Copyright AstraZeneca 2026
"""
This module contains metrics for evaluating click validity during click propagation.

The ClickValidityMetrics class tracks various performance metrics for propagated prompts.
including accuracy, and distances between propagated clicks and actual foreground locations.
"""

import logging

import numpy as np
import pandas as pd
import torch
from monai.data.meta_tensor import MetaTensor

from linguine.utils.bounding_boxes import BBox3D
from linguine.utils.misc import find_closest_point

LOGGER = logging.getLogger(__name__)


class ClickValidityMetrics:
    """A class for tracking and analyzing the accuracy of propagated prompts.

    This class maintains counters and metrics to evaluate how well clicks are filtered and propagated
    during LinGuinE. It tracks:
    - Correctly/incorrectly kept or removed clicks
    - Out of bounds clicks
    - Per-lesion valid/invalid click counts
    - Distances between propagated clicks and nearest ground truth voxels.

    The metrics can be used to compute standard classification metrics like precision,
    sensitivity (recall), specificity, and accuracy.
    """

    def __init__(
        self,
    ):
        """Initialize a new ClickValidityMetrics instance."""

        # These metrics track the total amount of clicks propagated correctly/incorrectly
        # over the entire dataset. This can help generally assess the performance of the prompt
        # propagation.
        self.correctly_kept = 0
        self.falsely_kept = 0
        self.out_of_bounds = 0

        # Stores as a list the number of valid and invalid clicks propagated for every lesion individually.
        # This will be added to the results.csv directly as a column.
        self.invalid_click_per_lesion_log = []
        self.valid_click_per_lesion_log = []
        self.distances = []

    def log_metrics(self) -> None:
        """Log the current state of all tracking metrics to the logger."""
        LOGGER.info(
            f"Of all clicks propagated so far, {self.correctly_kept}/{(self.correctly_kept + self.falsely_kept)} were valid."
        )
        LOGGER.info(
            f"Of all clicks propagated so far, {self.falsely_kept}/{(self.correctly_kept + self.falsely_kept)} were invalid."
        )
        LOGGER.info(f"{self.out_of_bounds} clicks were propagated out of bounds, and therefore were not considered.")
        LOGGER.info(f"On average, propagated clicks are {np.nanmean(self.distances)} voxels away from the labels.")
        LOGGER.info(f"The median propagated click is {np.nanmedian(self.distances)} voxels away from the label.")
        LOGGER.info(f"Accuracy: {self._compute_accuracy()}")

    def update(
        self,
        clicks: list[tuple[int, int, int]],
        target_label: MetaTensor,
    ) -> None:
        """Update metrics based on propagated clicks for a single lesion.

        Args:
            clicks: List of click coordinates (x,y,z) after registration.
            target_label: Binary tensor marking actual foreground locations, used as ground truth
                         for evaluating click validity.
        """
        n_valid_for_this_case = 0
        n_invalid_for_this_case = 0
        dists = []
        for click in clicks:
            # This click will be propagated.
            if target_label[0][0][click]:
                # It is a valid click.
                self.correctly_kept += 1
                n_valid_for_this_case += 1
                dists.append(0.0)
            else:
                # It is an invalid click.
                self.falsely_kept += 1
                n_invalid_for_this_case += 1
                dist = self._compute_dist_to_label(target_label, click)
                dists.append(dist)
                LOGGER.warning(f"Propagated click {click} falls outside of label by {dist} voxels.")
        if (n_invalid_for_this_case + n_valid_for_this_case) > 0:
            # Something was propagated, so this case will have a row in the
            # final csv with all results.
            self.invalid_click_per_lesion_log.append(n_invalid_for_this_case)
            self.valid_click_per_lesion_log.append(n_valid_for_this_case)
            self.distances.append(np.nanmean(dists))

    def update_perfect(self) -> None:
        """Updates metrics for a single valid click."""
        self.valid_click_per_lesion_log.append(1)
        self.invalid_click_per_lesion_log.append(0)
        self.distances.append(0.0)

    def update_bbox(
        self,
        target_bbox: BBox3D,
        target_label: MetaTensor,
    ) -> None:
        """Update metrics using the center of a bounding box as a click.

        Args:
            target_bbox: The bounding box whose center will be used as a click location.
            target_label: 5D Binary tensor marking actual foreground locations, used as ground truth
                         for evaluating click validity.
        """
        bbox_center = target_bbox.center.to_tuple()
        self.update(clicks=[bbox_center], target_label=target_label)

    def _compute_dist_to_label(self, target_label: MetaTensor, click: tuple[int, int, int]) -> float:
        """Computes the euclidean distance from the click to the nearest foreground voxel.

        Args:
            target_label: a MetaTensor containing the ground truth mask.
            click: the propagated click (x, y, z).

        Returns:
            A euclidean distance (in voxels).
        """
        if not target_label.any():
            return np.nan
        positives = torch.where(target_label[0][0])
        closest_coor = find_closest_point(positives=positives, target=click)
        dist = np.sqrt(np.sum([(closest_coor[i] - click[i]) ** 2 for i in range(3)]))
        return dist

    def save_metrics_csv(self, path: str) -> None:
        """Saves various classic metrics."""

        assert path.endswith(".csv"), "Path provided must be a csv."

        metrics: dict[str, float | int | None] = {
            "true_positives": [self.correctly_kept],
            "false_positives": [self.falsely_kept],
            "accuracy": [self._compute_accuracy()],
            "mean_dist_to_label": [np.nanmean(self.distances)],
        }

        df = pd.DataFrame(metrics)
        df.to_csv(path, index=False)
        LOGGER.info(f"Metrics saved to: {path}")

    def _compute_accuracy(self) -> float | None:
        """Computes the accuracy of the propagated prompts.
        What fraction of propagated clicks landed inside the tumour?"""
        n_correct = self.correctly_kept
        n_predicted = self.correctly_kept + self.falsely_kept
        if n_predicted != 0:
            return n_correct / n_predicted
