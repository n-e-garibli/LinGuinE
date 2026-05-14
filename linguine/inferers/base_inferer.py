# Copyright AstraZeneca 2026

from abc import abstractmethod

import numpy as np
import torch

from linguine.utils.bounding_boxes import BBox3D


class AbstractInferer:
    """Interface for performing segmentation model inference."""

    @property
    def spacing(self) -> tuple[float, float, float] | None:
        """Defines the image spacing that inferer expects. If None, it is assumed that
        the inferer method can output a prediction for an input of any spacing."""
        return None

    @abstractmethod
    def infer(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Generates a model prediction.

        Args:
            img: a 5D tensor containing the image to infer on.
            fg_clicks: a list of foreground guidance clicks to use (can be empty).
            bg_clicks: a list of background guidance clicks to use (can be empty).
            bboxes: a list of bounding boxes to use as guidance (can be empty).
            return_probs: whether to return foreground class probabilities instead
                of a boundary segmentation.
            filter_pred: whether to filter the prediction to only keep the connected
                component

        Returns:
            A numpy array with a binary model prediction or a torch tensor
            with the foreground class probabilities if return_probs is True.
        """
        pass
