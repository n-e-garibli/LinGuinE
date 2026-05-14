# Copyright AstraZeneca 2026
"""Module implementing a prompt selector based on unguided model predictions.

This prompt selector implementation uses predictions from a pre-trained
segmentation model to determine click validity in medical imaging data."""

from typing import Any

import torch
from monai.data.meta_tensor import MetaTensor

from linguine.constants import FILE_ID
from linguine.inferers.base_inferer import AbstractInferer
from linguine.inferers.roi_inferer import ROIInferer
from linguine.prompt_selectors.base_ps import PromptSelector
from linguine.utils.data import get_spacing_from_metatensor, prepare_inputs_for_inferer


class UnguidedModelPS(PromptSelector):
    """A prompt selector using unguided model predictions.

    This validates clicks by comparing the model's prediction
    probability (without any guidance signals) at each click location
    against a threshold.
    """

    def __init__(
        self,
        inferer: AbstractInferer | None = None,
    ):
        """Initialize the unguided model-based click validity classifier.

        Args:
            inferer: Inference handler for generating model predictions.
            image_loader: Optional image loader for loading image data when not
                directly provided.
        """
        self.inferer = inferer
        self._tiny_cache = {}
        self.inferer_is_roi_based: bool = isinstance(self.inferer, ROIInferer)

    def _get_pred(
        self,
        target_img: torch.Tensor,
        target_dict: dict[str, Any] | None,
        target_pred: torch.Tensor | None = None,
        target_clicks: list[tuple[int, int, int]] | None = None,
    ) -> torch.Tensor:
        """Get model predictions for the target image.

        Internal helper method that handles obtaining predictions either from
        pre-computed values or by running inference with the model.

        Args:
            target_img: Optional pre-loaded image tensor.
            target_dict: Dictionary containing image metadata and path information.
            target_clicks: A list of clicks to potentially define the ROI to predict on.
            target_pred: Optional pre-computed model predictions.

        Returns:
            torch.Tensor: Unguided model predictions for the target image.

        Raises:
            AssertionError: If neither predictions nor means to compute them
                (inferer/image_loader) are provided.
        """
        if target_pred is None:
            if target_dict is not None:
                if target_dict[FILE_ID] in self._tiny_cache:
                    return self._tiny_cache[target_dict[FILE_ID]]
            assert self.inferer is not None, "Prompt selector not equipped to get prediction from inputs."
            if self.inferer_is_roi_based:
                assert target_clicks is not None, (
                    "ROIInferers require clicks to be provided even if doing unguided inference."
                )
                target_clicks = self.filter_out_of_bounds_clicks(clicks=target_clicks, img_shape=target_img.shape[2:])
                if len(target_clicks) == 0:
                    # No valid target clicks, return all 0 prediction to avoid any propagation
                    return torch.zeros_like(target_img[0, 0])
            spaced_image, spaced_clicks, _ = prepare_inputs_for_inferer(
                desired_spacing=self.inferer.spacing, image=target_img, clicks=target_clicks
            )
            inferer_args = {"img": spaced_image, "bg_clicks": [], "bboxes": [], "return_probs": True}
            if self.inferer_is_roi_based:
                inferer_args["unguided"] = True
                inferer_args["fg_clicks"] = spaced_clicks
            else:
                inferer_args["fg_clicks"] = []
            target_pred = self.inferer.infer(**inferer_args)
            meta_tensor_target_pred = MetaTensor(x=target_pred.unsqueeze(0), affine=spaced_image.affine).unsqueeze(0)
            target_pred, _, _ = prepare_inputs_for_inferer(
                desired_spacing=get_spacing_from_metatensor(target_img), image=meta_tensor_target_pred
            )
            target_pred = target_pred[0, 0]
        else:
            assert target_pred.ndim == 3, "Unexpected prediction shape."

        if not self.inferer_is_roi_based and target_dict is not None:
            # The unguided prediction will be the same for this file
            # Store it to avoid recomputing if this scan has multiple
            # lesions.
            self._tiny_cache = {}
            self._tiny_cache[target_dict[FILE_ID]] = target_pred
        return target_pred

    def get_best_clicks(
        self,
        target_clicks: list[tuple[int, int, int]],
        target_dict: dict[str, Any] | None,
        target_pred: torch.Tensor | None = None,
        target_img: torch.Tensor | None = None,
        n_clicks: int = 1,
        *args,
        **kwargs,
    ) -> list[tuple[int, int, int]]:
        """Select clicks with highest model prediction probabilities.

        Ranks clicks based on the model's prediction probabilities and returns
        the top n_clicks clicks. Can optionally filter to only return clicks above
        the validity threshold.

        Args:
            target_clicks: List of click coordinates as (x, y, z) tuples.
            target_dict: Dictionary containing image metadata and path information.
            target_pred: Optional pre-computed model predictions.
            target_img: Optional pre-loaded image tensor.
            n_clicks: Maximum number of clicks to return. Defaults to 1.
            *args: Additional positional arguments for future extensions and
                compatibility with other prompt selectors.
            **kwargs: Additional keyword arguments for future extensions and
                compatibility with other prompt selectors.

        Returns:
            list[tuple[int, int, int]]: Up to n_clicks clicks with highest
                prediction probabilities. Returns empty list if no valid clicks found.
        """
        target_pred = self._get_pred(
            target_dict=target_dict, target_clicks=target_clicks, target_pred=target_pred, target_img=target_img
        )
        target_clicks = self.filter_out_of_bounds_clicks(
            clicks=target_clicks,
            img_shape=target_pred.shape,
        )
        if len(target_clicks) == 0:
            return []
        target_clicks.sort(key=lambda x: target_pred[x])
        top_n_clicks = target_clicks[-n_clicks:][::-1]
        return top_n_clicks
