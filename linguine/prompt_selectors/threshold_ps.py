# Copyright AstraZeneca 2026
"""Implementation of a threshold-based prompt selector.

This module provides a PS implementation that determines prompt validity based on
the image value of the target voxel location in medical imaging data."""

from typing import Any

import numpy as np
import torch

from linguine.prompt_selectors.base_ps import PromptSelector


class ThresholdPS(PromptSelector):
    """A prompt selector based on voxel intensity thresholding.

    This selector validates clicks by comparing the attenuation value of the
    target voxel against dual intensity thresholds. Clicks are considered valid
    if the voxel intensity at the click location falls within the specified
    range [l_threshold, u_threshold]. This can help exclude clicks in
    very dark regions (e.g., air, background) and very bright regions
    (e.g., bone, contrast agent).

    It is recommended to choose a range that is appropriate for the object being
    segmented.
    """

    def __init__(
        self,
        l_threshold: float | int = -650,
        u_threshold: float | int = 350,
        seed: int = 42,
    ):
        """Constructor.

        Args:
            l_threshold: The attenuation threshold above which a voxel is considered
                foreground.
            u_threshold: The upper attenuation threshold for foreground voxels.
            seed: a random seed for reproducible results.
        """
        self.l_threshold = l_threshold
        self.u_threshold = u_threshold
        assert l_threshold < u_threshold, "Invalid thresholds: l_threshold must be less than u_threshold."
        self.rng = np.random.default_rng(seed)

    def _filter_invalid_clicks(
        self,
        target_clicks: list[tuple[int, int, int]],
        target_img: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> list[tuple[int, int, int]]:
        """Filter clicks based on voxel intensity thresholds.

        Validates clicks by checking if the image value at each click
        location falls within the valid range [l_threshold, u_threshold].
        Clicks are also filtered for being within image bounds.

        Args:
            target_clicks: List of click coordinates as (x, y, z) tuples to validate.
            target_img: Optional 5D tensor containing the image data. If None, the
                image will be loaded using the image_loader.
            *args: Additional positional arguments for future extensions and
                compatibility with other prompt selectors.
            **kwargs: Additional keyword arguments for future extensions and
                compatibility with other prompt selectors.

        Returns:
            list[tuple[int, int, int]]: Filtered list of click coordinates that
                fall within image bounds and have values within the
                range [l_threshold, u_threshold].
        """
        assert target_img.ndim == 5, f"Expect 5D image, got {target_img.ndim}"
        target_img = target_img[0, 0]
        target_clicks = self.filter_out_of_bounds_clicks(clicks=target_clicks, img_shape=target_img.shape)
        valid_clicks: list[tuple[int, int, int]] = []
        for click in target_clicks:
            val = target_img[click]
            if self.u_threshold >= val >= self.l_threshold:
                valid_clicks.append(click)
        return valid_clicks

    def get_best_clicks(
        self,
        target_clicks: list[tuple[int, int, int]],
        target_img: torch.Tensor,
        target_dict: dict[str, Any] | None = None,
        n_clicks: int = 5,
        *args,
        **kwargs,
    ) -> list[tuple[int, int, int]]:
        """Randomly select a subset of valid clicks from the input list.

        Filters the input clicks using the threshold criterion and returns a
        random sample of the valid clicks.

        Args:
            target_clicks: List of click coordinates as (x, y, z) tuples.
            target_dict: Optional dictionary containing image metadata and path
                information.
            target_img: Tensor containing the image data.
            n_clicks: Maximum number of clicks to return. Defaults to 5.
            *args: Additional positional arguments for future extensions and
                compatibility with other prompt selectors.
            **kwargs: Additional keyword arguments for future extensions and
                compatibility with other prompt selectors.

        Returns:
            list[tuple[int, int, int]]: Up to n_clicks randomly selected valid
                click coordinates.
        """
        target_clicks = self.filter_out_of_bounds_clicks(clicks=target_clicks, img_shape=target_img.shape[2:])
        n_clicks = min(len(target_clicks), n_clicks)
        valid_clicks = self._filter_invalid_clicks(
            target_clicks=target_clicks,
            target_dict=target_dict,
            target_img=target_img,
        )
        if len(valid_clicks) < n_clicks:
            # We'll just have to sample a few invalid ones randomly too.
            invalid_clicks = set(target_clicks) - set(valid_clicks)
            chosen = self.rng.choice(invalid_clicks, n_clicks - len(valid_clicks))
            chosen = [tuple(x) for x in chosen]
            return valid_clicks + chosen

        # Randomly sample clicks inside the valid range to return.
        return [tuple(x) for x in self.rng.choice(valid_clicks, n_clicks)]
