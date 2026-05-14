# Copyright AstraZeneca 2026
"""Base class for click validity classifiers used in click propagation tasks."""

import torch


class PromptSelector:
    """A class for determining the validity of propagated click annotations.

    This class provides base functionality for validating clicks that have been
    propagated from a source image to a target image during registration. Its main
    purpose is to filter out invalid clicks that may result from the propagation
    process.

    The base implementation provides simple bounds checking to ensure clicks fall
    within the target image dimensions. Subclasses can extend this with more
    sophisticated validation logic.
    """

    @staticmethod
    def filter_out_of_bounds_clicks(
        clicks: list[tuple[int, int, int]],
        img_shape: tuple[int, int, int],
    ) -> list[tuple[int, int, int]]:
        """Filter out clicks that fall outside the image boundaries.

        Args:
            clicks: List of click coordinates as (x, y, z) tuples to validate.
            img_shape: Image dimensions as (x, y, z) against which
                to check the click coordinates.

        Returns:
            list[tuple[int, int, int]]: Filtered list containing only clicks that
                fall within the image boundaries.
        """
        new_clicks = []
        for click in clicks:
            if not any([x < 0 or x >= i for i, x in zip(img_shape, click)]):
                new_clicks.append(click)
        return new_clicks

    def get_best_clicks(
        self,
        target_clicks: list[tuple[int, int, int]],
        target_img: torch.Tensor,
        n_clicks: int,
        *args,
        **kwargs,
    ) -> list[tuple[int, int, int]]:
        """Select the most likely valid clicks from a list of candidates.

        In the base implementation, this simply returns the first n valid clicks
        that fall within the image boundaries. Subclasses may implement more
        sophisticated selection criteria.

        Args:
            target_clicks: List of candidate click coordinates as (x, y, z) tuples.
            target_img: A 5D tensor of shape (B, C, H, W, D).
            n_clicks: Number of best clicks to return.
            *args: Additional positional arguments for subclass implementations.
            **kwargs: Additional keyword arguments for subclass implementations.

        Returns:
            list[tuple[int, int, int]]: Up to n_clicks valid click coordinates,
                ordered by likelihood of being valid.
        """
        # For the basic prompt selector with no predictive capabilities,
        # the best click is any click that falls in bounds - first n clicks that are
        # in bounds will be returned.
        target_img_shape = tuple(target_img.shape[2:])
        valid_clicks = self.filter_out_of_bounds_clicks(clicks=target_clicks, img_shape=target_img_shape)
        return valid_clicks[:n_clicks]
