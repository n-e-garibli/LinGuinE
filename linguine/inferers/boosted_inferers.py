# Copyright AstraZeneca 2026

import numpy as np
import torch

from linguine.inferers.base_inferer import AbstractInferer
from linguine.utils.bounding_boxes import BBox3D, Point3D
from linguine.utils.misc import location_is_valid
from linguine.utils.pred_filtering import filter_prediction
from linguine.utils.sampling import find_mask_center


class BasicBoostedInferer(AbstractInferer):
    """Performs basic bootsted inference.

    This inferer will perform an initial inference with the input guidance
    and then resample a center click from the prediction. It will then
    reinfer the prediction using the resampled click as the fg guidance instead.
    The other guidance signals (bg clicks and bbox) will remain unchanged.
    """

    def __init__(self, base_inferer: AbstractInferer, *args, **kwargs):
        """Constructs the class.

        Args:
            base_inferer: An instantiated Inferer to use to obtain non boosted predictions.
        """
        self.base_inferer = base_inferer

    @property
    def spacing(self) -> tuple[float, float, float] | None:
        return self.base_inferer.spacing

    @property
    def roi(self) -> tuple[float, float, float] | None:
        if hasattr(self.base_inferer, "roi"):
            return self.base_inferer.roi
        return None

    def infer(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> np.ndarray | torch.Tensor:
        original_pred = self.base_inferer.infer(
            img=img,
            fg_clicks=fg_clicks,
            bg_clicks=bg_clicks,
            bboxes=bboxes,
            filter_pred=True,
            return_probs=False,
        )
        center = find_mask_center(original_pred)
        if len(center) == 0:
            return original_pred
        del original_pred
        new_fg_clicks = [center]
        new_pred = self.base_inferer.infer(img, new_fg_clicks, bg_clicks, bboxes, return_probs, filter_pred)
        return new_pred


class ResampleAdditiveBoostedInferer(BasicBoostedInferer):
    """Performs boosted inference with an inferer by adding a new fg click.

    This inferer will perform an initial inference with the input guidance
    and then resample a center click from the prediction. It will then
    reinfer the prediction using the resampled click as the fg guidance along
    with all the original guidance.
    """

    def infer(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> np.ndarray | torch.Tensor:
        original_pred = self.base_inferer.infer(
            img=img,
            fg_clicks=fg_clicks,
            bg_clicks=bg_clicks,
            bboxes=bboxes,
            filter_pred=True,
            return_probs=False,
        )
        center = find_mask_center(original_pred)
        if len(center) == 0:
            return original_pred
        new_fg_clicks = [center]
        new_fg_clicks.extend(fg_clicks)
        new_pred = self.base_inferer.infer(img, new_fg_clicks, bg_clicks, bboxes, return_probs, filter_pred)
        return new_pred


class MergeProbabilitiesBoostedInferer(BasicBoostedInferer):
    """Performs bootstrapped inference with an inferer by merging probabilities.

    This inferer will perform an initial inference with the input guidance
    and then resample a center click from the prediction.It will then
    reinfer the prediction using the resampled click as the fg guidance instead.
    The other guidance signals (bg clicks and bbox) will remain unchanged.
    Then the probability maps from the two inferences obtained are averaged and those
    voxels with probablity > 0.5 are taken to be foreground voxels in the final prediction.
    """

    def infer(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> np.ndarray | torch.Tensor:
        original_probs = super().infer(
            img=img,
            fg_clicks=fg_clicks,
            bg_clicks=bg_clicks,
            bboxes=bboxes,
            filter_pred=True,
            return_probs=False,
        )
        original_pred = (original_probs > 0.5).int()
        center = find_mask_center(original_pred)
        if len(center) == 0:
            return original_pred
        new_fg_clicks = [center]
        second_probs = self.base_inferer.infer(
            img,
            new_fg_clicks,
            bg_clicks,
            bboxes,
            return_probs=True,
            filter_pred=False,
        )
        merged_probs = (original_probs + second_probs) / 2
        if return_probs:
            return merged_probs

        pred = (merged_probs > 0.5).int()
        if len(fg_clicks) > 0 and filter_pred:
            # Ensure prediction only contains prediction for the relevant
            # lesion.
            pred, _ = filter_prediction(fg_clicks, pred, merge_bbox=False)
        return pred


class ClickEnsembleInferer(AbstractInferer):
    """
    Runs inference separately for each foreground click and returns an aggregated mask.

    - For each click in fg_clicks, call base_inferer.infer with that single click.
    - Aggregate across runs:
      * return_probs=False -> strict majority over binary predictions
      * return_probs=True  -> average probabilities across runs
    """

    def __init__(self, base_inferer: AbstractInferer):
        self.base_inferer = base_inferer

    @property
    def spacing(self) -> tuple[float, float, float] | None:
        return self.base_inferer.spacing

    @property
    def roi(self) -> tuple[float, float, float] | None:
        return getattr(self.base_inferer, "roi", None)

    def infer(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> np.ndarray | torch.Tensor:
        if not fg_clicks:
            return self.base_inferer.infer(
                img, [], bg_clicks, bboxes, return_probs=return_probs, filter_pred=filter_pred
            )

        # Run a prediction per click
        preds = []
        for click in fg_clicks:
            pred = self.base_inferer.infer(
                img=img,
                fg_clicks=[click],
                bg_clicks=bg_clicks,
                bboxes=bboxes,
                return_probs=return_probs,
                filter_pred=filter_pred,
            )
            preds.append(pred)

        first = preds[0]
        if isinstance(first, torch.Tensor):
            stacked = torch.stack(preds, dim=0)

            if return_probs:
                # Average probabilities
                probs = stacked.mean(dim=0).to(torch.float32)
                return probs

            counts = stacked.sum(dim=0)
            majority = (counts >= (len(preds) // 2 + 1)).to(torch.uint8)
            return majority

        else:
            stacked = np.stack(preds, axis=0)

            if return_probs:
                # Average probabilities
                probs = stacked.mean(dim=0).to(torch.float32)
                return probs

            counts = stacked.sum(axis=0)
            majority = (counts >= (len(preds) // 2 + 1)).astype(np.uint8)
            return majority


class PerturbationEnsembleInferer(ClickEnsembleInferer):
    """
    Performs inference for each perturbed click and returns the majority-vote mask.

    This inferer performs an initial inference with the input foreground guidance (from registration),
    then perturbs that click multiple times and re-infers for each perturbed click.
    The final prediction is the majority vote across all binary predictions.
    The other guidance signals (bg clicks and bbox) will remain unchanged.
    """

    def __init__(
        self,
        base_inferer: AbstractInferer,
        num_perturbations: int = 2,
        max_radius_mm: float = 5.0,
        distribution: str = "gaussian",  # "gaussian" or "uniform"
        include_original_click: bool = True,  # include the original click as one vote
    ):
        """
        Args:
            base_inferer: An instantiated Inferer for non-boosted predictions.
            num_perturbations: The number of times to perturb the foreground click.
                               The base_inferer will be called this many times (plus one
                               if include_seed is True) for each inference.
            max_radius_mm: The maximum radius in millimeters within which perturbations
                           will be sampled from the original foreground click.
            distribution: The statistical distribution to use for sampling perturbations.
                          Must be either "gaussian" (samples from a normal distribution
                          centered at the original click) or "uniform" (samples uniformly
                          within the max_radius_mm).
            include_original_click: If True, the original click is included as one vote.
        """
        super().__init__(base_inferer)
        self.num_perturbations = num_perturbations
        self.max_radius_mm = max_radius_mm
        self.distribution = distribution
        self.include_original_click = include_original_click

    def _sample_perturbations(self, click: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        pts = [list(click) for _ in range(self.num_perturbations)]

        for idx_click, s_ in enumerate(self.spacing):
            r_ = max(1, int(round(self.max_radius_mm / max(s_, 1e-6))))

            for idx_perturb in range(self.num_perturbations):
                if self.distribution == "gaussian":
                    d_ = int(np.round(np.random.normal(0, r_)))
                else:
                    d_ = np.random.randint(-r_, r_ + 1)

                # Adjust the specific axis of the specific perturbation point
                pts[idx_perturb][idx_click] = pts[idx_perturb][idx_click] + d_

        # Convert each perturbed point to a tuple
        out = [tuple(p) for p in pts]

        # Optionally include the original click
        if self.include_original_click:
            out.append(tuple(click))

        return out

    def infer(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> np.ndarray | torch.Tensor:
        if not fg_clicks:
            return self.base_inferer.infer(
                img, [], bg_clicks, bboxes, return_probs=return_probs, filter_pred=filter_pred
            )

        original_click = fg_clicks[0]  # Assume first fg click is the one to perturb

        candidates = self._sample_perturbations(original_click)

        # Validate clicks against image bounds
        valid_clicks = [pc for pc in candidates if location_is_valid(pc, img)]

        # Fallbacks if nothing valid
        if not valid_clicks:
            if location_is_valid(original_click, img):
                valid_clicks = [original_click]
            else:
                # No valid fg clicks at all; delegate with empty fg list
                return self.base_inferer.infer(
                    img, [], bg_clicks, bboxes, return_probs=return_probs, filter_pred=filter_pred
                )
        return super().infer(
            img=img,
            fg_clicks=valid_clicks,
            bg_clicks=bg_clicks,
            bboxes=bboxes,
            return_probs=return_probs,
            filter_pred=filter_pred,
        )


class OrientationEnsembleInferer(AbstractInferer):
    """Ensemble inferer that aggregates predictions across multiple orientations.

    Performs inference on images shaped [N, C, X, Y, Z] by:
    - Running the base inferer on rotations of the input (rot90 with k in {0, 1, 2} in the XY plane)
    - Aggregating results via mean (for probabilities) or majority vote (for masks)
    """

    def __init__(self, base_inferer, transforms: list[int] | None = None):
        """
        Args:
            base_inferer: An instantiated Inferer to use for obtaining predictions.
            transforms: Optional list of transform parameters to apply.
                        If None, defaults to rotations of k=0 - 0°, k=1 - 90°, and k=2 - 180° in the XY plane.
        """
        self.base_inferer = base_inferer
        # Rotations k=0,1,2
        self.transforms = transforms or [0, 1, 2]

    @property
    def spacing(self) -> tuple[float, float, float] | None:
        return getattr(self.base_inferer, "spacing", None)

    @property
    def roi(self) -> tuple[float, float, float] | None:
        return getattr(self.base_inferer, "roi", None)

    def _apply_tensor_transform(self, input_tensor: torch.Tensor, k: int) -> torch.Tensor:
        """
        Rotates the input tensor in the XY plane by k * 90 degrees.
        Args:
            input_tensor: A tensor with spatial dimensions at the end (..., X, Y, Z).
            k: Number of 90° rotations to apply. Normalized modulo 4 is used to ensure k is always in {0,1,2,3}.

        Returns:
            The rotated tensor.
        """
        k = k % 4
        # Rotate XY -> dims (-3, -2) because input_tensor[..., X, Y, Z]
        transformed_tensor = torch.rot90(input_tensor, k=k, dims=(-3, -2))
        return transformed_tensor

    def _invert_tensor_transform(self, input_tensor: torch.Tensor, k: int) -> torch.Tensor:
        """
        Inverts a prior XY-plane rotation applied by rot90.
        Args:
            input_tensor: A tensor with spatial dimensions at the end (..., X, Y, Z).
            k: Number of k rotations to apply to invert a prior XY-plane rotation applied by rot90

            Returns:
            The rotated tensor.
        """
        k_inv = (-k) % 4
        transformed_tensor = torch.rot90(input_tensor, k=k_inv, dims=(-3, -2))
        return transformed_tensor

    def _apply_click_transform(
        self, click: tuple[int, int, int], shape_xyz: tuple[int, int, int], k: int
    ) -> tuple[int, int, int]:
        """
        Rotates the guidance click in the XY plane by k * 90 degrees.
        Args:
            click: The original 3D coordinate (x, y, z) of the guidance click
            shape_xyz: shape of the image
            k: Number of 90° rotations to apply. Normalized modulo 4 is used to ensure k is always in {0,1,2,3}.

            Returns:
            The rotated 3D coordinate of the guidance click
        """
        x, y, z = click
        X, Y, Z = shape_xyz

        k = k % 4
        if k == 1:
            x, y = (Y - 1 - y), x
        elif k == 2:
            x, y = (X - 1 - x), (Y - 1 - y)
        elif k == 3:
            x, y = y, (X - 1 - x)
        return (x, y, z)

    def _apply_bbox_transform(self, b: BBox3D, shape_xyz: tuple[int, int, int], k: int) -> BBox3D:
        """
        Rotates a 3D bounding box in the XY plane by k * 90 degrees.
        Args:
            b: The input bounding box with min/max coordinates
            shape_xyz: shape of the image
            k: Number of 90° rotations to apply.

            Returns:
            The rotated bounding box
        """
        # Gather corners from the BBox3D properties
        corners = [
            (b.x_min, b.y_min, b.z_min),
            (b.x_min, b.y_min, b.z_max),
            (b.x_min, b.y_max, b.z_min),
            (b.x_min, b.y_max, b.z_max),
            (b.x_max, b.y_min, b.z_min),
            (b.x_max, b.y_min, b.z_max),
            (b.x_max, b.y_max, b.z_min),
            (b.x_max, b.y_max, b.z_max),
        ]

        # Apply point-wise transform to each corner
        transformed = [self._apply_click_transform(c, shape_xyz, k) for c in corners]

        # Split into coordinate lists
        xs, ys, zs = zip(*transformed)

        # Build new min/max points
        min_point = Point3D(min(xs), min(ys), min(zs))
        max_point = Point3D(max(xs), max(ys), max(zs))

        return BBox3D(min_point=min_point, max_point=max_point)

    def _ensure_float_probs(self, pred) -> torch.Tensor | np.ndarray:
        """
        Converts to float32 probabilities.
        Args:
        pred: The prediction array/tensor to convert.

        Returns:
        The input prediction with dtype float32 (torch.Tensor or numpy.ndarray).
        """
        # Convert to float32 probabilities
        if isinstance(pred, torch.Tensor):
            return pred.to(torch.float32)
        elif isinstance(pred, np.ndarray):
            return pred.astype(np.float32)
        else:
            raise TypeError(f"Unsupported prediction type: {type(pred)}")

    def _ensure_uint8_mask(self, pred) -> torch.Tensor | np.ndarray:
        """
        Converts to uint8 masks.
        Args:
        pred: The prediction array/tensor to convert.

        Returns:
        The input prediction with dtype uint8 (torch.Tensor or numpy.ndarray).
        """
        # Convert to uint8 masks
        if isinstance(pred, torch.Tensor):
            return pred.to(torch.uint8)
        elif isinstance(pred, np.ndarray):
            return pred.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported prediction type: {type(pred)}")

    def _stack_preds(self, preds) -> torch.Tensor | np.ndarray:
        """
        Stacks list of same-type tensors/arrays.
        Args:
        preds: A list of predictions of the same type (torch or numpy).

        Returns:
        A stacked torch.Tensor or numpy.ndarray with a new leading dimension (number of ensebmble elements).
        """
        # Stack list of same-type tensors/arrays
        first = preds[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(preds, dim=0)
        elif isinstance(first, np.ndarray):
            return np.stack(preds, axis=0)
        else:
            raise TypeError(f"Unsupported prediction type: {type(first)}")

    def _mean_aggregate(self, preds) -> torch.Tensor | np.ndarray:
        """
        Aggregates ensemble predictions by averaging.
        Args:
        preds: A list of predictions.

        Returns:
        The mean-aggregated prediction.
        """
        stacked = self._stack_preds(preds)
        if isinstance(stacked, torch.Tensor):
            return stacked.mean(dim=0)
        else:
            return stacked.mean(axis=0)

    def _majority_vote(self, preds) -> torch.Tensor | np.ndarray:
        """
        Aggregates ensemble binary predictions via majority vote.
        Args:
        preds: A list of binary predictions.

        Returns:
        A uint8 mask of the majority vote.
        """
        stacked = self._stack_preds(preds)
        if isinstance(stacked, torch.Tensor):
            votes = stacked.sum(dim=0)
            # Threshold at > E/2
            return (votes > (stacked.shape[0] // 2)).to(torch.uint8)
        else:
            votes = stacked.sum(axis=0)
            return (votes > (stacked.shape[0] // 2)).astype(np.uint8)

    def infer(
        self,
        img: torch.Tensor,  # expected shape [N, C, X, Y, Z] or [..., X, Y, Z] at end
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Generates an ensemble prediction across multiple orientations.

        Args:
            img: a 5D tensor containing the image to infer on.
            fg_clicks: a list of foreground guidance clicks to use (can be empty).
            bg_clicks: a list of background guidance clicks to use (can be empty).
            bboxes: a list of bounding boxes to use as guidance (can be empty).
            return_probs: whether to return foreground class probabilities instead of a binary segmentation.
            filter_pred: whether to filter the prediction to only keep the connected component.

        Returns:
        A numpy array or torch tensor with the aggregated prediction.
        """
        # Spatial dims are (X, Y, Z) at the end
        shape_xyz = (img.shape[-3], img.shape[-2], img.shape[-1])

        inverted_preds = []

        for k in self.transforms:
            # Transform image and guidance
            t_img = self._apply_tensor_transform(img, k)
            t_fg = [self._apply_click_transform(c, shape_xyz, k) for c in fg_clicks]
            t_bg = [self._apply_click_transform(c, shape_xyz, k) for c in bg_clicks]
            t_boxes = [self._apply_bbox_transform(b, shape_xyz, k) for b in bboxes]

            # Infer in transformed space
            pred_t = self.base_inferer.infer(
                img=t_img,
                fg_clicks=t_fg,
                bg_clicks=t_bg,
                bboxes=t_boxes,
                return_probs=return_probs,
                filter_pred=filter_pred,
            )

            # Invert orientation back to original
            if isinstance(pred_t, torch.Tensor):
                pred_o = self._invert_tensor_transform(pred_t, k)
            elif isinstance(pred_t, np.ndarray):
                k_inv = (-k) % 4
                pred_o = np.rot90(pred_t, k=k_inv, axes=(-3, -2)) if k_inv else pred_t
            else:
                raise TypeError(f"Unsupported prediction type from base_inferer: {type(pred_t)}")

            inverted_preds.append(pred_o)

        # Aggregate
        if return_probs:
            # Convert all to float32 and mean-aggregate
            converted = [self._ensure_float_probs(p) for p in inverted_preds]
            agg = self._mean_aggregate(converted)
            return agg  # float32 torch.Tensor or np.ndarray
        else:
            # Convert to uint8, majority vote
            converted = [self._ensure_uint8_mask(p) for p in inverted_preds]
            agg = self._majority_vote(converted)
            return agg  # uint8 torch.Tensor or np.ndarray
