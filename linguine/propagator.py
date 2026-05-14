# Copyright AstraZeneca 2026
"""Contains a propagator objects for running Linguine. These classes are responsible for propagations between a pair of scans."""

import logging
from typing import Any

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Compose, Resized, Spacingd

from linguine.config import LinguineConfig
from linguine.constants import LABEL
from linguine.disappearance_detectors import DisappearanceDetector
from linguine.inferers.base_inferer import AbstractInferer
from linguine.metrics import MetricsBundle, dice
from linguine.prompt_selectors import PromptSelector
from linguine.prompt_selectors.metrics import ClickValidityMetrics
from linguine.registration.point_extractors import PointExtractor
from linguine.registration.registrators import ImageRegistrator, PointSetRegistrator
from linguine.utils.bounding_boxes import BBox2D, BBox3D, EmptyBBox, Point3D
from linguine.utils.data import (
    crop_to_same_size,
    get_spacing_from_metatensor,
    prepare_inputs_for_inferer,
)
from linguine.utils.misc import filter_invalid_prompts
from linguine.utils.sampling import find_mask_center, sample_clicks

LOGGER = logging.getLogger(__name__)


class LinguineClickPropagator:
    """
    This class can be used to propagate a guidance click between two scans using registration
    and evaluate model performance.
    """

    def __init__(
        self,
        cfg: LinguineConfig,
        inferer: AbstractInferer,
        registrator: PointSetRegistrator | ImageRegistrator,
        prompt_selector: PromptSelector | None = None,
        disappearance_detector: DisappearanceDetector | None = None,
        point_extractor: PointExtractor | None = None,
    ):
        """Constructor.

        Args:
            cfg: The configuration of the linguine experiment to run.
            inferer: The inferer object that will be used for inference.
            registrator: The registrator object that will be used for registering the images.
            prompt_selector: An optional prompt selector to sample optimal prompts from the source label.
                If not provided, the center of the mask will always be propagated.
            disappearance_detector: An optional disappearance detector which will determine, after a propagation,
                whether the predicted tumour is actually still present in the image/is measurable.
            point_extractor: The point extractor object that will extract landmarks used in registration.
                only needed if the registrator used is a PointSetRegistrator.
        """
        self.cfg = cfg
        self.point_extractor = point_extractor
        self.registrator = registrator
        if isinstance(registrator, PointSetRegistrator):
            assert point_extractor is not None, "Point set based registration requires a point extractor."
        self.inferer = inferer
        self.prompt_selector = prompt_selector
        # This will store/compute metrics on how good the click propagation is (do the clicks land inside the tumour?)
        self.cv_metrics = ClickValidityMetrics()
        if disappearance_detector is None:
            # Initialize base class - this always returns "True" when considering whether tumour is there.
            # Protip: Encode RECIST data if you have some into your target image metatensor and
            # define a lovely DD that looks at that :)
            self.disappearance_detector = DisappearanceDetector()
        else:
            self.disappearance_detector = disappearance_detector

        self.rng = np.random.default_rng(cfg.analysis.seed)

    def __call__(
        self,
        target_img: MetaTensor,
        source_prompts: list[tuple[int, int, int]],
        source_label: MetaTensor,
        source_dict: dict[str, Any],
        target_dict: dict[str, Any],
        target_label: MetaTensor | None = None,
        source_img: MetaTensor | None = None,
    ) -> tuple[
        MetricsBundle | None,
        list[tuple[int, int, int]],
        MetaTensor | None,
    ]:
        """Propagates a click and obtains the metrics bundle for a model prediction
        obtained when the propagated click is used as guidance.

        self.cv_metrics will be updated every time this is called if self.cfg.predict_only
        is false.

        Args:
            target_img: 5D MetaTensor containing the target image (to infer on).
            source_prompts: the guidance clicks sampled from the source image
                that need to be propagated.
            source_label: 5D MetaTensor containing the source label for the lesion
                being considered. Will be used to sample clicks with a prompt selector.
                If not using a prompt selector, just pass an empty metatensor.
                Note that when the propagator is called within study segmentor objects
                the source label can be a model prediction from prompts (arg above) in the
                scenario where no ground truth mask is available for the source scan.
            source_dict: the data dictionary for the source scan.
                This isn't inherently necessary, but can be useful for custom objects
                or for identifying landmarks to use in a PointSetRegistrator. - pass {}
                if your linguine is vanilla.
            target_dict: the data dictionary for the target scan.
                This isn't inherently necessary, but can be useful for custom objects
                or for identifying landmarks to use in a PointSetRegistrator. - pass {}
                if your linguine is vanilla.
            target_label: optional 5D MetaTensor containing the target label.
                Will be used to compute metrics if provided.
            source_img: 5D MetaTensor containing the source image. Only needed
                for ImageRegistrators.

        Returns:
            The metrics bundle for the obtained prediction (None if propagation failed)
            The transformed clicks in the space of the target image.
            The prediction obtained with the transformed clicks as guidance (None if propagation failed)
        """
        target_clicks = self._prepare_target_clicks(
            source_dict=source_dict,
            target_dict=target_dict,
            source_clicks=source_prompts,
            source_img=source_img,
            source_label=source_label,
            target_img=target_img,
            target_label=target_label,
        )
        if len(target_clicks) == 0:
            # We can invoke the inferer unguided but not all inferers will support this
            # and we will have no way of identifying the pred for only the
            # lesion considered. Consider this a failed propagation - or, more optimistically,
            # the tumour is not in the image FOV!
            return None, [], None

        # Find dice score on target image with propagated clicks
        target_metrics, pred = self.get_metrics_and_prediction(
            img=target_img, label=target_label, fg_clicks=target_clicks, bboxes=[]
        )
        return target_metrics, target_clicks, pred

    def _prepare_target_clicks(
        self,
        source_dict: dict[str, Any],
        source_clicks: list[tuple[int, int, int]],
        source_label: MetaTensor,
        source_img: MetaTensor | None,
        target_dict: dict[str, Any],
        target_img: MetaTensor,
        target_label: torch.Tensor | None,
    ) -> list[tuple[int, int, int]]:
        """Determine what clicks are to be provided to the target image.

        Args:
            source_dict: the data dictionary for the source scan.
            source_clicks: the guidance clicks sampled from the source image
                that need to be propagated.
            source_label: 5D MetaTensor containing the source label for the lesion
                being considered. Will be used to resample clicks if prompt selection is enabled.
            source_img: 5D MetaTensor containing the source image. Only used for
                registration that take both images as input.
            target_dict: the data dictionary for the target scan.
            target_img: 5D Metatensor containing the target image.
            target_label: 5D Metatensor containing the target label.

        Returns:
            A list of guidance click to provide in the target image.
        """
        if self.cfg.registration.registrator == "perfect":
            self.cv_metrics.update_perfect()
            center = find_mask_center(target_label)
            if center == ():
                return []
            return [center]
        target_clicks: list[tuple[int, int, int]] = []
        if self.prompt_selector is None:
            # Propagate source clicks into the space of the target
            target_clicks = self._propagate_clicks(
                source_dict=source_dict,
                target_dict=target_dict,
                source_clicks=source_clicks,
                source_image=source_img,
                target_image=target_img,
                source_spacing=get_spacing_from_metatensor(source_label),
                target_spacing=get_spacing_from_metatensor(target_img),
            )
            LOGGER.info(f"Foreground clicks transformed from {source_clicks} to {target_clicks}")
            # Filter out OOB fg clicks! This will be done before inference too but just to be safe.
            target_clicks, _, _ = filter_invalid_prompts(
                img=target_img[0][0],
                fg_clicks=target_clicks,
                bg_clicks=[],
                bboxes=[],
            )
            if not self.cfg.predict_only:
                # Update metrics to reflect propagation with new clicks.
                self.cv_metrics.update(
                    clicks=target_clicks,
                    target_label=target_label,
                )
        else:
            # Below should've been caught at config init
            assert self.cfg.prompt_selection.mask_sampling.num_samples > 0
            target_clicks = self._find_best_clicks(
                source_dict=source_dict,
                target_dict=target_dict,
                source_label=source_label,
                target_img=target_img,
                source_img=source_img,
                target_spacing=get_spacing_from_metatensor(target_img),
                n_clicks=self.cfg.prompt_selection.n_clicks,
            )
            if len(target_clicks) == 0:
                LOGGER.warning(
                    f"Failed to sample any clicks in {self.cfg.prompt_selection.mask_sampling.num_samples} attempts. Not propagating anything."
                )
                return []
            LOGGER.info(f"After prompt selection, foreground clicks: {target_clicks} were selected.")
            if not self.cfg.predict_only:
                # Update metrics to reflect propagation with new clicks.
                self.cv_metrics.update(
                    clicks=target_clicks,
                    target_label=target_label,
                )

        return target_clicks

    def _find_best_clicks(
        self,
        source_dict: dict[str, Any],
        target_dict: dict[str, Any],
        source_label: MetaTensor,
        source_img: MetaTensor | None,
        target_img: MetaTensor,
        target_spacing: tuple[float, float, float],
        n_clicks: int = 1,
    ) -> list[tuple[int, int, int]]:
        """Look for optimal guidance click locations (according to prompt refiner) in the target image.

        Args:
            source_dict: the data dictionary for the source scan.
            target_dict: the data dictionary for the target scan.
            source_label: 5D MetaTensor containing the source label for the lesion
                being considered.
            target_img: 5D MetaTensor containing the target image.
            target_spacing: The spacing of the target image as (x, y, z) tuple.
            n_clicks: The number of clicks to find. Defaults to 1.

        Returns:
            A list of the valid guidance clicks in the form (x,y,z).
        """
        source_spacing = get_spacing_from_metatensor(source_label)
        new_source_clicks = sample_clicks(
            binary_array=source_label[0][0],
            config=self.cfg.prompt_selection.mask_sampling,
            spacing=source_spacing,
        )
        new_target_clicks = self._propagate_clicks(
            source_dict=source_dict,
            target_dict=target_dict,
            source_clicks=new_source_clicks,
            source_spacing=source_spacing,
            target_spacing=target_spacing,
            source_image=source_img,
            target_image=target_img,
        )
        return self.prompt_selector.get_best_clicks(
            target_clicks=new_target_clicks,
            source_dict=source_dict,
            target_dict=target_dict,
            target_img=target_img,
            source_clicks=new_source_clicks,
            n_clicks=n_clicks,
        )

    def get_metrics_and_prediction(
        self,
        img: MetaTensor,
        label: torch.Tensor | None,
        fg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D | BBox2D] | None = None,
    ) -> tuple[MetricsBundle | None, MetaTensor | None]:
        """Performs model inference and returns the metrics.

        Args:
            img: A 5D metatensor containing the image
            label: A 5D tensor containing the label for the image. If None,
                returned metrics will be nans.
            fg_clicks: Guidance clicks to be used as prompts to the model.
            bboxes: A list of bboxes to use as guidance.

        Returns:
            The metrics of the model prediction compared with the label.
            The prediction obtained with model inference in the form of
                a 5D binary metatensor. Resampled to match original image."""

        # Apply spacing transform as expected by the inferer
        transformed_img, fg_clicks, bboxes = prepare_inputs_for_inferer(
            desired_spacing=self.inferer.spacing, image=img, clicks=fg_clicks, bboxes=bboxes
        )
        LOGGER.info(f"Resampled clicks to match model expected spacing: {fg_clicks}")
        if bboxes and len(bboxes) > 0:
            LOGGER.info(f"Transformed bboxes to match model expected spacing: {bboxes}")

        fg_clicks, _, bboxes = filter_invalid_prompts(
            img=transformed_img[0][0],
            fg_clicks=fg_clicks,
            bg_clicks=[],
            bboxes=bboxes,
        )
        LOGGER.info("Began model inference on target scan...")
        pred = self.inferer.infer(
            img=transformed_img,
            fg_clicks=fg_clicks,
            bg_clicks=[],
            bboxes=bboxes,
        )

        # Get confidence scores for the clicks used
        metrics = MetricsBundle()
        if self.cfg.analysis.num_perturbations != 0:
            if len(fg_clicks) == 0:
                metrics.confidence = np.nan
            else:
                metrics.confidence = self._get_confidence(
                    fg_clicks=fg_clicks,
                    bboxes=bboxes,
                    transformed_img=transformed_img,
                    pred=pred,
                )

        pred_4D_metatensor = MetaTensor(pred, affine=transformed_img.affine).unsqueeze(0)
        del pred
        del transformed_img
        pred = self._resample_pred(
            img_spacing=get_spacing_from_metatensor(img), img_shape=img.shape[2:], pred=pred_4D_metatensor
        )
        del pred_4D_metatensor

        tumour_vanished = not self.disappearance_detector.tumour_present(
            lesion_pred=pred,
            target_image=img,
        )
        if tumour_vanished:
            pred = pred.zero_()

        if label is not None:
            metrics_pred = pred[0].cpu().numpy()
            metrics_label = label[0][0].cpu().numpy()
            metrics_spacing = get_spacing_from_metatensor(img)
            # Default to isotropic 1mm spacing if spacing is None
            if metrics_spacing is None:
                metrics_spacing = (1.0, 1.0, 1.0)
                LOGGER.warning("Could not determine image spacing, defaulting to (1.0, 1.0, 1.0) mm")
            metrics_pred, metrics_label = crop_to_same_size([metrics_pred, metrics_label])
            # Pass spacing to compute_metrics to calculate distances in mm
            metrics.compute_metrics(mask=metrics_pred, label=metrics_label, spacing=metrics_spacing)
        pred = pred.to(img.device).unsqueeze(0)
        return metrics, pred

    def _resample_pred(
        self, img_spacing: tuple[float, float, float], img_shape: tuple[int, ...], pred: MetaTensor
    ) -> MetaTensor:
        """Resample prediction to match original image spacing and shape.

        Args:
            img_spacing: The spacing of the original image as (x, y, z) tuple.
            img_shape: The spatial shape of the original image.
            pred: The 4D prediction tensor to be resampled (1, x, y, z).

        Returns:
            The resampled prediction tensor matching original image spacing and shape.
        """
        inverse_spacing_transform = Spacingd(
            pixdim=img_spacing,
            keys=[LABEL],
            mode=["nearest"],
        )
        resize_transform = Resized(
            keys=[LABEL],
            spatial_size=img_shape,
            mode=["nearest"],
        )
        pred_transforms = Compose([inverse_spacing_transform, resize_transform])
        return pred_transforms({LABEL: pred})[LABEL]

    def _propagate_clicks(
        self,
        source_dict: dict[str, Any],
        target_dict: dict[str, Any],
        source_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float],
        source_image: MetaTensor | None,
        target_image: MetaTensor | None,
        source_clicks: list[tuple[int, int, int]],
    ) -> list[tuple[int, int, int]]:
        """This function propagates input clicks from the space of the source scan into
        the space of the target.

        Args:
            source_dict: the data dictionary for the source scan.
            target_dict: the data dictionary for the target scan.
            source_prompts: the guidance clicks sampled from the source image
                that need to be propagated.

        Returns:
            The clicks transformed into the space of the target scan.
        """
        if len(source_clicks) == 0:
            return []
        LOGGER.info("Began performing registration...")
        if isinstance(self.registrator, PointSetRegistrator):
            source_landmarks = self.point_extractor.extract_points(source_dict)
            target_landmarks = self.point_extractor.extract_points(target_dict)
            try:
                target_clicks = self.registrator.map_coordinates(
                    source_spacing=source_spacing,
                    target_spacing=target_spacing,
                    source_landmarks=source_landmarks,
                    target_landmarks=target_landmarks,
                    coors=source_clicks,
                )
            except Exception as e:
                LOGGER.error(f"Registration failed: {e}")
                return []
        elif isinstance(self.registrator, ImageRegistrator):
            target_clicks = self.registrator.map_coordinates(
                source_image=source_image,
                target_image=target_image,
                coors=source_clicks,
                # Passing in as kwargs for any custom registrators that may
                # need to do something fancy. For example the GradIcon lung model
                # can grab lung masks from here.
                source_dict=source_dict,
                target_dict=target_dict,
            )
        return target_clicks

    def _get_confidence(
        self,
        fg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        transformed_img: MetaTensor,
        pred: np.ndarray,
    ) -> float:
        """
        Calculate confidence score for the provided clicks based on perturbations of the propagated click.
        If no fg clicks are provided or are filtered out, returns np.nan.

        Args:
            fg_clicks: List of foreground clicks used for inference.
            bboxes: List of bounding boxes used for inference.
            transformed_img: The image after spacing transform for inference.
            pred: The prediction obtained from the model.
        """
        perturbed_dices = []
        for _ in range(self.cfg.analysis.num_perturbations):
            perturbed_clicks = fg_clicks.copy()
            for idx in range(len(perturbed_clicks)):
                perturbed_click = list(perturbed_clicks[idx])
                perturbed_click[0] += self.rng.choice([-1, 1])
                perturbed_click[1] += self.rng.choice([-1, 1])
                perturbed_click[2] += self.rng.choice([-1, 1])
                perturbed_clicks[idx] = tuple(perturbed_click)

            perturbed_clicks, _, _ = filter_invalid_prompts(
                img=transformed_img[0][0],
                fg_clicks=perturbed_clicks,
                bg_clicks=[],
                bboxes=bboxes,
            )
            if len(perturbed_clicks) == 0:
                continue
            perturbed_pred = self.inferer.infer(
                img=transformed_img,
                fg_clicks=perturbed_clicks,
                bboxes=bboxes,
                bg_clicks=[],
            )

            perturbed_dices.append(dice(pred, perturbed_pred))

        if len(perturbed_dices) == 0:
            return np.nan

        return np.mean(perturbed_dices)


class LinguineBboxPropagator(LinguineClickPropagator):
    """
    This class propagates bounding boxes between two scans using registration
    and evaluates model performance using bbox guidance.

    Inherits from LinguineClickPropagator but overrides the __call__ method
    to propagate bounding box corners instead of clicks.
    """

    def __init__(
        self,
        cfg: LinguineConfig,
        inferer: AbstractInferer,
        registrator: PointSetRegistrator | ImageRegistrator,
        point_extractor: PointExtractor | None,
        bbox_2d: bool = False,
        prompt_selector: PromptSelector | None = None,
        disappearance_detector: DisappearanceDetector | None = None,
    ):
        """
        Args:
            cfg: The configuration of the linguine to be run.
            inferer: The inferer object that will be used for inference.
            registrator: The registrator object that will be used for registering the images.
            point_extractor: The point extractor object that will extract landmarks used in registration.
                only needed if the registrator used is a PointSetRegistrator.
            click_classifier: A click classifier to determine which propagated clicks are valid.
            logger: A logger to write logs to.
            bbox_2d: Whether to propagate 2D bounding boxes instead of 3D bounding boxes.
            disappearance_detector: An optional disappearance detector which will determine, after a propagation,
                whether the predicted tumour is actually still present in the image/is measurable.
        """
        super().__init__(
            cfg=cfg,
            inferer=inferer,
            registrator=registrator,
            point_extractor=point_extractor,
            prompt_selector=prompt_selector,
            disappearance_detector=disappearance_detector,
        )
        self.bbox_2d = bbox_2d

    def __call__(
        self,
        source_dict: dict[str, Any],
        target_dict: dict[str, Any],
        target_img: MetaTensor,
        target_label: MetaTensor | None,
        source_prompts: list[BBox3D | BBox2D],
        source_label: MetaTensor,
        source_img: MetaTensor | None = None,
    ) -> tuple[
        MetricsBundle | None,
        list[BBox3D | BBox2D],
        MetaTensor | None,
    ]:
        """Propagates a bounding box and obtains the metrics bundle for a model prediction
        obtained when the propagated bbox is used as guidance.

        Args:
            source_dict: the data dictionary for the source scan.
            target_dict: the data dictionary for the target scan.
            target_img: 5D MetaTensor containing the target image.
            target_label: optional 5D MetaTensor containing the target label.
                Will be used to compute metrics if provided.
            source_prompts: the bounding boxes sampled from the source image
            source_label: 5D MetaTensor containing the source label for the lesion
                being considered.

        Returns:
            The metrics bundle for the obtained prediction.
            The transformed bounding boxes in the space of the target image.
            The prediction obtained with the bounding box as guidance.
        """
        if len(source_prompts) > 1:
            raise NotImplementedError("Only single bbox propagation per lesion supported.")
        source_bbox = source_prompts[0]

        if isinstance(source_bbox, EmptyBBox):
            LOGGER.warning("Bounding box empty, not propagating.")
            return None, [], None

        # Propagate bbox corners to target space
        target_bbox = self._propagate_bbox(
            source_dict=source_dict,
            target_dict=target_dict,
            source_bbox=source_bbox,
            source_spacing=get_spacing_from_metatensor(source_label),
            target_spacing=get_spacing_from_metatensor(target_img),
            source_image=source_img,
            target_image=target_img,
        )

        if target_bbox is None:
            LOGGER.warning("Failed to propagate bounding box to target space.")
            return None, [], None

        LOGGER.info(f"Propagated bbox from {source_bbox.to_bounds()} to {target_bbox.to_bounds()}")
        if not self.cfg.predict_only:
            # Validity metrics will be computed as if the center of the bbox was a click.
            self.cv_metrics.update_bbox(target_bbox=target_bbox, target_label=target_label)

        # Perform inference with the propagated bounding box
        target_metrics, pred = self.get_metrics_and_prediction(
            img=target_img,
            label=target_label,
            fg_clicks=[],  # No clicks used for bbox propagation
            bboxes=[target_bbox],
        )
        return target_metrics, [target_bbox], pred

    def _propagate_bbox(
        self,
        source_dict: dict[str, Any],
        target_dict: dict[str, Any],
        source_bbox: BBox3D | BBox2D,
        source_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float],
        source_image: MetaTensor | None,
        target_image: MetaTensor | None,
    ) -> BBox3D | BBox2D | None:
        """Propagate a bounding box from source to target space.

        Args:
            source_dict: the data dictionary for the source scan.
            target_dict: the data dictionary for the target scan.
            source_bbox: the bounding box in source space.
            source_spacing: the spacing of the source image.
            target_spacing: the spacing of the target image.

        Returns:
            A BBox3D or BBox2D object in target space, or None if propagation fails.
        """
        # Extract the 8 corners of the bounding box
        corners = [
            (source_bbox.x_min, source_bbox.y_min, source_bbox.z_min),
            (source_bbox.x_min, source_bbox.y_min, source_bbox.z_max),
            (source_bbox.x_min, source_bbox.y_max, source_bbox.z_min),
            (source_bbox.x_min, source_bbox.y_max, source_bbox.z_max),
            (source_bbox.x_max, source_bbox.y_min, source_bbox.z_min),
            (source_bbox.x_max, source_bbox.y_min, source_bbox.z_max),
            (source_bbox.x_max, source_bbox.y_max, source_bbox.z_min),
            (source_bbox.x_max, source_bbox.y_max, source_bbox.z_max),
        ]

        # Propagate all corners using the registration
        transformed_corners = self._propagate_clicks(
            source_dict=source_dict,
            target_dict=target_dict,
            source_spacing=source_spacing,
            target_spacing=target_spacing,
            source_clicks=corners,
            source_image=source_image,
            target_image=target_image,
        )

        if len(transformed_corners) == 0:
            return None

        # Find the min and max coordinates from all transformed corners
        x_coords = [int(c[0]) for c in transformed_corners]
        y_coords = [int(c[1]) for c in transformed_corners]
        z_coords = [int(c[2]) for c in transformed_corners]

        try:
            if self.bbox_2d:
                # For 2D bbox, we need to ensure one dimension has the same min/max
                # Preserve the fixed dimension from the source bbox
                num_coords = len(x_coords)
                if isinstance(source_bbox, BBox2D):
                    fixed_dim = source_bbox.fixed_dimension
                    if fixed_dim == "x":
                        # Use mean of transformed x coordinates as the fixed value
                        fixed_x = int(sum(x_coords) / num_coords)
                        target_bbox = BBox2D(
                            min_point=Point3D(fixed_x, min(y_coords), min(z_coords)),
                            max_point=Point3D(fixed_x, max(y_coords), max(z_coords)),
                        )
                    elif fixed_dim == "y":
                        # Use mean of transformed y coordinates as the fixed value
                        fixed_y = int(sum(y_coords) / num_coords)
                        target_bbox = BBox2D(
                            min_point=Point3D(min(x_coords), fixed_y, min(z_coords)),
                            max_point=Point3D(max(x_coords), fixed_y, max(z_coords)),
                        )
                    else:  # fixed_dim == "z"
                        # Use mean of transformed z coordinates as the fixed value
                        fixed_z = int(sum(z_coords) / num_coords)
                        target_bbox = BBox2D(
                            min_point=Point3D(min(x_coords), min(y_coords), fixed_z),
                            max_point=Point3D(max(x_coords), max(y_coords), fixed_z),
                        )
                else:
                    # Fallback: If source is not BBox2D but bbox_2d=True, default to fixing z dimension
                    # This should not normally happen as get_bounding_box should create BBox2D when bbox_2d=True
                    LOGGER.warning("Source bbox is not BBox2D but bbox_2d=True. Defaulting to fixing z dimension.")
                    fixed_z = int(sum(z_coords) / num_coords)
                    target_bbox = BBox2D(
                        min_point=Point3D(min(x_coords), min(y_coords), fixed_z),
                        max_point=Point3D(max(x_coords), max(y_coords), fixed_z),
                    )
            else:
                target_bbox = BBox3D(
                    min_point=Point3D(min(x_coords), min(y_coords), min(z_coords)),
                    max_point=Point3D(max(x_coords), max(y_coords), max(z_coords)),
                )
            return target_bbox
        except (ValueError, AssertionError) as e:
            LOGGER.error(f"Failed to create valid bounding box from transformed corners: {e}")
            return None
