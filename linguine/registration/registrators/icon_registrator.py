"""Contains classes for performing registration within LinGuinE using ICON based models.

Some of the helpers in this file were adapted from https://github.com/uncbiag/icon."""

import logging
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from icon_registration.config import device as ICON_DEVICE
from icon_registration.network_wrappers import RegistrationModule
from icon_registration.pretrained_models.lung_ct import make_network
from monai.data.meta_tensor import MetaTensor

from linguine.image_loaders.base_image_loader import BaseImageLoader
from linguine.registration.registrators import ImageRegistrator
from linguine.utils.misc import location_is_valid

LOGGER = logging.getLogger(__name__)

LUNG_MODEL_WEIGHTS_DOWNLOAD = (
    "https://github.com/uncbiag/ICON/releases/download/pretrained_models_v1.0.0/lung_model_weights_step_2.trch"
)
UNIGRADICON_WEIGHTS_DOWNLOAD = (
    "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch"
)


class IconRegistrator(ImageRegistrator):
    """A registrator that uses ICON neural networks for image registration.

    This class wraps an ICON RegistrationModule to provide coordinate mapping
    functionality between source and target images.
    """

    def __init__(self, icon_module: RegistrationModule, crop_foreground: bool = False):
        """Initialize the IconRegistrator with an ICON module.

        Args:
            icon_module: A RegistrationModule with an assigned identity map.
            crop_foreground: Whether to crop images to foreground region before registration.
        """
        if not hasattr(icon_module, "identity_map"):
            raise ValueError(
                "Input gradicon module must have an assigned identity map already! Ensure its instantiated properly before use."
            )
        self.model = icon_module
        self.identity_map = icon_module.identity_map
        self.crop_foreground = crop_foreground

    @staticmethod
    def _get_roi_bounds(image: torch.Tensor) -> list[tuple[int, int]]:
        """Get the bounding box of non-zero regions in the image.

        Args:
            image: A 5D tensor (1,1,x,y,z) containing the image.

        Returns:
            List of (min, max+1) tuples for slicing along each spatial dimension.
        """
        # Get non-zero mask for the first channel
        nonzero_mask = image[0, 0] != 0

        # Find non-zero indices along each axis
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False)

        if len(nonzero_indices) == 0:
            # If image is all zeros, return full bounds
            return [(0, s) for s in image.shape[2:]]

        # Get min and max indices for each dimension
        mins = nonzero_indices.min(dim=0)[0]
        maxs = nonzero_indices.max(dim=0)[0]

        # Return as list of (min, max+1) tuples for slicing
        return [(mins[i].item(), maxs[i].item() + 1) for i in range(3)]

    @staticmethod
    def _crop_image(image: torch.Tensor, bounds: list[tuple[int, int]]) -> torch.Tensor:
        """Crop image using the provided bounds.

        Args:
            image: A 5D tensor (1,1,x,y,z) to crop.
            bounds: List of (min, max) tuples for each spatial dimension.

        Returns:
            The cropped image tensor.
        """
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        z_min, z_max = bounds[2]
        return image[:, :, x_min:x_max, y_min:y_max, z_min:z_max]

    def map_coordinates(
        self,
        source_image: MetaTensor,
        target_image: MetaTensor,
        coors: list[tuple[int, int, int]],
        round_: bool = True,
        **kwargs,
    ) -> list[tuple[int, int, int]]:
        """Maps coordinates from the space of the source image to the space of the target.

        Args:
            source_image: A 5D MetaTensor (1,1,x,y,z) containing the source image.
            target_image: A 5D MetaTensor (1,1,x,y,z) containing the target image.
            coors: a list of coordinates (x,y,z) to map.
            round_: a bool specifying whether to round the coordinates to integers for easy indexing

        Returns:
            The coordinates in the space of the target image in the same format as they were inputted.
        """
        shape = self.identity_map.shape
        if source_image.shape[1] > 1:
            LOGGER.warning(
                f"Your source image has {source_image.shape[1]} channels, will only consider the first for registration."
            )
            source_image = source_image[:, 0:1, :, :, :]
        if target_image.shape[1] > 1:
            LOGGER.warning(
                f"Your target image has {target_image.shape[1]} channels, will only consider the first for registration."
            )
            target_image = target_image[:, 0:1, :, :, :]

        if self.crop_foreground:
            # Get ROI bounds for both images
            source_bounds = self._get_roi_bounds(source_image)
            target_bounds = self._get_roi_bounds(target_image)
        else:
            # Use full image shape for bounds
            source_bounds = [(0, s) for s in source_image.shape[2:]]
            target_bounds = [(0, s) for s in target_image.shape[2:]]

        # Crop images to their ROIs
        source_cropped = self._crop_image(source_image, source_bounds)
        target_cropped = self._crop_image(target_image, target_bounds)

        source_resized = F.interpolate(source_cropped, size=shape[2:], mode="trilinear", align_corners=False)
        source_resized = source_resized.to(ICON_DEVICE)
        target_resized = F.interpolate(target_cropped, size=shape[2:], mode="trilinear", align_corners=False)
        target_resized = target_resized.to(ICON_DEVICE)

        with torch.no_grad():
            vectorfield = self.model(target_resized.float(), source_resized.float())(self.identity_map)
        new_coors = []
        for coor in coors:
            # Transform coordinates from full source space to cropped source space
            cropped_source_coor = [coor[i] - source_bounds[i][0] for i in range(3)]
            if not location_is_valid(cropped_source_coor, source_cropped[0, 0]):
                # This coordinate is not in the cropped area.
                continue

            coor = [
                round(c * shape[2 + i] / source_cropped.shape[2:][i]) for c, i in zip(cropped_source_coor, range(3))
            ]
            if not location_is_valid(coor, vectorfield[0][0]):
                # Somehow ended up out of bounds, skip.
                continue
            target_coor_normalized_space = vectorfield[0][:, coor[0], coor[1], coor[2]].to("cpu")
            target_coor_cropped_space = target_coor_normalized_space * (torch.tensor(target_cropped.shape[2:]) - 1)
            target_coor_original_space = target_coor_cropped_space + torch.tensor(
                [target_bounds[i][0] for i in range(3)]
            ).to("cpu")

            if round_:
                new_coor = tuple([round(x.item()) for x in target_coor_original_space])
            else:
                new_coor = tuple([x.item() for x in target_coor_original_space])
            new_coors.append(new_coor)
        return new_coors


class UniGradIconRegistrator(IconRegistrator):
    """A GradICON foundation model for medical image registration.
    See https://arxiv.org/abs/2403.05780 for more details.

    """

    def __init__(self, crop_foreground: bool = False):
        """Initialize the UniGradIconRegistrator with a pretrained model.

        Args:
            crop_foreground: Whether to crop images to foreground region before registration.
        """
        # Function from lung ct pretrained model used to avoid unigradicon dependency. This is fine because they have the same architectures.
        net = make_network()
        # This is the shape of the identity map for the pretrained model.
        net.assign_identity_map([1, 1, 175, 175, 175])
        net.regis_net.load_state_dict(self._get_weights())
        # Icon repo does this again after loading weights, so doing this just in case
        net.assign_identity_map([1, 1, 175, 175, 175])
        net.to(ICON_DEVICE)
        net.eval()
        super().__init__(net.regis_net, crop_foreground=crop_foreground)

    @staticmethod
    def _preprocess_image(image: MetaTensor) -> MetaTensor:
        """Image preprocessing necessary for unigradicon in CT."""
        pp_image = image.clone()
        # Clip to be between -1000 and 1000
        pp_image[pp_image < -1000] = -1000
        pp_image[pp_image > 1000] = 1000
        # Normalize to 0-1 range
        pp_image = (pp_image + 1000) / 2000
        return pp_image

    def _get_weights(self):
        """Download or load the pretrained unigradicon weights.

        Adapted from https://github.com/uncbiag/uniGradICON/blob/d73ab76966e5d930ac26da6b0b6aaab86b9fa2dd/src/unigradicon/__init__.py#L263-L279"""

        # Store weights in user cache directory
        cache_dir = Path.home() / ".cache" / "linguine" / "icon_models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        weights_location = cache_dir / "unigradicon1.0" / "Step_2_final.trch"

        if not weights_location.exists():
            LOGGER.info("Downloading pretrained unigradicon model")
            weights_location.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url=UNIGRADICON_WEIGHTS_DOWNLOAD, filename=str(weights_location))
        trained_weights = torch.load(str(weights_location), map_location=torch.device("cpu"), weights_only=True)
        return trained_weights

    def map_coordinates(
        self,
        source_image: MetaTensor,
        target_image: MetaTensor,
        coors: list[tuple[int, int, int]],
        round_=True,
        **kwargs,
    ) -> list[tuple[int, int, int]]:
        """Maps coordinates from the space of the source image to the space of the target image.

        This method extends the base map_coordinates functionality by applying unigradicon specific
        preprocessing to both source and target images before performing registration.

        Args:
            source_image: A 5D MetaTensor (1,1,x,y,z) containing the source image.
            target_image: A 5D MetaTensor (1,1,x,y,z) containing the target image.
            coors: A list or numpy array of coordinates (x,y,z) to map.
            round_: A bool specifying whether to round the coordinates to integers for easy indexing.
            **kwargs: Additional keyword arguments for compatibility.

        Returns:
            The coordinates in the space of the target image in the same format as they were inputted.
        """
        source_image = self._preprocess_image(source_image)
        target_image = self._preprocess_image(target_image)
        return super().map_coordinates(source_image, target_image, coors, round_)


class GradIconLungRegistrator(IconRegistrator):
    """A specialized GradICON registrator for lung CT image registration.

    This class extends GradIconRegistrator with lung-specific preprocessing
    using a pretrained lung CT registration model.
    """

    def __init__(self, crop_foreground: bool = False):
        """Initialize a lung CT gradicon with a pretrained model.

        Args:
            crop_foreground: Whether to crop images to foreground region before registration.
        """
        net = make_network()
        # This is the shape of the identity map for the pretrained model.
        net.assign_identity_map([1, 1, 175, 175, 175])
        net.regis_net.load_state_dict(self._get_weights())
        # Icon repo does this again after loading weights, so doing this just in case
        net.assign_identity_map([1, 1, 175, 175, 175])
        net.to(ICON_DEVICE)
        net.eval()
        super().__init__(net.regis_net, crop_foreground=crop_foreground)

    def _get_weights(self):
        """Download or load the pretrained grad icon lung CT model weights.

        Adapted from icon_registration.pretrained_models.lung_ct.LungCT_registration_model
        """
        cache_dir = Path.home() / ".cache" / "linguine" / "icon_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        lung_model_path = cache_dir / "lung_model" / "lung_model_weights.trch"
        if not lung_model_path.exists():
            LOGGER.info("Downloading pretrained lung CT gradicon model")
            lung_model_path.parent.mkdir(exist_ok=True, parents=True)
            urllib.request.urlretrieve(
                url=LUNG_MODEL_WEIGHTS_DOWNLOAD,
                filename=str(lung_model_path),
            )

        trained_weights = torch.load(
            str(lung_model_path),
            map_location=torch.device("cpu"),
        )
        return trained_weights

    @staticmethod
    @lru_cache(maxsize=2)
    def _preprocess_image(img: torch.Tensor, ts_path: str | Path) -> torch.Tensor:
        """Preprocess a lung CT image for registration.

        This method applies lung-specific preprocessing including normalization,
        tissue segmentation filtering, and intensity windowing.

        Args:
            img: The input CT image tensor.
            scan_dict: Dictionary containing image and segmentation data, must include "ts" key.

        Returns:
            The preprocessed lung CT image tensor.
        """
        loader = BaseImageLoader(device=ICON_DEVICE)
        # Load the total segmentator image. If you're here because this doesn't work for your data
        # please define a child class for this registrator where you load your segmentations appropriately
        # and pass the registrator into the pipeline as a custom registrator!
        _, loaded_ts = loader({"image": ts_path, "label": ts_path})
        if loaded_ts.shape != img.shape:
            raise ValueError(
                f"Shape mismatch between loaded TS segmentation and image! TS loaded shape: {loaded_ts.shape}, image shape: {img.shape}"
            )

        # Normalise image.
        pp_image = img.clone()
        pp_image[pp_image < -1000] = -1000
        pp_image[pp_image > 0] = 0
        pp_image = (pp_image + 1000) / 1000

        # Filter out everything that isnt lung. In TS segmentations, values 9-15 correspond to lung lobes.
        loaded_ts[(loaded_ts >= 15) | (loaded_ts <= 9)] = 0.0
        loaded_ts[loaded_ts > 0] = 1
        pp_image[loaded_ts < 1.0] = 0
        return pp_image

    def map_coordinates(
        self,
        source_image: MetaTensor,
        target_image: MetaTensor,
        coors: list[tuple[int, int, int]],
        source_dict: dict[str, Any],
        target_dict: dict[str, Any],
        round_=True,
        **kwargs,
    ) -> list[tuple[int, int, int]]:
        """Maps coordinates from the space of the source lung CT image to the space of the target lung CT image.

        This method extends the base map_coordinates functionality by applying lung-specific
        preprocessing to both source and target images before performing registration.

        Args:
            source_image: A 5D MetaTensor (1,1,x,y,z) containing the source lung CT image.
            target_image: A 5D MetaTensor (1,1,x,y,z) containing the target lung CT image.
            coors: A list or numpy array of coordinates (x,y,z) to map.
            source_dict: Dictionary containing source image data, must include "ts" segmentation.
            target_dict: Dictionary containing target image data, must include "ts" segmentation.
            round_: A bool specifying whether to round the coordinates to integers for easy indexing.
            **kwargs: Additional keyword arguments for compatibility.

        Returns:
            The coordinates in the space of the target image in the same format as they were inputted.
        """
        assert "ts" in source_dict, "Grad icon lung registration needs a ts segmentation."
        assert "ts" in target_dict, "Grad icon lung registration needs a ts segmentation."
        source_image = self._preprocess_image(source_image, source_dict["ts"])
        target_image = self._preprocess_image(target_image, target_dict["ts"])
        return super().map_coordinates(source_image, target_image, coors, round_)


class UniGradIconWithROIRefinement(UniGradIconRegistrator):
    """A UniGradIcon registrator with ROI-based refinement.

    This class extends UniGradIconRegistrator by implementing a two-stage registration process:
    1. First performs registration on the full images to get initial coordinate estimates
    2. Then crops ROIs around the estimated locations and re-registers at higher resolution
    """

    def map_coordinates(
        self,
        source_image: MetaTensor,
        target_image: MetaTensor,
        coors: list[tuple[int, int, int]],
        round_=True,
        **kwargs,
    ) -> list[tuple[int, int, int]]:
        """Maps coordinates using a two-stage ROI refinement approach for improved accuracy.

        This method implements a two-stage registration process:
        1. Performs initial registration on full images to get rough coordinate estimates
        2. Identifies ROIs around the median estimated location in both images
        3. Re-registers using identified ROIs only to achieve higher resolution

        Args:
            source_image: A 5D MetaTensor (1,1,x,y,z) containing the source image.
            target_image: A 5D MetaTensor (1,1,x,y,z) containing the target image.
            coors: A list of coordinates (x,y,z) to map.
            round_: A bool specifying whether to round the coordinates to integers for easy indexing.
            **kwargs: Additional keyword arguments for compatibility.

        Returns:
            The coordinates in the space of the target image in the same format as inputted.
        """

        initial_coors = super().map_coordinates(source_image, target_image, coors, round_)
        if len(initial_coors) == 0:
            # Everything ended up out of bounds after initial registration.
            return []
        # Find the median coordinate
        median_coor_source = np.median(np.array(coors), axis=0).astype(int)
        median_coor_target = np.median(np.array(initial_coors), axis=0).astype(int)
        # Crop ROI for source and target images
        source_roi_mask = self._create_roi_mask(source_image, median_coor_source)
        target_roi_mask = self._create_roi_mask(target_image, median_coor_target)
        roi_source_image = source_image.clone()
        # -1001 is chosen because these values will go to 0 when unigradicon preprocessing is applied.
        roi_source_image[source_roi_mask] = -1001
        roi_target_image = target_image.clone()
        roi_target_image[target_roi_mask] = -1001
        self.crop_foreground = True
        clicks = super().map_coordinates(roi_source_image, roi_target_image, coors, round_)
        self.crop_foreground = False
        return clicks

    def _create_roi_mask(self, image: MetaTensor, center: np.ndarray):
        """Create a boolean mask to isolate ROI around a center point for refinement registration.

        This method creates a mask that identifies regions outside the ROI (True values),
        which can be used to masked out the less relevant areas of the image and focus registration
        on the ROI. The ROI is centered around the provided point and has dimensions matching
        the identity map shape.

        Args:
            image: A 5D MetaTensor (1,1,x,y,z) containing the image to mask.
            center: A numpy array of (x,y,z) coordinates specifying the ROI center.

        Returns:
            A boolean mask tensor with the same shape as input image, where True indicates
            regions outside the ROI that should be masked out.
        """
        roi_shape = np.array(self.identity_map.shape[2:])
        img_shape = np.array(image.shape[2:])  # (x, y, z)
        mask = torch.ones_like(image)
        start = center - roi_shape // 2
        end = start + roi_shape
        # Shift start/end if out of bounds
        start = np.maximum(start, 0)
        end = np.minimum(end, img_shape)
        # If image is smaller than ROI, adjust start/end to fit as much as possible
        for i in range(3):
            if end[i] - start[i] < roi_shape[i]:
                if start[i] == 0:
                    end[i] = min(roi_shape[i], img_shape[i])
                elif end[i] == img_shape[i]:
                    start[i] = max(img_shape[i] - roi_shape[i], 0)
        slices = tuple([slice(None), slice(None)] + [slice(s, e) for s, e in zip(start, end)])
        mask[slices] = 0
        return mask.bool()
