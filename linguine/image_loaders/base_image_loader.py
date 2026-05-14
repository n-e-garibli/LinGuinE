# Copyright AstraZeneca 2026

from typing import Any

from monai import transforms
from monai.data.meta_tensor import MetaTensor

from linguine.constants import IMAGE, LABEL
from linguine.utils.data import get_path_from_data_dict_entry


class BaseImageLoader:
    def __init__(self, device: str = "cpu"):
        """Constructor.

        Args:
            device: device to load the data onto.
        """
        self.device = device

    def __call__(self, data_dict: dict[str, Any]) -> tuple[MetaTensor, MetaTensor | None]:
        """Loads target image and label from the parent directory.

        Args:
            data_dict: A data dictionary containing image and optionally label paths.

        Returns:
            A tuple of 5D metatensors (with batch dimension) corresponding to the
            image and label respectively. Label tensor is None if no label path provided.
            If MONAI transforms already produce 5D tensors, they are returned as-is.
            If they produce 4D tensors, a batch dimension is added to make them 5D.
        """
        img_path = get_path_from_data_dict_entry(data_dict, IMAGE)
        label_path = get_path_from_data_dict_entry(data_dict, LABEL)
        data = {IMAGE: img_path}
        if label_path is not None:
            data_dict_keys = [IMAGE, LABEL]
            data[LABEL] = label_path
        else:
            data_dict_keys = [IMAGE]

        # Basic MONAI transforms to load the image as a 5D MetaTensor
        transforms_to_apply = [
            transforms.LoadImaged(image_only=False, keys=data_dict_keys),
            transforms.EnsureChannelFirstd(keys=data_dict_keys),
            transforms.Orientationd(keys=data_dict_keys, as_closest_canonical=True),
            transforms.EnsureTyped(keys=data_dict_keys),
        ]
        trans = transforms.Compose(transforms_to_apply)
        loaded_data = trans(data)

        # Add batch dimension only if tensor is 4D to make it 5D as expected by the rest of the codebase
        image_tensor = loaded_data[IMAGE].to(self.device)
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension for 4D -> 5D

        if label_path is None:
            return image_tensor, None
        else:
            label_tensor = loaded_data[LABEL].to(self.device)
            if label_tensor.ndim == 4:
                label_tensor = label_tensor.unsqueeze(0)  # Add batch dimension for 4D -> 5D
            return image_tensor, label_tensor
