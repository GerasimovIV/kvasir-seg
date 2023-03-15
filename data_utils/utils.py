from pathlib import Path
from typing import Any, Dict, Union

import albumentations as A
import numpy as np
import torch
import yaml
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm


def plot_mask_over_image(
    image: Union[torch.Tensor, Image.Image], mask: Union[torch.Tensor, Image.Image]
) -> None:
    to_tensor = ToTensor()

    masked_image = image

    if isinstance(image, Image.Image):
        masked_image = to_tensor(masked_image)

    masked_image = (masked_image * 255).to(torch.uint8)

    if isinstance(mask, torch.Tensor):
        mask_ = mask
    else:
        mask_ = to_tensor(mask)
        if len(mask_.shape) > 2:
            mask_ = mask[0, :, :]
        mask_ = mask_ == 1

    to_pil = ToPILImage()
    masked_image = draw_segmentation_masks(
        masked_image, mask_, alpha=0.6, colors=(0, 200, 0)
    )
    masked_image = to_pil(masked_image)
    masked_image.show()


def _count_target(mask: torch.Tensor) -> int:
    assert len(mask.shape) == 2, f"wrong shape of mask, got mask.shape = {mask.shape}"
    return mask.sum()


def _count_without_black(image: torch.Tensor) -> int:
    assert (
        len(image.shape) == 3 and image.shape[0] == 3
    ), f"wrong shape of image, got image.shape = {image.shape}"
    mask = torch.ones(image.shape[1:])
    image_0 = image[0, :, :] == 0.0
    image_1 = image[1, :, :] == 0.0
    image_2 = image[2, :, :] == 0.0
    mask_zeros = image_0 * image_1 * image_2
    mask[mask_zeros] = 0.0
    return mask.sum()


def _count_background(image: torch.Tensor, mask: torch.Tensor) -> int:
    return _count_without_black(image) - _count_target(mask)


def collect_statistic(dataset: Dataset) -> Dict[str, int]:
    object_areas = []
    # background_areas = []
    # black_regions_areas = []
    objects_counts = []

    for data_ in tqdm(dataset, desc="processing dataset"):
        image = data_["input"]
        mask = data_["target"]
        bboxes = data_["bboxes"] if "bboxes" in data_ else []

        object_area = _count_target(mask)
        useful_area = _count_without_black(image)
        amount_of_objects = len(bboxes)

        object_areas.append(object_area / useful_area)
        objects_counts.append(amount_of_objects)

    result = {
        "object_areas": object_areas,
        "objects_counts": objects_counts,
    }

    return result


augmentation_funcs = {
    "Resize": A.Resize,
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "GaussianBlur": A.GaussianBlur,
    "Rotate": A.Rotate,
}


def blue_bold(x):
    return colored(f"{x}", "blue", attrs=["bold"])


def red_bold(x):
    return colored(f"{x}", "red", attrs=["bold"])


class Augmentator(object):
    def __init__(
        self, transforms_config: Union[Dict[str, Any], Union[str, Path]]
    ) -> None:

        self.transforms = []
        print(blue_bold("[Augmentator]:"), "initialization")
        if isinstance(transforms_config, str) or isinstance(transforms_config, Path):
            transforms_config = Path(transforms_config)
            with open(f"{transforms_config}") as file:
                transforms_config = yaml.safe_load(file)

        for name, params in transforms_config.items():
            if name in augmentation_funcs:
                print("\t" + blue_bold(name))
                for k, v in params.items():
                    print(f"\t\t{k}: {v}")
                self.transforms.append(augmentation_funcs[name](**params))

            else:
                print(
                    red_bold("Warning!"), f"the augmentation with {name} was not found"
                )

        self.len_transforms = len(self.transforms)

    def __call__(self) -> torch.nn.modules.module.Module:
        if self.len_transforms == 0:
            return A.Compose([])

        k = np.random.randint(low=1, high=self.len_transforms + 1)
        list_seq = np.random.choice(self.transforms, size=k, replace=False)
        seq = A.Compose(list_seq)
        return seq
