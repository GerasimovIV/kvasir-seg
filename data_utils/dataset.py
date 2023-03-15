import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .utils import Augmentator


def red_bold(x):
    return colored(f"{x}", "red", attrs=["bold"])


def blue_bold(x):
    return colored(f"{x}", "blue", attrs=["bold"])


class KvasirDatasetBase(Dataset):
    """
    Basic Dataset class for Kvasir-SEG
    Initialises dataset with images and masks

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says (default 45)
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        bboxes_file: Optional[str] = "kavsir_bboxes.json",
        image_format: str = "jpg",
        mask_format: str = "jpg",
        augment_conf: Union[Dict[str, Any], Union[str, Path]] = {},
    ) -> None:
        self.image_format = image_format
        self.mask_format = mask_format
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"
        self.bboxes_file = self.root_dir / bboxes_file

        # prepare data
        self.data_images = {}

        for name in tqdm(glob(f"{self.images_dir}/*.{self.image_format}")):
            try:
                name_key = Path(name).name.split(".")[0]
                image_raw = Image.open(name)
                self.data_images[name_key] = {"pil_image_raw": image_raw}
            except Exception:
                print(
                    f"{red_bold('error during reading image (this will be skipped):')}: {name}"
                )

        for name in tqdm(glob(f"{self.masks_dir}/*.{self.mask_format}")):
            try:
                name_key = Path(name).name.split(".")[0]
                if name_key not in self.data_images:
                    print("key error")
                    self.data_images[name_key]
                image_mask = Image.open(name)
                self.data_images[name_key]["pil_image_mask"] = image_mask

                size_image = self.data_images[name_key]["pil_image_raw"].size
                size_mask = self.data_images[name_key]["pil_image_mask"].size
                assert (
                    size_image == size_mask
                ), f"the mask and image size should be the same, name: {name_key}"

            except Exception:
                print(
                    f"{red_bold('error during reading image (this will be skipped):')}: {name}"
                )

        self.data_names = list(self.data_images.keys())
        self.data_names.sort()

        # prepare bboxes
        try:
            with open(self.bboxes_file) as f:
                bboxes = json.load(f)

                for name_key in bboxes:
                    if name_key in self.data_images:
                        self.data_images[name_key]["bboxes"] = bboxes.get(name_key)
        except FileNotFoundError:
            print(
                f"Warning! {self.bboxes_file} file not found, dataset will be without bboxes"
            )
        except KeyError:
            print(f"Warning! {self.bboxes_file} doesn't consist of bbox for {name_key}")

        # prepare augmentations
        self.augmentator = Augmentator(augment_conf)
        self.to_tensor = ToTensor()

    def __getitem__(
        self, idx: int, return_bboxes: bool = True
    ) -> Dict["str", Union[torch.Tensor, List[torch.Tensor]]]:
        input_tensor = self.data_images[self.data_names[idx]].get("pil_image_raw")
        target_mask_tensor = self.data_images[self.data_names[idx]].get(
            "pil_image_mask"
        )

        input_tensor = np.array(input_tensor)
        target_mask_tensor = np.array(target_mask_tensor)

        augmentations = self.augmentator()

        # print(input_tensor.size, target_mask_tensor.size)
        transformed = augmentations(image=input_tensor, mask=target_mask_tensor)

        input_tensor = transformed["image"]
        target_mask_tensor = transformed["mask"]

        input_tensor = self.to_tensor(input_tensor)
        target_mask_tensor = self.to_tensor(target_mask_tensor)[0, :, :]
        target_mask_tensor = target_mask_tensor == 1.0

        result = {"input": input_tensor, "target": target_mask_tensor}

        if "bboxes" in self.data_images[self.data_names[idx]] and return_bboxes:
            curr_bboxes = self.data_images[self.data_names[idx]].get("bboxes")
            height_orig = curr_bboxes["height"]
            weight_orig = curr_bboxes["width"]

            normalized_bboxes = []

            for bbox in curr_bboxes["bbox"]:
                normalized_bboxes.append(
                    torch.tensor(
                        [
                            bbox["xmin"] / weight_orig,
                            bbox["ymin"] / height_orig,
                            bbox["xmax"] / weight_orig,
                            bbox["ymax"] / height_orig,
                        ]
                    )
                )
            result["bboxes"] = normalized_bboxes

        return result

    def __len__(self) -> int:
        raise len(self.data_images)


class KvasirDataset(KvasirDatasetBase):
    def __getitem__(self, idx) -> Dict["str", Union[torch.Tensor, List[torch.Tensor]]]:
        result = super().__getitem__(idx, return_bboxes=False)
        return result
