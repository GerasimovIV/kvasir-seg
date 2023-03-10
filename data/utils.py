import torch
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import draw_segmentation_masks
from typing import Dict, Union, List, Tuple, Optional
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

def plot_mask_over_image(
    image: Union[torch.Tensor, Image.Image], 
    mask: Union[torch.Tensor, Image.Image]
) -> None:
    to_tensor = ToTensor()
    if isinstance(image, Image.Image):
        masked_image = to_tensor(masked_image)
    else:
        masked_image = image
        
    masked_image = (masked_image * 255).to(torch.uint8)
        
    if isinstance(mask, torch.Tensor):
        mask_ = mask
    else:
        mask_ = to_tensor(mask)
        if len(mask_.shape) > 2:
            mask_ = mask[0, :, :]
        mask_ = mask_ == 1
        
    to_pil = ToPILImage()
    masked_image = draw_segmentation_masks(masked_image, mask_, alpha=0.7, colors=(0, 200, 0))
    masked_image = to_pil(masked_image)
    masked_image.show()
    
    
def _count_target(mask: torch.Tensor) -> int:
    assert len(mask.shape) == 2, f'wrong shape of mask, got mask.shape = {mask.shape}'
    return mask.sum()


def _count_without_black(image: torch.Tensor) -> int:
    assert len(image.shape) == 3 and image.shape[0] == 3 , f'wrong shape of image, got image.shape = {image.shape}'
    mask = torch.ones(image.shape[1:])
    image_0 = image[0, :, :] == 0.
    image_1 = image[1, :, :] == 0.
    image_2 = image[2, :, :] == 0.
    mask_zeros = image_0 * image_1 * image_2
    mask[mask_zeros] = 0.
    return mask.sum()

def _count_background(image: torch.Tensor, mask: torch.Tensor) -> int:
    return count_without_black(image) - count_target(mask)


def collect_statistic(dataset: Dataset) -> Dict[str, int]:
    object_areas = []
    # background_areas = []
    # black_regions_areas = []
    objects_counts = []

    for data_ in tqdm(dataset, desc='processing dataset'):
        image = data_['input']
        mask = data_['target']
        bboxes = data_['bboxes'] if 'bboxes' in data_ else [] 

        object_area = _count_target(mask)
        useful_area = _count_without_black(image)
        amount_of_objects = len(bboxes)
        
        object_areas.append(object_area / useful_area)
        objects_counts.append(amount_of_objects)
        

    result = {
        'object_areas': object_areas,
        'objects_counts': objects_counts,
    }
    
    return result