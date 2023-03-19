from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from .losses import load_loss


class UpDownSampler(object):
    def __init__(self, resource: Union[Path, str, Dict[str, Any]]):
        config = resource
        if isinstance(resource, Path) or isinstance(resource, str):
            resource = Path(resource)

            with open(resource, "r") as file:
                config = yaml.safe_load(file)

        assert "model" in config, f"wrong resource {resource} construction"

        self.downsample_target = config["model"]["downsample_target"]

    def __call__(self, input: Tensor, target: Tensor) -> Tuple[Tensor]:
        if self.downsample_target:
            size = input.shape[-2:]
            target = target.unsqueeze(1).float()
            target = F.interpolate(target, size=size, mode="nearest").squeeze(1).long()
        else:
            size = target.shape[-2:]
            input = F.interpolate(input, size=size, mode="nearest")

        return input, target

    def resize(self, input: Tensor, size: Sequence[int]):
        input = F.interpolate(input, size=size, mode="nearest")
        return input


class ModelWrapper(nn.Module):
    def to(self, device):
        super().to(device)
        self.loss.to(device)

    def add_loss_function(self, loss: nn.Module) -> None:
        self.loss = loss

    def set_up_type_upsample(self, resource: Union[Path, str, Dict[str, Any]]):
        """
        if downsample_target == True,
        then target will be downsamplet to size of model
        and loss will be computed with smallest tensors
        and vise wersa
        """
        self.sampler = UpDownSampler(resource)

    def predict_and_resize(self, input: Tensor, size: Sequence[int]) -> Tensor:
        pred = self.predict(input)
        pred = self.sampler.resize(pred, size)
        return pred


class WrappedSegformerForSemanticSegmentation(
    SegformerForSemanticSegmentation, ModelWrapper
):
    def forward(self, input: Tensor, target: Tensor) -> Dict[str, Tensor]:
        logits = self.predict(input)
        logits, target = self.sampler(input=logits, target=target)
        assert (
            logits.shape[-2:] == target.shape[-2:]
        ), "Something wrong in UpDownSampler"

        loss_result = self.loss(input=logits, target=target)
        return dict(logits=logits, loss=loss_result)

    def predict(self, input: Tensor) -> Tensor:
        return super().forward(input).logits


class WrappedUnetPlusPlus(smp.UnetPlusPlus, ModelWrapper):
    def forward(self, input: Tensor, target: Tensor) -> Dict[str, Tensor]:
        logits = self.predict(input)
        logits, target = self.sampler(input=logits, target=target)
        assert (
            logits.shape[-2:] == target.shape[-2:]
        ), "Something wrong in UpDownSampler"

        loss_result = self.loss(input=logits, target=target)
        return dict(logits=logits, loss=loss_result)

    def predict(slef, input: Tensor) -> Tensor:
        label = super().forward(input)
        return label


def loadSegformerForSemanticSegmentation(
    resource: Union[Path, str, Dict[str, Any]],
    pre_trained_name: str = "nvidia/mit-b5",
) -> nn.Module:

    loss = load_loss(resource)

    model = WrappedSegformerForSemanticSegmentation.from_pretrained(pre_trained_name)
    model.add_loss_function(loss)

    config = model.config
    model.decode_head.classifier = nn.Conv2d(
        config.decoder_hidden_size, 2, kernel_size=1
    )
    model.set_up_type_upsample(resource)
    return model


def loadUnetPlusPlus(
    resource: Union[Path, str, Dict[str, Any]],
    pre_trained_name: str = "imagenet",
    encoder_name: str = "resnet34",
) -> nn.Module:
    loss = load_loss(resource)

    model = WrappedUnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=pre_trained_name,
        in_channels=3,
        classes=2,
    )

    model.add_loss_function(loss)
    model.set_up_type_upsample(resource)
    return model


models_load_functions = {
    "SegformerForSemanticSegmentation": loadSegformerForSemanticSegmentation,
    "UnetPlusPlus": loadUnetPlusPlus,
}


def load_model(resource: Union[Path, str, Dict[str, Any]]):
    config = resource
    if isinstance(resource, Path) or isinstance(resource, str):
        resource = Path(resource)

        with open(resource, "r") as file:
            config = yaml.safe_load(file)

        assert "model" in config, f"wrong resource {resource} construction"

    name = config["model"]["name"]
    args = config["model"]["load_args"]

    model = models_load_functions[name](resource, **args)

    return model


def load_weights_from_checkpoint(path: Union[str, Path], model: nn.Module) -> None:
    path = Path(path)
    if path.is_file():
        checkpoint_path = path

    else:
        checkpoint_path = path / "pytorch_model.bin"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    res = model.load_state_dict(checkpoint)
    print(f"{model.__class__.__name__}: {res}")
