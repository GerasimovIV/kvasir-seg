from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


class WrappedSegformerForSemanticSegmentation(SegformerForSemanticSegmentation):
    def to(self, device):
        # print('XCFGHHJKLML<')
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

    def forward(self, input: Tensor, target: Tensor) -> Dict[str, Tensor]:
        logits = self.predict(input).logits
        logits, target = self.sampler(input=logits, target=target)
        assert (
            logits.shape[-2:] == target.shape[-2:]
        ), "Something wrong in UpDownSampler"

        loss_result = self.loss(input=logits, target=target)
        return dict(logits=logits, loss=loss_result)

    def predict(slef, input: Tensor) -> Tensor:
        return super().forward(input)


def loadSegformerForSemanticSegmentation(
    resource: Union[Path, str, Dict[str, Any]], pre_trained_name: str = "nvidia/mit-b5"
):
    seg_former_config_args = {
        # "num_channels": 3,
        # "num_encoder_blocks": 4,
        # "depths": [3, 6, 40, 3],
        # "sr_ratios": [8, 4, 2, 1],
        # "hidden_sizes": [64, 128, 320, 512],
        # "patch_sizes": [7, 3, 3, 3],
        # "strides": [4, 2, 2, 2],
        # "num_attention_heads": [1, 2, 5, 8],
        # "mlp_ratios": [4, 4, 4, 4],
        # "hidden_act": "gelu",
        # "hidden_dropout_prob": 0.0,
        # "attention_probs_dropout_prob": 0.0,
        # "classifier_dropout_prob": 0.1,
        # "initializer_range": 0.02,
        # "drop_path_rate": 0.1,
        # "layer_norm_eps": 1e-06,
        # "decoder_hidden_size": 768,
    }
    # config = SegformerConfig(**seg_former_config_args)
    loss = load_loss(resource)
    # model = WrappedSegformerForSemanticSegmentation(config, loss)
    model = WrappedSegformerForSemanticSegmentation.from_pretrained(pre_trained_name)
    model.add_loss_function(loss)
    # model = WrappedSegformerForSemanticSegmentation(pre_trained_name, loss)
    config = model.config
    model.decode_head.classifier = nn.Conv2d(
        config.decoder_hidden_size, 2, kernel_size=1
    )
    model.set_up_type_upsample(resource)
    return model


models_load_functions = {
    "SegformerForSemanticSegmentation": loadSegformerForSemanticSegmentation
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
