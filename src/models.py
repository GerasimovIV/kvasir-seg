from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch.nn as nn
import yaml
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def loadSegformerForSemanticSegmentation(pre_trained_name: str = "nvidia/mit-b5"):
    seg_former_config_args = {
        "num_channels": 3,
        "num_encoder_blocks": 4,
        "depths": [3, 6, 40, 3],
        "sr_ratios": [8, 4, 2, 1],
        "hidden_sizes": [64, 128, 320, 512],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "num_attention_heads": [1, 2, 5, 8],
        "mlp_ratios": [4, 4, 4, 4],
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "drop_path_rate": 0.1,
        "layer_norm_eps": 1e-06,
        "decoder_hidden_size": 768,
    }
    config = SegformerConfig(**seg_former_config_args)
    model = SegformerForSemanticSegmentation(config)
    model.segformer = model.segformer.from_pretrained(pre_trained_name)
    return model


models_load_functions = {
    "SegformerForSemanticSegmentation": loadSegformerForSemanticSegmentation
}


def load_model(resource: Union[Path, str, Dict[str, Any]]):
    if isinstance(resource, Path) or isinstance(resource, str):
        resource = Path(resource)

        with open(resource, "r") as file:
            config = yaml.safe_load(file)

        assert "model" in config, f"wrong resource {resource} construction"

    name = config["model"]["name"]
    args = config["model"]["load_args"]

    model = models_load_functions[name](**args)

    return model
