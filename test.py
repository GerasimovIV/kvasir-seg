from pathlib import Path
from typing import Any, Dict, Sequence, Union

import torch
import yaml
from tabulate import tabulate
from tqdm import tqdm

from src.data_utils.dataset import load_dataset
from src.metrics import ComputeMetrics
from src.models import load_model, load_weights_from_checkpoint


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_model(
    resource: Union[str, Path, Dict[str, Any]], return_pred: bool = False
) -> Dict[str, Any]:
    test_config = resource
    if isinstance(resource, str) or isinstance(resource, Path):
        resource = Path(resource)
        with open(resource, "r") as file:
            test_config = yaml.safe_load(file)

    model = load_model(test_config)
    path_to_checkpoint = test_config["checkpoint"]
    device = test_config["device"]
    load_weights_from_checkpoint(path_to_checkpoint, model)
    compute_metrics = ComputeMetrics(test_config)
    test_dataset = load_dataset(test_config)

    model.to(device)
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for data in tqdm(test_dataset, desc="getting predictions"):
            input = data["input"].unsqueeze(0).to(device)
            target = data["target"].unsqueeze(0).to(device)

            logits = model.predict_and_resize(input, target.shape[-2:])
            test_preds.append(logits.to("cpu"))
            test_labels.append(target.to("cpu"))

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    metrics = compute_metrics([test_preds, test_labels])
    model_params_count = count_parameters(model)

    metrics = {
        "model_name": model.__class__.__name__,
        "params": model_params_count,
        "metrics": metrics,
    }

    if return_pred:
        metrics["test_preds"] = test_preds
        metrics["test_labels"] = test_labels

    return metrics


def test_models(
    models: Sequence[Dict], resource: Union[str, Path, Dict[str, Any]]
) -> Dict:

    test_config = resource
    if isinstance(resource, str) or isinstance(resource, Path):
        resource = Path(resource)
        with open(resource, "r") as file:
            test_config = yaml.safe_load(file)

    result_metrics = []

    for model_params in models:
        test_config["model"] = model_params.get("model")
        test_config["checkpoint"] = model_params.get("checkpoint")

        metrics = test_model(test_config)
        metrics["model_name"] = model_params.get("name_display")
        result_metrics.append(metrics)

    return result_metrics


if __name__ == "__main__":

    models = [
        {
            "model": {
                "name": "UnetPlusPlus",
                "downsample_target": False,
                "load_args": {
                    "pre_trained_name": "imagenet",
                    "encoder_name": "resnet34",
                },
            },
            "checkpoint": "./expirements/unet_plus_plus_baseline/checkpoint-1000",
            "name_display": "CE+Dice, Unet++ (resnet34)",
        },
        {
            "model": {
                "name": "UnetPlusPlus",
                "downsample_target": False,
                "load_args": {
                    "pre_trained_name": "imagenet",
                    "encoder_name": "resnet34",
                },
            },
            "checkpoint": "./expirements/UnetPlusPlus_resnet34_CE_Boundary/checkpoint-1000/",
            "name_display": "CE+Boundary, Unet++ (resnet34)",
        },
        {
            "model": {
                "name": "UnetPlusPlus",
                "downsample_target": False,
                "load_args": {
                    "pre_trained_name": "imagenet",
                    "encoder_name": "resnet101",
                },
            },
            "checkpoint": "./expirements/unet_plus_plus_resnet101/checkpoint-1000",
            "name_display": "CE+Dice, Unet++ (resnet101)",
        },
        {
            "model": {
                "name": "UnetPlusPlus",
                "downsample_target": False,
                "load_args": {
                    "pre_trained_name": "imagenet",
                    "encoder_name": "resnet101",
                },
            },
            "checkpoint": "./expirements/UnetPlusPlus_resnet101_CE_Boundary/checkpoint-1000/",
            "name_display": "CE+Boundary, Unet++ (resnet101)",
        },
        {
            "model": {
                "name": "SegformerForSemanticSegmentation",
                "downsample_target": False,
                "load_args": {"pre_trained_name": "nvidia/mit-b0"},
            },
            "checkpoint": "./expirements/segformer_mit-b0/checkpoint-1000",
            "name_display": "CE+Dice, Segformer (nvidia/mit-b0)",
        },
        {
            "model": {
                "name": "SegformerForSemanticSegmentation",
                "downsample_target": False,
                "load_args": {"pre_trained_name": "nvidia/mit-b0"},
            },
            "checkpoint": "./expirements/SegformerForSemanticSegmentation_mit-b0_CE_Boundary/checkpoint-1000/",
            "name_display": "CE+Boundary, Segformer (nvidia/mit-b0)",
        },
    ]

    metrics = test_models(models=models, resource="./test_config.yaml")

    head_metrics = list(metrics[0]["metrics"].keys())
    head_metrics.sort()
    headers = ["model"] + head_metrics

    table = [
        [metrics[i]["model_name"] + f' ({metrics[i]["params"]/1e+6:.3}M)']
        + [f"{metrics[i]['metrics'][k]:.3}" for k in headers[1:]]
        for i in range(len(metrics))
    ]
    print(tabulate(table, headers=headers, tablefmt="grid"))
