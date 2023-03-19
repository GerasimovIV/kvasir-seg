from pathlib import Path
from typing import Any, Dict, Sequence, Union

import torch
import yaml
from tabulate import tabulate
from tqdm import tqdm

from src.data_utils.dataset import load_dataset
from src.metrics import ComputeMetrics
from src.models import load_model, load_weights_from_checkpoint
from train import CustomTrainer


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
        result_metrics.append(metrics)

    return result_metrics


if __name__ == "__main__":
    metrics = test_model("./test_config.yaml")
    table = [[k, f"{v:.3}"] for k, v in metrics.items()]
    print(tabulate(table))
