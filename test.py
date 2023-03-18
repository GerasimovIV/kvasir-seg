from pathlib import Path
from typing import Any, Dict, Union

import torch
import yaml
from tabulate import tabulate
from tqdm import tqdm

from src.data_utils.dataset import load_dataset
from src.metrics import ComputeMetrics
from src.models import load_model, load_weights_from_checkpoint
from train import CustomTrainer


def test_model(
    resource: Union[str, Path, Dict[str, Any]], return_pred: bool = False
) -> Dict[str, float]:
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

            logits = model.predict(input)
            test_preds.append(logits.to("cpu"))
            test_labels.append(target.to("cpu"))

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    metrics = compute_metrics([test_preds, test_labels])

    if return_pred:
        return metrics, test_preds, test_labels
    return metrics


if __name__ == "__main__":
    metrics = test_model("./test_config.yaml")
    table = [[k, f"{v:.3}"] for k, v in metrics.items()]
    print(tabulate(table))
