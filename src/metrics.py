from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torchmetrics import Dice, JaccardIndex
from transformers import EvalPrediction


class ComputeMetrics(object):
    def __init__(self, resource: Union[Path, str, Dict[str, Any]]) -> None:
        config = resource
        if isinstance(resource, Path) or isinstance(resource, str):
            resource = Path(resource)

            with open(resource, "r") as file:
                config = yaml.safe_load(file)

        assert "compute_metrics" in config, f"wrong resource {resource} construction"

        self.thresholds = config["compute_metrics"]["thresholds"]
        self.dice = Dice(num_classes=1)
        self.iou = JaccardIndex(task="binary")

    def compute_group_area_by_relarea(self, label: float) -> int:
        for i, interval in enumerate(self.thresholds):
            if interval[0] <= label < interval[1]:
                return i
        return i

    def compute_group_area_by_target(self, label: Union[Tensor, np.ndarray]) -> int:
        object_area = label.sum()
        image_area = np.product(label.shape[-2:])

        relative_area = object_area / image_area

        for i, interval in enumerate(self.thresholds):
            if interval[0] <= relative_area < interval[1]:

                return i
        return i

    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions, labels = eval_pred
        b, _, _, _ = predictions.shape

        area_groups = {i: {"dice": [], "iou": []} for i in range(len(self.thresholds))}

        for i in range(b):
            pred = Tensor(predictions[i : i + 1]).float()
            label = Tensor(labels[i : i + 1]).long()

            pred = F.softmax(pred, dim=1)

            group_area = self.compute_group_area_by_target(label)

            area_groups[group_area]["dice"].append(
                self.dice(pred[:, 1].flatten(), label.flatten())
            )

            pred = pred.argmax(dim=1)
            area_groups[group_area]["iou"].append(self.iou(pred, label))

        metrics_result = {}
        dice_all = []
        iou_all = []

        for gr in area_groups:
            dice_all += area_groups[gr]["dice"]
            iou_all += area_groups[gr]["iou"]
            metrics_result[f"dice_group_{gr}"] = np.mean(area_groups[gr]["dice"])
            metrics_result[f"iou_group_{gr}"] = np.mean(area_groups[gr]["iou"])

        metrics_result["dice"] = np.mean(dice_all)
        metrics_result["iou"] = np.mean(iou_all)

        return metrics_result
