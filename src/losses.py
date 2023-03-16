from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torchgeometry.losses import DiceLoss

from .boundary_loss import BoundaryLoss


class WrappedCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)


class WrappedDiceLoss(DiceLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target)


class WrappedBoundaryLoss(BoundaryLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return super().forward(input, target)


losses_funcs = {
    "CrossEntropy": WrappedCrossEntropy,
    "DiceLoss": WrappedDiceLoss,
    "BoundaryLoss": WrappedBoundaryLoss,
}


class LossCompose(nn.Module):
    def __init__(
        self, losses: Sequence[nn.Module], weights: Optional[Sequence[float]] = None
    ):
        super().__init__()
        if weights is None:
            weights = [1.0 for i in range(len(losses))]

        assert len(losses) == len(weights), "losses and their weights should be matched"
        self.losses = losses
        self.weights = weights

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        device = input.device
        loss_sum = torch.tensor(0.0, requires_grad=True, device=device)

        for w, loss_f in zip(self.weights, self.losses):
            loss_sum += w * loss_f(input, target)

        return loss_sum


def load_loss(resource: Union[Path, str, Dict[str, Any]]) -> LossCompose:
    """
    loss loader from configureation, suppose that resource
    contains field 'loss_config'
    """
    config = resource
    if isinstance(resource, Path) or isinstance(resource, str):
        resource = Path(resource)

        with open(resource, "r") as file:
            config = yaml.safe_load(file)

        assert "loss_config" in config, f"wrong resource {resource} construction"

    loss_config = config.get("loss_config")
    losses_list = []

    for i in range(len(loss_config["losses"])):
        name = list(loss_config["losses"][i].keys())[0]
        params = loss_config["losses"][i].get(name)
        params = {} if params is None else params
        for k in params:
            if isinstance(params[k], List):
                params[k] = Tensor(params[k])
        loss = losses_funcs[name](**params)

        losses_list.append(loss)

    weights = loss_config["weights"]

    loss_compose = LossCompose(losses=losses_list, weights=weights)

    return loss_compose
