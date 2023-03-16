from pathlib import Any, Dict, Path, Union

import torch.nn as nn
import torch.optim as optim
import yaml

optimizers = {"SGD": optim.SGD, "Adam": optim.Adam}


def load_optimizer(
    resource: Union[Path, str, Dict[str, Any]], model: nn.Module
) -> optim.optimizer.Optimizer:
    """
    loss loader from configureation, suppose that resource
    contains field 'loss_config'
    """

    if isinstance(resource, Path) or isinstance(resource, str):
        resource = Path(resource)

        with open(resource, "r") as file:
            config = yaml.safe_load(file)

        assert "optimizer" in config, f"wrong resource {resource} construction"

    name = config["optimizer"]["name"]
    params = config["optimizer"]["params"]

    optimizer = optimizers[name](model.parameters(), **params)
    return optimizer
