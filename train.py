from pathlib import Path
from typing import Optional, Sequence, Union

import yaml
from torch import nn
from torch.utils.data import Subset
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup

import wandb
from src.data_utils.dataset import load_datasets
from src.losses import load_loss
from src.metrics import ComputeMetrics
from src.models import load_model
from src.optimizers import load_optimizer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        outputs = model(**inputs)
        # compute custom loss (suppose one has 3 labels with different weights)
        loss = outputs["loss"]
        outputs = outputs["logits"]
        return (loss, outputs) if return_outputs else loss


def train_setup(train_config_path: Union[str, Path] = r"./train_config.yaml"):
    with open(train_config_path) as file:
        train_config = yaml.safe_load(file)

    train_dataset, test_dataset = load_datasets(train_config)

    # train_dataset = Subset(train_dataset, [0, 1, 2, 4])
    # test_dataset = Subset(test_dataset, [9, 8, 7, 6, 5])

    model = load_model(train_config)
    optimizer = load_optimizer(resource=train_config, model=model)
    compute_metrics = ComputeMetrics(train_config)

    trainer_args_config = train_config["trainer_args"]
    trainer_args = TrainingArguments(**trainer_args_config)

    wandb_init_params = train_config["wandb_init"]
    wandb.init(**wandb_init_params)

    sheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=200,
        num_training_steps=trainer_args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=(optimizer, sheduler),
        compute_metrics=compute_metrics,
    )

    return trainer
