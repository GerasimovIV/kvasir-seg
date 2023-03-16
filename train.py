from pathlib import Path
from typing import Optional, Sequence, Union

import yaml
from torch import nn
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup

import wandb
from data_utils.dataset import load_datasets
from src.losses import load_loss
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

    model = load_model(train_config)
    train_dataset, test_dataset = load_datasets(train_config)
    optimizer = load_optimizer(resource=train_config, model=model)
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
    )

    return trainer
