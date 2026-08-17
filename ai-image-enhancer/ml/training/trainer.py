"""Training loop utilities for SRCNN."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .train_step import train_step
from .validation import validate_model


@dataclass
class TrainingHistory:
    """Training and validation losses recorded for each epoch."""

    train_loss: list[float]
    validation_loss: list[float]


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    epochs: int = 1,
) -> TrainingHistory:
    """Train a model for a number of epochs and validate after each epoch.

    Args:
        model: The PyTorch model to train.
        train_dataloader: DataLoader that yields training input and target batches.
        validation_dataloader: DataLoader that yields validation input and target batches.
        criterion: Loss function used during training and validation.
        optimizer: Optimizer used to update model parameters.
        epochs: Number of full passes through the training data.

    Returns:
        A TrainingHistory containing average training and validation losses
        for each epoch.

    Raises:
        ValueError: If epochs is not a positive integer or a DataLoader is empty.
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    train_losses: list[float] = []
    validation_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        batch_losses: list[float] = []

        for input_tensor, target_tensor in train_dataloader:
            loss = train_step(
                model,
                input_tensor,
                target_tensor,
                criterion,
                optimizer,
            )
            batch_losses.append(loss)

        if not batch_losses:
            raise ValueError("DataLoader produced no training batches.")

        average_train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(average_train_loss)

        validation_loss = validate_model(
            model,
            validation_dataloader,
            criterion,
        )
        validation_losses.append(validation_loss)

    model.train()
    return TrainingHistory(
        train_loss=train_losses,
        validation_loss=validation_losses,
    )
