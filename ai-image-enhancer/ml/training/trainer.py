"""Training loop utilities for SRCNN."""

from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .train_step import train_step


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    epochs: int = 1,
) -> list[float]:
    """Train a model for a number of epochs using a DataLoader.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader that yields input and target tensor batches.
        criterion: Loss function used during training.
        optimizer: Optimizer used to update model parameters.
        epochs: Number of full passes through the training data.

    Returns:
        A list containing the average loss for each epoch.

    Raises:
        ValueError: If epochs is not a positive integer.
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    epoch_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        batch_losses: list[float] = []

        for input_tensor, target_tensor in dataloader:
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

        average_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(average_loss)

    return epoch_losses
