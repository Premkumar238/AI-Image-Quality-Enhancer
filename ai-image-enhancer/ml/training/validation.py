"""Validation utilities for super-resolution models."""

from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def validate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    """Evaluate a model on a validation DataLoader without updating weights.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader that yields input and target tensor batches.
        criterion: Loss function used to compare predictions and targets.

    Returns:
        The average validation loss as a Python float.

    Raises:
        ValueError: If the DataLoader produces no batches.
    """
    model.eval()
    batch_losses: list[float] = []

    with torch.no_grad():
        for input_tensor, target_tensor in dataloader:
            prediction = model(input_tensor)
            loss = criterion(prediction, target_tensor)
            batch_losses.append(float(loss.item()))

    if not batch_losses:
        raise ValueError("DataLoader produced no validation batches.")

    return sum(batch_losses) / len(batch_losses)
