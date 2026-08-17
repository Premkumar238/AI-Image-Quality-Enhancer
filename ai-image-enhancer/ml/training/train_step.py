"""Single-step training utilities."""

from typing import Callable

import torch
from torch import nn, optim


def train_step(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
) -> float:
    """Perform one SRCNN training step.

    Args:
        model: The PyTorch model to train.
        input_tensor: Low-resolution input batch.
        target_tensor: High-resolution target batch with the same shape as the model output.
        criterion: Loss function that compares prediction and target.
        optimizer: Optimizer used to update model parameters.

    Returns:
        The training loss as a Python float.

    Raises:
        TypeError: If input or target is not a PyTorch tensor.
        ValueError: If input and target shapes are incompatible.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a PyTorch tensor.")

    if not isinstance(target_tensor, torch.Tensor):
        raise TypeError("target_tensor must be a PyTorch tensor.")

    if input_tensor.shape != target_tensor.shape:
        raise ValueError(
            "input_tensor and target_tensor must have the same shape for this training step."
        )

    model.train()
    optimizer.zero_grad()

    prediction = model(input_tensor)
    loss = criterion(prediction, target_tensor)

    loss.backward()
    optimizer.step()

    return float(loss.item())
