"""Optimizer utilities for model training."""

from numbers import Real

import torch
from torch import nn, optim


def create_optimizer(model: nn.Module, learning_rate: float = 0.001) -> optim.Adam:
    """Create an Adam optimizer for a PyTorch model.

    Args:
        model: The PyTorch model whose parameters will be optimized.
        learning_rate: Step size used when updating model weights.

    Returns:
        An Adam optimizer configured for the model parameters.

    Raises:
        TypeError: If learning_rate is not numeric.
        ValueError: If learning_rate is not a positive number.
    """
    if not isinstance(learning_rate, Real):
        raise TypeError("learning_rate must be a number.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be a positive number.")

    return optim.Adam(model.parameters(), lr=learning_rate)
