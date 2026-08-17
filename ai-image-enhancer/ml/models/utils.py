"""Utility functions for model inspection."""

from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
