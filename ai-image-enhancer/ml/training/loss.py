"""Loss function utilities for model training."""

from torch import nn


def create_loss_function() -> nn.MSELoss:
    """Create the loss function used for SRCNN training.

    Returns:
        An MSELoss instance for comparing predicted and target images.
    """
    return nn.MSELoss()
