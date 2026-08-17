"""Checkpoint saving and loading utilities for PyTorch models."""

from pathlib import Path

import torch
from torch import nn


def save_model(model: nn.Module, file_path: str | Path) -> None:
    """Save a model's weights to a checkpoint file.

    Args:
        model: The PyTorch model whose weights will be saved.
        file_path: Destination path for the checkpoint file.

    Raises:
        TypeError: If model is not a PyTorch nn.Module.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch nn.Module.")

    checkpoint_path = Path(file_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_path)


def load_model(
    model: nn.Module,
    file_path: str | Path,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load model weights from a checkpoint file.

    Args:
        model: An initialized model with the correct architecture.
        file_path: Path to the saved checkpoint file.
        device: Device used when loading the checkpoint.

    Returns:
        The same model instance with loaded weights.

    Raises:
        TypeError: If model is not a PyTorch nn.Module.
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch nn.Module.")

    checkpoint_path = Path(file_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model
