"""Tests for training loss functions."""

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import create_loss_function


def test_create_loss_function_returns_mse():
    criterion = create_loss_function()

    assert isinstance(criterion, nn.MSELoss)


def test_identical_tensors_produce_zero_loss():
    criterion = create_loss_function()
    tensor = torch.tensor([0.5, 0.25, 0.75])

    loss = criterion(tensor, tensor)

    assert loss.item() == 0.0


def test_different_tensors_produce_positive_loss():
    criterion = create_loss_function()
    prediction = torch.tensor([0.0, 0.0])
    target = torch.tensor([1.0, 1.0])

    loss = criterion(prediction, target)

    assert loss.item() > 0.0


def test_loss_with_image_shaped_tensors():
    criterion = create_loss_function()
    prediction = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)

    loss = criterion(prediction, target)

    assert loss.shape == torch.Size([])


def test_loss_is_scalar_tensor():
    criterion = create_loss_function()
    prediction = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)

    loss = criterion(prediction, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_loss_is_finite():
    criterion = create_loss_function()
    prediction = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)

    loss = criterion(prediction, target)

    assert torch.isfinite(loss).item()


def test_known_mse_example():
    criterion = create_loss_function()
    prediction = torch.tensor([0.0, 1.0])
    target = torch.tensor([1.0, 1.0])

    loss = criterion(prediction, target)

    assert loss.item() == 0.5
