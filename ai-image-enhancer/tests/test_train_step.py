"""Tests for single-step training utilities."""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import SRCNN
from ml.training import create_loss_function, create_optimizer, train_step


def _create_training_setup():
    model = SRCNN()
    criterion = create_loss_function()
    optimizer = create_optimizer(model)
    input_tensor = torch.randn(2, 3, 32, 32)
    target_tensor = torch.randn(2, 3, 32, 32)
    return model, criterion, optimizer, input_tensor, target_tensor


def test_train_step_returns_float():
    model, criterion, optimizer, input_tensor, target_tensor = _create_training_setup()

    loss_value = train_step(model, input_tensor, target_tensor, criterion, optimizer)

    assert isinstance(loss_value, float)


def test_train_step_loss_is_finite():
    model, criterion, optimizer, input_tensor, target_tensor = _create_training_setup()

    loss_value = train_step(model, input_tensor, target_tensor, criterion, optimizer)

    assert loss_value >= 0.0
    assert loss_value < float("inf")


def test_train_step_parameters_have_gradients():
    model, criterion, optimizer, input_tensor, target_tensor = _create_training_setup()

    train_step(model, input_tensor, target_tensor, criterion, optimizer)

    assert any(parameter.grad is not None for parameter in model.parameters())


def test_train_step_updates_parameters():
    model, criterion, optimizer, input_tensor, target_tensor = _create_training_setup()
    parameters_before = [parameter.clone().detach() for parameter in model.parameters()]

    train_step(model, input_tensor, target_tensor, criterion, optimizer)

    parameters_after = list(model.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after, strict=True)
    )


def test_train_step_sets_training_mode():
    model, criterion, optimizer, input_tensor, target_tensor = _create_training_setup()
    model.eval()

    train_step(model, input_tensor, target_tensor, criterion, optimizer)

    assert model.training


def test_train_step_invalid_input_type():
    model, criterion, optimizer, _, target_tensor = _create_training_setup()

    with pytest.raises(TypeError, match="input_tensor must be a PyTorch tensor"):
        train_step(model, [[1.0]], target_tensor, criterion, optimizer)


def test_train_step_invalid_target_type():
    model, criterion, optimizer, input_tensor, _ = _create_training_setup()

    with pytest.raises(TypeError, match="target_tensor must be a PyTorch tensor"):
        train_step(model, input_tensor, [[1.0]], criterion, optimizer)


def test_train_step_incompatible_shapes():
    model, criterion, optimizer, input_tensor, _ = _create_training_setup()
    incompatible_target = torch.randn(2, 3, 16, 16)

    with pytest.raises(ValueError, match="input_tensor and target_tensor must have the same shape"):
        train_step(model, input_tensor, incompatible_target, criterion, optimizer)
