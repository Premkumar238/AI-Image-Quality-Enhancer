"""Tests for training optimizer utilities."""

import sys
from pathlib import Path

import pytest
import torch
from torch import optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import SRCNN
from ml.training import create_loss_function, create_optimizer


def test_create_optimizer_returns_adam():
    model = SRCNN()
    optimizer = create_optimizer(model)

    assert isinstance(optimizer, optim.Adam)


def test_optimizer_contains_model_parameters():
    model = SRCNN()
    optimizer = create_optimizer(model)

    model_param_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_param_ids = {id(group["params"][0]) for group in optimizer.param_groups}

    assert model_param_ids.intersection(optimizer_param_ids)


def test_default_learning_rate():
    model = SRCNN()
    optimizer = create_optimizer(model)

    assert optimizer.param_groups[0]["lr"] == 0.001


def test_custom_learning_rate():
    model = SRCNN()
    optimizer = create_optimizer(model, learning_rate=0.01)

    assert optimizer.param_groups[0]["lr"] == 0.01


def test_zero_learning_rate_raises_error():
    model = SRCNN()

    with pytest.raises(ValueError, match="learning_rate must be a positive number"):
        create_optimizer(model, learning_rate=0)


def test_negative_learning_rate_raises_error():
    model = SRCNN()

    with pytest.raises(ValueError, match="learning_rate must be a positive number"):
        create_optimizer(model, learning_rate=-0.001)


def test_non_numeric_learning_rate_raises_error():
    model = SRCNN()

    with pytest.raises(TypeError, match="learning_rate must be a number"):
        create_optimizer(model, learning_rate="0.001")


def test_optimizer_can_perform_basic_step():
    model = SRCNN()
    optimizer = create_optimizer(model)
    criterion = create_loss_function()
    input_tensor = torch.randn(1, 3, 16, 16)
    target = torch.randn(1, 3, 16, 16)

    prediction = model(input_tensor)
    loss = criterion(prediction, target)

    loss.backward()
    optimizer.step()

    assert loss.item() >= 0.0
