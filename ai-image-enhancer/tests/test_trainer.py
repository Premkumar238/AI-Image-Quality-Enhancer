"""Tests for the SRCNN training loop."""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import SRCNN
from ml.training import (
    TrainingHistory,
    create_loss_function,
    create_optimizer,
    train_model,
    train_step,
    validate_model,
)


class TinySyntheticDataset(Dataset):
    """Small in-memory dataset for fast trainer tests."""

    def __init__(self, num_samples: int = 4, height: int = 16, width: int = 16) -> None:
        self.num_samples = num_samples
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        input_tensor = torch.randn(3, self.height, self.width, generator=generator)
        target_tensor = torch.randn(3, self.height, self.width, generator=generator)
        return input_tensor, target_tensor


def _create_training_components(batch_size: int = 2, num_samples: int = 6):
    model = SRCNN()
    train_dataset = TinySyntheticDataset(num_samples=num_samples, height=16, width=16)
    validation_dataset = TinySyntheticDataset(num_samples=num_samples, height=16, width=16)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    criterion = create_loss_function()
    optimizer = create_optimizer(model)
    return model, train_dataloader, validation_dataloader, criterion, optimizer


def test_train_model_one_epoch_returns_one_training_loss():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    history = train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs=1,
    )

    assert isinstance(history, TrainingHistory)
    assert len(history.train_loss) == 1


def test_train_model_one_epoch_returns_one_validation_loss():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    history = train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs=1,
    )

    assert len(history.validation_loss) == 1


def test_train_model_three_epochs_returns_three_training_losses():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    history = train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs=3,
    )

    assert len(history.train_loss) == 3


def test_train_model_three_epochs_returns_three_validation_losses():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    history = train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs=3,
    )

    assert len(history.validation_loss) == 3


def test_train_model_losses_are_floats():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    history = train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs=2,
    )

    assert all(isinstance(loss, float) for loss in history.train_loss)
    assert all(isinstance(loss, float) for loss in history.validation_loss)


def test_train_model_losses_are_finite():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    history = train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs=2,
    )

    for loss in history.train_loss + history.validation_loss:
        assert loss >= 0.0
        assert loss < float("inf")


def test_train_model_updates_parameters():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )
    parameters_before = [parameter.clone().detach() for parameter in model.parameters()]

    train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs=1,
    )

    parameters_after = list(model.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after, strict=True)
    )


def test_train_model_validation_does_not_modify_parameters():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    model.train()
    for input_tensor, target_tensor in train_dataloader:
        train_step(model, input_tensor, target_tensor, criterion, optimizer)

    parameters_before_validation = [
        parameter.clone().detach() for parameter in model.parameters()
    ]

    validate_model(model, validation_dataloader, criterion)

    parameters_after_validation = list(model.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            parameters_before_validation,
            parameters_after_validation,
            strict=True,
        )
    )


def test_train_model_zero_epochs_raises_error():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        train_model(
            model,
            train_dataloader,
            validation_dataloader,
            criterion,
            optimizer,
            epochs=0,
        )


def test_train_model_negative_epochs_raises_error():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        train_model(
            model,
            train_dataloader,
            validation_dataloader,
            criterion,
            optimizer,
            epochs=-1,
        )


def test_train_model_leaves_model_in_training_mode():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components()
    )

    train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs=2,
    )

    assert model.training


def test_train_model_supports_multiple_batches():
    model, train_dataloader, validation_dataloader, criterion, optimizer = (
        _create_training_components(batch_size=2, num_samples=8)
    )

    history = train_model(
        model,
        train_dataloader,
        validation_dataloader,
        criterion,
        optimizer,
        epochs=1,
    )

    assert len(history.train_loss) == 1
    assert len(history.validation_loss) == 1
