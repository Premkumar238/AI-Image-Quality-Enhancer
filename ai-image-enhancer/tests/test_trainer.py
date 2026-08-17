"""Tests for the SRCNN training loop."""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import SRCNN
from ml.training import create_loss_function, create_optimizer, train_model


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


def _create_training_components(batch_size: int = 2):
    model = SRCNN()
    dataset = TinySyntheticDataset(num_samples=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = create_loss_function()
    optimizer = create_optimizer(model)
    return model, dataloader, criterion, optimizer


def test_train_model_returns_list_for_one_epoch():
    model, dataloader, criterion, optimizer = _create_training_components()

    losses = train_model(model, dataloader, criterion, optimizer, epochs=1)

    assert isinstance(losses, list)


def test_train_model_one_epoch_returns_one_loss():
    model, dataloader, criterion, optimizer = _create_training_components()

    losses = train_model(model, dataloader, criterion, optimizer, epochs=1)

    assert len(losses) == 1


def test_train_model_loss_is_finite_float():
    model, dataloader, criterion, optimizer = _create_training_components()

    losses = train_model(model, dataloader, criterion, optimizer, epochs=1)

    assert isinstance(losses[0], float)
    assert losses[0] >= 0.0
    assert losses[0] < float("inf")


def test_train_model_two_epochs_returns_two_losses():
    model, dataloader, criterion, optimizer = _create_training_components()

    losses = train_model(model, dataloader, criterion, optimizer, epochs=2)

    assert len(losses) == 2


def test_train_model_zero_epochs_raises_error():
    model, dataloader, criterion, optimizer = _create_training_components()

    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        train_model(model, dataloader, criterion, optimizer, epochs=0)


def test_train_model_negative_epochs_raises_error():
    model, dataloader, criterion, optimizer = _create_training_components()

    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        train_model(model, dataloader, criterion, optimizer, epochs=-1)


def test_train_model_updates_parameters():
    model, dataloader, criterion, optimizer = _create_training_components()
    parameters_before = [parameter.clone().detach() for parameter in model.parameters()]

    train_model(model, dataloader, criterion, optimizer, epochs=1)

    parameters_after = list(model.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after, strict=True)
    )
