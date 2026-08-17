"""Tests for model validation utilities."""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import SRCNN
from ml.training import create_loss_function, validate_model


class TinySyntheticDataset(Dataset):
    """Small in-memory dataset for fast validation tests."""

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


class EmptyDataset(Dataset):
    """Dataset with no samples for empty DataLoader tests."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise IndexError(index)


def _create_validation_components(batch_size: int = 2, num_samples: int = 4):
    model = SRCNN()
    dataset = TinySyntheticDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = create_loss_function()
    return model, dataloader, criterion


def test_validate_model_returns_float():
    model, dataloader, criterion = _create_validation_components()

    loss_value = validate_model(model, dataloader, criterion)

    assert isinstance(loss_value, float)


def test_validate_model_loss_is_finite():
    model, dataloader, criterion = _create_validation_components()

    loss_value = validate_model(model, dataloader, criterion)

    assert loss_value >= 0.0
    assert loss_value < float("inf")


def test_validate_model_does_not_change_parameters():
    model, dataloader, criterion = _create_validation_components()
    parameters_before = [parameter.clone().detach() for parameter in model.parameters()]

    validate_model(model, dataloader, criterion)

    parameters_after = list(model.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after, strict=True)
    )


def test_validate_model_leaves_model_in_evaluation_mode():
    model, dataloader, criterion = _create_validation_components()
    model.train()

    validate_model(model, dataloader, criterion)

    assert not model.training


def test_validate_model_works_with_batch_size_one():
    model, dataloader, criterion = _create_validation_components(batch_size=1, num_samples=2)

    loss_value = validate_model(model, dataloader, criterion)

    assert isinstance(loss_value, float)
    assert loss_value >= 0.0


def test_validate_model_works_with_multiple_batches():
    model, dataloader, criterion = _create_validation_components(batch_size=2, num_samples=6)

    loss_value = validate_model(model, dataloader, criterion)

    assert isinstance(loss_value, float)
    assert loss_value >= 0.0


def test_validate_model_empty_dataloader_raises_error():
    model = SRCNN()
    criterion = create_loss_function()
    empty_dataloader = DataLoader(EmptyDataset(), batch_size=1, shuffle=False, num_workers=0)

    with pytest.raises(ValueError, match="DataLoader produced no validation batches"):
        validate_model(model, empty_dataloader, criterion)
