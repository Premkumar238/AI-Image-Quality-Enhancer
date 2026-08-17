"""Tests for the super-resolution DataLoader utility."""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import SuperResolutionDataset, create_dataloader


def _create_rgb_image(path: Path, size: tuple[int, int], color: str) -> None:
    Image.new("RGB", size, color=color).save(path)


def _create_dataset_directory(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    _create_rgb_image(image_dir / "first.png", (100, 100), "red")
    _create_rgb_image(image_dir / "second.png", (100, 100), "blue")
    _create_rgb_image(image_dir / "third.png", (100, 100), "green")

    return image_dir


def test_dataloader_returns_batches(tmp_path):
    dataset = SuperResolutionDataset(_create_dataset_directory(tmp_path), scale_factor=4)
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)

    input_batch, target_batch = next(iter(dataloader))

    assert input_batch is not None
    assert target_batch is not None


def test_dataloader_input_batch_shape(tmp_path):
    dataset = SuperResolutionDataset(_create_dataset_directory(tmp_path), scale_factor=4)
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)

    input_batch, _ = next(iter(dataloader))

    assert input_batch.shape == (2, 3, 25, 25)


def test_dataloader_target_batch_shape(tmp_path):
    dataset = SuperResolutionDataset(_create_dataset_directory(tmp_path), scale_factor=4)
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)

    _, target_batch = next(iter(dataloader))

    assert target_batch.shape == (2, 3, 100, 100)


def test_dataloader_respects_batch_size(tmp_path):
    dataset = SuperResolutionDataset(_create_dataset_directory(tmp_path), scale_factor=4)
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)

    batches = list(dataloader)

    assert len(batches) == 2
    assert batches[0][0].shape[0] == 2
    assert batches[1][0].shape[0] == 1


def test_dataloader_shuffle_false(tmp_path):
    dataset = SuperResolutionDataset(_create_dataset_directory(tmp_path), scale_factor=4)
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    first_pass = [batch[0].clone() for batch in dataloader]
    second_pass = [batch[0].clone() for batch in dataloader]

    assert len(first_pass) == len(second_pass)
    for first_batch, second_batch in zip(first_pass, second_pass, strict=True):
        assert torch.equal(first_batch, second_batch)


def test_dataloader_invalid_batch_size(tmp_path):
    dataset = SuperResolutionDataset(_create_dataset_directory(tmp_path), scale_factor=4)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        create_dataloader(dataset, batch_size=0)


def test_dataloader_invalid_num_workers(tmp_path):
    dataset = SuperResolutionDataset(_create_dataset_directory(tmp_path), scale_factor=4)

    with pytest.raises(ValueError, match="num_workers must be a non-negative integer"):
        create_dataloader(dataset, batch_size=2, num_workers=-1)
