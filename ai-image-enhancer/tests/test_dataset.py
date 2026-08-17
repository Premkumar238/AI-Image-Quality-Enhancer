"""Tests for the super-resolution dataset."""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import SuperResolutionDataset


def _create_rgb_image(path: Path, size: tuple[int, int], color: str) -> None:
    Image.new("RGB", size, color=color).save(path)


def _create_dataset_directory(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    _create_rgb_image(image_dir / "first.png", (100, 100), "red")
    _create_rgb_image(image_dir / "second.jpg", (80, 60), "blue")
    _create_rgb_image(image_dir / "third.jpeg", (120, 120), "green")
    (image_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    (image_dir / "broken.png").write_text("not an image", encoding="utf-8")

    return image_dir


def test_dataset_finds_valid_image_files(tmp_path):
    image_dir = _create_dataset_directory(tmp_path)
    dataset = SuperResolutionDataset(image_dir, scale_factor=4)

    assert len(dataset.image_paths) == 3


def test_dataset_ignores_unsupported_files(tmp_path):
    image_dir = _create_dataset_directory(tmp_path)
    dataset = SuperResolutionDataset(image_dir, scale_factor=4)

    suffixes = {path.suffix.lower() for path in dataset.image_paths}
    assert ".txt" not in suffixes


def test_dataset_length(tmp_path):
    image_dir = _create_dataset_directory(tmp_path)
    dataset = SuperResolutionDataset(image_dir, scale_factor=4)

    assert len(dataset) == 3


def test_dataset_getitem_works(tmp_path):
    image_dir = _create_dataset_directory(tmp_path)
    dataset = SuperResolutionDataset(image_dir, scale_factor=4)

    input_tensor, target_tensor = dataset[0]

    assert input_tensor is not None
    assert target_tensor is not None


def test_dataset_input_is_tensor(tmp_path):
    image_dir = _create_dataset_directory(tmp_path)
    dataset = SuperResolutionDataset(image_dir, scale_factor=4)

    input_tensor, _ = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)


def test_dataset_target_is_tensor(tmp_path):
    image_dir = _create_dataset_directory(tmp_path)
    dataset = SuperResolutionDataset(image_dir, scale_factor=4)

    _, target_tensor = dataset[0]

    assert isinstance(target_tensor, torch.Tensor)


def test_dataset_input_shape(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _create_rgb_image(image_dir / "sample.png", (100, 100), "red")

    dataset = SuperResolutionDataset(image_dir, scale_factor=4)
    input_tensor, _ = dataset[0]

    assert input_tensor.shape == (3, 25, 25)


def test_dataset_target_shape_is_scaled(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _create_rgb_image(image_dir / "sample.png", (100, 100), "red")

    dataset = SuperResolutionDataset(image_dir, scale_factor=4)
    input_tensor, target_tensor = dataset[0]

    assert target_tensor.shape == (3, 100, 100)
    assert target_tensor.shape[1] == input_tensor.shape[1] * 4
    assert target_tensor.shape[2] == input_tensor.shape[2] * 4


def test_dataset_input_values_between_zero_and_one(tmp_path):
    image_dir = _create_dataset_directory(tmp_path)
    dataset = SuperResolutionDataset(image_dir, scale_factor=4)

    input_tensor, _ = dataset[0]

    assert input_tensor.min().item() >= 0.0
    assert input_tensor.max().item() <= 1.0


def test_dataset_target_values_between_zero_and_one(tmp_path):
    image_dir = _create_dataset_directory(tmp_path)
    dataset = SuperResolutionDataset(image_dir, scale_factor=4)

    _, target_tensor = dataset[0]

    assert target_tensor.min().item() >= 0.0
    assert target_tensor.max().item() <= 1.0


def test_dataset_scale_factor_2(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _create_rgb_image(image_dir / "sample.png", (40, 40), "yellow")

    dataset = SuperResolutionDataset(image_dir, scale_factor=2)
    input_tensor, target_tensor = dataset[0]

    assert input_tensor.shape == (3, 20, 20)
    assert target_tensor.shape == (3, 40, 40)


def test_dataset_scale_factor_4(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _create_rgb_image(image_dir / "sample.png", (80, 80), "purple")

    dataset = SuperResolutionDataset(image_dir, scale_factor=4)
    input_tensor, target_tensor = dataset[0]

    assert input_tensor.shape == (3, 20, 20)
    assert target_tensor.shape == (3, 80, 80)


def test_dataset_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        SuperResolutionDataset(missing_dir)


def test_dataset_empty_image_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No supported images found"):
        SuperResolutionDataset(empty_dir)
