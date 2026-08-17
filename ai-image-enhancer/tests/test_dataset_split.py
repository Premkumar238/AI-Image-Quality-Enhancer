"""Tests for the dataset splitting utility."""

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import DatasetSplitResult, split_dataset


def _create_rgb_image(path: Path, size: tuple[int, int] = (32, 32), color: str = "red") -> None:
    Image.new("RGB", size, color=color).save(path)


def _create_source_directory(tmp_path: Path, image_count: int = 10) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    for index in range(image_count):
        extension = [".jpg", ".jpeg", ".png"][index % 3]
        _create_rgb_image(
            source_dir / f"image_{index:02d}{extension}",
            color=["red", "green", "blue"][index % 3],
        )

    (source_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    (source_dir / "archive.zip").write_bytes(b"not an image")

    return source_dir


def test_split_dataset_detects_source_directory(tmp_path):
    source_dir = _create_source_directory(tmp_path)
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    result = split_dataset(source_dir, train_dir, validation_dir)

    assert isinstance(result, DatasetSplitResult)
    assert result.total == 10


def test_split_dataset_includes_jpg_images(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _create_rgb_image(source_dir / "photo.jpg")

    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    with pytest.raises(ValueError, match="At least 2 supported images"):
        split_dataset(source_dir, train_dir, validation_dir)

    _create_rgb_image(source_dir / "photo2.jpg")
    result = split_dataset(source_dir, train_dir, validation_dir)

    assert result.total == 2
    assert any(path.suffix.lower() == ".jpg" for path in train_dir.iterdir()) or any(
        path.suffix.lower() == ".jpg" for path in validation_dir.iterdir()
    )


def test_split_dataset_includes_jpeg_images(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _create_rgb_image(source_dir / "photo.jpeg")
    _create_rgb_image(source_dir / "photo2.jpeg")

    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    result = split_dataset(source_dir, train_dir, validation_dir)

    copied_names = {path.name for path in train_dir.iterdir()} | {
        path.name for path in validation_dir.iterdir()
    }
    assert "photo.jpeg" in copied_names
    assert "photo2.jpeg" in copied_names
    assert result.total == 2


def test_split_dataset_includes_png_images(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _create_rgb_image(source_dir / "photo.png")
    _create_rgb_image(source_dir / "photo2.png")

    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    result = split_dataset(source_dir, train_dir, validation_dir)

    copied_names = {path.name for path in train_dir.iterdir()} | {
        path.name for path in validation_dir.iterdir()
    }
    assert "photo.png" in copied_names
    assert result.total == 2


def test_split_dataset_ignores_unsupported_files(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=6)
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    result = split_dataset(source_dir, train_dir, validation_dir)

    copied_names = {path.name for path in train_dir.iterdir()} | {
        path.name for path in validation_dir.iterdir()
    }

    assert "notes.txt" not in copied_names
    assert "archive.zip" not in copied_names
    assert result.total == 6


def test_split_dataset_splits_images_correctly(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=10)
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    result = split_dataset(source_dir, train_dir, validation_dir, validation_ratio=0.2)

    assert result.total == 10
    assert result.train + result.validation == 10
    assert result.train == 8
    assert result.validation == 2
    assert len(list(train_dir.iterdir())) == 8
    assert len(list(validation_dir.iterdir())) == 2


def test_split_dataset_validation_ratio_0_2(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=10)
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    result = split_dataset(source_dir, train_dir, validation_dir, validation_ratio=0.2)

    assert result.validation == 2
    assert result.train == 8


def test_split_dataset_validation_ratio_0_3(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=10)
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    result = split_dataset(source_dir, train_dir, validation_dir, validation_ratio=0.3)

    assert result.validation == 3
    assert result.train == 7


def test_split_dataset_same_seed_produces_same_split(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=10)

    first_train_dir = tmp_path / "train_first"
    first_validation_dir = tmp_path / "validation_first"
    second_train_dir = tmp_path / "train_second"
    second_validation_dir = tmp_path / "validation_second"

    split_dataset(source_dir, first_train_dir, first_validation_dir, seed=42)
    split_dataset(source_dir, second_train_dir, second_validation_dir, seed=42)

    first_train_names = sorted(path.name for path in first_train_dir.iterdir())
    second_train_names = sorted(path.name for path in second_train_dir.iterdir())
    first_validation_names = sorted(path.name for path in first_validation_dir.iterdir())
    second_validation_names = sorted(path.name for path in second_validation_dir.iterdir())

    assert first_train_names == second_train_names
    assert first_validation_names == second_validation_names


def test_split_dataset_different_seed_can_produce_different_split(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=20)

    first_train_dir = tmp_path / "train_seed_1"
    first_validation_dir = tmp_path / "validation_seed_1"
    second_train_dir = tmp_path / "train_seed_2"
    second_validation_dir = tmp_path / "validation_seed_2"

    split_dataset(source_dir, first_train_dir, first_validation_dir, seed=1)
    split_dataset(source_dir, second_train_dir, second_validation_dir, seed=99)

    first_train_names = sorted(path.name for path in first_train_dir.iterdir())
    second_train_names = sorted(path.name for path in second_train_dir.iterdir())

    assert first_train_names != second_train_names


def test_split_dataset_creates_destination_directories(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=4)
    train_dir = tmp_path / "nested" / "train"
    validation_dir = tmp_path / "nested" / "validation"

    split_dataset(source_dir, train_dir, validation_dir)

    assert train_dir.exists()
    assert validation_dir.exists()


def test_split_dataset_preserves_original_files(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=5)
    original_names = sorted(path.name for path in source_dir.iterdir() if path.is_file())
    original_contents = {
        path.name: path.read_bytes()
        for path in source_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    split_dataset(source_dir, train_dir, validation_dir)

    assert sorted(path.name for path in source_dir.iterdir() if path.is_file()) == original_names
    for name, content in original_contents.items():
        assert (source_dir / name).read_bytes() == content


def test_split_dataset_preserves_original_filenames(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=6)
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    split_dataset(source_dir, train_dir, validation_dir)

    copied_names = {path.name for path in train_dir.iterdir()} | {
        path.name for path in validation_dir.iterdir()
    }
    source_names = {
        path.name
        for path in source_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    assert copied_names == source_names


def test_split_dataset_two_images_keeps_non_empty_training_set(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _create_rgb_image(source_dir / "one.png")
    _create_rgb_image(source_dir / "two.png")

    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"

    result = split_dataset(source_dir, train_dir, validation_dir, validation_ratio=0.2)

    assert result.train >= 1
    assert result.validation >= 1


def test_split_dataset_invalid_validation_ratio_zero(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=4)

    with pytest.raises(ValueError, match="validation_ratio must be greater than 0 and less than 1"):
        split_dataset(source_dir, tmp_path / "train", tmp_path / "validation", validation_ratio=0)


def test_split_dataset_invalid_validation_ratio_one(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=4)

    with pytest.raises(ValueError, match="validation_ratio must be greater than 0 and less than 1"):
        split_dataset(source_dir, tmp_path / "train", tmp_path / "validation", validation_ratio=1)


def test_split_dataset_invalid_validation_ratio_negative(tmp_path):
    source_dir = _create_source_directory(tmp_path, image_count=4)

    with pytest.raises(ValueError, match="validation_ratio must be greater than 0 and less than 1"):
        split_dataset(
            source_dir,
            tmp_path / "train",
            tmp_path / "validation",
            validation_ratio=-0.1,
        )


def test_split_dataset_missing_source_directory(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        split_dataset(missing_dir, tmp_path / "train", tmp_path / "validation")


def test_split_dataset_empty_source_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No supported images found"):
        split_dataset(empty_dir, tmp_path / "train", tmp_path / "validation")


def test_split_dataset_single_image_raises_error(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _create_rgb_image(source_dir / "only.png")

    with pytest.raises(ValueError, match="At least 2 supported images"):
        split_dataset(source_dir, tmp_path / "train", tmp_path / "validation")
