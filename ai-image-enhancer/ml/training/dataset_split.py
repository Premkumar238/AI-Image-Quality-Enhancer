"""Dataset splitting utilities for training and validation images."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class DatasetSplitResult:
    """Summary of a train/validation dataset split."""

    total: int
    train: int
    validation: int


def _find_supported_image_paths(directory: Path) -> list[Path]:
    """Return sorted paths to supported image files in a directory."""
    image_paths: list[Path] = []

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(path)

    return image_paths


def _validate_validation_ratio(validation_ratio: float) -> None:
    """Validate the requested validation split ratio."""
    if not isinstance(validation_ratio, (int, float)) or isinstance(validation_ratio, bool):
        raise ValueError("validation_ratio must be greater than 0 and less than 1.")

    if validation_ratio <= 0 or validation_ratio >= 1:
        raise ValueError("validation_ratio must be greater than 0 and less than 1.")


def _calculate_validation_count(total_images: int, validation_ratio: float) -> int:
    """Calculate validation count while keeping both splits non-empty."""
    validation_count = int(round(total_images * validation_ratio))
    validation_count = max(1, validation_count)
    validation_count = min(validation_count, total_images - 1)
    return validation_count


def split_dataset(
    image_directory: str | Path,
    train_directory: str | Path,
    validation_directory: str | Path,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetSplitResult:
    """Split image files into training and validation directories.

    Supported image extensions are ``.jpg``, ``.jpeg``, and ``.png``.
    Files are copied into the destination directories; the source images
    are preserved.

    Args:
        image_directory: Directory containing source image files.
        train_directory: Destination directory for training images.
        validation_directory: Destination directory for validation images.
        validation_ratio: Fraction of images reserved for validation.
        seed: Random seed used for deterministic shuffling.

    Returns:
        A DatasetSplitResult with the total, training, and validation counts.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        ValueError: If validation_ratio is invalid, no supported images are
            found, or fewer than two images are available for splitting.
    """
    _validate_validation_ratio(validation_ratio)

    source_directory = Path(image_directory)
    train_path = Path(train_directory)
    validation_path = Path(validation_directory)

    if not source_directory.exists():
        raise FileNotFoundError(f"Image directory not found: {source_directory}")

    if not source_directory.is_dir():
        raise FileNotFoundError(f"Path is not a directory: {source_directory}")

    image_paths = _find_supported_image_paths(source_directory)

    if not image_paths:
        raise ValueError(f"No supported images found in directory: {source_directory}")

    if len(image_paths) < 2:
        raise ValueError(
            "At least 2 supported images are required for a train/validation split."
        )

    shuffled_paths = image_paths.copy()
    random.Random(seed).shuffle(shuffled_paths)

    validation_count = _calculate_validation_count(len(shuffled_paths), validation_ratio)
    validation_paths = shuffled_paths[:validation_count]
    train_paths = shuffled_paths[validation_count:]

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)

    for path in train_paths:
        shutil.copy2(path, train_path / path.name)

    for path in validation_paths:
        shutil.copy2(path, validation_path / path.name)

    return DatasetSplitResult(
        total=len(image_paths),
        train=len(train_paths),
        validation=len(validation_paths),
    )
