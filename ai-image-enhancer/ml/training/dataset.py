"""PyTorch Dataset for super-resolution training."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.preprocessing import (
    create_training_pair,
    load_image,
    normalize_image,
    numpy_to_tensor,
    pil_to_numpy,
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _find_supported_image_paths(directory: Path) -> list[Path]:
    """Return sorted paths to supported image files in a directory."""
    image_paths: list[Path] = []

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(path)

    return image_paths


def _is_valid_image(path: Path) -> bool:
    """Return True if the image can be loaded successfully."""
    try:
        load_image(path)
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


def _pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a (3, H, W) float tensor."""
    array = pil_to_numpy(image)
    normalized = normalize_image(array)
    tensor = numpy_to_tensor(normalized)
    return tensor.squeeze(0)


class SuperResolutionDataset(Dataset):
    """Dataset that creates low-resolution and high-resolution training pairs."""

    def __init__(self, image_dir: str | Path, scale_factor: int = 4) -> None:
        """Initialize the dataset from a directory of high-resolution images.

        Args:
            image_dir: Directory containing high-resolution training images.
            scale_factor: Factor used to create low-resolution inputs.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If no supported images are found.
        """
        directory = Path(image_dir)

        if not directory.exists():
            raise FileNotFoundError(f"Image directory not found: {directory}")

        if not directory.is_dir():
            raise FileNotFoundError(f"Path is not a directory: {directory}")

        candidate_paths = _find_supported_image_paths(directory)
        self.image_paths = [path for path in candidate_paths if _is_valid_image(path)]

        if not self.image_paths:
            raise ValueError(f"No supported images found in directory: {directory}")

        if scale_factor not in {2, 3, 4}:
            raise ValueError("scale_factor must be 2, 3, or 4.")

        self.scale_factor = scale_factor

    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one low-resolution input and high-resolution target pair."""
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(f"Index {index} is out of range for dataset of size {len(self)}")

        image_path = self.image_paths[index]

        try:
            high_resolution_image = load_image(image_path)
            pair = create_training_pair(
                high_resolution_image,
                scale_factor=self.scale_factor,
            )
            input_tensor = _pil_image_to_tensor(pair.input_image)
            target_tensor = _pil_image_to_tensor(pair.target_image)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise RuntimeError(
                f"Failed to load training sample from: {image_path}"
            ) from exc

        return input_tensor, target_tensor
