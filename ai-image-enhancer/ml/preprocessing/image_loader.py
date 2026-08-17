"""Image loading utilities for the preprocessing pipeline."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk and return it as an RGB PIL Image.

    Args:
        image_path: Path to the image file on disk.

    Returns:
        A PIL Image object converted to RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid image.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")

    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"File is not a valid image: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to open image file: {path}") from exc
