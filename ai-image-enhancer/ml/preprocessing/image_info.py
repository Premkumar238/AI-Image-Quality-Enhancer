"""Image information utilities for the preprocessing pipeline."""

from PIL import Image


def get_image_info(image: Image.Image) -> dict[str, int | str]:
    """Return basic metadata for a PIL Image.

    Args:
        image: A loaded PIL Image object.

    Returns:
        A dictionary containing width, height, channels, and mode.
    """
    width, height = image.size

    return {
        "width": width,
        "height": height,
        "channels": len(image.getbands()),
        "mode": image.mode,
    }
