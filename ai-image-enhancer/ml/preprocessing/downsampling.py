"""Downsampling utilities for creating low-resolution training images."""

from PIL import Image


def create_low_resolution_image(
    image: Image.Image,
    scale_factor: int = 4,
) -> Image.Image:
    """Create a low-resolution version of an RGB image.

    Args:
        image: A PIL Image in RGB mode.
        scale_factor: Factor by which to reduce width and height.
            For example, a scale factor of 4 reduces a 1024x1024 image to 256x256.

    Returns:
        A new PIL Image in RGB mode with reduced dimensions.
        The original image is not modified.

    Raises:
        TypeError: If the input is not a PIL Image.
        ValueError: If scale_factor is invalid or the result would be too small.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    if not isinstance(scale_factor, int) or isinstance(scale_factor, bool):
        raise ValueError("scale_factor must be a positive integer.")

    if scale_factor <= 0:
        raise ValueError("scale_factor must be a positive integer.")

    width, height = image.size
    new_width = width // scale_factor
    new_height = height // scale_factor

    if new_width < 1 or new_height < 1:
        raise ValueError(
            f"scale_factor {scale_factor} is too large for image size {width}x{height}."
        )

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS).convert("RGB")
