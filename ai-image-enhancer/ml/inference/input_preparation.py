"""Input preparation utilities for SRCNN inference."""

from PIL import Image

from ml.preprocessing import create_low_resolution_image


def prepare_srcnn_input(image: Image.Image, scale_factor: int = 4) -> Image.Image:
    """Prepare a high-resolution image for SRCNN inference.

    The image is first downsampled to simulate a low-resolution version, then
    upscaled back to the original dimensions using bicubic interpolation.

    Args:
        image: A PIL Image in RGB mode representing the original image.
        scale_factor: Factor used to create the temporary low-resolution image.

    Returns:
        A PIL Image in RGB mode with the same size as the original image.
        The original input image is not modified.

    Raises:
        TypeError: If the input is not a PIL Image.
        ValueError: If scale_factor is invalid or too large for the image size.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    if not isinstance(scale_factor, int) or isinstance(scale_factor, bool):
        raise ValueError("scale_factor must be a positive integer.")

    if scale_factor <= 0:
        raise ValueError("scale_factor must be a positive integer.")

    original_size = image.size
    rgb_image = image.convert("RGB")

    low_resolution_image = create_low_resolution_image(
        rgb_image,
        scale_factor=scale_factor,
    )

    prepared_image = low_resolution_image.resize(
        original_size,
        Image.Resampling.BICUBIC,
    ).convert("RGB")

    return prepared_image
