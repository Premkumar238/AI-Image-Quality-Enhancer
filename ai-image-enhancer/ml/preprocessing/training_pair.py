"""Training pair creation utilities for supervised learning."""

from dataclasses import dataclass

from PIL import Image

from .downsampling import create_low_resolution_image


@dataclass
class TrainingPair:
    """A supervised training pair for image super-resolution."""

    input_image: Image.Image
    target_image: Image.Image


def create_training_pair(
    image: Image.Image,
    scale_factor: int = 4,
) -> TrainingPair:
    """Create a low-resolution input and high-resolution target training pair.

    Args:
        image: A PIL Image in RGB mode representing the high-resolution image.
        scale_factor: Factor by which to reduce the input image dimensions.

    Returns:
        A TrainingPair containing:
            input_image: The low-resolution PIL Image.
            target_image: The original high-resolution PIL Image.

    Raises:
        TypeError: If the input is not a PIL Image.
        ValueError: If scale_factor is invalid or the result would be too small.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    input_image = create_low_resolution_image(image, scale_factor=scale_factor)
    target_image = image.convert("RGB")

    return TrainingPair(input_image=input_image, target_image=target_image)
