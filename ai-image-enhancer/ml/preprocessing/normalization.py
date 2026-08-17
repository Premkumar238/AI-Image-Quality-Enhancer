"""Image normalization utilities for the preprocessing pipeline."""

import numpy as np


def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """Normalize an RGB image array from 0-255 to 0.0-1.0.

    Args:
        image_array: A NumPy array with shape (height, width, 3).
            Pixel values are expected to be in the range 0-255.

    Returns:
        A float32 NumPy array with the same shape and values in 0.0-1.0.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the input does not have valid RGB image dimensions.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (height, width, 3).")

    height, width, _ = image_array.shape
    if height <= 0 or width <= 0:
        raise ValueError("Image height and width must be greater than zero.")

    return image_array.astype(np.float32) / 255.0
