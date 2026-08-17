"""Array conversion utilities for the preprocessing pipeline."""

import numpy as np
from PIL import Image


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a NumPy array.

    Args:
        image: A loaded PIL Image in RGB mode.

    Returns:
        A NumPy array with shape (height, width, 3) and dtype uint8.
        Pixel values remain in the range 0 to 255.

    Raises:
        TypeError: If the input is not a PIL Image.
        ValueError: If the image is not RGB with three channels.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    array = np.asarray(image)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Image must be RGB with shape (height, width, 3).")

    return array
