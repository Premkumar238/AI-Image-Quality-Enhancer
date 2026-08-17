"""Tensor conversion utilities for the preprocessing pipeline."""

import numpy as np
import torch


def numpy_to_tensor(image_array: np.ndarray) -> torch.Tensor:
    """Convert a normalized RGB NumPy array to a PyTorch tensor.

    Args:
        image_array: A NumPy array with shape (height, width, 3).
            Values are expected to be normalized to the range 0.0-1.0.

    Returns:
        A float32 PyTorch tensor with shape (1, 3, height, width).

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the input does not have valid RGB image dimensions.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if image_array.ndim != 3:
        raise ValueError("Input must be a 3-dimensional array.")

    if image_array.shape[2] != 3:
        raise ValueError("Input must have 3 channels.")

    height, width, _ = image_array.shape
    if height <= 0 or width <= 0:
        raise ValueError("Image height and width must be greater than zero.")

    tensor = torch.from_numpy(image_array.astype(np.float32))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return tensor
