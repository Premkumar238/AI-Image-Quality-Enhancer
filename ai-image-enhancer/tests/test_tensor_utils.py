"""Tests for tensor conversion utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import numpy_to_tensor


def test_numpy_to_tensor_conversion():
    array = np.random.rand(100, 100, 3).astype(np.float32)
    tensor = numpy_to_tensor(array)

    assert isinstance(tensor, torch.Tensor)


def test_numpy_to_tensor_dtype():
    array = np.zeros((100, 100, 3), dtype=np.float32)
    tensor = numpy_to_tensor(array)

    assert tensor.dtype == torch.float32


def test_numpy_to_tensor_shape():
    array = np.zeros((100, 100, 3), dtype=np.float32)
    tensor = numpy_to_tensor(array)

    assert tensor.shape == (1, 3, 100, 100)


def test_numpy_to_tensor_preserves_values():
    array = np.zeros((10, 10, 3), dtype=np.float32)
    array[5, 5] = np.array([0.25, 0.5, 0.75], dtype=np.float32)

    tensor = numpy_to_tensor(array)

    assert tensor[0, 0, 5, 5].item() == pytest.approx(0.25)
    assert tensor[0, 1, 5, 5].item() == pytest.approx(0.5)
    assert tensor[0, 2, 5, 5].item() == pytest.approx(0.75)


def test_numpy_to_tensor_black_image():
    array = np.zeros((32, 32, 3), dtype=np.float32)
    tensor = numpy_to_tensor(array)

    assert torch.all(tensor == 0.0)


def test_numpy_to_tensor_white_image():
    array = np.ones((32, 32, 3), dtype=np.float32)
    tensor = numpy_to_tensor(array)

    assert torch.all(tensor == 1.0)


def test_numpy_to_tensor_invalid_input():
    with pytest.raises(TypeError, match="Input must be a NumPy array"):
        numpy_to_tensor([[0.0, 0.0, 0.0]])


def test_numpy_to_tensor_invalid_channels():
    array = np.zeros((100, 100, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="Input must have 3 channels"):
        numpy_to_tensor(array)
