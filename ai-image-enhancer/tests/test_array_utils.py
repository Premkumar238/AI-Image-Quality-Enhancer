"""Tests for array conversion utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import pil_to_numpy


def test_pil_to_numpy_rgb_conversion():
    image = Image.new("RGB", (50, 40), color=(10, 20, 30))
    array = pil_to_numpy(image)

    assert isinstance(array, np.ndarray)


def test_pil_to_numpy_shape():
    image = Image.new("RGB", (100, 100), color="green")
    array = pil_to_numpy(image)

    assert array.shape == (100, 100, 3)


def test_pil_to_numpy_dtype():
    image = Image.new("RGB", (32, 32), color="blue")
    array = pil_to_numpy(image)

    assert array.dtype == np.uint8


def test_pil_to_numpy_pixel_range():
    image = Image.new("RGB", (16, 16), color=(255, 128, 0))
    array = pil_to_numpy(image)

    assert array.min() >= 0
    assert array.max() <= 255


def test_pil_to_numpy_invalid_input():
    with pytest.raises(TypeError, match="Input must be a PIL Image"):
        pil_to_numpy("not-an-image")
