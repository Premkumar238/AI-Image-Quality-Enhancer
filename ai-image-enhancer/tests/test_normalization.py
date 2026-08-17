"""Tests for image normalization utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import normalize_image, pil_to_numpy


def test_normalize_black_pixel():
    array = np.array([[[0, 0, 0]]], dtype=np.uint8)
    normalized = normalize_image(array)

    np.testing.assert_array_equal(normalized, np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32))


def test_normalize_white_pixel():
    array = np.array([[[255, 255, 255]]], dtype=np.uint8)
    normalized = normalize_image(array)

    np.testing.assert_array_equal(normalized, np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32))


def test_normalize_mid_gray_pixel():
    array = np.array([[[128, 128, 128]]], dtype=np.uint8)
    normalized = normalize_image(array)

    expected = 128 / 255.0
    assert normalized[0, 0, 0] == pytest.approx(expected)


def test_normalize_output_dtype():
    array = np.array([[[64, 128, 192]]], dtype=np.uint8)
    normalized = normalize_image(array)

    assert normalized.dtype == np.float32


def test_normalize_shape_unchanged():
    array = np.zeros((100, 80, 3), dtype=np.uint8)
    normalized = normalize_image(array)

    assert normalized.shape == (100, 80, 3)


def test_normalize_value_range():
    array = np.array([[[0, 128, 255]]], dtype=np.uint8)
    normalized = normalize_image(array)

    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_invalid_input():
    with pytest.raises(TypeError, match="Input must be a NumPy array"):
        normalize_image([[0, 0, 0]])


def test_normalize_generated_rgb_image():
    image = Image.new("RGB", (32, 24), color=(255, 128, 0))
    array = pil_to_numpy(image)
    normalized = normalize_image(array)

    assert normalized.shape == (24, 32, 3)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
