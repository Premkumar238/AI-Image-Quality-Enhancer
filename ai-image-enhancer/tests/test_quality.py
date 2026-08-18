"""Tests for photo denoise, polish, and enhancement helpers."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import denoise_image, enhance_photo, refine_image


def test_denoise_image_keeps_size_and_mode():
    image = Image.new("RGB", (48, 32), color=(90, 40, 40))
    output = denoise_image(image)

    assert output.mode == "RGB"
    assert output.size == (48, 32)


def test_refine_image_keeps_size_and_mode():
    image = Image.new("RGB", (40, 40), color=(120, 130, 140))
    output = refine_image(image)

    assert output.mode == "RGB"
    assert output.size == (40, 40)


def test_enhance_photo_upscales_and_returns_rgb():
    image = Image.new("RGB", (32, 24), color=(80, 90, 100))

    output = enhance_photo(image, scale_factor=2)

    assert output.mode == "RGB"
    assert output.size == (64, 48)
    array = np.array(output)
    assert array.min() >= 0
    assert array.max() <= 255
