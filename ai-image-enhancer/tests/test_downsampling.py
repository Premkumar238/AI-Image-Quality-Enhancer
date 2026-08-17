"""Tests for downsampling utilities."""

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import create_low_resolution_image


def test_downsample_1024_to_256():
    image = Image.new("RGB", (1024, 1024), color="red")
    low_res = create_low_resolution_image(image, scale_factor=4)

    assert low_res.size == (256, 256)


def test_downsample_800x600_scale_factor_2():
    image = Image.new("RGB", (800, 600), color="blue")
    low_res = create_low_resolution_image(image, scale_factor=2)

    assert low_res.size == (400, 300)


def test_downsample_preserves_rgb_mode():
    image = Image.new("RGB", (120, 120), color="green")
    low_res = create_low_resolution_image(image, scale_factor=2)

    assert low_res.mode == "RGB"


def test_downsample_does_not_modify_original():
    image = Image.new("RGB", (512, 512), color="yellow")
    original_size = image.size

    create_low_resolution_image(image, scale_factor=4)

    assert image.size == original_size


def test_downsample_scale_factor_2():
    image = Image.new("RGB", (100, 100), color="white")
    low_res = create_low_resolution_image(image, scale_factor=2)

    assert low_res.size == (50, 50)


def test_downsample_scale_factor_3():
    image = Image.new("RGB", (900, 600), color="white")
    low_res = create_low_resolution_image(image, scale_factor=3)

    assert low_res.size == (300, 200)


def test_downsample_scale_factor_4():
    image = Image.new("RGB", (400, 400), color="white")
    low_res = create_low_resolution_image(image, scale_factor=4)

    assert low_res.size == (100, 100)


def test_downsample_invalid_scale_factor():
    image = Image.new("RGB", (100, 100), color="red")

    with pytest.raises(ValueError, match="scale_factor must be a positive integer"):
        create_low_resolution_image(image, scale_factor=0)

    with pytest.raises(ValueError, match="scale_factor must be a positive integer"):
        create_low_resolution_image(image, scale_factor=-2)


def test_downsample_scale_factor_too_large():
    image = Image.new("RGB", (10, 10), color="red")

    with pytest.raises(ValueError, match="scale_factor 20 is too large"):
        create_low_resolution_image(image, scale_factor=20)


def test_downsample_invalid_input():
    with pytest.raises(TypeError, match="Input must be a PIL Image"):
        create_low_resolution_image("not-an-image")


def test_downsample_output_is_pil_image():
    image = Image.new("RGB", (64, 64), color="purple")
    low_res = create_low_resolution_image(image, scale_factor=2)

    assert isinstance(low_res, Image.Image)
