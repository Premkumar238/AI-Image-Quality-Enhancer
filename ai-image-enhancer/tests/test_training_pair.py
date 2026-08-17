"""Tests for training pair creation utilities."""

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import create_training_pair


def test_training_pair_1024_scale_factor_4():
    image = Image.new("RGB", (1024, 1024), color="red")
    pair = create_training_pair(image, scale_factor=4)

    assert pair.input_image.size == (256, 256)
    assert pair.target_image.size == (1024, 1024)


def test_training_pair_returns_pil_images():
    image = Image.new("RGB", (200, 200), color="blue")
    pair = create_training_pair(image, scale_factor=2)

    assert isinstance(pair.input_image, Image.Image)
    assert isinstance(pair.target_image, Image.Image)


def test_training_pair_rgb_mode():
    image = Image.new("RGB", (120, 120), color="green")
    pair = create_training_pair(image, scale_factor=3)

    assert pair.input_image.mode == "RGB"
    assert pair.target_image.mode == "RGB"


def test_training_pair_target_has_original_dimensions():
    image = Image.new("RGB", (800, 600), color="yellow")
    pair = create_training_pair(image, scale_factor=2)

    assert pair.target_image.size == (800, 600)


def test_training_pair_input_dimensions_reduced():
    image = Image.new("RGB", (800, 600), color="yellow")
    pair = create_training_pair(image, scale_factor=2)

    assert pair.input_image.size == (400, 300)


def test_training_pair_does_not_modify_original():
    image = Image.new("RGB", (512, 512), color="white")
    original_size = image.size

    create_training_pair(image, scale_factor=4)

    assert image.size == original_size


def test_training_pair_scale_factor_2():
    image = Image.new("RGB", (100, 100), color="red")
    pair = create_training_pair(image, scale_factor=2)

    assert pair.input_image.size == (50, 50)
    assert pair.target_image.size == (100, 100)


def test_training_pair_scale_factor_4():
    image = Image.new("RGB", (400, 400), color="red")
    pair = create_training_pair(image, scale_factor=4)

    assert pair.input_image.size == (100, 100)
    assert pair.target_image.size == (400, 400)


def test_training_pair_invalid_image_input():
    with pytest.raises(TypeError, match="Input must be a PIL Image"):
        create_training_pair("not-an-image")


def test_training_pair_invalid_scale_factor():
    image = Image.new("RGB", (100, 100), color="red")

    with pytest.raises(ValueError, match="scale_factor must be a positive integer"):
        create_training_pair(image, scale_factor=0)
