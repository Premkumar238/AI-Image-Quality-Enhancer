"""Tests for SRCNN input preparation."""

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import prepare_srcnn_input


def test_prepare_srcnn_input_1024x1024_scale_factor_4():
    image = Image.new("RGB", (1024, 1024), color="red")

    prepared_image = prepare_srcnn_input(image, scale_factor=4)

    assert isinstance(prepared_image, Image.Image)
    assert prepared_image.mode == "RGB"
    assert prepared_image.size == (1024, 1024)


def test_prepare_srcnn_input_scale_factor_2():
    image = Image.new("RGB", (100, 100), color="blue")

    prepared_image = prepare_srcnn_input(image, scale_factor=2)

    assert prepared_image.size == (100, 100)


def test_prepare_srcnn_input_scale_factor_4():
    image = Image.new("RGB", (400, 400), color="green")

    prepared_image = prepare_srcnn_input(image, scale_factor=4)

    assert prepared_image.size == (400, 400)


def test_prepare_srcnn_input_non_square_image():
    image = Image.new("RGB", (800, 600), color="yellow")

    prepared_image = prepare_srcnn_input(image, scale_factor=4)

    assert prepared_image.size == (800, 600)


def test_prepare_srcnn_input_does_not_modify_original():
    image = Image.new("RGB", (800, 600), color=(10, 20, 30))
    original_size = image.size

    prepare_srcnn_input(image, scale_factor=4)

    assert image.size == original_size


def test_prepare_srcnn_input_invalid_scale_factor_zero():
    image = Image.new("RGB", (64, 64), color="white")

    with pytest.raises(ValueError, match="scale_factor must be a positive integer"):
        prepare_srcnn_input(image, scale_factor=0)


def test_prepare_srcnn_input_invalid_scale_factor_negative():
    image = Image.new("RGB", (64, 64), color="white")

    with pytest.raises(ValueError, match="scale_factor must be a positive integer"):
        prepare_srcnn_input(image, scale_factor=-2)


def test_prepare_srcnn_input_invalid_input():
    with pytest.raises(TypeError, match="Input must be a PIL Image"):
        prepare_srcnn_input("not-an-image")
