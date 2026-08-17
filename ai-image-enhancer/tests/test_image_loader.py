"""Tests for image loading utilities."""

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import load_image


def test_load_valid_image(tmp_path):
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (32, 24), color="blue").save(image_path)

    image = load_image(image_path)

    assert isinstance(image, Image.Image)
    assert image.size == (32, 24)


def test_load_image_is_rgb(tmp_path):
    image_path = tmp_path / "grayscale.png"
    Image.new("L", (16, 16), color=128).save(image_path)

    image = load_image(image_path)

    assert image.mode == "RGB"


def test_load_image_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.png"

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        load_image(missing_path)


def test_load_image_invalid_file(tmp_path):
    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_text("this is not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid image"):
        load_image(invalid_path)
