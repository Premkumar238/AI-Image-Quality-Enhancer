"""Tests for image information utilities."""

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import get_image_info


def test_get_image_info_100x100_rgb():
    image = Image.new("RGB", (100, 100), color="red")
    info = get_image_info(image)

    assert info["width"] == 100
    assert info["height"] == 100
    assert info["channels"] == 3
    assert info["mode"] == "RGB"


def test_get_image_info_1920x1080_rgb():
    image = Image.new("RGB", (1920, 1080), color="blue")
    info = get_image_info(image)

    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["channels"] == 3
    assert info["mode"] == "RGB"
