"""Tests for Real-ESRGAN helper utilities."""

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.realesrgan_enhance import is_realesrgan_ready


def test_realesrgan_weights_present_after_download():
    assert is_realesrgan_ready()


def test_enhance_photo_uses_ready_pipeline():
    from ml.inference import enhance_photo

    image = Image.new("RGB", (32, 32), color=(100, 110, 120))
    output = enhance_photo(image, scale_factor=2)
    assert output.mode == "RGB"
    assert output.size == (64, 64)
