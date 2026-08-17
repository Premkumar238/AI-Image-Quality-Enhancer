"""Tests for SRCNN inference."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import predict_image
from ml.models import SRCNN


def test_predict_image_returns_pil_image():
    model = SRCNN()
    input_image = Image.new("RGB", (32, 32), color="red")

    output_image = predict_image(model, input_image)

    assert isinstance(output_image, Image.Image)


def test_predict_image_output_mode_is_rgb():
    model = SRCNN()
    input_image = Image.new("RGB", (32, 32), color="blue")

    output_image = predict_image(model, input_image)

    assert output_image.mode == "RGB"


def test_predict_image_output_size_is_32x32():
    model = SRCNN()
    input_image = Image.new("RGB", (32, 32), color="green")

    output_image = predict_image(model, input_image)

    assert output_image.size == (32, 32)


def test_predict_image_does_not_modify_input():
    model = SRCNN()
    input_image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    original_array = np.array(input_image).copy()

    predict_image(model, input_image)

    assert np.array_equal(np.array(input_image), original_array)


def test_predict_image_runs_without_grad():
    model = SRCNN()
    input_image = Image.new("RGB", (32, 32), color="yellow")

    with torch.no_grad():
        output_image = predict_image(model, input_image)

    assert output_image.size == (32, 32)


def test_predict_image_output_has_valid_pixel_values():
    model = SRCNN()
    input_image = Image.new("RGB", (32, 32), color="purple")

    output_image = predict_image(model, input_image)
    output_array = np.array(output_image)

    assert output_array.min() >= 0
    assert output_array.max() <= 255


def test_predict_image_handles_finite_output():
    model = SRCNN()
    input_image = Image.new("RGB", (32, 32), color=(128, 64, 32))

    output_image = predict_image(model, input_image)
    output_array = np.array(output_image, dtype=np.float32)

    assert np.isfinite(output_array).all()
