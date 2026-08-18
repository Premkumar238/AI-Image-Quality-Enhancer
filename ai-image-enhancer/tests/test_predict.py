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


def test_predict_image_64x64_rgb():
    model = SRCNN()
    input_image = Image.new("RGB", (64, 64), color="red")

    output_image = predict_image(model, input_image)

    assert isinstance(output_image, Image.Image)
    assert output_image.mode == "RGB"
    assert output_image.size == (64, 64)


def test_predict_image_scale_factor_2():
    model = SRCNN()
    input_image = Image.new("RGB", (64, 64), color="blue")

    output_image = predict_image(model, input_image, scale_factor=2)

    assert output_image.size == (64, 64)


def test_predict_image_scale_factor_4():
    model = SRCNN()
    input_image = Image.new("RGB", (64, 64), color="green")

    output_image = predict_image(model, input_image, scale_factor=4)

    assert output_image.size == (64, 64)


def test_predict_image_non_square_80x60():
    model = SRCNN()
    input_image = Image.new("RGB", (80, 60), color="yellow")

    output_image = predict_image(model, input_image, scale_factor=2)

    assert output_image.size == (80, 60)


def test_predict_image_does_not_modify_input():
    model = SRCNN()
    input_image = Image.new("RGB", (64, 64), color=(10, 20, 30))
    original_array = np.array(input_image).copy()

    predict_image(model, input_image, scale_factor=4)

    assert np.array_equal(np.array(input_image), original_array)


def test_predict_image_runs_on_cpu():
    model = SRCNN()
    input_image = Image.new("RGB", (64, 64), color="purple")

    output_image = predict_image(model, input_image, scale_factor=4)

    assert output_image.size == (64, 64)


def test_predict_image_runs_without_grad():
    model = SRCNN()
    input_image = Image.new("RGB", (64, 64), color="orange")

    with torch.no_grad():
        output_image = predict_image(model, input_image, scale_factor=4)

    assert output_image.size == (64, 64)


def test_predict_image_output_has_valid_pixel_values():
    model = SRCNN()
    input_image = Image.new("RGB", (64, 64), color="cyan")

    output_image = predict_image(model, input_image, scale_factor=4)
    output_array = np.array(output_image)

    assert output_array.min() >= 0
    assert output_array.max() <= 255


def test_predict_image_handles_finite_output():
    model = SRCNN()
    input_image = Image.new("RGB", (64, 64), color=(128, 64, 32))

    output_image = predict_image(model, input_image, scale_factor=4)
    output_array = np.array(output_image, dtype=np.float32)

    assert np.isfinite(output_array).all()


def test_predict_image_upscale_increases_dimensions():
    model = SRCNN()
    input_image = Image.new("RGB", (32, 24), color="red")

    output_image = predict_image(model, input_image, scale_factor=2, upscale=True)

    assert output_image.size == (64, 48)
    assert output_image.mode == "RGB"
