"""Tests for the SRCNN model."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import SRCNN, count_parameters


def test_srcnn_can_be_instantiated():
    model = SRCNN()

    assert model is not None


def test_srcnn_is_nn_module():
    model = SRCNN()

    assert isinstance(model, nn.Module)


def test_srcnn_default_configuration():
    model = SRCNN()

    assert model.in_channels == 3
    assert model.out_channels == 3
    assert model.num_features == 64


def test_srcnn_custom_configuration():
    model = SRCNN(in_channels=1, out_channels=1, num_features=32)

    assert model.in_channels == 1
    assert model.out_channels == 1
    assert model.num_features == 32


def test_srcnn_has_reconstruction_layer():
    model = SRCNN()

    assert hasattr(model, "reconstruction")
    assert isinstance(model.reconstruction, nn.Conv2d)


def test_reconstruction_input_channels():
    model = SRCNN()

    assert model.reconstruction.in_channels == 64


def test_reconstruction_output_channels():
    model = SRCNN()

    assert model.reconstruction.out_channels == 3


def test_reconstruction_kernel_size():
    model = SRCNN()

    assert model.reconstruction.kernel_size == (5, 5)


def test_reconstruction_stride():
    model = SRCNN()

    assert model.reconstruction.stride == (1, 1)


def test_reconstruction_padding():
    model = SRCNN()

    assert model.reconstruction.padding == (2, 2)


def test_non_linear_mapping_exists():
    model = SRCNN()

    assert hasattr(model, "non_linear_mapping")


def test_non_linear_mapping_is_conv2d():
    model = SRCNN()

    assert isinstance(model.non_linear_mapping, nn.Conv2d)


def test_non_linear_mapping_input_channels():
    model = SRCNN()

    assert model.non_linear_mapping.in_channels == 64


def test_non_linear_mapping_output_channels():
    model = SRCNN()

    assert model.non_linear_mapping.out_channels == 64


def test_non_linear_mapping_kernel_size():
    model = SRCNN()

    assert model.non_linear_mapping.kernel_size == (5, 5)


def test_non_linear_mapping_stride():
    model = SRCNN()

    assert model.non_linear_mapping.stride == (1, 1)


def test_non_linear_mapping_padding():
    model = SRCNN()

    assert model.non_linear_mapping.padding == (2, 2)


def test_feature_extraction_exists():
    model = SRCNN()

    assert hasattr(model, "feature_extraction")


def test_feature_extraction_is_conv2d():
    model = SRCNN()

    assert isinstance(model.feature_extraction, nn.Conv2d)


def test_feature_extraction_kernel_size():
    model = SRCNN()

    assert model.feature_extraction.kernel_size == (9, 9)


def test_feature_extraction_input_channels():
    model = SRCNN()

    assert model.feature_extraction.in_channels == 3


def test_feature_extraction_output_channels():
    model = SRCNN()

    assert model.feature_extraction.out_channels == 64


def test_feature_extraction_stride():
    model = SRCNN()

    assert model.feature_extraction.stride == (1, 1)


def test_feature_extraction_padding():
    model = SRCNN()

    assert model.feature_extraction.padding == (4, 4)


def test_forward_pass_output_shape():
    model = SRCNN()
    dummy_input = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 3, 32, 32)


def test_complete_forward_pass():
    model = SRCNN()
    dummy_input = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 2
    assert output.shape[1] == 3
    assert output.shape[2] == 32
    assert output.shape[3] == 32
    assert not torch.isnan(output).any()


def test_standard_rgb_image_64x64():
    model = SRCNN()
    input_tensor = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (1, 3, 64, 64)


def test_batch_of_four_images():
    model = SRCNN()
    input_tensor = torch.randn(4, 3, 64, 64)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (4, 3, 64, 64)


def test_different_image_size_128x128():
    model = SRCNN()
    input_tensor = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (2, 3, 128, 128)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = SRCNN()
    input_tensor = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (batch_size, 3, 64, 64)


@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [(1, 64, 64), (2, 32, 32), (4, 128, 128)],
)
def test_output_has_three_rgb_channels(batch_size, height, width):
    model = SRCNN()
    input_tensor = torch.randn(batch_size, 3, height, width)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape[1] == 3


@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [(1, 64, 64), (2, 48, 48), (3, 128, 96)],
)
def test_spatial_dimensions_preserved(batch_size, height, width):
    model = SRCNN()
    input_tensor = torch.randn(batch_size, 3, height, width)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape[2] == height
    assert output.shape[3] == width


def test_output_dtype():
    model = SRCNN()
    input_tensor = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.dtype == torch.float32


def test_no_nan_or_infinity():
    model = SRCNN()
    input_tensor = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        output = model(input_tensor)

    assert not torch.isnan(output).any().item()
    assert not torch.isinf(output).any().item()


def test_model_parameter_count():
    model = SRCNN()
    parameter_count = count_parameters(model)

    expected_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert parameter_count == expected_count
    assert parameter_count > 0


def test_evaluation_mode_inference():
    model = SRCNN()
    model.eval()
    input_tensor = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (1, 3, 64, 64)
    assert not model.training


def test_cpu_execution():
    model = SRCNN()
    model.eval()
    input_tensor = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.device.type == "cpu"
    assert output.shape == (1, 3, 64, 64)
