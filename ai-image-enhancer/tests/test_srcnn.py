"""Tests for the SRCNN model skeleton."""

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import SRCNN


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


def test_srcnn_has_layer_placeholders():
    model = SRCNN()

    assert hasattr(model, "feature_extraction")
    assert hasattr(model, "non_linear_mapping")
    assert hasattr(model, "reconstruction")
