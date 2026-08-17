"""Tests for model checkpoint saving and loading."""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import SRCNN, load_model, save_model


def test_save_model_creates_checkpoint_file(tmp_path):
    model = SRCNN()
    checkpoint_path = tmp_path / "srcnn.pth"

    save_model(model, checkpoint_path)

    assert checkpoint_path.exists()


def test_loaded_model_matches_saved_state_dict(tmp_path):
    original_model = SRCNN()
    checkpoint_path = tmp_path / "srcnn.pth"
    save_model(original_model, checkpoint_path)

    loaded_model = SRCNN()
    load_model(loaded_model, checkpoint_path)

    for original_key, loaded_key in zip(
        original_model.state_dict().keys(),
        loaded_model.state_dict().keys(),
        strict=True,
    ):
        assert original_key == loaded_key
        assert torch.equal(
            original_model.state_dict()[original_key],
            loaded_model.state_dict()[loaded_key],
        )


def test_save_model_creates_nested_directories(tmp_path):
    model = SRCNN()
    checkpoint_path = tmp_path / "nested" / "models" / "srcnn.pth"

    save_model(model, checkpoint_path)

    assert checkpoint_path.exists()


def test_load_model_missing_checkpoint_raises_error(tmp_path):
    model = SRCNN()
    missing_path = tmp_path / "missing" / "srcnn.pth"

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_model(model, missing_path)


def test_load_model_on_cpu(tmp_path):
    model = SRCNN()
    checkpoint_path = tmp_path / "srcnn.pth"
    save_model(model, checkpoint_path)

    loaded_model = SRCNN()
    load_model(loaded_model, checkpoint_path, device="cpu")

    for parameter in loaded_model.parameters():
        assert parameter.device.type == "cpu"


def test_checkpoint_contains_model_weights(tmp_path):
    model = SRCNN()
    checkpoint_path = tmp_path / "srcnn.pth"
    save_model(model, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert isinstance(checkpoint, dict)
    assert "feature_extraction.weight" in checkpoint
    assert "reconstruction.weight" in checkpoint
