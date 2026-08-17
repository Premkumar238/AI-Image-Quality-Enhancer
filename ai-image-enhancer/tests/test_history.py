"""Tests for training history saving and loading."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import (
    TrainingHistory,
    load_training_history,
    save_training_history,
)


def _sample_history() -> TrainingHistory:
    return TrainingHistory(
        train_loss=[0.42, 0.31, 0.25],
        validation_loss=[0.45, 0.34, 0.28],
    )


def test_save_training_history_creates_json_file(tmp_path):
    history = _sample_history()
    file_path = tmp_path / "history.json"

    save_training_history(history, file_path)

    assert file_path.exists()


def test_load_training_history_returns_saved_history(tmp_path):
    history = _sample_history()
    file_path = tmp_path / "history.json"

    save_training_history(history, file_path)
    loaded_history = load_training_history(file_path)

    assert loaded_history == history


def test_save_training_history_creates_nested_directories(tmp_path):
    history = _sample_history()
    file_path = tmp_path / "nested" / "runs" / "history.json"

    save_training_history(history, file_path)

    assert file_path.exists()
    assert load_training_history(file_path) == history


def test_load_training_history_missing_file(tmp_path):
    missing_file = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Training history file not found"):
        load_training_history(missing_file)


def test_load_training_history_invalid_json(tmp_path):
    file_path = tmp_path / "invalid.json"
    file_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_training_history(file_path)


def test_save_training_history_missing_train_loss(tmp_path):
    history = {"validation_loss": [0.5, 0.4]}

    with pytest.raises(ValueError, match="train_loss"):
        save_training_history(history, tmp_path / "history.json")


def test_save_training_history_missing_validation_loss(tmp_path):
    history = {"train_loss": [0.5, 0.4]}

    with pytest.raises(ValueError, match="validation_loss"):
        save_training_history(history, tmp_path / "history.json")


def test_save_training_history_train_loss_not_list(tmp_path):
    history = {
        "train_loss": 0.5,
        "validation_loss": [0.6, 0.4],
    }

    with pytest.raises(ValueError, match="'train_loss' must be a list"):
        save_training_history(history, tmp_path / "history.json")


def test_save_training_history_validation_loss_not_list(tmp_path):
    history = {
        "train_loss": [0.5, 0.4],
        "validation_loss": 0.3,
    }

    with pytest.raises(ValueError, match="'validation_loss' must be a list"):
        save_training_history(history, tmp_path / "history.json")


def test_save_training_history_different_list_lengths(tmp_path):
    history = {
        "train_loss": [0.5, 0.4, 0.3],
        "validation_loss": [0.6, 0.4],
    }

    with pytest.raises(ValueError, match="same length"):
        save_training_history(history, tmp_path / "history.json")


def test_save_and_load_empty_history_lists(tmp_path):
    history = TrainingHistory(train_loss=[], validation_loss=[])
    file_path = tmp_path / "empty_history.json"

    save_training_history(history, file_path)
    loaded_history = load_training_history(file_path)

    assert loaded_history.train_loss == []
    assert loaded_history.validation_loss == []


def test_saved_json_is_readable_and_utf8(tmp_path):
    history = _sample_history()
    file_path = tmp_path / "history.json"

    save_training_history(history, file_path)

    text = file_path.read_text(encoding="utf-8")
    parsed = json.loads(text)

    assert parsed["train_loss"] == history.train_loss
    assert parsed["validation_loss"] == history.validation_loss
    assert "\n" in text
