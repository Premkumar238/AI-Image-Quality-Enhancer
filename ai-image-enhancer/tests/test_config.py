"""Tests for centralized training configuration."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml import TrainingConfig


def test_default_configuration_works():
    config = TrainingConfig()

    assert config.batch_size == 4
    assert config.epochs == 10


def test_custom_configuration_works():
    config = TrainingConfig(
        dataset_directory="data/all",
        train_directory="data/train",
        validation_directory="data/validation",
        checkpoint_path="output/model.pth",
        history_path="output/history.json",
        batch_size=8,
        learning_rate=0.0005,
        epochs=5,
        scale_factor=2,
        validation_ratio=0.25,
        num_workers=2,
        seed=7,
    )

    assert config.batch_size == 8
    assert config.learning_rate == 0.0005
    assert config.epochs == 5
    assert config.scale_factor == 2
    assert config.validation_ratio == 0.25
    assert config.num_workers == 2
    assert config.seed == 7
    assert config.dataset_directory == Path("data/all")


def test_default_batch_size_is_four():
    config = TrainingConfig()

    assert config.batch_size == 4


def test_default_learning_rate():
    config = TrainingConfig()

    assert config.learning_rate == 0.001


def test_default_epochs():
    config = TrainingConfig()

    assert config.epochs == 10


def test_default_scale_factor():
    config = TrainingConfig()

    assert config.scale_factor == 4


def test_default_validation_ratio():
    config = TrainingConfig()

    assert config.validation_ratio == 0.2


def test_default_num_workers():
    config = TrainingConfig()

    assert config.num_workers == 0


def test_default_seed():
    config = TrainingConfig()

    assert config.seed == 42


def test_batch_size_zero_raises_value_error():
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=0)


def test_negative_batch_size_raises_value_error():
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=-1)


def test_learning_rate_zero_raises_value_error():
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=0)


def test_negative_learning_rate_raises_value_error():
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=-0.001)


def test_epochs_zero_raises_value_error():
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(epochs=0)


def test_negative_epochs_raises_value_error():
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(epochs=-3)


def test_invalid_scale_factor_raises_value_error():
    with pytest.raises(ValueError, match="scale_factor"):
        TrainingConfig(scale_factor=5)


def test_validation_ratio_zero_raises_value_error():
    with pytest.raises(ValueError, match="validation_ratio"):
        TrainingConfig(validation_ratio=0)


def test_validation_ratio_one_raises_value_error():
    with pytest.raises(ValueError, match="validation_ratio"):
        TrainingConfig(validation_ratio=1)


def test_negative_num_workers_raises_value_error():
    with pytest.raises(ValueError, match="num_workers"):
        TrainingConfig(num_workers=-1)


def test_negative_seed_raises_value_error():
    with pytest.raises(ValueError, match="seed"):
        TrainingConfig(seed=-1)
