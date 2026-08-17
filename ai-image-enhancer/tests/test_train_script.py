"""Tests for the training command-line script."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train.py"


def load_train_module():
    """Load the training script as a module without executing main."""
    spec = importlib.util.spec_from_file_location("train_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _create_dataset_images(directory: Path, count: int = 4) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", (32, 32), color=(index * 40, 80, 120))
        image.save(directory / f"image_{index}.png")


def test_set_seed_produces_reproducible_random_values():
    train_module = load_train_module()

    train_module.set_seed(42)
    python_value = random.random()
    numpy_value = float(np.random.rand())
    torch_value = float(torch.rand(1))

    train_module.set_seed(42)
    assert random.random() == python_value
    assert float(np.random.rand()) == numpy_value
    assert float(torch.rand(1)) == torch_value


def test_parse_args_uses_optional_arguments():
    train_module = load_train_module()

    args = train_module.parse_args(
        [
            "--dataset",
            "data/source",
            "--epochs",
            "3",
            "--batch-size",
            "2",
        ]
    )

    assert args.dataset == "data/source"
    assert args.epochs == 3
    assert args.batch_size == 2
    assert args.train_dir is None


def test_build_training_config_applies_cli_overrides():
    train_module = load_train_module()
    args = train_module.parse_args(
        [
            "--dataset",
            "custom/data",
            "--train-dir",
            "custom/train",
            "--validation-dir",
            "custom/validation",
            "--checkpoint",
            "custom/model.pth",
            "--history",
            "custom/history.json",
            "--epochs",
            "5",
            "--batch-size",
            "8",
            "--learning-rate",
            "0.0005",
            "--scale-factor",
            "2",
            "--validation-ratio",
            "0.25",
            "--num-workers",
            "1",
            "--seed",
            "7",
        ]
    )

    config = train_module.build_training_config(args)

    assert config.dataset_directory == Path("custom/data")
    assert config.train_directory == Path("custom/train")
    assert config.validation_directory == Path("custom/validation")
    assert config.checkpoint_path == Path("custom/model.pth")
    assert config.history_path == Path("custom/history.json")
    assert config.epochs == 5
    assert config.batch_size == 8
    assert config.learning_rate == 0.0005
    assert config.scale_factor == 2
    assert config.validation_ratio == 0.25
    assert config.num_workers == 1
    assert config.seed == 7


def test_build_training_config_rejects_invalid_values():
    train_module = load_train_module()
    args = train_module.parse_args(["--epochs", "0"])

    with pytest.raises(ValueError, match="epochs"):
        train_module.build_training_config(args)


def test_train_script_can_be_imported_without_starting_training():
    train_module = load_train_module()

    assert hasattr(train_module, "main")
    assert hasattr(train_module, "run_training")
    assert callable(train_module.main)


def test_run_training_small_end_to_end(tmp_path):
    train_module = load_train_module()

    dataset_dir = tmp_path / "dataset"
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    checkpoint_path = tmp_path / "output" / "model.pth"
    history_path = tmp_path / "output" / "history.json"

    _create_dataset_images(dataset_dir, count=4)

    config = train_module.TrainingConfig(
        dataset_directory=dataset_dir,
        train_directory=train_dir,
        validation_directory=validation_dir,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        batch_size=1,
        epochs=1,
        scale_factor=2,
        validation_ratio=0.25,
        num_workers=0,
        seed=42,
    )

    exit_code = train_module.run_training(config)

    assert exit_code == 0
    assert checkpoint_path.exists()
    assert history_path.exists()

    history_data = json.loads(history_path.read_text(encoding="utf-8"))
    assert "train_loss" in history_data
    assert "validation_loss" in history_data
    assert len(history_data["train_loss"]) == 1
    assert len(history_data["validation_loss"]) == 1


def test_run_training_missing_dataset_directory(tmp_path):
    train_module = load_train_module()

    config = train_module.TrainingConfig(
        dataset_directory=tmp_path / "missing",
        train_directory=tmp_path / "train",
        validation_directory=tmp_path / "validation",
        checkpoint_path=tmp_path / "model.pth",
        history_path=tmp_path / "history.json",
        epochs=1,
        batch_size=1,
    )

    exit_code = train_module.run_training(config)

    assert exit_code == 1
