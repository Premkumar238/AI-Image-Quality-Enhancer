"""Training history saving and loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .trainer import TrainingHistory


def _history_to_mapping(history: TrainingHistory | dict[str, Any]) -> dict[str, Any]:
    """Convert supported history inputs to a dictionary."""
    if isinstance(history, TrainingHistory):
        return {
            "train_loss": history.train_loss,
            "validation_loss": history.validation_loss,
        }

    if isinstance(history, dict):
        return history

    raise TypeError("history must be a TrainingHistory instance or dictionary.")


def _validate_history_data(data: dict[str, Any]) -> TrainingHistory:
    """Validate and normalize training history data."""
    if "train_loss" not in data:
        raise ValueError("Training history must include 'train_loss'.")

    if "validation_loss" not in data:
        raise ValueError("Training history must include 'validation_loss'.")

    train_loss = data["train_loss"]
    validation_loss = data["validation_loss"]

    if not isinstance(train_loss, list):
        raise ValueError("'train_loss' must be a list.")

    if not isinstance(validation_loss, list):
        raise ValueError("'validation_loss' must be a list.")

    if len(train_loss) != len(validation_loss):
        raise ValueError("'train_loss' and 'validation_loss' must have the same length.")

    return TrainingHistory(
        train_loss=[float(value) for value in train_loss],
        validation_loss=[float(value) for value in validation_loss],
    )


def save_training_history(
    history: TrainingHistory | dict[str, Any],
    file_path: str | Path,
) -> None:
    """Save training history to a JSON file.

    Args:
        history: Training history containing train and validation losses.
        file_path: Destination path for the JSON file.

    Raises:
        TypeError: If history is not a supported type.
        ValueError: If the history structure is invalid.
    """
    history_data = _validate_history_data(_history_to_mapping(history))
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "train_loss": history_data.train_loss,
        "validation_loss": history_data.validation_loss,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def load_training_history(file_path: str | Path) -> TrainingHistory:
    """Load training history from a JSON file.

    Args:
        file_path: Path to the saved JSON history file.

    Returns:
        The loaded TrainingHistory instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or history data.
    """
    input_path = Path(file_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Training history file not found: {input_path}")

    try:
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in training history file: {input_path}") from exc

    if not isinstance(data, dict):
        raise ValueError("Training history JSON must contain an object at the top level.")

    return _validate_history_data(data)
