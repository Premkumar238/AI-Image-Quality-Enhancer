"""Command-line script for training the SRCNN super-resolution model."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import TrainingConfig
from ml.inference import prepare_srcnn_input
from ml.models import SRCNN, save_model
from ml.preprocessing import load_image, normalize_image, numpy_to_tensor, pil_to_numpy
from ml.training import (
    SuperResolutionDataset,
    TrainingHistory,
    create_dataloader,
    create_loss_function,
    create_optimizer,
    save_training_history,
    split_dataset,
    train_model,
)


class PreparedSuperResolutionDataset(SuperResolutionDataset):
    """SuperResolutionDataset with SRCNN-compatible prepared inputs."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = load_image(self.image_paths[index])
        prepared_input = prepare_srcnn_input(image, scale_factor=self.scale_factor)

        input_tensor = numpy_to_tensor(normalize_image(pil_to_numpy(prepared_input))).squeeze(0)
        target_tensor = numpy_to_tensor(normalize_image(pil_to_numpy(image))).squeeze(0)
        return input_tensor, target_tensor


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train the SRCNN super-resolution model.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Source image directory.")
    parser.add_argument("--train-dir", dest="train_dir", type=str, default=None)
    parser.add_argument("--validation-dir", dest="validation_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--history", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--scale-factor", dest="scale_factor", type=int, default=None)
    parser.add_argument(
        "--validation-ratio",
        dest="validation_ratio",
        type=float,
        default=None,
    )
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Build a TrainingConfig from defaults and optional CLI overrides."""
    defaults = TrainingConfig()
    overrides = {
        "dataset_directory": args.dataset,
        "train_directory": args.train_dir,
        "validation_directory": args.validation_dir,
        "checkpoint_path": args.checkpoint,
        "history_path": args.history,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "scale_factor": args.scale_factor,
        "validation_ratio": args.validation_ratio,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }

    config_values = {
        field_name: value if value is not None else getattr(defaults, field_name)
        for field_name, value in overrides.items()
    }

    return TrainingConfig(**config_values)


def run_training(config: TrainingConfig) -> int:
    """Run the full SRCNN training pipeline."""
    device = torch.device("cpu")
    print(f"Using device: {device}")

    set_seed(config.seed)

    if not config.dataset_directory.exists():
        print(
            f"Error: Dataset directory not found: {config.dataset_directory}",
            file=sys.stderr,
        )
        return 1

    print("Starting training...")
    split_result = split_dataset(
        config.dataset_directory,
        config.train_directory,
        config.validation_directory,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )
    print(f"Training images: {split_result.train}")
    print(f"Validation images: {split_result.validation}")

    train_dataset = PreparedSuperResolutionDataset(
        config.train_directory,
        scale_factor=config.scale_factor,
    )
    validation_dataset = PreparedSuperResolutionDataset(
        config.validation_directory,
        scale_factor=config.scale_factor,
    )

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_dataloader = create_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = SRCNN().to(device)
    criterion = create_loss_function()
    optimizer = create_optimizer(model, learning_rate=config.learning_rate)

    history = TrainingHistory(train_loss=[], validation_loss=[])

    for epoch in range(1, config.epochs + 1):
        epoch_history = train_model(
            model,
            train_dataloader,
            validation_dataloader,
            criterion,
            optimizer,
            epochs=1,
        )
        history.train_loss.extend(epoch_history.train_loss)
        history.validation_loss.extend(epoch_history.validation_loss)

        print(f"Epoch {epoch}/{config.epochs}")
        print(f"Training loss: {epoch_history.train_loss[-1]:.6f}")
        print(f"Validation loss: {epoch_history.validation_loss[-1]:.6f}")
        print()

    save_model(model, config.checkpoint_path)
    save_training_history(history, config.history_path)

    print("Training complete.")
    print(f"Model saved to: {config.checkpoint_path}")
    print(f"History saved to: {config.history_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the training script."""
    args = parse_args(argv)
    config = build_training_config(args)
    return run_training(config)


if __name__ == "__main__":
    sys.exit(main())
