"""Central configuration for machine learning training."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration settings for SRCNN training and dataset preparation.

    This class stores paths and training hyperparameters in one place and
    validates their values when the configuration is created.
    """

    dataset_directory: str | Path = "datasets/images"
    train_directory: str | Path = "datasets/train"
    validation_directory: str | Path = "datasets/validation"
    checkpoint_path: str | Path = "models/srcnn.pth"
    history_path: str | Path = "models/training_history.json"
    batch_size: int = 4
    learning_rate: float = 0.001
    epochs: int = 10
    scale_factor: int = 4
    validation_ratio: float = 0.2
    num_workers: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise ValueError("batch_size must be a positive integer.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        if not isinstance(self.learning_rate, (int, float)) or isinstance(
            self.learning_rate,
            bool,
        ):
            raise ValueError("learning_rate must be greater than 0.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool):
            raise ValueError("epochs must be a positive integer.")

        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer.")

        if not isinstance(self.scale_factor, int) or isinstance(self.scale_factor, bool):
            raise ValueError("scale_factor must be 2, 3, or 4.")

        if self.scale_factor not in {2, 3, 4}:
            raise ValueError("scale_factor must be 2, 3, or 4.")

        if not isinstance(self.validation_ratio, (int, float)) or isinstance(
            self.validation_ratio,
            bool,
        ):
            raise ValueError("validation_ratio must be greater than 0 and less than 1.")

        if self.validation_ratio <= 0 or self.validation_ratio >= 1:
            raise ValueError("validation_ratio must be greater than 0 and less than 1.")

        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool):
            raise ValueError("num_workers must be a non-negative integer.")

        if self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be a non-negative integer.")

        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        self.dataset_directory = Path(self.dataset_directory)
        self.train_directory = Path(self.train_directory)
        self.validation_directory = Path(self.validation_directory)
        self.checkpoint_path = Path(self.checkpoint_path)
        self.history_path = Path(self.history_path)
