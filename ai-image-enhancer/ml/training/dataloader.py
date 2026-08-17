"""DataLoader utilities for super-resolution training."""

from torch.utils.data import DataLoader

from .dataset import SuperResolutionDataset


def create_dataloader(
    dataset: SuperResolutionDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader for a SuperResolutionDataset.

    Args:
        dataset: An initialized SuperResolutionDataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples each epoch.
        num_workers: Number of subprocesses used for data loading.

    Returns:
        A PyTorch DataLoader that yields batched input and target tensors.

    Raises:
        TypeError: If dataset is not a SuperResolutionDataset.
        ValueError: If batch_size or num_workers is invalid.
    """
    if not isinstance(dataset, SuperResolutionDataset):
        raise TypeError("dataset must be a SuperResolutionDataset instance.")

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
