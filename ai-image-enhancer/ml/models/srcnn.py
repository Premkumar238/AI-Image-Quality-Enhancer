"""SRCNN model definition for image super-resolution."""

import torch
from torch import nn


class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network (SRCNN).

    SRCNN is a deep learning model designed to reconstruct a higher-quality
    image from a lower-quality input image.

    The full model will eventually contain three main stages:
        1. Feature extraction
        2. Non-linear mapping
        3. Reconstruction

    Currently, the feature extraction and non-linear mapping stages are implemented.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
    ) -> None:
        """Initialize the SRCNN model configuration.

        Args:
            in_channels: Number of input image channels. RGB images use 3.
            out_channels: Number of output image channels. RGB images use 3.
            num_features: Number of feature maps used in the hidden layers.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        self.feature_extraction = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_features,
            kernel_size=9,
            stride=1,
            padding=4,
        )
        self.relu = nn.ReLU()

        self.non_linear_mapping = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        # Placeholder for future SRCNN layer.
        self.reconstruction: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature extraction and non-linear mapping to the input tensor."""
        x = self.feature_extraction(x)
        x = self.relu(x)
        x = self.non_linear_mapping(x)
        x = self.relu(x)
        return x
