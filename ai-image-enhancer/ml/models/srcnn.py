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

    This class currently defines the model configuration and structure only.
    The convolution layers will be added in a later step.
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

        # Placeholders for future SRCNN layers.
        self.feature_extraction: nn.Module | None = None
        self.non_linear_mapping: nn.Module | None = None
        self.reconstruction: nn.Module | None = None
