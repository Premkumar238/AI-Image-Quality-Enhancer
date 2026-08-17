"""Model definitions for image super-resolution."""

from .srcnn import SRCNN
from .utils import count_parameters

__all__ = ["SRCNN", "count_parameters"]
