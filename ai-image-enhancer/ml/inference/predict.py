"""Image prediction utilities for SRCNN inference."""

import torch
from PIL import Image
from torch import nn

from ml.preprocessing import normalize_image, numpy_to_tensor, pil_to_numpy, tensor_to_pil

from .input_preparation import prepare_srcnn_input


def predict_image(
    model: nn.Module,
    image: Image.Image,
    scale_factor: int = 4,
) -> Image.Image:
    """Run SRCNN inference on a PIL RGB image.

    Args:
        model: An initialized SRCNN model provided by the caller.
        image: A PIL Image to enhance. It will be converted to RGB if needed.
        scale_factor: Factor used to prepare the low-resolution input image.

    Returns:
        A new PIL Image in RGB mode with the same dimensions as the input.
        The original input image is not modified.

    Raises:
        TypeError: If the model is not a PyTorch module or image is not a PIL Image.
        ValueError: If scale_factor is invalid or too large for the image size.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch nn.Module.")

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.")

    original_size = image.size
    prepared_image = prepare_srcnn_input(image, scale_factor=scale_factor)

    image_array = pil_to_numpy(prepared_image)
    normalized_array = normalize_image(image_array)
    input_tensor = numpy_to_tensor(normalized_array)

    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = tensor_to_pil(output_tensor)

    if output_image.size != original_size:
        raise ValueError("Model output dimensions do not match the input image dimensions.")

    return output_image
