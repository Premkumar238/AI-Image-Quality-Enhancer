"""Image prediction utilities for SRCNN inference."""

import torch
from PIL import Image
from torch import nn

from ml.preprocessing import normalize_image, numpy_to_tensor, pil_to_numpy, tensor_to_pil


def predict_image(model: nn.Module, image: Image.Image) -> Image.Image:
    """Run SRCNN inference on a PIL RGB image.

    Args:
        model: An initialized SRCNN model provided by the caller.
        image: A PIL Image to enhance. It will be converted to RGB if needed.

    Returns:
        A new PIL Image in RGB mode produced by the model.
        The original input image is not modified.

    Raises:
        TypeError: If the model is not a PyTorch module or image is not a PIL Image.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch nn.Module.")

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.")

    rgb_image = image.convert("RGB")

    image_array = pil_to_numpy(rgb_image)
    normalized_array = normalize_image(image_array)
    input_tensor = numpy_to_tensor(normalized_array)

    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)

    return tensor_to_pil(output_tensor)
