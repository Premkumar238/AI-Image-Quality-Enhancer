"""Image prediction utilities for SRCNN inference."""

import torch
from PIL import Image
from torch import nn

from ml.preprocessing import normalize_image, numpy_to_tensor, pil_to_numpy, tensor_to_pil

from .input_preparation import prepare_srcnn_input

MAX_OUTPUT_SIDE = 2048


def _upscaled_size(
    width: int,
    height: int,
    scale_factor: int,
    max_side: int = MAX_OUTPUT_SIDE,
) -> tuple[int, int]:
    """Return the output size after upscaling, capped to keep inference practical."""
    target_width = width * scale_factor
    target_height = height * scale_factor
    longest_side = max(target_width, target_height)

    if longest_side > max_side:
        ratio = max_side / longest_side
        target_width = max(1, int(round(target_width * ratio)))
        target_height = max(1, int(round(target_height * ratio)))

    return target_width, target_height


def predict_image(
    model: nn.Module,
    image: Image.Image,
    scale_factor: int = 4,
    upscale: bool = False,
) -> Image.Image:
    """Run SRCNN inference on a PIL RGB image.

    Args:
        model: An initialized SRCNN model provided by the caller.
        image: A PIL Image to enhance. It will be converted to RGB if needed.
        scale_factor: Factor used to prepare the low-resolution input image,
            or to enlarge the image when ``upscale`` is True.
        upscale: If True, enlarge the image by ``scale_factor`` with bicubic
            interpolation, then refine it with SRCNN. If False, simulate a
            low-resolution input and restore the original dimensions.

    Returns:
        A new PIL Image in RGB mode. The original input image is not modified.

    Raises:
        TypeError: If the model is not a PyTorch module or image is not a PIL Image.
        ValueError: If scale_factor is invalid or too large for the image size.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch nn.Module.")

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.")

    rgb_image = image.convert("RGB")

    if upscale:
        if scale_factor not in {2, 3, 4}:
            raise ValueError("scale_factor must be 2, 3, or 4.")
        expected_size = _upscaled_size(*rgb_image.size, scale_factor)
        prepared_image = rgb_image.resize(expected_size, Image.Resampling.BICUBIC)
    else:
        expected_size = rgb_image.size
        prepared_image = prepare_srcnn_input(rgb_image, scale_factor=scale_factor)

    image_array = pil_to_numpy(prepared_image)
    normalized_array = normalize_image(image_array)
    input_tensor = numpy_to_tensor(normalized_array)

    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = tensor_to_pil(output_tensor)

    if output_image.size != expected_size:
        raise ValueError("Model output dimensions do not match the expected image dimensions.")

    return output_image
