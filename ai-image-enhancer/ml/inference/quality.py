"""Classical photo enhancement: denoise, upscale, and reduce blur."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def is_opencv_available() -> bool:
    """Return True when OpenCV is installed for denoise/deblur processing."""
    return cv2 is not None

MAX_OUTPUT_SIDE = 2560


def _upscaled_size(
    width: int,
    height: int,
    scale_factor: int,
    max_side: int = MAX_OUTPUT_SIDE,
) -> tuple[int, int]:
    target_width = width * scale_factor
    target_height = height * scale_factor
    longest_side = max(target_width, target_height)

    if longest_side > max_side:
        ratio = max_side / longest_side
        target_width = max(1, int(round(target_width * ratio)))
        target_height = max(1, int(round(target_height * ratio)))

    return target_width, target_height


def _longest_side(image: Image.Image) -> int:
    return max(image.size)


def denoise_image(image: Image.Image) -> Image.Image:
    """Remove grain and color speckle before upscaling."""
    rgb_image = image.convert("RGB")
    array = np.array(rgb_image)
    longest_side = max(array.shape[0], array.shape[1])

    if cv2 is None:
        smoothed = rgb_image.filter(ImageFilter.MedianFilter(size=3))
        return smoothed.filter(ImageFilter.SMOOTH_MORE)

    if longest_side <= 360:
        luma_strength, chroma_strength = 14, 18
        chroma_kernel = 5
    elif longest_side <= 720:
        luma_strength, chroma_strength = 9, 12
        chroma_kernel = 3
    else:
        luma_strength, chroma_strength = 5, 8
        chroma_kernel = 3

    array = cv2.fastNlMeansDenoisingColored(
        array,
        None,
        luma_strength,
        chroma_strength,
        7,
        21,
    )

    yuv = cv2.cvtColor(array, cv2.COLOR_RGB2YUV)
    luma, chroma_u, chroma_v = cv2.split(yuv)
    chroma_u = cv2.GaussianBlur(chroma_u, (chroma_kernel, chroma_kernel), 0)
    chroma_v = cv2.GaussianBlur(chroma_v, (chroma_kernel, chroma_kernel), 0)
    array = cv2.cvtColor(cv2.merge((luma, chroma_u, chroma_v)), cv2.COLOR_YUV2RGB)
    return Image.fromarray(array)


def upscale_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Enlarge an image while keeping edges smooth."""
    if scale_factor not in {2, 3, 4}:
        raise ValueError("scale_factor must be 2, 3, or 4.")

    target_size = _upscaled_size(*image.size, scale_factor)
    if cv2 is None:
        return image.resize(target_size, Image.Resampling.LANCZOS)

    array = np.array(image.convert("RGB"))
    upscaled = cv2.resize(array, target_size, interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(upscaled)


def _sharpen_luminance(array: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    lightness, chroma_a, chroma_b = cv2.split(lab)
    blurred = cv2.GaussianBlur(lightness, (0, 0), sigma)
    sharpened = cv2.addWeighted(lightness, 1 + amount, blurred, -amount, 0)
    return cv2.cvtColor(cv2.merge((sharpened, chroma_a, chroma_b)), cv2.COLOR_LAB2RGB)


def reduce_blur(image: Image.Image) -> Image.Image:
    """Reduce softness and blur while preserving natural skin tones."""
    rgb_image = image.convert("RGB")
    array = np.array(rgb_image)

    if cv2 is None:
        polished = rgb_image.filter(ImageFilter.UnsharpMask(radius=1.6, percent=120, threshold=3))
        return ImageEnhance.Sharpness(polished).enhance(1.2)

    array = cv2.edgePreservingFilter(array, flags=cv2.RECURS_FILTER, sigma_s=50, sigma_r=0.35)

    yuv = cv2.cvtColor(array, cv2.COLOR_RGB2YUV)
    luma, chroma_u, chroma_v = cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    luma = clahe.apply(luma)
    array = cv2.cvtColor(cv2.merge((luma, chroma_u, chroma_v)), cv2.COLOR_YUV2RGB)

    array = _sharpen_luminance(array, amount=0.55, sigma=1.2)
    array = cv2.bilateralFilter(array, d=7, sigmaColor=28, sigmaSpace=28)

    yuv = cv2.cvtColor(array, cv2.COLOR_RGB2YUV)
    luma, chroma_u, chroma_v = cv2.split(yuv)
    chroma_u = cv2.GaussianBlur(chroma_u, (3, 3), 0)
    chroma_v = cv2.GaussianBlur(chroma_v, (3, 3), 0)
    array = cv2.cvtColor(cv2.merge((luma, chroma_u, chroma_v)), cv2.COLOR_YUV2RGB)
    return Image.fromarray(array)


def refine_image(image: Image.Image) -> Image.Image:
    """Final light polish after blur reduction."""
    rgb_image = image.convert("RGB")

    if cv2 is None:
        rgb_image = ImageOps.autocontrast(rgb_image, cutoff=0.5)
        return rgb_image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=4))

    array = np.array(rgb_image)
    array = _sharpen_luminance(array, amount=0.25, sigma=0.9)
    rgb_image = Image.fromarray(array)
    rgb_image = ImageEnhance.Contrast(rgb_image).enhance(1.06)
    rgb_image = ImageEnhance.Color(rgb_image).enhance(1.04)
    return rgb_image


def enhance_photo(
    image: Image.Image,
    scale_factor: int = 4,
    model=None,
) -> Image.Image:
    """Denoise, upscale, and reduce blur using a stable classical pipeline.

    The optional ``model`` argument is kept for compatibility but is not used.
    The trained SRCNN checkpoint was adding color noise on small blurry photos,
    so the website uses reliable OpenCV processing instead.
    """
    del model

    cleaned = denoise_image(image)
    upscaled = upscale_image(cleaned, scale_factor=scale_factor)
    deblurred = reduce_blur(upscaled)
    return refine_image(deblurred)
