"""Post-processing utilities that improve sharpness, contrast, and noise."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _to_uint8_image(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def denoise_image(image: Image.Image) -> Image.Image:
    """Remove grain and color speckle from the original photo.

    Denoising is applied before upscaling so noise is not enlarged.
    """
    rgb_image = image.convert("RGB")
    array = np.array(rgb_image)
    longest_side = max(array.shape[0], array.shape[1])

    if cv2 is None:
        smoothed = rgb_image.filter(ImageFilter.MedianFilter(size=3))
        return smoothed.filter(ImageFilter.SMOOTH_MORE)

    if longest_side <= 360:
        luma_strength, chroma_strength = 11, 16
        chroma_kernel = 5
    elif longest_side <= 720:
        luma_strength, chroma_strength = 7, 10
        chroma_kernel = 3
    else:
        luma_strength, chroma_strength = 4, 7
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


def _luminance_unsharp(array: np.ndarray, amount: float = 0.42, sigma: float = 1.05) -> np.ndarray:
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    lightness, chroma_a, chroma_b = cv2.split(lab)
    blurred = cv2.GaussianBlur(lightness, (0, 0), sigma)
    sharpened = cv2.addWeighted(lightness, 1 + amount, blurred, -amount, 0)
    return cv2.cvtColor(cv2.merge((sharpened, chroma_a, chroma_b)), cv2.COLOR_LAB2RGB)


def _blend_detail_into_clean(
    clean_image: Image.Image,
    detail_image: Image.Image,
    detail_amount: float = 0.22,
) -> Image.Image:
    """Keep clean colors from Lanczos and borrow a little structure from SRCNN."""
    clean = np.array(clean_image.convert("RGB"), dtype=np.float32)
    detail = np.array(detail_image.convert("RGB"), dtype=np.float32)

    if cv2 is None:
        mixed = clean * (1 - detail_amount) + detail * detail_amount
        return _to_uint8_image(mixed)

    clean_yuv = cv2.cvtColor(clean.astype(np.uint8), cv2.COLOR_RGB2YUV)
    detail_yuv = cv2.cvtColor(detail.astype(np.uint8), cv2.COLOR_RGB2YUV)
    clean_y, chroma_u, chroma_v = cv2.split(clean_yuv)
    detail_y, _, _ = cv2.split(detail_yuv)
    merged_y = cv2.addWeighted(clean_y, 1 - detail_amount, detail_y, detail_amount, 0)
    merged = cv2.cvtColor(cv2.merge((merged_y, chroma_u, chroma_v)), cv2.COLOR_YUV2RGB)
    return Image.fromarray(merged)


def refine_image(image: Image.Image) -> Image.Image:
    """Mild polish after upscaling: edge sharpening without amplifying grain."""
    rgb_image = image.convert("RGB")
    array = np.array(rgb_image)

    if cv2 is not None:
        yuv = cv2.cvtColor(array, cv2.COLOR_RGB2YUV)
        luma, chroma_u, chroma_v = cv2.split(yuv)
        chroma_u = cv2.GaussianBlur(chroma_u, (3, 3), 0)
        chroma_v = cv2.GaussianBlur(chroma_v, (3, 3), 0)
        array = cv2.cvtColor(cv2.merge((luma, chroma_u, chroma_v)), cv2.COLOR_YUV2RGB)
        array = _luminance_unsharp(array)
        rgb_image = Image.fromarray(array)
    else:
        rgb_image = ImageOps.autocontrast(rgb_image, cutoff=0.5)
        rgb_image = rgb_image.filter(
            ImageFilter.UnsharpMask(radius=1.1, percent=70, threshold=4)
        )

    rgb_image = ImageEnhance.Contrast(rgb_image).enhance(1.08)
    rgb_image = ImageEnhance.Color(rgb_image).enhance(1.05)
    rgb_image = ImageEnhance.Sharpness(rgb_image).enhance(1.1)
    return rgb_image


def enhance_photo(model, image: Image.Image, scale_factor: int = 4) -> Image.Image:
    """Denoise, enlarge, then lightly sharpen a photo.

    SRCNN is used only as a small detail mix so a weakly trained checkpoint
    cannot reintroduce color noise after upscaling.
    """
    from ml.inference.predict import predict_image

    cleaned = denoise_image(image)
    srcnn_output = predict_image(
        model,
        cleaned,
        scale_factor=scale_factor,
        upscale=True,
    )
    lanczos = cleaned.resize(srcnn_output.size, Image.Resampling.LANCZOS)
    blended = _blend_detail_into_clean(lanczos, srcnn_output)
    return refine_image(blended)
