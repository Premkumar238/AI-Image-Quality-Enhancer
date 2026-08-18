"""Post-processing utilities that improve sharpness, contrast, and noise."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def refine_image(image: Image.Image) -> Image.Image:
    """Improve a photo after super-resolution.

    Applies denoising, local contrast, sharpening, and a mild color boost so
    blurry or low-quality images look clearer.
    """
    rgb_image = image.convert("RGB")
    array = np.array(rgb_image)

    if cv2 is not None:
        array = cv2.fastNlMeansDenoisingColored(array, None, 4, 4, 7, 21)
        lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        lightness, chroma_a, chroma_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        array = cv2.cvtColor(
            cv2.merge((lightness, chroma_a, chroma_b)),
            cv2.COLOR_LAB2RGB,
        )
        rgb_image = Image.fromarray(array)
    else:
        rgb_image = ImageOps.autocontrast(rgb_image, cutoff=1)

    rgb_image = rgb_image.filter(
        ImageFilter.UnsharpMask(radius=1.8, percent=185, threshold=1)
    )
    rgb_image = ImageEnhance.Sharpness(rgb_image).enhance(1.45)
    rgb_image = ImageEnhance.Contrast(rgb_image).enhance(1.14)
    rgb_image = ImageEnhance.Color(rgb_image).enhance(1.1)
    rgb_image = ImageEnhance.Brightness(rgb_image).enhance(1.03)
    return rgb_image
