"""Load the SRCNN model and run image enhancement for the API."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import predict_image, refine_image
from ml.models import SRCNN, load_model

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "srcnn.pth"

_model: SRCNN | None = None
_model_loaded = False


def is_model_loaded() -> bool:
    """Return True when trained checkpoint weights are loaded."""
    return _model_loaded


def get_model() -> SRCNN:
    """Return the in-memory SRCNN model, creating it if needed."""
    global _model
    if _model is None:
        initialize_model()
    assert _model is not None
    return _model


def initialize_model(checkpoint_path: Path | None = None) -> None:
    """Create the SRCNN model and load checkpoint weights when available."""
    global _model, _model_loaded

    _model = SRCNN()
    _model.eval()
    _model_loaded = False

    path = checkpoint_path or DEFAULT_CHECKPOINT
    if path.exists():
        load_model(_model, path, device="cpu")
        _model_loaded = True


def enhance_image_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
    scale_factor: int = 4,
) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """Enhance an uploaded image and return encoded bytes plus media type.

    Returns:
        A tuple of (image_bytes, media_type, original_size, output_size).
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    original_size = rgb_image.size
    enhanced = predict_image(
        get_model(),
        rgb_image,
        scale_factor=scale_factor,
        upscale=True,
    )
    enhanced = refine_image(enhanced)

    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image_format = "JPEG"
        media_type = "image/jpeg"
    else:
        image_format = "PNG"
        media_type = "image/png"

    buffer = io.BytesIO()
    save_kwargs: dict[str, object] = {"format": image_format}
    if image_format == "JPEG":
        save_kwargs["quality"] = 95
    enhanced.save(buffer, **save_kwargs)
    return buffer.getvalue(), media_type, original_size, enhanced.size
