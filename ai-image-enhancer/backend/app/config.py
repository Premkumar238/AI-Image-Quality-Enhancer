"""Backend configuration settings."""

from app.ml_service import is_model_loaded as _is_model_loaded

APP_NAME = "AI Image Quality Enhancer API"


def is_model_loaded() -> bool:
    """Return whether trained model weights are currently loaded."""
    return _is_model_loaded()
