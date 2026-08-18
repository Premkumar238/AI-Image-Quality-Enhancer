"""Tests for the backend application."""

import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.main import app

client = TestClient(app)


def _create_image_bytes(size=(64, 64), image_format="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format=image_format)
    return buffer.getvalue()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Image Quality Enhancer API is running"}


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["application_name"] == "AI Image Quality Enhancer API"
    assert isinstance(payload["model_loaded"], bool)


def test_enhance_endpoint_returns_image():
    image_bytes = _create_image_bytes()
    response = client.post(
        "/api/enhance",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"scale_factor": "2"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

    output = Image.open(io.BytesIO(response.content))
    assert output.mode == "RGB"
    assert output.size == (128, 128)
    assert response.headers["x-output-size"] == "128x128"


def test_enhance_endpoint_rejects_invalid_file_type():
    response = client.post(
        "/api/enhance",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_enhance_endpoint_rejects_invalid_scale_factor():
    image_bytes = _create_image_bytes()
    response = client.post(
        "/api/enhance",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"scale_factor": "8"},
    )

    assert response.status_code == 400
