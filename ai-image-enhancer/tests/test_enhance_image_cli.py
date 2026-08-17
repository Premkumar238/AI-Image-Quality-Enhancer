"""Tests for the enhance_image command-line script."""

import subprocess
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "enhance_image.py"


def run_enhance_cli(
    input_path: Path,
    output_path: Path,
    *,
    scale_factor: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the enhance_image CLI as a subprocess."""
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        str(input_path),
        str(output_path),
    ]

    if scale_factor is not None:
        command.extend(["--scale-factor", str(scale_factor)])

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def create_test_image(path: Path, size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Create and save a temporary RGB test image."""
    image = Image.new("RGB", size, color=(120, 80, 200))
    image.save(path, format="PNG")
    return image


def test_enhance_image_cli_success(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    input_image = create_test_image(input_path)

    result = run_enhance_cli(input_path, output_path)

    assert result.returncode == 0
    assert output_path.exists()

    output_image = Image.open(output_path)
    assert output_image.mode == "RGB"
    assert output_image.size == input_image.size


def test_enhance_image_cli_scale_factor_2(tmp_path):
    input_path = tmp_path / "input_scale_2.png"
    output_path = tmp_path / "output_scale_2.png"
    input_image = create_test_image(input_path, size=(80, 60))

    result = run_enhance_cli(input_path, output_path, scale_factor=2)

    assert result.returncode == 0
    assert output_path.exists()

    output_image = Image.open(output_path)
    assert output_image.size == input_image.size


def test_enhance_image_cli_scale_factor_4(tmp_path):
    input_path = tmp_path / "input_scale_4.png"
    output_path = tmp_path / "output_scale_4.png"
    input_image = create_test_image(input_path, size=(64, 64))

    result = run_enhance_cli(input_path, output_path, scale_factor=4)

    assert result.returncode == 0
    assert output_path.exists()

    output_image = Image.open(output_path)
    assert output_image.size == input_image.size


def test_enhance_image_cli_missing_input(tmp_path):
    input_path = tmp_path / "missing.png"
    output_path = tmp_path / "output.png"

    result = run_enhance_cli(input_path, output_path)

    assert result.returncode != 0
    assert not output_path.exists()
    assert "Error:" in result.stderr
    assert "not found" in result.stderr.lower()


def test_enhance_image_cli_invalid_scale_factor(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    create_test_image(input_path)

    result = run_enhance_cli(input_path, output_path, scale_factor=0)

    assert result.returncode != 0
    assert not output_path.exists()
    assert "Error:" in result.stderr
    assert "scale_factor" in result.stderr.lower()
