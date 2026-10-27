"""Real-ESRGAN enhancement for Cutout.pro-style upscaling."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .rrdbnet import RRDBNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "RealESRGAN_x4plus.pth"
WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)

_upsampler: RealESRGANEnhancer | None = None


class RealESRGANEnhancer:
    """Load RealESRGAN_x4plus and run tiled inference on CPU/GPU."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS,
        tile: int = 256,
        tile_pad: int = 10,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.tile = tile
        self.tile_pad = tile_pad
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = 4
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        self._load_weights()
        self.model.eval()
        self.model.to(self.device)

    def _load_weights(self) -> None:
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Real-ESRGAN weights not found at {self.weights_path}. "
                "Run: python scripts/download_realesrgan.py"
            )

        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(self.weights_path, map_location="cpu")
        except Exception:
            checkpoint = torch.load(
                self.weights_path,
                map_location="cpu",
                weights_only=False,
            )

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
        else:
            state_dict = checkpoint

        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key[6:] if key.startswith("module.") else key] = value
        self.model.load_state_dict(cleaned, strict=True)

    @torch.inference_mode()
    def enhance(self, image: Image.Image, outscale: float = 4.0) -> Image.Image:
        rgb = image.convert("RGB")
        array = np.array(rgb).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.tile > 0 and max(rgb.size) > self.tile:
            output = self._tile_process(tensor)
        else:
            output = self.model(tensor)

        output = output.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = Image.fromarray((output * 255.0).round().astype(np.uint8), mode="RGB")

        if abs(outscale - self.scale) > 1e-3:
            target = (
                max(1, int(round(rgb.size[0] * outscale))),
                max(1, int(round(rgb.size[1] * outscale))),
            )
            result = result.resize(target, Image.Resampling.LANCZOS)
        return result

    def _tile_process(self, img: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output = img.new_zeros((batch, channels, output_height, output_width))

        tiles_x = (width + self.tile - 1) // self.tile
        tiles_y = (height + self.tile - 1) // self.tile

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.tile
                ofs_y = y * self.tile
                input_end_x = min(ofs_x + self.tile, width)
                input_end_y = min(ofs_y + self.tile, height)
                input_start_x = max(ofs_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y = max(ofs_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                tile = img[:, :, input_start_y:input_end_y_pad, input_start_x:input_end_x_pad]
                tile_output = self.model(tile)

                output_start_x = ofs_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = ofs_y * self.scale
                output_end_y = input_end_y * self.scale

                tile_output_start_x = (ofs_x - input_start_x) * self.scale
                tile_output_end_x = tile_output_start_x + (input_end_x - ofs_x) * self.scale
                tile_output_start_y = (ofs_y - input_start_y) * self.scale
                tile_output_end_y = tile_output_start_y + (input_end_y - ofs_y) * self.scale

                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = (
                    tile_output[
                        :,
                        :,
                        tile_output_start_y:tile_output_end_y,
                        tile_output_start_x:tile_output_end_x,
                    ]
                )
        return output


def download_weights(destination: Path = DEFAULT_WEIGHTS) -> Path:
    """Download RealESRGAN_x4plus.pth if missing."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return destination

    print(f"Downloading Real-ESRGAN weights to {destination} ...")
    tmp_path = destination.with_suffix(".pth.partial")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp_path)
    tmp_path.replace(destination)
    print("Download complete.")
    return destination


def is_realesrgan_ready(weights_path: Path = DEFAULT_WEIGHTS) -> bool:
    path = Path(weights_path)
    return path.exists() and path.stat().st_size > 1_000_000


def get_enhancer() -> RealESRGANEnhancer:
    global _upsampler
    if _upsampler is None:
        download_weights()
        _upsampler = RealESRGANEnhancer()
    return _upsampler


def enhance_with_realesrgan(image: Image.Image, scale_factor: int = 4) -> Image.Image:
    """Upscale and restore an image with Real-ESRGAN x4plus."""
    if scale_factor not in {2, 3, 4}:
        raise ValueError("scale_factor must be 2, 3, or 4.")
    enhancer = get_enhancer()
    return enhancer.enhance(image, outscale=float(scale_factor))
