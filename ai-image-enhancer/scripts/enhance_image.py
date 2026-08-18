"""Command-line script for enhancing images with SRCNN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference import predict_image, refine_image
from ml.models import SRCNN, load_model
from ml.preprocessing import load_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the enhancement script."""
    parser = argparse.ArgumentParser(
        description="Enhance an image using the SRCNN super-resolution model.",
    )
    parser.add_argument(
        "input_path",
        help="Path to the input image file.",
    )
    parser.add_argument(
        "output_path",
        help="Path where the enhanced image will be saved.",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=4,
        help="Upscale factor. The output image will be this many times larger (default: 4).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a trained model checkpoint file.",
    )
    return parser.parse_args()


def validate_scale_factor(scale_factor: int) -> None:
    """Validate the scale factor before running inference.

    Args:
        scale_factor: User-provided downsampling factor.

    Raises:
        ValueError: If scale_factor is not a positive integer.
    """
    if not isinstance(scale_factor, int) or isinstance(scale_factor, bool):
        raise ValueError("scale_factor must be a positive integer.")

    if scale_factor <= 0:
        raise ValueError("scale_factor must be a positive integer.")


def save_enhanced_image(image: Image.Image, output_path: Path) -> None:
    """Save an enhanced image, creating parent directories when needed.

    The output format is chosen from the file extension when possible.

    Args:
        image: Enhanced PIL image to save.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extension = output_path.suffix.lower()
    format_map = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }

    image_format = format_map.get(extension)
    if image_format is None:
        image.save(output_path)
        return

    save_kwargs: dict[str, object] = {"format": image_format}
    if image_format == "JPEG":
        save_kwargs["quality"] = 95

    image.save(output_path, **save_kwargs)


def main() -> int:
    """Run the image enhancement CLI."""
    args = parse_args()

    try:
        validate_scale_factor(args.scale_factor)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    print("Loading image...")
    try:
        image = load_image(input_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Loading model...")
    model = SRCNN()

    if args.checkpoint:
        try:
            load_model(model, args.checkpoint, device="cpu")
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        print(
            "Warning: No checkpoint provided. Using a randomly initialized model. "
            "The output will not reflect trained enhancement quality.",
            file=sys.stderr,
        )

    print("Running inference...")
    try:
        enhanced_image = predict_image(
            model,
            image,
            scale_factor=args.scale_factor,
            upscale=True,
        )
        enhanced_image = refine_image(enhanced_image)
    except (TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Saving result...")
    try:
        save_enhanced_image(enhanced_image, output_path)
    except OSError as exc:
        print(f"Error: Unable to save output image: {exc}", file=sys.stderr)
        return 1

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
