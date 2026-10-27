"""Download pretrained Real-ESRGAN weights used by the website enhancer."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.realesrgan_enhance import DEFAULT_WEIGHTS, download_weights


def main() -> int:
    path = download_weights(DEFAULT_WEIGHTS)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Ready: {path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
