"""
03_display_rgb_image.py
-----------------------
Build a true-colour (RGB) composite and save it as a PNG, using shared helpers
from utils (percentile stretch + save_png).

Band ordering assumption
------------------------
Defaults assume B2 (Blue), B3 (Green), B4 (Red), B8 (NIR). rasterio band
indices are 1-based. Adjust the constants below to match your stack.

Usage:
    python 03_display_rgb_image.py [path/to/image.tif]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import DEFAULT_IMAGE, read_band, percentile_stretch, save_png  # noqa: E402

RED_BAND = 3
GREEN_BAND = 2
BLUE_BAND = 1


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    rgb = np.dstack([
        percentile_stretch(read_band(path, RED_BAND)),
        percentile_stretch(read_band(path, GREEN_BAND)),
        percentile_stretch(read_band(path, BLUE_BAND)),
    ])

    out = save_png(rgb, "rgb_composite.png",
                   title="Sentinel-2 True-Colour Composite (2-98% stretch)")
    print(f"Saved RGB composite -> {out}")


if __name__ == "__main__":
    main()
