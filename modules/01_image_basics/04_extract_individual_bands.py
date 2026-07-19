"""
04_extract_individual_bands.py
------------------------------
Split a multi-band image into individual single-band GeoTIFFs (georeferenced)
plus grayscale PNG previews, using shared helpers from utils.

Usage:
    python 04_extract_individual_bands.py [path/to/image.tif]
"""

import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import DEFAULT_IMAGE, write_singleband, save_png  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        for b in range(1, src.count + 1):
            band = src.read(b).astype("float32")

            tif = write_singleband(band, f"bands/band_{b:02d}.tif", profile,
                                   dtype=src.dtypes[0])
            p_low, p_high = np.nanpercentile(band, (2, 98))
            png = save_png(band, f"bands/band_{b:02d}.png", title=f"Band {b}",
                           cmap="gray", vmin=p_low, vmax=p_high)
            print(f"Band {b:02d} -> {tif.name}, {png.name}")

    print("\nDone. See outputs/bands/")


if __name__ == "__main__":
    main()
