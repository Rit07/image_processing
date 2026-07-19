"""
06_save_processed_image.py
--------------------------
Compute NDVI and save it as a georeferenced GeoTIFF plus a coloured PNG,
using shared helpers from utils.

    NDVI = (NIR - RED) / (NIR + RED)

Band ordering assumption
------------------------
Defaults assume B4 (Red) = band 3 and B8 (NIR) = band 4. Adjust to your stack.

Usage:
    python 06_save_processed_image.py [path/to/image.tif]
"""

import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import DEFAULT_IMAGE, read_band, write_singleband, save_png  # noqa: E402

RED_BAND = 3
NIR_BAND = 4


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    red = read_band(path, RED_BAND)
    nir = read_band(path, NIR_BAND)
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom).astype("float32")

    with rasterio.open(path) as src:
        profile = src.profile.copy()

    tif = write_singleband(ndvi, "ndvi.tif", profile,
                           dtype="float32", nodata=np.nan, description="NDVI")
    print(f"Saved NDVI GeoTIFF -> {tif}")

    png = save_png(ndvi, "ndvi.png", title="NDVI",
                   cmap="RdYlGn", colorbar=True, vmin=-1, vmax=1)
    print(f"Saved NDVI preview -> {png}")

    valid = ndvi[~np.isnan(ndvi)]
    print(f"\nNDVI range: {valid.min():.3f} to {valid.max():.3f} "
          f"(mean {valid.mean():.3f})")


if __name__ == "__main__":
    main()
