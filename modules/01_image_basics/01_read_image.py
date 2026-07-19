"""
01_read_image.py
-----------------
Open a Sentinel-2 GeoTIFF and read it into a NumPy array.

Usage:
    python 01_read_image.py [path/to/image.tif]
"""

import sys
from pathlib import Path

import numpy as np
import rasterio

# Make the shared utils package importable from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import DEFAULT_IMAGE  # noqa: E402


def read_image(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        array = src.read()
        print(f"Opened: {path.name}")
        print(f"  Driver      : {src.driver}")
        print(f"  Band count  : {src.count}")
        print(f"  Size (WxH)  : {src.width} x {src.height}")
        print(f"  Data type   : {src.dtypes[0]}")
        print(f"  Array shape : {array.shape}  (bands, rows, cols)")
    return array


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    array = read_image(path)
    print("\nQuick pixel summary (all bands combined):")
    print(f"  min = {np.nanmin(array)}")
    print(f"  max = {np.nanmax(array)}")
    print(f"  mean = {np.nanmean(array):.2f}")


if __name__ == "__main__":
    main()
