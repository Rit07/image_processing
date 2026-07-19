"""
02_image_metadata.py
---------------------
Inspect the geospatial metadata that makes a GeoTIFF more than a plain image:
coordinate reference system (CRS), affine transform, pixel resolution,
geographic bounds, nodata value, and per-band data types.

Usage:
    python 02_image_metadata.py [path/to/image.tif]
"""

import sys
from pathlib import Path

import rasterio

DEFAULT_IMAGE = Path(__file__).resolve().parents[2] / "data" / "sample" / "sentinel2.tif"


def describe(path: Path) -> None:
    with rasterio.open(path) as src:
        res_x, res_y = src.res  # pixel size in CRS units (metres for UTM)

        print(f"File            : {path.name}")
        print(f"Driver          : {src.driver}")
        print(f"Dimensions      : {src.width} x {src.height} (W x H)")
        print(f"Band count      : {src.count}")
        print(f"Data types      : {src.dtypes}")
        print(f"NoData value    : {src.nodata}")
        print()
        print(f"CRS             : {src.crs}")
        print(f"CRS (EPSG)      : {src.crs.to_epsg() if src.crs else 'n/a'}")
        print(f"Pixel size      : {res_x} x {res_y} (CRS units)")
        print()
        print("Bounds (CRS units):")
        b = src.bounds
        print(f"  left   = {b.left}")
        print(f"  bottom = {b.bottom}")
        print(f"  right  = {b.right}")
        print(f"  top    = {b.top}")
        print()
        print("Affine transform (maps pixel -> map coordinates):")
        print(f"  {src.transform}")

        # Optional band descriptions, if the file author set them.
        if any(src.descriptions):
            print("\nBand descriptions:")
            for i, desc in enumerate(src.descriptions, start=1):
                print(f"  Band {i}: {desc}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    describe(path)


if __name__ == "__main__":
    main()
