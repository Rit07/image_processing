"""
raster_io.py
------------
Shared raster helpers used across every module in this repo.

Write a helper once here, and every script can import it:

    from utils import read_band, percentile_stretch, save_png

This keeps the individual topic scripts short and focused on the concept
they're demonstrating, instead of repeating I/O boilerplate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Repo root = two levels up from this file (utils/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "data" / "sample" / "sentinel2.tif"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def read_raster(path: Path = DEFAULT_IMAGE):
    """Read all bands into a (bands, rows, cols) array and return (array, profile)."""
    with rasterio.open(path) as src:
        return src.read(), src.profile.copy()


def read_band(path: Path = DEFAULT_IMAGE, band: int = 1) -> np.ndarray:
    """Read a single band (1-based index) as a float32 array."""
    with rasterio.open(path) as src:
        return src.read(band).astype("float32")


def percentile_stretch(band: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Rescale a band to 0-1 using percentile clipping, ignoring NaNs."""
    band = band.astype("float32")
    p_low, p_high = np.nanpercentile(band, (low, high))
    return np.clip((band - p_low) / (p_high - p_low + 1e-9), 0, 1)


def save_png(array: np.ndarray, filename: str, title: str = "",
             cmap: str | None = None, colorbar: bool = False,
             vmin=None, vmax=None) -> Path:
    """Save an array (2-D grayscale/coloured, or 3-D RGB) as a PNG in outputs/."""
    out_path = OUTPUTS_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    im = plt.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def write_singleband(array: np.ndarray, filename: str, profile: dict,
                     dtype: str = "float32", nodata=None, description: str = "") -> Path:
    """Write a single-band GeoTIFF to outputs/, preserving CRS and transform."""
    out_path = OUTPUTS_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = profile.copy()
    profile.update(count=1, dtype=dtype)
    if nodata is not None:
        profile.update(nodata=nodata)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        if description:
            dst.set_band_description(1, description)
    return out_path
