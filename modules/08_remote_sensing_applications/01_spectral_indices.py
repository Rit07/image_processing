"""
01_spectral_indices.py
=====================

Module 07 - Feature Extraction · spectral indices.

Normalised-difference indices combine two spectral bands into a single value in
[-1, 1] that highlights a land-cover property:

    NDVI = (NIR - Red)   / (NIR + Red)     vegetation vigour
    NDWI = (Green - NIR) / (Green + NIR)   open water (McFeeters)
    NDBI = (SWIR - NIR)  / (SWIR + NIR)     built-up / bare surfaces

They all share one operation — the normalised difference — which is exposed as a
reusable helper so you can build any custom index.

Bands may be passed as 2-D NumPy arrays or single-band PIL images; they are
converted to float internally so integer band data does not overflow.

Public functions
----------------
    normalized_difference(band_a, band_b)   -> float array in [-1, 1]
    ndvi(nir, red) / ndwi(green, nir) / ndbi(swir, nir)
    index_to_image(index, colormap=None)    -> PIL.Image  (L or RGB)

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _as_band(x) -> np.ndarray:
    """Coerce a NumPy array or single-band PIL image to a float64 2-D array."""
    if isinstance(x, Image.Image):
        if x.mode != "L":
            x = x.convert("L")
        return np.asarray(x, dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Each band must be a 2-D array or single-band image.")
    return arr


# --------------------------------------------------------------------------- #
# Reusable band math
# --------------------------------------------------------------------------- #
def normalized_difference(band_a, band_b) -> np.ndarray:
    """
    (A - B) / (A + B), evaluated safely (0 where the denominator is 0) and
    clipped to [-1, 1]. This is the building block for every index below.
    """
    a = _as_band(band_a)
    b = _as_band(band_b)
    if a.shape != b.shape:
        raise ValueError(f"Band shapes differ: {a.shape} vs {b.shape}.")
    denom = a + b
    out = np.zeros_like(a)
    nonzero = denom != 0
    out[nonzero] = (a[nonzero] - b[nonzero]) / denom[nonzero]
    return np.clip(out, -1.0, 1.0)


def ndvi(nir, red) -> np.ndarray:
    """Normalised Difference Vegetation Index."""
    return normalized_difference(nir, red)


def ndwi(green, nir) -> np.ndarray:
    """Normalised Difference Water Index (McFeeters)."""
    return normalized_difference(green, nir)


def ndbi(swir, nir) -> np.ndarray:
    """Normalised Difference Built-up Index."""
    return normalized_difference(swir, nir)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def _apply_ramp(norm01: np.ndarray, stops) -> np.ndarray:
    """Interpolate an (N,) array in [0,1] through colour stops -> (H,W,3) uint8."""
    positions = np.array([p for p, _ in stops])
    colours = np.array([c for _, c in stops], dtype=np.float64)
    flat = norm01.ravel()
    rgb = np.empty((flat.size, 3))
    for ch in range(3):
        rgb[:, ch] = np.interp(flat, positions, colours[:, ch])
    return rgb.reshape(norm01.shape + (3,)).astype(np.uint8)


# a brown -> yellow -> green ramp, useful for NDVI-style indices
_VEG_RAMP = [
    (0.00, (120, 80, 40)),
    (0.50, (230, 220, 120)),
    (0.75, (120, 190, 90)),
    (1.00, (20, 110, 40)),
]


def index_to_image(index: np.ndarray, colormap: str | None = None) -> Image.Image:
    """
    Render an index array (values in [-1, 1]) as an image.

    colormap=None  -> grayscale (L), -1 maps to black, +1 to white.
    colormap='veg' -> brown/yellow/green ramp (RGB), good for vegetation.
    """
    index = np.asarray(index, dtype=np.float64)
    norm01 = (np.clip(index, -1, 1) + 1.0) / 2.0
    if colormap is None:
        return Image.fromarray((norm01 * 255).astype(np.uint8), mode="L")
    if colormap == "veg":
        return Image.fromarray(_apply_ramp(norm01, _VEG_RAMP), mode="RGB")
    raise ValueError("colormap must be None or 'veg'.")


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    h, w = 40, 60
    # Left third = vegetation (high NIR), middle = water (high green/low NIR),
    # right = built-up (high SWIR).
    red = np.full((h, w), 40.0)
    nir = np.full((h, w), 40.0)
    green = np.full((h, w), 40.0)
    swir = np.full((h, w), 40.0)

    nir[:, 0:20] = 200        # vegetation reflects strongly in NIR
    green[:, 20:40] = 120     # water brighter in green than NIR
    nir[:, 20:40] = 30
    swir[:, 40:60] = 180      # built-up bright in SWIR
    nir[:, 40:60] = 90
    for band in (red, nir, green, swir):
        band += rng.normal(0, 3, band.shape)

    v = ndvi(nir, red)
    water = ndwi(green, nir)
    built = ndbi(swir, nir)

    print(f"NDVI  vegetation-zone mean: {v[:, 0:20].mean():+.2f} (should be high +)")
    print(f"NDWI  water-zone mean     : {water[:, 20:40].mean():+.2f} (should be +)")
    print(f"NDBI  built-zone mean     : {built[:, 40:60].mean():+.2f} (should be +)")
    assert v[:, 0:20].mean() > 0.3
    assert water[:, 20:40].mean() > 0.2
    assert built[:, 40:60].mean() > 0.2

    gray = index_to_image(v)
    colour = index_to_image(v, colormap="veg")
    assert gray.mode == "L" and colour.mode == "RGB"
    assert gray.size == (w, h)

    custom = normalized_difference(swir, green)   # arbitrary custom index
    assert custom.shape == (h, w)
    print("custom band-math index -> OK")

    print("\nSpectral indices ran successfully.")
