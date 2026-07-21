"""
01_thresholding.py
=================

Module 06 - Image Segmentation · thresholding.

The simplest form of segmentation: split pixels into foreground/background by
intensity.

    manual_threshold  — you supply the cut value.
    otsu_threshold    — the cut value is found automatically by maximising the
                        between-class variance of the intensity histogram
                        (Otsu's method), which works well for bimodal scenes.

Public functions
----------------
    manual_threshold(image, threshold=127, invert=False)      -> Image
    otsu_value(image)                                         -> int
    otsu_threshold(image, invert=False, return_value=False)   -> Image | (Image, int)

All return a binary (0/255) grayscale PIL image.

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _to_gray_array(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8)


def _binarise(gray: np.ndarray, threshold: int, invert: bool,
              strict: bool = False) -> Image.Image:
    # strict=True uses > threshold (Otsu's <= t vs > t split);
    # strict=False uses >= threshold (inclusive, intuitive for manual cuts).
    fg = gray > threshold if strict else gray >= threshold
    mask = ~fg if invert else fg
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")


def manual_threshold(image: Image.Image, threshold: int = 127,
                     invert: bool = False) -> Image.Image:
    """Foreground = pixels >= threshold (or < threshold if invert=True)."""
    if not (0 <= threshold <= 255):
        raise ValueError("threshold must be in 0..255.")
    return _binarise(_to_gray_array(image), threshold, invert)


def otsu_value(image: Image.Image) -> int:
    """
    Compute Otsu's optimal global threshold (0..255).

    Maximises between-class variance sigma_b^2(t) = w0*w1*(mu0 - mu1)^2 over all
    candidate thresholds t, using the intensity histogram.
    """
    gray = _to_gray_array(image)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / gray.size

    omega = np.cumsum(prob)                       # class-0 weight up to t
    mu = np.cumsum(prob * np.arange(256))         # cumulative mean
    mu_total = mu[-1]

    # Guard against divide-by-zero at the ends where a class is empty.
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_total * omega - mu) ** 2 / denom
    sigma_b2 = np.nan_to_num(sigma_b2)
    return int(np.argmax(sigma_b2))


def otsu_threshold(image: Image.Image, invert: bool = False,
                   return_value: bool = False):
    """Binarise using the automatically chosen Otsu threshold."""
    t = otsu_value(image)
    binary = _binarise(_to_gray_array(image), t, invert, strict=True)
    return (binary, t) if return_value else binary


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Bimodal scene: dark background (~60) + bright object (~190) + noise.
    canvas = np.full((80, 80), 60, dtype=np.float64)
    canvas[25:60, 25:60] = 190
    canvas += rng.normal(0, 10, canvas.shape)
    test = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="L")

    t = otsu_value(test)
    print(f"Otsu threshold chosen: {t}  (any value in the ~90-160 valley separates)")
    assert 85 <= t <= 165, "threshold should fall in the gap between the two modes"

    man = manual_threshold(test, 125)
    ots = otsu_threshold(test)
    for label, img in (("manual", man), ("otsu", ots)):
        assert img.size == test.size and img.mode == "L"
        fg = int((np.asarray(img) > 0).sum())
        print(f"{label:6s} -> OK ({fg} fg px)")
        # object is a 35x35 = 1225 px block; foreground should be close to it
        assert 1000 < fg < 1500

    print("\nThresholding ran successfully.")
