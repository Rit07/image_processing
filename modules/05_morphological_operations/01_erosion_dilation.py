"""
01_erosion_dilation.py
=====================

Module 05 - Morphological Operations · the two fundamental operators.

Morphology probes a binary image with a small "structuring element" (SE).

    Erosion  — a pixel stays foreground only if the SE fits entirely inside the
               foreground. Shrinks regions, removes thin protrusions/speckle.
    Dilation — a pixel becomes foreground if the SE touches any foreground.
               Grows regions, fills small gaps.

Structuring-element builders (square / disk / cross) are provided so callers can
tune the neighbourhood shape.

Public functions
----------------
    square(size=3) / disk(radius=1) / cross(size=3)   -> bool SE array
    erode(image, selem=None, iterations=1)            -> Image  (binary 0/255)
    dilate(image, selem=None, iterations=1)           -> Image  (binary 0/255)

Inputs are binarised at `threshold` (foreground = pixels >= threshold).

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Binary <-> image helpers
# --------------------------------------------------------------------------- #
def _to_binary(image: Image.Image, threshold: int = 127) -> np.ndarray:
    """Convert any PIL image to a boolean foreground mask."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8) >= threshold


def _to_image(binary: np.ndarray) -> Image.Image:
    """Boolean mask -> 0/255 grayscale PIL image."""
    return Image.fromarray(np.where(binary, 255, 0).astype(np.uint8), mode="L")


# --------------------------------------------------------------------------- #
# Structuring elements
# --------------------------------------------------------------------------- #
def square(size: int = 3) -> np.ndarray:
    """Solid square SE of side `size` (odd)."""
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer.")
    return np.ones((size, size), dtype=bool)


def disk(radius: int = 1) -> np.ndarray:
    """Approximate circular SE of the given radius."""
    if radius < 1:
        raise ValueError("radius must be >= 1.")
    coords = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(coords, coords)
    return (xx ** 2 + yy ** 2) <= radius ** 2


def cross(size: int = 3) -> np.ndarray:
    """Plus-shaped SE of side `size` (odd)."""
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer.")
    se = np.zeros((size, size), dtype=bool)
    c = size // 2
    se[c, :] = True
    se[:, c] = True
    return se


# --------------------------------------------------------------------------- #
# Core morphology (vectorised over the SE offsets)
# --------------------------------------------------------------------------- #
def _morph(binary: np.ndarray, selem: np.ndarray, op: str) -> np.ndarray:
    """
    Single erosion or dilation pass.

    The image border is treated as background: dilation cannot invent
    foreground beyond the edge, and erosion trims foreground touching it.
    """
    kh, kw = selem.shape
    ph, pw = kh // 2, kw // 2
    h, w = binary.shape
    padded = np.pad(binary, ((ph, ph), (pw, pw)), constant_values=False)

    if op == "dilate":
        out = np.zeros((h, w), dtype=bool)
        combine = np.logical_or
    elif op == "erode":
        out = np.ones((h, w), dtype=bool)
        combine = np.logical_and
    else:  # pragma: no cover
        raise ValueError("op must be 'erode' or 'dilate'.")

    for i in range(kh):
        for j in range(kw):
            if selem[i, j]:
                shifted = padded[i:i + h, j:j + w]
                if op == "erode":
                    out = combine(out, shifted)
                else:
                    out = combine(out, shifted)
    return out


def _repeat(binary, selem, op, iterations):
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    result = binary
    for _ in range(iterations):
        result = _morph(result, selem, op)
    return result


def erode(image: Image.Image, selem: np.ndarray | None = None,
          iterations: int = 1, threshold: int = 127) -> Image.Image:
    """Binary erosion. Default SE is a 3x3 square."""
    selem = square(3) if selem is None else selem.astype(bool)
    binary = _to_binary(image, threshold)
    return _to_image(_repeat(binary, selem, "erode", iterations))


def dilate(image: Image.Image, selem: np.ndarray | None = None,
           iterations: int = 1, threshold: int = 127) -> Image.Image:
    """Binary dilation. Default SE is a 3x3 square."""
    selem = square(3) if selem is None else selem.astype(bool)
    binary = _to_binary(image, threshold)
    return _to_image(_repeat(binary, selem, "dilate", iterations))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    canvas = np.zeros((60, 60), dtype=np.uint8)
    canvas[20:40, 20:40] = 255            # a 20x20 solid block
    test = Image.fromarray(canvas, mode="L")
    fg = lambda im: int((np.asarray(im) > 0).sum())
    base = fg(test)

    eroded = erode(test)
    dilated = dilate(test)
    print(f"original foreground : {base}")
    print(f"after erosion       : {fg(eroded)}  (should shrink)")
    print(f"after dilation      : {fg(dilated)}  (should grow)")
    assert fg(eroded) < base < fg(dilated)

    for se_name, se in (("square", square(3)), ("disk", disk(2)), ("cross", cross(3))):
        out = dilate(test, selem=se)
        assert out.size == test.size and out.mode == "L"
        print(f"SE {se_name:6s} dilation -> OK ({fg(out)} px)")

    print("\nErosion / dilation ran successfully.")
