"""
02_opening_closing.py
====================

Module 05 - Morphological Operations · opening and closing.

These compose the two fundamental operators:

    Opening  = erosion then dilation.
               Removes small foreground speckle (isolated bright spots, thin
               spurs) while keeping the overall size of larger regions.
    Closing  = dilation then erosion.
               Fills small holes and narrow gaps inside foreground regions,
               again roughly preserving their outer size.

Both use the same structuring-element builders as script 01.

Public functions
----------------
    square(size=3) / disk(radius=1) / cross(size=3)
    opening(image, selem=None, iterations=1)   -> Image  (binary 0/255)
    closing(image, selem=None, iterations=1)   -> Image  (binary 0/255)

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Binary <-> image helpers
# --------------------------------------------------------------------------- #
def _to_binary(image: Image.Image, threshold: int = 127) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8) >= threshold


def _to_image(binary: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(binary, 255, 0).astype(np.uint8), mode="L")


# --------------------------------------------------------------------------- #
# Structuring elements
# --------------------------------------------------------------------------- #
def square(size: int = 3) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer.")
    return np.ones((size, size), dtype=bool)


def disk(radius: int = 1) -> np.ndarray:
    if radius < 1:
        raise ValueError("radius must be >= 1.")
    coords = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(coords, coords)
    return (xx ** 2 + yy ** 2) <= radius ** 2


def cross(size: int = 3) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer.")
    se = np.zeros((size, size), dtype=bool)
    c = size // 2
    se[c, :] = True
    se[:, c] = True
    return se


# --------------------------------------------------------------------------- #
# Core morphology
# --------------------------------------------------------------------------- #
def _morph(binary: np.ndarray, selem: np.ndarray, op: str) -> np.ndarray:
    kh, kw = selem.shape
    ph, pw = kh // 2, kw // 2
    h, w = binary.shape
    padded = np.pad(binary, ((ph, ph), (pw, pw)), constant_values=False)

    if op == "dilate":
        out = np.zeros((h, w), dtype=bool)
    else:
        out = np.ones((h, w), dtype=bool)

    for i in range(kh):
        for j in range(kw):
            if selem[i, j]:
                shifted = padded[i:i + h, j:j + w]
                out = np.logical_or(out, shifted) if op == "dilate" \
                    else np.logical_and(out, shifted)
    return out


def _erode(binary, selem, iterations):
    for _ in range(iterations):
        binary = _morph(binary, selem, "erode")
    return binary


def _dilate(binary, selem, iterations):
    for _ in range(iterations):
        binary = _morph(binary, selem, "dilate")
    return binary


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def opening(image: Image.Image, selem: np.ndarray | None = None,
            iterations: int = 1, threshold: int = 127) -> Image.Image:
    """Erosion followed by dilation — clears small foreground speckle."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    selem = square(3) if selem is None else selem.astype(bool)
    binary = _to_binary(image, threshold)
    binary = _dilate(_erode(binary, selem, iterations), selem, iterations)
    return _to_image(binary)


def closing(image: Image.Image, selem: np.ndarray | None = None,
            iterations: int = 1, threshold: int = 127) -> Image.Image:
    """Dilation followed by erosion — fills small holes and gaps."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    selem = square(3) if selem is None else selem.astype(bool)
    binary = _to_binary(image, threshold)
    binary = _erode(_dilate(binary, selem, iterations), selem, iterations)
    return _to_image(binary)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    canvas = np.zeros((60, 60), dtype=np.uint8)
    canvas[15:45, 15:45] = 255                       # solid block
    canvas[25:30, 25:30] = 0                         # a hole inside it
    speckle = rng.random(canvas.shape) < 0.04        # scattered noise
    canvas[speckle] = 255
    test = Image.fromarray(canvas, mode="L")

    opened = opening(test)
    closed = closing(test)
    for label, img in (("opening", opened), ("closing", closed)):
        assert img.size == test.size and img.mode == "L"
        print(f"{label:8s} -> OK ({int((np.asarray(img) > 0).sum())} px)")

    # Opening should knock out most isolated speckle pixels outside the block.
    outside = np.zeros((60, 60), dtype=bool)
    outside[:15, :] = outside[45:, :] = outside[:, :15] = outside[:, 45:] = True
    before = int((np.asarray(test)[outside] > 0).sum())
    after = int((np.asarray(opened)[outside] > 0).sum())
    print(f"speckle outside block: {before} -> {after} after opening")
    assert after <= before

    print("\nOpening / closing ran successfully.")
