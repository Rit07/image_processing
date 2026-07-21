"""
03_mask_cleanup.py
=================

Module 05 - Morphological Operations · practical mask cleanup.

Ties the operators together into a realistic post-classification step: take a
feature image (or an already-binary mask), threshold it, then tidy the result:

    1. Opening   -> remove isolated speckle / salt noise.
    2. Closing   -> fill small holes and bridge narrow gaps.
    3. Small-object removal (optional) -> drop connected components below a
       minimum pixel area, using a from-scratch flood-fill labelling.

Public functions
----------------
    clean_mask(image, threshold=127, selem=None, iterations=1,
               remove_speckle=True, fill_holes=True, min_size=0) -> Image
    remove_small_objects(image, min_size, threshold=127)          -> Image

Both return a binary (0/255) grayscale PIL image.

Dependencies: numpy, Pillow
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Binary <-> image helpers  +  structuring elements
# --------------------------------------------------------------------------- #
def _to_binary(image: Image.Image, threshold: int = 127) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8) >= threshold


def _to_image(binary: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(binary, 255, 0).astype(np.uint8), mode="L")


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


# --------------------------------------------------------------------------- #
# Core morphology
# --------------------------------------------------------------------------- #
def _morph(binary: np.ndarray, selem: np.ndarray, op: str) -> np.ndarray:
    kh, kw = selem.shape
    ph, pw = kh // 2, kw // 2
    h, w = binary.shape
    padded = np.pad(binary, ((ph, ph), (pw, pw)), constant_values=False)
    out = np.zeros((h, w), dtype=bool) if op == "dilate" else np.ones((h, w), dtype=bool)
    for i in range(kh):
        for j in range(kw):
            if selem[i, j]:
                s = padded[i:i + h, j:j + w]
                out = np.logical_or(out, s) if op == "dilate" else np.logical_and(out, s)
    return out


def _opening(binary, selem, iterations):
    for _ in range(iterations):
        binary = _morph(binary, selem, "erode")
    for _ in range(iterations):
        binary = _morph(binary, selem, "dilate")
    return binary


def _closing(binary, selem, iterations):
    for _ in range(iterations):
        binary = _morph(binary, selem, "dilate")
    for _ in range(iterations):
        binary = _morph(binary, selem, "erode")
    return binary


# --------------------------------------------------------------------------- #
# Connected-component labelling (8-connectivity, BFS flood fill)
# --------------------------------------------------------------------------- #
def _label(binary: np.ndarray):
    """Return (labels, sizes) where labels[i,j] is a component id (0 = bg)."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    sizes: list[int] = [0]                       # index 0 reserved for background
    current = 0
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for r in range(h):
        for c in range(w):
            if binary[r, c] and labels[r, c] == 0:
                current += 1
                count = 0
                queue = deque([(r, c)])
                labels[r, c] = current
                while queue:
                    y, x = queue.popleft()
                    count += 1
                    for dy, dx in neighbours:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w \
                                and binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            queue.append((ny, nx))
                sizes.append(count)
    return labels, sizes


def remove_small_objects(image: Image.Image, min_size: int,
                         threshold: int = 127) -> Image.Image:
    """Drop connected foreground components smaller than `min_size` pixels."""
    binary = _to_binary(image, threshold)
    labels, sizes = _label(binary)
    keep = np.array([s >= min_size for s in sizes], dtype=bool)
    keep[0] = False                              # background never kept
    return _to_image(keep[labels])


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def clean_mask(
    image: Image.Image,
    threshold: int = 127,
    selem: np.ndarray | None = None,
    iterations: int = 1,
    remove_speckle: bool = True,
    fill_holes: bool = True,
    min_size: int = 0,
) -> Image.Image:
    """
    Threshold an image and clean the resulting mask.

    remove_speckle : apply an opening.
    fill_holes     : apply a closing.
    min_size       : if > 0, also remove connected components below this area.
    """
    selem = square(3) if selem is None else selem.astype(bool)
    binary = _to_binary(image, threshold)

    if remove_speckle:
        binary = _opening(binary, selem, iterations)
    if fill_holes:
        binary = _closing(binary, selem, iterations)

    if min_size > 0:
        labels, sizes = _label(binary)
        keep = np.array([s >= min_size for s in sizes], dtype=bool)
        keep[0] = False
        binary = keep[labels]

    return _to_image(binary)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(2)
    canvas = np.zeros((80, 80), dtype=np.uint8)
    canvas[20:60, 20:60] = 255                       # main feature
    canvas[35:40, 35:40] = 0                         # a hole
    # scattered small speckle clusters elsewhere
    for _ in range(15):
        y, x = rng.integers(0, 78, size=2)
        canvas[y:y + 2, x:x + 2] = 255
    test = Image.fromarray(canvas, mode="L")
    fg = lambda im: int((np.asarray(im) > 0).sum())

    cleaned = clean_mask(test, remove_speckle=True, fill_holes=True, min_size=20)
    assert cleaned.size == test.size and cleaned.mode == "L"
    print(f"raw mask foreground     : {fg(test)}")
    print(f"cleaned mask foreground : {fg(cleaned)}")

    # After cleanup, essentially one big component should remain.
    labels, sizes = _label(np.asarray(cleaned) > 0)
    big = [s for s in sizes[1:] if s >= 20]
    print(f"components >= 20 px kept : {len(big)}")
    assert len(big) == 1, "expected a single dominant region after cleanup"

    only_small = remove_small_objects(test, min_size=10)
    print(f"remove_small_objects -> OK ({fg(only_small)} px kept)")

    print("\nMask cleanup ran successfully.")
