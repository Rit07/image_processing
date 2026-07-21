"""
01_sobel_prewitt.py
===================

Module 04 - Edge Detection · gradient-based operators.

First-derivative edge detection. Both Sobel and Prewitt estimate the intensity
gradient in the x and y directions using 3x3 kernels; the edge strength at each
pixel is the magnitude of that gradient. Sobel weights the centre row/column
more heavily (better noise handling); Prewitt uses uniform weights.

Public functions
----------------
    sobel(image, threshold=None, return_direction=False)
    prewitt(image, threshold=None, return_direction=False)

Each returns a grayscale PIL.Image.Image of edge magnitude (or a binary edge
map if `threshold` is given). With return_direction=True a (magnitude_image,
direction_array) tuple is returned, direction in degrees.

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _to_gray_array(image: Image.Image) -> np.ndarray:
    """Convert any PIL image to a float64 grayscale (H, W) array."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.float64)


def _convolve(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-size 2D convolution (reflect-padded) via sliding windows."""
    pad = kernel.shape[0] // 2
    padded = np.pad(channel, pad, mode="reflect")
    win = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", win, kernel)


def _normalise(array: np.ndarray) -> np.ndarray:
    """Scale an array so its maximum maps to 255 (0..255 float)."""
    peak = array.max()
    if peak <= 0:
        return np.zeros_like(array)
    return array / peak * 255.0


def _finish(magnitude: np.ndarray, threshold) -> Image.Image:
    """Turn a magnitude array into a grayscale or binary PIL image."""
    mag = _normalise(magnitude)
    if threshold is not None:
        mag = np.where(mag >= threshold, 255.0, 0.0)
    return Image.fromarray(np.clip(mag, 0, 255).astype(np.uint8), mode="L")


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
_PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _gradient_edges(image, kx, ky, threshold, return_direction):
    gray = _to_gray_array(image)
    gx = _convolve(gray, kx)
    gy = _convolve(gray, ky)
    magnitude = np.hypot(gx, gy)                       # sqrt(gx^2 + gy^2)
    edge_img = _finish(magnitude, threshold)
    if return_direction:
        direction = np.rad2deg(np.arctan2(gy, gx))    # -180..180 degrees
        return edge_img, direction
    return edge_img


def sobel(image: Image.Image, threshold=None, return_direction: bool = False):
    """Sobel edge magnitude. `threshold` (0-255) yields a binary edge map."""
    return _gradient_edges(image, _SOBEL_X, _SOBEL_Y, threshold, return_direction)


def prewitt(image: Image.Image, threshold=None, return_direction: bool = False):
    """Prewitt edge magnitude. `threshold` (0-255) yields a binary edge map."""
    return _gradient_edges(image, _PREWITT_X, _PREWITT_Y, threshold, return_direction)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Bright square on a dark field -> strong, predictable edges.
    canvas = np.zeros((80, 80), dtype=np.uint8)
    canvas[20:60, 20:60] = 220
    test = Image.fromarray(canvas, mode="L")

    for name, fn in (("sobel", sobel), ("prewitt", prewitt)):
        mag = fn(test)
        binary = fn(test, threshold=60)
        edges = int((np.asarray(binary) > 0).sum())
        assert mag.size == test.size and mag.mode == "L"
        print(f"{name:8s} -> OK  ({edges} edge pixels above threshold)")

    _, direction = sobel(test, return_direction=True)
    print(f"direction array shape: {direction.shape}")
    print("\nGradient edge operators ran successfully.")
