"""
02_laplacian.py
==============

Module 04 - Edge Detection · second-derivative operator.

The Laplacian sums the second derivatives in x and y, responding to rapid
intensity change regardless of direction. Because a bare Laplacian is very
sensitive to noise, this script also offers Laplacian-of-Gaussian (LoG): smooth
first with a Gaussian, then take the Laplacian. Optional zero-crossing detection
locates the precise edge positions where the LoG response changes sign.

Public functions
----------------
    laplacian(image, connectivity=8, threshold=None)
    laplacian_of_gaussian(image, sigma=1.0, kernel_size=5, zero_crossing=False)

Returns a grayscale PIL.Image.Image (binary if thresholded / zero-crossing).

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
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.float64)


def _convolve(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.shape[0] // 2
    padded = np.pad(channel, pad, mode="reflect")
    win = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", win, kernel)


def _gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    ax = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _normalise(array: np.ndarray) -> np.ndarray:
    peak = array.max()
    return array / peak * 255.0 if peak > 0 else np.zeros_like(array)


def _to_L(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
_LAP_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
_LAP_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Zero-crossing detection
# --------------------------------------------------------------------------- #
def _zero_crossings(response: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Mark pixels where the Laplacian response changes sign between opposite
    neighbours (a true edge location). Returns a 0/255 array.
    """
    padded = np.pad(response, 1, mode="reflect")
    h, w = response.shape
    out = np.zeros((h, w), dtype=np.float64)

    # opposite neighbour pairs: (left,right), (up,down), and the two diagonals
    pairs = [
        (padded[1:h + 1, 0:w],     padded[1:h + 1, 2:w + 2]),   # horizontal
        (padded[0:h,     1:w + 1], padded[2:h + 2, 1:w + 1]),   # vertical
        (padded[0:h,     0:w],     padded[2:h + 2, 2:w + 2]),   # main diagonal
        (padded[0:h,     2:w + 2], padded[2:h + 2, 0:w]),       # anti-diagonal
    ]
    for a, b in pairs:
        crossing = (a * b < 0) & (np.abs(a - b) > threshold)
        out[crossing] = 255.0
    return out


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def laplacian(image: Image.Image, connectivity: int = 8, threshold=None) -> Image.Image:
    """
    Plain Laplacian edge response.

    connectivity : 4 (axial neighbours) or 8 (includes diagonals).
    threshold    : if given (0-255), returns a binary edge map.
    """
    if connectivity == 4:
        kernel = _LAP_4
    elif connectivity == 8:
        kernel = _LAP_8
    else:
        raise ValueError("connectivity must be 4 or 8.")

    gray = _to_gray_array(image)
    response = _convolve(gray, kernel)
    magnitude = _normalise(np.abs(response))
    if threshold is not None:
        magnitude = np.where(magnitude >= threshold, 255.0, 0.0)
    return _to_L(magnitude)


def laplacian_of_gaussian(
    image: Image.Image,
    sigma: float = 1.0,
    kernel_size: int = 5,
    zero_crossing: bool = False,
) -> Image.Image:
    """
    Laplacian-of-Gaussian: Gaussian smoothing followed by the Laplacian.

    zero_crossing=True returns a thin 0/255 edge map at sign changes;
    otherwise returns the normalised |LoG| response.
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd integer.")

    gray = _to_gray_array(image)
    smoothed = _convolve(gray, _gaussian_kernel(kernel_size, sigma))
    response = _convolve(smoothed, _LAP_8)

    if zero_crossing:
        return _to_L(_zero_crossings(response))
    return _to_L(_normalise(np.abs(response)))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    canvas = np.zeros((80, 80), dtype=np.uint8)
    canvas[20:60, 20:60] = 220
    test = Image.fromarray(canvas, mode="L")

    for label, img in (
        ("laplacian-4", laplacian(test, connectivity=4)),
        ("laplacian-8", laplacian(test, connectivity=8)),
        ("laplacian-thr", laplacian(test, threshold=40)),
        ("LoG", laplacian_of_gaussian(test, sigma=1.2)),
        ("LoG-zerocross", laplacian_of_gaussian(test, sigma=1.2, zero_crossing=True)),
    ):
        assert img.size == test.size and img.mode == "L"
        n = int((np.asarray(img) > 0).sum())
        print(f"{label:15s} -> OK  ({n} responsive pixels)")

    print("\nSecond-derivative edge detection ran successfully.")
