"""
03_canny.py
==========

Module 04 - Edge Detection · the Canny edge detector.

A full from-scratch implementation of the classic multi-stage pipeline:

    1. Grayscale + Gaussian smoothing      (suppress noise)
    2. Intensity gradient via Sobel         (magnitude + direction)
    3. Non-maximum suppression              (thin edges to 1px ridges)
    4. Double threshold                     (classify strong / weak / none)
    5. Hysteresis edge tracking             (keep weak edges linked to strong)

Public function
---------------
    canny(image, low_threshold=50, high_threshold=150,
          sigma=1.0, kernel_size=5) -> PIL.Image.Image   (binary 0/255)

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


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Stage 3 — non-maximum suppression (vectorised)
# --------------------------------------------------------------------------- #
def _non_max_suppression(mag: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    angle = angle_deg % 180                          # gradient direction, 0..180
    p = np.pad(mag, 1, mode="constant", constant_values=0)

    center     = p[1:h + 1, 1:w + 1]
    left       = p[1:h + 1, 0:w]
    right      = p[1:h + 1, 2:w + 2]
    up         = p[0:h,     1:w + 1]
    down       = p[2:h + 2, 1:w + 1]
    up_left    = p[0:h,     0:w]
    up_right   = p[0:h,     2:w + 2]
    down_left  = p[2:h + 2, 0:w]
    down_right = p[2:h + 2, 2:w + 2]

    q = np.zeros_like(mag)   # neighbour "ahead" along the gradient
    r = np.zeros_like(mag)   # neighbour "behind" along the gradient

    m0   = (angle < 22.5) | (angle >= 157.5)          # horizontal gradient
    m45  = (angle >= 22.5) & (angle < 67.5)
    m90  = (angle >= 67.5) & (angle < 112.5)          # vertical gradient
    m135 = (angle >= 112.5) & (angle < 157.5)

    q[m0], r[m0]     = right[m0],     left[m0]
    q[m45], r[m45]   = up_right[m45], down_left[m45]
    q[m90], r[m90]   = up[m90],       down[m90]
    q[m135], r[m135] = up_left[m135], down_right[m135]

    keep = (center >= q) & (center >= r)
    return np.where(keep, mag, 0.0)


# --------------------------------------------------------------------------- #
# Stage 5 — hysteresis
# --------------------------------------------------------------------------- #
def _dilate8(mask: np.ndarray) -> np.ndarray:
    """8-connected binary dilation via OR of the nine shifted copies."""
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for di in range(3):
        for dj in range(3):
            out |= p[di:di + h, dj:dj + w]
    return out


def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Promote weak pixels that are 8-connected to a strong edge, iteratively."""
    edges = strong.copy()
    while True:
        grown = _dilate8(edges) & weak & (~edges)
        if not grown.any():
            break
        edges |= grown
    return np.where(edges, 255.0, 0.0)


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def canny(
    image: Image.Image,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    sigma: float = 1.0,
    kernel_size: int = 5,
) -> Image.Image:
    """
    Canny edge detector.

    low_threshold / high_threshold : hysteresis bounds on gradient magnitude
        (0-255 after the gradient is rescaled). high > low is required.
    sigma, kernel_size : Gaussian pre-smoothing parameters.
    """
    if not (0 <= low_threshold < high_threshold):
        raise ValueError("Require 0 <= low_threshold < high_threshold.")
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd integer.")

    # 1. smooth
    gray = _to_gray_array(image)
    smoothed = _convolve(gray, _gaussian_kernel(kernel_size, sigma))

    # 2. gradient
    gx = _convolve(smoothed, _SOBEL_X)
    gy = _convolve(smoothed, _SOBEL_Y)
    mag = np.hypot(gx, gy)
    peak = mag.max()
    if peak > 0:                                       # rescale to 0-255
        mag = mag / peak * 255.0
    angle = np.rad2deg(np.arctan2(gy, gx))

    # 3. thin
    thin = _non_max_suppression(mag, angle)

    # 4. double threshold
    strong = thin >= high_threshold
    weak = (thin >= low_threshold) & (thin < high_threshold)

    # 5. link
    result = _hysteresis(strong, weak)
    return Image.fromarray(result.astype(np.uint8), mode="L")


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    canvas = np.zeros((100, 100), dtype=np.float64)
    canvas[25:75, 25:75] = 200                         # square
    canvas += rng.normal(0, 12, canvas.shape)          # light noise
    test = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="L")

    edges = canny(test, low_threshold=40, high_threshold=120, sigma=1.4)
    n = int((np.asarray(edges) > 0).sum())
    assert edges.size == test.size and edges.mode == "L"
    print(f"canny -> OK  ({n} edge pixels detected)")
    assert n > 0, "expected to find edges on the square"
    print("\nCanny edge detector ran successfully.")
