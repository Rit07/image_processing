"""
image_filtering.py
==================

Module 3 - Image Filtering (Smoothing).

Reusable, self-contained smoothing filters implemented with NumPy and PIL.
Each filter is written from scratch (via sliding windows) rather than calling
PIL's built-ins, so the underlying operation is transparent. All functions
accept and return a PIL.Image.Image, so they slot directly alongside the
earlier modules.

Public functions
----------------
    mean_blur(image, kernel_size=3)               -> Image
    gaussian_blur(image, kernel_size=5, sigma=1.0) -> Image
    median_filter(image, kernel_size=3)           -> Image
    apply_filter(image, method, **kwargs)         -> Image   (dispatcher)

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _to_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a float64 NumPy array (H, W) or (H, W, C)."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    return np.asarray(image, dtype=np.float64)


def _to_image(array: np.ndarray, mode: str) -> Image.Image:
    """Clip to [0, 255], cast to uint8 and rebuild a PIL image."""
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode=mode)


def _validate_kernel_size(kernel_size: int) -> int:
    """Ensure the kernel size is a positive, odd integer."""
    if not isinstance(kernel_size, (int, np.integer)):
        raise TypeError("kernel_size must be an integer.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer (1, 3, 5, ...).")
    return int(kernel_size)


def _pad(channel: np.ndarray, pad: int) -> np.ndarray:
    """Reflect-pad a 2D channel so output keeps the original size (no dark border)."""
    return np.pad(channel, pad, mode="reflect")


def _windows(channel: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Return a view of shape (H, W, kernel_size, kernel_size) holding every
    neighbourhood centred on each pixel of the (padded) channel.
    """
    pad = kernel_size // 2
    padded = _pad(channel, pad)
    return sliding_window_view(padded, (kernel_size, kernel_size))


def _apply_per_channel(array: np.ndarray, fn) -> np.ndarray:
    """
    Apply `fn` (a function mapping a 2D channel -> 2D channel) to a grayscale
    (2D) or multichannel (3D) array, channel by channel.
    """
    if array.ndim == 2:                       # grayscale
        return fn(array)
    # (H, W, C) -> stack results back along the channel axis
    channels = [fn(array[..., c]) for c in range(array.shape[2])]
    return np.stack(channels, axis=-1)


def _gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """Build a normalised 2D Gaussian kernel."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    ax = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()              # normalise so brightness is preserved


def _convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a single 2D channel with a 2D kernel (same-size output)."""
    win = _windows(channel, kernel.shape[0])
    # win: (H, W, k, k) ; kernel: (k, k) -> weighted sum over the window
    return np.einsum("ijkl,kl->ij", win, kernel)


# --------------------------------------------------------------------------- #
# Public filters
# --------------------------------------------------------------------------- #
def mean_blur(image: Image.Image, kernel_size: int = 3) -> Image.Image:
    """
    Box / average blur: replace each pixel with the mean of its neighbourhood.

    Parameters
    ----------
    image : PIL.Image.Image
    kernel_size : int, odd (default 3)  -- larger = stronger blur.
    """
    k = _validate_kernel_size(kernel_size)
    kernel = np.full((k, k), 1.0 / (k * k))
    array = _to_array(image)
    out = _apply_per_channel(array, lambda ch: _convolve_channel(ch, kernel))
    return _to_image(out, image.mode)


def gaussian_blur(image: Image.Image, kernel_size: int = 5, sigma: float = 1.0) -> Image.Image:
    """
    Gaussian blur: weighted average that favours nearby pixels.

    Parameters
    ----------
    image : PIL.Image.Image
    kernel_size : int, odd (default 5)
    sigma : float > 0 (default 1.0)  -- spread of the bell curve; higher = smoother.
    """
    k = _validate_kernel_size(kernel_size)
    kernel = _gaussian_kernel(k, sigma)
    array = _to_array(image)
    out = _apply_per_channel(array, lambda ch: _convolve_channel(ch, kernel))
    return _to_image(out, image.mode)


def median_filter(image: Image.Image, kernel_size: int = 3) -> Image.Image:
    """
    Median filter: replace each pixel with the median of its neighbourhood.
    Excellent for removing salt-and-pepper noise while keeping edges sharp.

    Parameters
    ----------
    image : PIL.Image.Image
    kernel_size : int, odd (default 3)
    """
    k = _validate_kernel_size(kernel_size)

    def _median(ch: np.ndarray) -> np.ndarray:
        win = _windows(ch, k)                 # (H, W, k, k)
        return np.median(win, axis=(-2, -1))

    array = _to_array(image)
    out = _apply_per_channel(array, _median)
    return _to_image(out, image.mode)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
_FILTERS = {
    "mean": mean_blur,
    "box": mean_blur,          # alias
    "gaussian": gaussian_blur,
    "median": median_filter,
}


def apply_filter(image: Image.Image, method: str, **kwargs) -> Image.Image:
    """
    Convenience dispatcher.

    Examples
    --------
    >>> apply_filter(img, "gaussian", kernel_size=7, sigma=2.0)
    >>> apply_filter(img, "median", kernel_size=5)
    """
    key = method.lower().strip()
    if key not in _FILTERS:
        valid = ", ".join(sorted(_FILTERS))
        raise ValueError(f"Unknown method '{method}'. Choose from: {valid}.")
    return _FILTERS[key](image, **kwargs)


# --------------------------------------------------------------------------- #
# Quick self-test / demo (only runs when the file is executed directly)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Build a small noisy test image so the module can be verified stand-alone.
    rng = np.random.default_rng(0)
    base = np.linspace(0, 255, 64 * 64).reshape(64, 64).astype(np.uint8)
    noisy = base.copy()
    mask = rng.random(noisy.shape) < 0.05          # 5% salt-and-pepper noise
    noisy[mask] = rng.choice([0, 255], size=mask.sum())
    test_img = Image.fromarray(noisy, mode="L")

    for name in ("mean", "gaussian", "median"):
        result = apply_filter(test_img, name, kernel_size=3)
        assert result.size == test_img.size, f"{name}: size changed!"
        assert result.mode == test_img.mode, f"{name}: mode changed!"
        print(f"{name:9s} -> OK  (output size {result.size}, mode {result.mode})")

    print("\nAll smoothing filters ran successfully.")
