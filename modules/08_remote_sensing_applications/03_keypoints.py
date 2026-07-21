"""
03_keypoints.py
==============

Module 07 - Feature Extraction · corner & blob keypoints.

Two detectors, both implemented from scratch:

    harris_corners  — the Harris corner response R = det(M) - k*trace(M)^2 from
                      the structure tensor M of image gradients, followed by
                      thresholding and non-maximum suppression.

    orb_keypoints   — the detection + description stages behind ORB:
                      FAST corner detection, orientation by intensity centroid,
                      and a steered (rotation-aware) BRIEF binary descriptor.
                      This mirrors ORB's design; production ORB adds a scale
                      pyramid and Harris re-ranking, omitted here for clarity.

    fast_corners    — the FAST detector on its own.
    draw_keypoints  — overlay detected points on the image for inspection.

Keypoints are returned as a list of (row, col) tuples (with an angle for ORB).

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw


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


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _local_maxima(response: np.ndarray, size: int, threshold: float) -> list:
    """Return (row, col) of pixels that are the strict max in a size x size window
    and exceed `threshold`."""
    pad = size // 2
    padded = np.pad(response, pad, mode="constant", constant_values=-np.inf)
    win = sliding_window_view(padded, (size, size))
    local_max = win.max(axis=(-2, -1))
    keep = (response == local_max) & (response > threshold)
    ys, xs = np.where(keep)
    return list(zip(ys.tolist(), xs.tolist()))


# --------------------------------------------------------------------------- #
# Harris
# --------------------------------------------------------------------------- #
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def harris_corners(image: Image.Image, k: float = 0.04,
                   threshold_ratio: float = 0.01, sigma: float = 1.0,
                   nms_size: int = 5) -> list:
    """
    Detect Harris corners.

    k               : sensitivity constant (0.04-0.06 typical).
    threshold_ratio : keep responses above this fraction of the max response.
    sigma           : Gaussian window for summing the structure tensor.
    nms_size        : non-maximum-suppression neighbourhood.
    Returns a list of (row, col) corner locations.
    """
    gray = _to_gray_array(image)
    ix = _convolve(gray, _SOBEL_X)
    iy = _convolve(gray, _SOBEL_Y)

    win = _gaussian_kernel(max(3, int(sigma * 3) | 1), sigma)
    sxx = _convolve(ix * ix, win)
    syy = _convolve(iy * iy, win)
    sxy = _convolve(ix * iy, win)

    det = sxx * syy - sxy ** 2
    trace = sxx + syy
    response = det - k * trace ** 2

    peak = response.max()
    if peak <= 0:
        return []
    return _local_maxima(response, nms_size, threshold_ratio * peak)


# --------------------------------------------------------------------------- #
# FAST  (the ORB detector)
# --------------------------------------------------------------------------- #
# 16-pixel Bresenham circle of radius 3, ordered clockwise as (dy, dx).
_FAST_CIRCLE = [
    (-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3), (2, 2), (3, 1),
    (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1),
]


def fast_corners(image: Image.Image, threshold: int = 20, n: int = 9,
                 nms_size: int = 5) -> list:
    """
    FAST-n corner detection with non-maximum suppression on the corner score.

    A pixel is a corner if at least `n` contiguous pixels on the radius-3 circle
    are all brighter than Ip+threshold or all darker than Ip-threshold.
    """
    gray = _to_gray_array(image)
    h, w = gray.shape
    if h < 7 or w < 7:
        return []

    ip = gray[3:h - 3, 3:w - 3]
    stacks = np.stack(
        [gray[3 + dy:h - 3 + dy, 3 + dx:w - 3 + dx] for dy, dx in _FAST_CIRCLE],
        axis=0,
    )
    brighter = stacks >= (ip + threshold)
    darker = stacks <= (ip - threshold)

    def max_run(mask):
        # longest contiguous True run around the circle, handling wrap-around
        doubled = np.concatenate([mask, mask], axis=0)
        run = np.zeros_like(ip)
        best = np.zeros_like(ip)
        for i in range(doubled.shape[0]):
            run = np.where(doubled[i], run + 1, 0)
            best = np.maximum(best, run)
        return best

    corner = (max_run(brighter) >= n) | (max_run(darker) >= n)
    score_inner = np.where(corner, np.abs(stacks - ip).sum(axis=0), 0.0)

    score = np.zeros((h, w), dtype=np.float64)
    score[3:h - 3, 3:w - 3] = score_inner
    return _local_maxima(score, nms_size, 0.0)


# --------------------------------------------------------------------------- #
# Orientation + steered BRIEF  (the ORB descriptor)
# --------------------------------------------------------------------------- #
def _orientation(gray, y, x, radius=3):
    """Intensity-centroid angle (radians) of the patch around a keypoint."""
    ys = np.arange(-radius, radius + 1).reshape(-1, 1)
    xs = np.arange(-radius, radius + 1).reshape(1, -1)
    patch = gray[y - radius:y + radius + 1, x - radius:x + radius + 1]
    if patch.shape != (2 * radius + 1, 2 * radius + 1):
        return 0.0
    m01 = np.sum(ys * patch)
    m10 = np.sum(xs * patch)
    return float(np.arctan2(m01, m10))


def _brief_pattern(n_bits=256, patch=15, seed=7):
    """Fixed random point pairs within a patch (Gaussian-distributed)."""
    rng = np.random.default_rng(seed)
    spread = patch / 5.0
    p = np.clip(rng.normal(0, spread, (n_bits, 2)), -patch // 2, patch // 2)
    q = np.clip(rng.normal(0, spread, (n_bits, 2)), -patch // 2, patch // 2)
    return p, q


def orb_keypoints(image: Image.Image, threshold: int = 20, n_bits: int = 256,
                  patch: int = 15, max_keypoints: int | None = None):
    """
    ORB-style keypoints: FAST detection + orientation + steered BRIEF.

    Returns (keypoints, descriptors) where keypoints is a list of
    (row, col, angle_radians) and descriptors is an (N, n_bits) uint8 bit array.
    """
    gray = _to_gray_array(image)
    h, w = gray.shape
    corners = fast_corners(image, threshold=threshold)

    p, q = _brief_pattern(n_bits, patch)
    half = patch // 2
    keypoints, descriptors = [], []

    for (y, x) in corners:
        if not (half <= y < h - half and half <= x < w - half):
            continue
        theta = _orientation(gray, y, x, radius=min(3, half))
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # steer the sampling pattern by the keypoint orientation
        def sample(points):
            ry = points[:, 0] * cos_t - points[:, 1] * sin_t
            rx = points[:, 0] * sin_t + points[:, 1] * cos_t
            yy = np.clip((y + ry).round().astype(int), 0, h - 1)
            xx = np.clip((x + rx).round().astype(int), 0, w - 1)
            return gray[yy, xx]

        bits = (sample(p) < sample(q)).astype(np.uint8)
        keypoints.append((y, x, theta))
        descriptors.append(bits)

    if max_keypoints is not None and len(keypoints) > max_keypoints:
        keypoints = keypoints[:max_keypoints]
        descriptors = descriptors[:max_keypoints]

    desc = np.array(descriptors, dtype=np.uint8) if descriptors \
        else np.empty((0, n_bits), dtype=np.uint8)
    return keypoints, desc


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def draw_keypoints(image: Image.Image, keypoints, radius: int = 3,
                   colour=(255, 0, 0)) -> Image.Image:
    """Draw small circles at each keypoint (row, col[, angle])."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for kp in keypoints:
        y, x = kp[0], kp[1]
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     outline=colour, width=1)
        if len(kp) >= 3:                              # draw orientation ray
            theta = kp[2]
            draw.line([x, y,
                       x + int(2 * radius * np.cos(theta)),
                       y + int(2 * radius * np.sin(theta))], fill=colour, width=1)
    return canvas


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # A grid of bright squares on a dark field: convex 90-degree corners are
    # exactly what Harris and FAST are built to detect.
    canvas = np.full((100, 100), 30, dtype=np.uint8)
    for cy in (20, 50, 80):
        for cx in (20, 50, 80):
            canvas[cy - 6:cy + 6, cx - 6:cx + 6] = 220
    test = Image.fromarray(canvas, mode="L")

    harris = harris_corners(test, threshold_ratio=0.05)
    print(f"Harris corners found: {len(harris)}")
    assert len(harris) > 10, "square grid should yield many corners"

    fast = fast_corners(test, threshold=30)
    print(f"FAST corners found  : {len(fast)}")
    assert len(fast) > 10

    kps, desc = orb_keypoints(test, threshold=30)
    print(f"ORB keypoints       : {len(kps)}, descriptor shape: {desc.shape}")
    assert desc.shape[0] == len(kps)
    if len(kps):
        assert desc.shape[1] == 256

    overlay = draw_keypoints(test, kps)
    assert overlay.mode == "RGB" and overlay.size == test.size
    print("keypoint overlay -> OK")

    print("\nKeypoint detection ran successfully.")
