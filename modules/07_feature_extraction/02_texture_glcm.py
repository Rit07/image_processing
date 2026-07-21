"""
02_texture_glcm.py
=================

Module 07 - Feature Extraction · GLCM texture.

The Gray-Level Co-occurrence Matrix (GLCM) counts how often pairs of gray
levels occur at a fixed offset (distance + direction). Haralick statistics
derived from it summarise texture — smooth vs coarse, regular vs random — which
is a strong discriminator for land cover that looks similar in colour alone
(e.g. grassland vs forest canopy).

Pixels are first quantised to `levels` gray levels to keep the matrix small.

Public functions
----------------
    compute_glcm(image, distance=1, angle=0, levels=8,
                 symmetric=True, normed=True)        -> (levels, levels) array
    contrast / dissimilarity / homogeneity / energy /
    correlation / entropy (glcm)                      -> float
    glcm_features(image, distance=1, angle=0, levels=8) -> dict
    texture_map(image, feature='contrast', window=15, levels=8,
                distance=1, angle=0)                  -> PIL.Image

Note: texture_map slides a window over the image and builds a GLCM per window,
so it is O(H*W*window^2) — use modest windows / image sizes.

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_ANGLE_OFFSETS = {
    0: (0, 1),      # horizontal, neighbour to the right
    45: (-1, 1),    # up-right
    90: (-1, 0),    # vertical, neighbour above
    135: (-1, -1),  # up-left
}


def _to_gray_array(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8)


def _quantise(gray: np.ndarray, levels: int) -> np.ndarray:
    """Map 0..255 down to 0..levels-1."""
    q = (gray.astype(np.int64) * levels) // 256
    return np.clip(q, 0, levels - 1)


def _glcm_from_quantised(q: np.ndarray, offset, levels, symmetric, normed):
    di, dj = offset
    h, w = q.shape
    # overlapping region of reference pixels and their offset neighbours
    r0, r1 = max(0, -di), h - max(0, di)
    c0, c1 = max(0, -dj), w - max(0, dj)
    ref = q[r0:r1, c0:c1]
    nbr = q[r0 + di:r1 + di, c0 + dj:c1 + dj]

    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (ref.ravel(), nbr.ravel()), 1.0)

    if symmetric:
        glcm += glcm.T
    if normed:
        total = glcm.sum()
        if total > 0:
            glcm /= total
    return glcm


def compute_glcm(image: Image.Image, distance: int = 1, angle: int = 0,
                 levels: int = 8, symmetric: bool = True,
                 normed: bool = True) -> np.ndarray:
    """Build the gray-level co-occurrence matrix for one offset."""
    if angle not in _ANGLE_OFFSETS:
        raise ValueError("angle must be one of 0, 45, 90, 135 degrees.")
    if distance < 1:
        raise ValueError("distance must be >= 1.")
    di, dj = _ANGLE_OFFSETS[angle]
    offset = (di * distance, dj * distance)
    q = _quantise(_to_gray_array(image), levels)
    return _glcm_from_quantised(q, offset, levels, symmetric, normed)


# --------------------------------------------------------------------------- #
# Haralick features
# --------------------------------------------------------------------------- #
def _ij(levels):
    i = np.arange(levels).reshape(-1, 1)
    j = np.arange(levels).reshape(1, -1)
    return i, j


def contrast(glcm: np.ndarray) -> float:
    i, j = _ij(glcm.shape[0])
    return float(np.sum(glcm * (i - j) ** 2))


def dissimilarity(glcm: np.ndarray) -> float:
    i, j = _ij(glcm.shape[0])
    return float(np.sum(glcm * np.abs(i - j)))


def homogeneity(glcm: np.ndarray) -> float:
    """Inverse difference moment: high when pairs are close in gray level."""
    i, j = _ij(glcm.shape[0])
    return float(np.sum(glcm / (1.0 + (i - j) ** 2)))


def energy(glcm: np.ndarray) -> float:
    """Angular second moment's root: high for uniform/orderly texture."""
    return float(np.sqrt(np.sum(glcm ** 2)))


def entropy(glcm: np.ndarray) -> float:
    nz = glcm[glcm > 0]
    return float(-np.sum(nz * np.log2(nz)))


def correlation(glcm: np.ndarray) -> float:
    levels = glcm.shape[0]
    i, j = _ij(levels)
    mu_i = np.sum(i * glcm)
    mu_j = np.sum(j * glcm)
    sigma_i = np.sqrt(np.sum(glcm * (i - mu_i) ** 2))
    sigma_j = np.sqrt(np.sum(glcm * (j - mu_j) ** 2))
    if sigma_i == 0 or sigma_j == 0:
        return 1.0          # perfectly uniform image -> defined as fully correlated
    return float(np.sum(glcm * (i - mu_i) * (j - mu_j)) / (sigma_i * sigma_j))


_FEATURES = {
    "contrast": contrast,
    "dissimilarity": dissimilarity,
    "homogeneity": homogeneity,
    "energy": energy,
    "entropy": entropy,
    "correlation": correlation,
}


def glcm_features(image: Image.Image, distance: int = 1, angle: int = 0,
                  levels: int = 8) -> dict:
    """Return all texture measures for the image's GLCM as a dict."""
    glcm = compute_glcm(image, distance, angle, levels)
    return {name: fn(glcm) for name, fn in _FEATURES.items()}


# --------------------------------------------------------------------------- #
# Windowed texture map
# --------------------------------------------------------------------------- #
def texture_map(image: Image.Image, feature: str = "contrast", window: int = 15,
                levels: int = 8, distance: int = 1, angle: int = 0) -> Image.Image:
    """
    Compute one GLCM feature over a sliding window, producing a per-pixel
    texture image (rescaled to 0-255). Edges use reflect padding.
    """
    if feature not in _FEATURES:
        raise ValueError(f"feature must be one of {sorted(_FEATURES)}.")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3.")

    fn = _FEATURES[feature]
    di, dj = _ANGLE_OFFSETS[angle]
    offset = (di * distance, dj * distance)
    q = _quantise(_to_gray_array(image), levels)
    pad = window // 2
    qp = np.pad(q, pad, mode="reflect")
    h, w = q.shape
    out = np.zeros((h, w), dtype=np.float64)

    for r in range(h):
        for c in range(w):
            patch = qp[r:r + window, c:c + window]
            glcm = _glcm_from_quantised(patch, offset, levels, True, True)
            out[r, c] = fn(glcm)

    peak = out.max()
    scaled = out / peak * 255 if peak > 0 else out
    return Image.fromarray(scaled.astype(np.uint8), mode="L")


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Smooth gradient (low contrast) vs random noise (high contrast).
    smooth = np.tile(np.linspace(0, 255, 64), (64, 1)).astype(np.uint8)
    noisy = rng.integers(0, 256, (64, 64)).astype(np.uint8)
    smooth_img = Image.fromarray(smooth, "L")
    noisy_img = Image.fromarray(noisy, "L")

    fs = glcm_features(smooth_img, levels=8)
    fn = glcm_features(noisy_img, levels=8)
    print("feature        smooth     noisy")
    for key in ("contrast", "homogeneity", "energy", "entropy"):
        print(f"{key:13s} {fs[key]:8.3f}  {fn[key]:8.3f}")

    # Noise should be higher contrast & entropy, lower homogeneity than a ramp.
    assert fn["contrast"] > fs["contrast"]
    assert fs["homogeneity"] > fn["homogeneity"]
    assert fn["entropy"] > fs["entropy"]

    tmap = texture_map(noisy_img, feature="contrast", window=9)
    assert tmap.mode == "L" and tmap.size == noisy_img.size
    print("texture_map -> OK")

    print("\nGLCM texture features ran successfully.")
