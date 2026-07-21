"""
02_kmeans_clustering.py
======================

Module 06 - Image Segmentation · unsupervised k-means clustering.

Groups pixels into `k` clusters by colour/intensity similarity — a common way
to derive rough land-cover classes (water, vegetation, soil, built-up ...)
without labelled training data. Works on RGB or grayscale input.

The algorithm (Lloyd's) is implemented from scratch with a k-means++ seeding
for stable, well-separated clusters.

Public function
---------------
    kmeans_segment(image, k=4, max_iter=100, tol=1e-4, seed=0,
                   return_labels=False) -> Image | (Image, labels, centroids)

By default returns an image where every pixel is repainted with its cluster's
mean colour (a posterised land-cover map).

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _features(image: Image.Image):
    """Return (feature_matrix (N, C), (H, W), mode)."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float64)
    h, w = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    return arr.reshape(-1, channels), (h, w), image.mode


def _kmeans_pp_init(x: np.ndarray, k: int, rng) -> np.ndarray:
    """k-means++ seeding: spread initial centroids apart."""
    n = x.shape[0]
    centroids = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            [np.sum((x - c) ** 2, axis=1) for c in centroids], axis=0
        )
        total = d2.sum()
        if total == 0:                              # all remaining points identical
            centroids.append(x[rng.integers(n)])
            continue
        probs = d2 / total
        centroids.append(x[rng.choice(n, p=probs)])
    return np.array(centroids)


def kmeans_segment(
    image: Image.Image,
    k: int = 4,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    return_labels: bool = False,
):
    """
    Segment `image` into `k` colour clusters.

    return_labels=True -> (segmented_image, labels(H,W), centroids(k,C)).
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    x, (h, w), mode = _features(image)
    rng = np.random.default_rng(seed)

    centroids = _kmeans_pp_init(x, k, rng)
    labels = np.zeros(x.shape[0], dtype=np.int32)

    for _ in range(max_iter):
        # assignment: nearest centroid (squared Euclidean distance)
        dists = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)

        # update: mean of assigned points (keep old centroid if a cluster empties)
        new_centroids = np.array([
            x[new_labels == c].mean(axis=0) if np.any(new_labels == c)
            else centroids[c]
            for c in range(k)
        ])

        shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)).max()
        centroids, labels = new_centroids, new_labels
        if shift <= tol:
            break

    # repaint each pixel with its cluster mean colour
    segmented = centroids[labels].reshape(
        (h, w) if mode == "L" else (h, w, centroids.shape[1])
    )
    seg_img = Image.fromarray(np.clip(segmented, 0, 255).astype(np.uint8), mode=mode)

    if return_labels:
        return seg_img, labels.reshape(h, w), centroids
    return seg_img


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Synthetic 3-class colour scene (e.g. water / vegetation / soil).
    img = np.zeros((60, 90, 3), dtype=np.float64)
    img[:, 0:30] = [30, 60, 140]      # bluish
    img[:, 30:60] = [40, 130, 50]     # greenish
    img[:, 60:90] = [150, 110, 60]    # brownish
    img += rng.normal(0, 8, img.shape)
    test = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")

    seg, labels, centroids = kmeans_segment(test, k=3, seed=1, return_labels=True)
    assert seg.size == test.size and seg.mode == "RGB"
    n_clusters = len(np.unique(labels))
    print(f"clusters found: {n_clusters}")
    print("centroid colours (approx):")
    for c in np.round(centroids).astype(int):
        print(f"   {tuple(int(v) for v in c)}")
    assert n_clusters == 3

    gray = test.convert("L")
    seg_g = kmeans_segment(gray, k=3, seed=1)
    assert seg_g.mode == "L"
    print("grayscale k-means -> OK")

    print("\nK-means clustering ran successfully.")
