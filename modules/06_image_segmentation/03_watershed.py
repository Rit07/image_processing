"""
03_watershed.py
==============

Module 06 - Image Segmentation · watershed.

Treats the image (its gradient, really) as a topographic surface and "floods"
basins outward from marker seeds; where floods from different markers meet, a
watershed line is drawn. This is the classic way to split touching objects that
simple thresholding merges together.

Pipeline (all from scratch):
    1. Gradient surface via Sobel magnitude.
    2. Automatic markers (when none supplied):
         Otsu threshold -> chamfer distance transform -> peak cores as
         foreground seeds (one label each) + background seed.
    3. Meyer's priority-queue flooding grows the seeds over the surface.

Public functions
----------------
    generate_markers(image, fg_ratio=0.5) -> int label array (0 = unknown)
    watershed(image, markers=None, return_labels=False) -> Image | labels

By default returns an RGB visualisation: each region a distinct colour, with
watershed boundaries drawn in red.

Note: the distance transform and flooding use Python-level loops, so very large
images will be slow; it is written for clarity over raw speed.

Dependencies: numpy, Pillow
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

WSHED = -1  # label for watershed boundary pixels


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def _to_gray_array(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image.Image instance.")
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.float64)


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    padded = np.pad(gray, 1, mode="reflect")
    win = sliding_window_view(padded, (3, 3))
    gx = np.einsum("ijkl,kl->ij", win, kx)
    gy = np.einsum("ijkl,kl->ij", win, ky)
    return np.hypot(gx, gy)


def _otsu_mask(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    prob = hist / g.size
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    t = int(np.argmax(np.nan_to_num(sigma_b2)))
    return g > t                                     # class split is <= t vs > t


def _distance_transform(binary: np.ndarray) -> np.ndarray:
    """Two-pass chamfer distance to the nearest background pixel (approx. Euclid)."""
    h, w = binary.shape
    INF = 1e9
    d = np.where(binary, INF, 0.0)
    diag = np.sqrt(2.0)
    # forward pass
    for y in range(h):
        for x in range(w):
            if binary[y, x]:
                best = d[y, x]
                if y > 0:
                    best = min(best, d[y - 1, x] + 1.0)
                    if x > 0:
                        best = min(best, d[y - 1, x - 1] + diag)
                    if x < w - 1:
                        best = min(best, d[y - 1, x + 1] + diag)
                if x > 0:
                    best = min(best, d[y, x - 1] + 1.0)
                d[y, x] = best
    # backward pass
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if binary[y, x]:
                best = d[y, x]
                if y < h - 1:
                    best = min(best, d[y + 1, x] + 1.0)
                    if x < w - 1:
                        best = min(best, d[y + 1, x + 1] + diag)
                    if x > 0:
                        best = min(best, d[y + 1, x - 1] + diag)
                if x < w - 1:
                    best = min(best, d[y, x + 1] + 1.0)
                d[y, x] = best
    return d


def _label_components(binary: np.ndarray, start: int = 1):
    """8-connected labelling. Returns (labels, n_components)."""
    from collections import deque
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    cur = start - 1
    for r in range(h):
        for c in range(w):
            if binary[r, c] and labels[r, c] == 0:
                cur += 1
                q = deque([(r, c)])
                labels[r, c] = cur
                while q:
                    y, x = q.popleft()
                    for dy, dx in nbrs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w \
                                and binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            q.append((ny, nx))
    return labels, cur - (start - 1)


def _dilate(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    h, w = binary.shape
    out = binary.copy()
    for _ in range(iterations):
        p = np.pad(out, 1, constant_values=False)
        grown = np.zeros_like(out)
        for di in range(3):
            for dj in range(3):
                grown |= p[di:di + h, dj:dj + w]
        out = grown
    return out


# --------------------------------------------------------------------------- #
# Marker generation
# --------------------------------------------------------------------------- #
def generate_markers(image: Image.Image, fg_ratio: float = 0.5) -> np.ndarray:
    """
    Build a marker array automatically.

    Label 1 = background; labels >= 2 = distinct object cores; 0 = unknown
    (to be resolved by flooding). `fg_ratio` sets how strict the "sure
    foreground" peak test is (fraction of the max distance).
    """
    gray = _to_gray_array(image)
    fg = _otsu_mask(gray)

    sure_bg = _dilate(fg, iterations=3)              # everything possibly object
    dist = _distance_transform(fg)
    peak = dist.max()
    sure_fg = dist >= (fg_ratio * peak) if peak > 0 else np.zeros_like(fg)
    unknown = sure_bg & ~sure_fg

    core_labels, _ = _label_components(sure_fg, start=1)
    markers = core_labels + 1                        # cores -> >=2, rest -> 1
    markers[unknown] = 0                             # unknown resolved by flood
    return markers.astype(np.int32)


# --------------------------------------------------------------------------- #
# Meyer priority-flooding watershed
# --------------------------------------------------------------------------- #
def _flood(surface: np.ndarray, markers: np.ndarray) -> np.ndarray:
    h, w = surface.shape
    labels = markers.copy()
    in_q = np.zeros((h, w), dtype=bool)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    heap: list = []
    counter = 0

    def push(y, x):
        nonlocal counter
        if not in_q[y, x]:
            heapq.heappush(heap, (surface[y, x], counter, y, x))
            in_q[y, x] = True
            counter += 1

    # seed: unlabeled neighbours of every labelled marker pixel
    ys, xs = np.where(labels > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == 0:
                push(ny, nx)

    while heap:
        _, _, y, x = heapq.heappop(heap)
        found = set()
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                lab = labels[ny, nx]
                if lab > 0:
                    found.add(lab)
        if len(found) == 1:
            labels[y, x] = found.pop()
        elif len(found) > 1:
            labels[y, x] = WSHED
            continue                                 # boundaries don't propagate
        else:
            continue
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == 0:
                push(ny, nx)
    return labels


# --------------------------------------------------------------------------- #
# Colourising
# --------------------------------------------------------------------------- #
def _colourise(labels: np.ndarray) -> Image.Image:
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    unique = [u for u in np.unique(labels) if u > 0]
    palette = {u: rng.integers(40, 256, size=3) for u in unique}
    palette[1] = np.array([50, 50, 50])              # background = dark grey
    for u, colour in palette.items():
        rgb[labels == u] = colour
    rgb[labels == WSHED] = [255, 0, 0]               # boundaries in red
    return Image.fromarray(rgb, mode="RGB")


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def watershed(image: Image.Image, markers: np.ndarray | None = None,
              return_labels: bool = False):
    """
    Marker-based watershed segmentation.

    markers : optional int seed array (0 = unknown, >=1 = seeds). Auto-generated
              when omitted.
    return_labels=True returns the raw int label array instead of an RGB image.
    """
    gray = _to_gray_array(image)
    surface = _sobel_magnitude(gray)
    if markers is None:
        markers = generate_markers(image)
    labels = _flood(surface, markers)
    return labels if return_labels else _colourise(labels)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Two bright blobs joined by a thin waist -> one connected region that
    # simple thresholding cannot split, but watershed can.
    h = w = 100
    yy, xx = np.mgrid[0:h, 0:w]
    blob1 = (yy - 30) ** 2 + (xx - 45) ** 2 <= 17 ** 2
    blob2 = (yy - 62) ** 2 + (xx - 45) ** 2 <= 17 ** 2
    canvas = np.where(blob1 | blob2, 210, 40).astype(np.uint8)
    test = Image.fromarray(canvas, mode="L")

    markers = generate_markers(test)
    cores = len([u for u in np.unique(markers) if u >= 2])
    print(f"object cores detected: {cores}  (expected 2 for the two blobs)")

    labels = watershed(test, return_labels=True)
    regions = [u for u in np.unique(labels) if u >= 2]
    boundary_px = int((labels == WSHED).sum())
    print(f"regions after flooding: {len(regions)}")
    print(f"watershed boundary pixels: {boundary_px}")
    assert len(regions) == 2, "the two blobs should be separated"
    assert boundary_px > 0, "expected a dividing watershed line"

    vis = watershed(test)
    assert vis.mode == "RGB" and vis.size == test.size
    print("colour visualisation -> OK")

    print("\nWatershed segmentation ran successfully.")
