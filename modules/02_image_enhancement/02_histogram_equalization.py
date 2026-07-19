"""
02_histogram_equalization.py
----------------------------
Histogram equalization redistributes pixel intensities so they spread more
evenly across the range, boosting contrast in a data-driven way.

Two variants are compared:
  * Global equalization      — one mapping applied to the whole image.
  * CLAHE (adaptive)         — Contrast Limited Adaptive Histogram Equalization,
    which equalizes in local tiles and limits over-amplification of noise.
    Usually preserves local detail much better than global equalization.

Requires scikit-image (see requirements.txt).

Usage:
    python 02_histogram_equalization.py [path/to/image.tif]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import DEFAULT_IMAGE, OUTPUTS_DIR, read_band, percentile_stretch  # noqa: E402

BAND = 4  # 1-based; NIR in the default stack


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Normalize to 0-1 first; skimage equalization expects float in [0, 1].
    base = percentile_stretch(read_band(path, BAND), 2, 98)

    global_eq = exposure.equalize_hist(base)
    clahe = exposure.equalize_adapthist(base, clip_limit=0.03)

    images = [base, global_eq, clahe]
    titles = ["Percentile-stretched", "Global equalization", "CLAHE (adaptive)"]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, (img, title) in enumerate(zip(images, titles)):
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(title)
        axes[0, col].axis("off")
        axes[1, col].hist(img.ravel(), bins=100, color="steelblue")
        axes[1, col].set_xlabel("Pixel value")
        axes[1, col].set_ylabel("Count")

    fig.suptitle(f"Histogram equalization (band {BAND})")
    fig.tight_layout()
    out = OUTPUTS_DIR / "enhancement_histogram_equalization.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison -> {out}")


if __name__ == "__main__":
    main()
