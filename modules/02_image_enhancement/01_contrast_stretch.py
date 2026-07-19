"""
01_contrast_stretch.py
----------------------
Contrast stretching remaps pixel values so they span the full display range,
making dark, low-contrast satellite imagery easier to interpret.

Two common approaches are compared:
  * Linear (min-max) stretch  — rescales the true min/max to 0-1. Simple, but
    a few extreme pixels can dominate and leave the image flat.
  * Percentile (2-98%) stretch — clips the darkest/brightest 2% first, which
    usually gives much better visual contrast.

Usage:
    python 01_contrast_stretch.py [path/to/image.tif]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import DEFAULT_IMAGE, OUTPUTS_DIR, read_band, percentile_stretch  # noqa: E402

# Band to demonstrate on (1-based). Band 4 = NIR in the default stack.
BAND = 4


def linear_stretch(band: np.ndarray) -> np.ndarray:
    """Rescale the full min-max range to 0-1."""
    band = band.astype("float32")
    b_min, b_max = np.nanmin(band), np.nanmax(band)
    return np.clip((band - b_min) / (b_max - b_min + 1e-9), 0, 1)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    band = read_band(path, BAND)
    # Baseline: raw values scaled by a fixed sensor range (no stretch) — this is
    # how the unenhanced image typically looks: dark and low-contrast.
    original = np.clip(band / 10000.0, 0, 1)
    linear = linear_stretch(band)
    perc = percentile_stretch(band, 2, 98)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(
        axes,
        [original, linear, perc],
        ["Original (no stretch)", "Linear stretch", "Percentile 2-98% stretch"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Contrast stretching (band {BAND})")
    fig.tight_layout()
    out = OUTPUTS_DIR / "enhancement_contrast_stretch.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison -> {out}")


if __name__ == "__main__":
    main()
