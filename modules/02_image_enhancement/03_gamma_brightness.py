"""
03_gamma_brightness.py
----------------------
Gamma correction and brightness/contrast tuning — simple point operations that
adjust how pixel values map to displayed brightness.

  * Gamma correction: output = input ** gamma
        gamma < 1  brightens shadows;  gamma > 1  darkens them.
  * Brightness/contrast: output = contrast * input + brightness

All operations run on a 0-1 normalized band so the parameters are intuitive.

Usage:
    python 03_gamma_brightness.py [path/to/image.tif]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import DEFAULT_IMAGE, OUTPUTS_DIR, read_band, percentile_stretch  # noqa: E402

BAND = 4  # 1-based


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(img ** gamma, 0, 1)


def adjust_brightness_contrast(img: np.ndarray, contrast: float, brightness: float) -> np.ndarray:
    return np.clip(contrast * img + brightness, 0, 1)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    base = percentile_stretch(read_band(path, BAND), 2, 98)

    panels = [
        (base, "Original"),
        (adjust_gamma(base, 0.5), "Gamma 0.5 (brighter)"),
        (adjust_gamma(base, 2.0), "Gamma 2.0 (darker)"),
        (adjust_brightness_contrast(base, 1.0, 0.15), "Brightness +0.15"),
        (adjust_brightness_contrast(base, 1.5, 0.0), "Contrast x1.5"),
        (adjust_brightness_contrast(base, 1.5, 0.1), "Contrast x1.5 + bright"),
    ]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Gamma & brightness/contrast (band {BAND})")
    fig.tight_layout()
    out = OUTPUTS_DIR / "enhancement_gamma_brightness.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison -> {out}")


if __name__ == "__main__":
    main()
