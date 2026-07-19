"""
05_pixel_value_analysis.py
--------------------------
Per-band descriptive statistics and pixel-value histograms.

Usage:
    python 05_pixel_value_analysis.py [path/to/image.tif]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import DEFAULT_IMAGE, OUTPUTS_DIR  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path) as src:
        nodata = src.nodata
        n = src.count
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

        print(f"{'Band':<6}{'min':>10}{'max':>10}{'mean':>12}{'std':>12}"
              f"{'p2':>10}{'p98':>10}")
        print("-" * 70)

        for b in range(1, n + 1):
            band = src.read(b).astype("float32")
            if nodata is not None:
                band = np.where(band == nodata, np.nan, band)
            valid = band[~np.isnan(band)]

            p2, p98 = np.percentile(valid, (2, 98))
            print(f"{b:<6}{valid.min():>10.1f}{valid.max():>10.1f}"
                  f"{valid.mean():>12.2f}{valid.std():>12.2f}{p2:>10.1f}{p98:>10.1f}")

            ax = axes[0][b - 1]
            ax.hist(valid.ravel(), bins=100, color="steelblue")
            ax.set_title(f"Band {b}")
            ax.set_xlabel("Pixel value")
            ax.set_ylabel("Count")

        fig.suptitle("Per-band pixel value distributions")
        fig.tight_layout()
        out_path = OUTPUTS_DIR / "band_histograms.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved histograms -> {out_path}")


if __name__ == "__main__":
    main()
