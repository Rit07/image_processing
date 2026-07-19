# 02 · Image Enhancement ✅

Improving the visual quality and interpretability of imagery by adjusting how
pixel values map to brightness — without changing the underlying geometry.

## Scripts

| Script | What it demonstrates |
| --- | --- |
| `01_contrast_stretch.py` | Linear (min-max) vs percentile (2–98%) contrast stretching. |
| `02_histogram_equalization.py` | Global histogram equalization vs adaptive CLAHE, with histograms. |
| `03_gamma_brightness.py` | Gamma correction and brightness/contrast point operations. |

## Run

```bash
python modules/02_image_enhancement/01_contrast_stretch.py
```

Each script takes an optional image path and otherwise uses
`data/sample/sentinel2.tif`. Comparison figures are written to `outputs/`.

> Note: `02_histogram_equalization.py` requires `scikit-image`
> (already in `requirements.txt`).
