# 01 · Image Basics ✅

Foundations of working with georeferenced raster data: reading imagery,
inspecting metadata, visualizing bands, analyzing pixel values, and writing a
new georeferenced product.

## Scripts

| Script | What it demonstrates |
| --- | --- |
| `01_read_image.py` | Open a GeoTIFF and read all bands into a NumPy array. |
| `02_image_metadata.py` | CRS, affine transform, resolution, bounds, data types. |
| `03_display_rgb_image.py` | True-colour composite with a 2–98% percentile stretch. |
| `04_extract_individual_bands.py` | Split bands into georeferenced single-band files + previews. |
| `05_pixel_value_analysis.py` | Per-band statistics and histograms. |
| `06_save_processed_image.py` | Compute NDVI and save it as a GeoTIFF preserving CRS/transform. |

## Run

```bash
python modules/01_image_basics/01_read_image.py
```

Each script takes an optional image path and otherwise uses `data/sample/sentinel2.tif`.
Outputs are written to `outputs/`.
