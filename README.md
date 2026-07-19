# Remote Sensing Image Processing

A hands-on, module-by-module portfolio of image-processing techniques applied to
[Sentinel-2](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)
satellite imagery — built to demonstrate practical geospatial and remote-sensing
skills in Python, from raster fundamentals up to full analysis workflows.

Each module is a self-contained folder of small, well-documented scripts. Shared
logic lives in a reusable `utils/` package, so the codebase stays clean as it
grows. Built with `rasterio`, `numpy`, and `matplotlib`.

## What this project demonstrates

- **Geospatial data handling** — reading and writing georeferenced rasters
  (GeoTIFF), preserving CRS and affine transforms, and interpreting metadata.
- **Remote-sensing fundamentals** — multi-band imagery, spectral bands, true-colour
  composites, and spectral indices such as NDVI.
- **Image processing** — contrast enhancement, filtering, edge detection,
  morphology, and segmentation (progressively added across modules).
- **Clean software practices** — modular structure, a shared utilities package to
  avoid code duplication, per-module documentation, and reproducible outputs.

## Progress

| # | Module | Status |
| --- | --- | --- |
| 01 | [Image Basics](modules/01_image_basics) | ✅ Done |
| 02 | [Image Enhancement](modules/02_image_enhancement) | 🚧 Planned |
| 03 | [Image Filtering](modules/03_image_filtering) | 🚧 Planned |
| 04 | [Edge Detection](modules/04_edge_detection) | 🚧 Planned |
| 05 | [Morphological Operations](modules/05_morphological_operations) | 🚧 Planned |
| 06 | [Image Segmentation](modules/06_image_segmentation) | 🚧 Planned |
| 07 | [Feature Extraction](modules/07_feature_extraction) | 🚧 Planned |
| 08 | [Remote Sensing Applications](modules/08_remote_sensing_applications) | 🚧 Planned |

## Repository structure

```
remote-sensing-image-processing/
├── data/
│   └── sample/sentinel2.tif      # sample Sentinel-2 scene (multi-band GeoTIFF)
├── utils/
│   ├── __init__.py
│   └── raster_io.py              # shared helpers: read, stretch, save PNG/GeoTIFF
├── modules/
│   ├── 01_image_basics/          # each module is a folder of scripts + a README
│   ├── 02_image_enhancement/
│   ├── 03_image_filtering/
│   ├── 04_edge_detection/
│   ├── 05_morphological_operations/
│   ├── 06_image_segmentation/
│   ├── 07_feature_extraction/
│   └── 08_remote_sensing_applications/
├── outputs/                      # generated images/rasters (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`rasterio` bundles its own GDAL via pip wheels, so no separate GDAL install is
needed on most systems.

## Usage

Run any script directly; each takes an optional image path and otherwise falls
back to the sample scene. Generated files are written to `outputs/`.

```bash
python modules/01_image_basics/03_display_rgb_image.py
python modules/01_image_basics/06_save_processed_image.py path/to/your.tif
```

## How it's built

The repo is organised so that adding a new technique is always the same small,
predictable step:

- **Modular layout.** Every topic lives in its own numbered folder under
  `modules/`, each with a short README explaining the concept and listing its
  scripts. Numbering keeps the learning path in order and easy to browse.
- **Shared utilities.** Reusable logic — opening rasters, percentile contrast
  stretching, and saving PNGs and georeferenced GeoTIFFs — lives once in
  `utils/raster_io.py`. Scripts import from it (`from utils import ...`) instead
  of repeating boilerplate, which keeps each script short and focused on the
  technique it demonstrates.
- **Reproducible outputs.** Scripts read from a sample scene by default and write
  results to `outputs/`, which is git-ignored so generated files never clutter
  the repository.
- **Consistent script pattern.** Every script accepts an optional image path,
  documents its assumptions (e.g. band ordering), and can be run standalone.

To add a module: create the next numbered folder, drop in scripts that import
from `utils`, write a short module README, and flip its row in the progress
table above.

## A note on band ordering

Sentinel-2 stacks don't have a universal band order. Scripts that need specific
bands declare them as constants at the top (e.g. `RED_BAND`, `NIR_BAND`); adjust
these to match your file. Run `modules/01_image_basics/02_image_metadata.py`
first to confirm your band count and order.

## Tech stack

Python · rasterio · NumPy · matplotlib

## License

MIT
