# Actin polarity quantification pipeline

Analysis pipeline for quantifying the polarized (front/back) distribution of a
fluorescent marker's intensity relative to a segmented cell's shape, across
time-lapse microscopy movies (e.g. a Lifeact/actin-labelled channel).

## What it does

For each cell movie, the pipeline:

1. Reads a paired segmentation mask and intensity image (`.tif` stacks).
2. For every timepoint, computes the geometric centroid and the
   intensity-weighted centroid of the cell mask, and derives a **polarity
   vector** (direction + magnitude) describing where signal is skewed
   relative to the cell's center.
3. Splits the cell into 4 angular **quadrants** (front, back, and two sides)
   oriented along that polarity vector:
   - `max` — a single fixed orientation, taken from the timepoint with the
     largest polarity magnitude across the whole movie.
   - `adaptive` — orientation recomputed per timepoint.
4. Optionally restricts measurement to a membrane **ring** (a rim of given
   thickness) instead of the whole cell interior.
5. Sums marker intensity per quadrant per timepoint, and tracks polarity
   magnitude, frame-to-frame polarity angle change, and the normalized
   front-quadrant (Q1) intensity over time.
6. Aggregates per-cell summary statistics and produces per-cell diagnostic
   plots plus cross-condition comparison plots.

## Repository structure

```
analyze.py     # CLI entry point — run this
cell.py        # CellSeries / CellTimepoint: core segmentation & measurement logic
helper.py      # image I/O and geometry/math utilities (centroid, polarity, masks, ...)
plotlib.py     # plotting (per-cell time series, cross-condition comparisons)
data/          # input: one subfolder per cell (see "Input data" below)
output/        # generated: CSV + PDF results (created automatically)
```

## Requirements

Python 3 with:

- `numpy`
- `scipy`
- `scikit-image`
- `pandas`
- `matplotlib`
- `seaborn`
- `tifffile`

There is currently no `requirements.txt`/`environment.yml` — install these
packages manually (e.g. via `pip install numpy scipy scikit-image pandas
matplotlib seaborn tifffile`).

## Input data

Each cell to analyze needs its own subfolder under `data/` (or under the
directory passed via `--data_dir`). Every subfolder must contain exactly:

- one `.tif` file with `mask` in its filename (case-insensitive) — the binary
  segmentation mask
- one other `.tif` file — the intensity/fluorescence channel

Example:

```
data/
└── cell1/
    ├── Example_Lifeact.tif   # intensity channel
    └── Example_Mask.tif      # segmentation mask
```

Supported image shapes:

- `(T, Y, X)` — used directly.
- `(T, C, Y, X)` — currently requires `channel_of_interest` (and/or
  `time_clip`) to be set in code; these are not yet exposed as CLI arguments.

Cells are assigned a **condition** by case-insensitive substring match of
each entry in `--conditions` against the folder name (e.g. a folder named
`cell1_ctrl` matches condition `ctrl`).

## Usage

```
python analyze.py [--data_dir PATH] [--conditions "cond1,cond2"]
```

- `--data_dir` — path to the directory containing cell folders (default:
  `data/` next to `analyze.py`).
- `--conditions` — comma-separated condition names matched against each cell
  folder's name.

## Output

Results are written to `output/` (created automatically, alongside `data/`):

- `output/cell_measurements.csv` — one row per cell with aggregate stats
  (mean/std/max/min of polarity magnitude, polarity angle change, and
  normalized Q1 intensity).
- `output/cell_measurements.pdf` — cross-condition strip + point comparison
  plots, one panel per feature.
- `output/quadrant_analysis_plots/<uid>__polarity_time_series.pdf` — one
  per-cell diagnostic figure per cell (annotated timepoint images plus
  polarity/intensity time-series plots).

## Known limitations

- Pixel size is currently hardcoded in `analyze.py` (`6.5/63` µm/px, i.e. a
  63x objective with a 6.5 µm camera pixel size) rather than a CLI option.
- `channel_of_interest` and `time_clip` (for selecting a channel or time
  range from 4D `(T, C, Y, X)` stacks) are supported by `CellSeries` but not
  yet exposed via `analyze.py`'s CLI.
- No dependency manifest (`requirements.txt`/`environment.yml`) or license
  file is included yet.
