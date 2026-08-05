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
6. Smooths the front-quadrant intensity and polarity magnitude (Savitzky-Golay)
   and classifies each timepoint into **phases** based on the smoothed signal:
   - **polarizing** — sustained increase over ≥`min_phase_length` frames.
   - **stalling** — sustained near-zero change (within `stalling_sigma` SDs)
     over ≥`min_phase_length` frames.
   - **polar** — sustained value above `polarity_threshold` over
     ≥`min_phase_length` frames.
   - **local extrema** — per-frame local maxima/minima of the signal.
   For polarizing/stalling/polar phases, per-phase duration, delta, and slope
   are summarized (count, mean duration, mean/max delta, mean slope); the time
   from the nearest preceding local minimum to the start of each polar phase
   ("ramp-on time") is also computed.
7. Aggregates per-cell summary statistics and produces per-cell diagnostic
   plots plus cross-condition comparison plots.

## Repository structure

```
analyze.py     # CLI entry point — run this
cell.py        # CellSeries / CellTimepoint: core segmentation & measurement logic
calculator.py  # Phase quantification: polarizing/stalling/polar-threshold runs, local extrema, ramp-on timing
smoothing.py   # Savitzky-Golay / spline smoothing helpers used for phase detection
helper.py      # image I/O and geometry/math utilities (centroid, polarity, masks, ...)
plotlib.py     # plotting (per-cell time series, cross-condition comparisons)
proto.py       # Marimo scratch notebook for interactive prototyping — not used by analyze.py
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

### Tunable analysis parameters

The following `CellSeries` constructor parameters control quadrant/ring
geometry and phase-quantification behavior. They currently have no CLI flag —
change the `CellSeries(...)` call in `analyze.py` (or call `CellSeries`
directly from a script) to override them:

| Parameter | Default | Purpose |
|---|---|---|
| `pixel_size` | `1.0` | Physical pixel size (µm/px), used for ring/center mask thickness |
| `outer_ring_thickness` | `None` | Rim thickness for the "Total" measurement mask (peripheral ring vs. whole cell) |
| `quadrant_method` | `"max"` | `"max"` (fixed orientation from peak-polarity timepoint) or `"adaptive"` (per-timepoint orientation) |
| `normalization_area` | `"Total"` | Mask area used to normalize Q1 intensity (`Total`, `Center`, or a quadrant) |
| `polarity_column` | `"Q1_norm_smooth"` | Column used throughout phase detection (polarizing/stalling/polar) |
| `time_resolution` | `1.0` | Time per frame, used to compute the `time` column |
| `polarity_threshold` | `0.275` | Threshold defining a "polar" phase |
| `min_phase_length` | `4` | Minimum consecutive frames to count as a polarizing/stalling/polar phase |
| `stalling_sigma` | `0.5` | Std-dev band (in units of the diff's SD) defining "flat" for stalling detection |
| `savgol_polyorder` | `4` | Polynomial order for Savitzky-Golay smoothing |
| `savgol_window_length` | `None` | Explicit SG window length (auto-derived from series length if `None`) |
| `channel_of_interest` | `None` | Channel index to select from a 4D `(T, C, Y, X)` image/mask |
| `time_clip` | `None` | `(start, end)` timepoint range to clip the movie to |

## Output

Results are written to `output/` (created automatically, alongside `data/`):

- `output/cell_measurements.csv` — one row per cell with aggregate stats:
  polarity magnitude (mean/std/max/min), polarity angle change
  (mean/std/max/min), normalized Q1 intensity (mean/std/max/min), phase
  counts and mean duration/delta/slope for both polarizing and stalling
  phases, mean ramp-on time from local minimum to polar onset, and
  polarity-angle-change stats restricted to non-polar frames.
- `output/cell_data/<uid>_<name>.csv` — full per-timepoint dataframe for one
  cell (raw and smoothed Q1/polarity-magnitude signals, their diffs, and the
  boolean `polarizing`/`stalling`/`polar`/`local_max`/`local_min` phase
  columns).
- `output/cell_measurements.pdf` — cross-condition strip + point comparison
  plots, one panel per feature.
- `output/quadrant_analysis_plots/<uid>__<name>_polarity_time_series.pdf` —
  one per-cell diagnostic figure per cell: annotated timepoint images with
  quadrant overlays, polarity magnitude/angle time series, and the smoothed
  polarity-column time series with polarizing/stalling/polar phases shaded.

## Known limitations

- Pixel size is currently hardcoded in `analyze.py` (`6.5/63` µm/px, i.e. a
  63x objective with a 6.5 µm camera pixel size) rather than a CLI option.
- `channel_of_interest`, `time_clip`, and the phase-quantification parameters
  (`polarity_threshold`, `min_phase_length`, `stalling_sigma`,
  `savgol_polyorder`, `savgol_window_length`, `time_resolution`) are all
  supported by `CellSeries` but not yet exposed via `analyze.py`'s CLI.
- `analyze.py` defines a few module-level constants (`polarity_threshold`,
  `polarity_feature`, `normalize_area`, `backtrack_frame_cutoff`) that are
  currently unused/dead — they are not wired into the `CellSeries(...)` call.
- No dependency manifest (`requirements.txt`/`environment.yml`) or license
  file is included yet.
