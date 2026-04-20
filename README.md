# Urban-Computing_Detroit-Crowd-Prediction
CSCI-SHU 205 Topics in Computer Science: Urban Computing Final Project 2026 Spring

## What this repo does

This project builds **Autoformer/Transformer-ready time series datasets** for **Detroit crowd (visit) prediction** from Dewey/SafeGraph-style Weekly Patterns data.

You currently have two main modeling datasets:

- **Weekly prediction dataset**: `data/autoformer_weekly_preprocessed.csv`
  - **1 row = 1 week**
  - **Target**: `OT` (weekly total visits)
  - **Extra channels**: within-week hourly shape `h_0..h_167` + calendar + volatility features
- **Hourly prediction dataset**: `data/autoformer_hourly_preprocessed.csv`
  - **1 row = 1 hour**
  - **Target**: `OT` (hourly total visits)
  - **Extra channels**: hour/day-of-week/month cyclical features + rolling volatility features

## Autoformer input compatibility (does this count as “preprocessing”?)

Yes. The generated CSVs are already in the **typical `Dataset_Custom` layout** used by Autoformer-style repos:

- The **first column is `date`** (timestamp)
- The **last column is `OT`** (prediction target)
- All middle columns are **numeric covariates / channels**

So this workflow **does constitute the data preprocessing step** needed to train Autoformer (you still need to configure training args like `freq`, `target`, and `features` in your Autoformer codebase).

## End-to-end workflow

### Step 0. Setup (recommended)

Create and activate a virtual environment, then install dependencies.

Minimum packages used by the scripts:
- `duckdb`
- `pandas`
- `numpy`
- `pyarrow`
- (optional) `pyproj` for grid aggregation
- (optional) `folium` for map visualization

### Step 1. Filter Michigan (or other) data to Detroit only

This creates the local working Parquet file that all later scripts read.

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.csv.gz" --output data\detroit_filtered.parquet
```

If your source is Parquet shards:

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.parquet" --format parquet --output data\detroit_filtered.parquet
```

Output:
- `data/detroit_filtered.parquet` (large; intentionally ignored by git)

### Step 2. Quick sanity checks (optional but useful)

Confirm the Parquet schema and time coverage:

```bash
python scripts/probe_parquet_sample.py
python scripts/summarize_detroit_time_range.py
python scripts/peek_visits_by_hour.py
```

### Step 3A. Build the weekly Autoformer dataset (weekly prediction)

This is the “full” weekly pipeline (not the simplified one):
- weekly totals (`VISIT_COUNTS`, `VISITOR_COUNTS`) via DuckDB
- within-week hourly profile from `VISITS_BY_EACH_HOUR` expanded to `h_0..h_167`
- calendar + volatility features
- target `OT = weekly visits`

```bash
python scripts/preprocess_weekly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

Output:
- `data/autoformer_weekly_preprocessed.csv`

### Step 3B. Build the hourly Autoformer dataset (hourly prediction)

This expands weekly `VISITS_BY_EACH_HOUR` vectors into a **true hourly timestamp series**:
- **1 row = 1 hour**
- target `OT = hourly visits`

```bash
python scripts/preprocess_hourly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

Output:
- `data/autoformer_hourly_preprocessed.csv`

### Step 4 (optional). Spatial aggregation datasets

CBG-level weekly totals (long table):

```bash
python scripts/aggregate_cbg_weekly.py --date-start 2025-01-01 --date-end 2025-12-31 --output data\cbg_weekly_2025.parquet
```

Regular grid aggregation (default 100m × 100m) per week:

```bash
python scripts/aggregate_grid_weekly.py --cell-meters 100 --date-start 2025-01-01 --date-end 2025-12-31 --output data\grid100_weekly.parquet
```

Map visualization (HTML):

```bash
python scripts/visualize_grid_osm.py --input data\grid100_weekly.parquet --week 2025-01-06 --mode heatmap --output data\grid_map.html
```

## Notes

- **Large files** (e.g. `data/detroit_filtered.parquet`) are intentionally ignored by git.
- The weekly dataset encodes within-week structure as `h_0..h_167` channels; the hourly dataset instead uses **real hourly timestamps**.
