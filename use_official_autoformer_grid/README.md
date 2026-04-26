## use_official_autoformer_grid

Grid (100m x 100m) weekly **crowdflow** prediction for Detroit using the **official** Autoformer repo.

### Goal

Predict weekly **visits** (`VISIT_COUNTS` aggregated to grid cells) for the **Top-100** most visited 100m grid cells in 2025.

### Inputs

- `data/grid100_weekly_2025.parquet`
  - columns: `week_start, gx, gy, grid_id, visits, visitors, cell_lon, cell_lat`

### Step 1. Prepare per-grid CSVs (Autoformer `Dataset_Custom`)

This writes 100 small CSVs like `grid_<grid_id>.csv` with columns:
- `date` (week_start)
- `OT` (grid weekly visits)

```powershell
cd "E:\Urban Computing Final Project\Try_0412"
.\.venv\Scripts\python.exe .\use_official_autoformer_grid\prepare_top100_grid_weekly_csvs.py --top-k 100
```

Optional (recommended): apply `log1p` transform to reduce spike impact:

```powershell
.\.venv\Scripts\python.exe .\use_official_autoformer_grid\prepare_top100_grid_weekly_csvs.py --top-k 100 --target-transform log1p --out-dir use_official_autoformer_grid/data/grid_weekly_top100_visits_log1p
```

Outputs:
- `use_official_autoformer_grid/data/grid_weekly_top100_visits/grid_topk_manifest.csv` (grid_id + gx/gy + lon/lat + totals)
- `use_official_autoformer_grid/data/grid_weekly_top100_visits/grid_<grid_id>.csv` (one per grid)

### Step 2. Train Autoformer per grid (Top-100)

```powershell
powershell -ExecutionPolicy Bypass -File .\use_official_autoformer_grid\runs\train_top100_grids_weekly_12to4.ps1
```

If you used a different data root (e.g. log1p directory), set `GRID_DATA_ROOT`:

```powershell
$env:GRID_DATA_ROOT = "E:\Urban Computing Final Project\Try_0412\use_official_autoformer_grid\data\grid_weekly_top100_visits_log1p"
powershell -ExecutionPolicy Bypass -File .\use_official_autoformer_grid\runs\train_top100_grids_weekly_12to4.ps1
```

Notes:
- This runs **one model per grid** (simplest panel strategy).
- Uses `features=S` (univariate) so `enc_in=1`.
- Checkpoints are written under `use_official_autoformer_grid/checkpoints/`.

### Step 3. Export predictions for QGIS (combined CSV)

After training some/all grids, export predictions into **one table**:

```powershell
.\.venv\Scripts\python.exe .\use_official_autoformer_grid\export_grid_predictions_2025.py --target-year 2025
```

If you trained on log1p data, export and invert back to raw visits:

```powershell
.\.venv\Scripts\python.exe .\use_official_autoformer_grid\export_grid_predictions_2025.py --target-year 2025 --grid-data-root use_official_autoformer_grid/data/grid_weekly_top100_visits_log1p --target-transform log1p
```

Outputs:
- `use_official_autoformer_grid/outputs/grid_top100_weekly_visits_all_2025_pred_by_date.csv`
  - one row per (`grid_id`, `date`) with `y_pred_last` and `y_pred_mean`
- `use_official_autoformer_grid/outputs/grid_top100_weekly_visits_all_2025_pred_long.csv`
  - long-form (keeps `horizon` and `window_start`)

### QGIS usage (join)

- Use `grid_topk_manifest.csv` as the **grid location table** (it contains `gx, gy, cell_lon, cell_lat`).
- Use exported predictions (to be added in next step) as a **time table** keyed by `grid_id` + `date`.
- In QGIS you can:
  - load manifest as points (`cell_lon`, `cell_lat`) or build polygons from `gx,gy` if needed
  - join predictions on `grid_id`

### Step 4. Export a GPKG grid layer (100m polygons)

This creates a ready-to-load GeoPackage containing **100m x 100m squares** (CRS = EPSG:32617) with `grid_id` for joins.

```powershell
.\.venv\Scripts\python.exe .\use_official_autoformer_grid\export_grid_top100_gpkg.py
```

Output:
- `use_official_autoformer_grid/outputs/grid_top100_100m.gpkg`

Optional: bake predictions for one week directly into the GPKG layer:

```powershell
.\.venv\Scripts\python.exe .\use_official_autoformer_grid\export_grid_top100_gpkg.py --date 2025-11-03
```

