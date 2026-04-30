## panel_training_0426

Panel (multi-series) training for Detroit grid weekly crowdflow using the official Autoformer model code, without the "one-model-per-grid" approach.

This folder builds a **panel dataset** from Top-K grids (default 100) and trains **one shared Autoformer** model across all grids.

### Key idea

- Each training sample comes from **one grid** and one sliding window.
- We add **static grid covariates** (e.g. `gx, gy, cell_lon, cell_lat`, and 2024 summary stats) and optional **global covariates** (e.g. citywide weekly total visits) as extra channels.
- The target is still `OT` (weekly visits). We use `features=MS` (multivariate -> univariate).

### Inputs

- Grid weekly parquet (long table): `data/grid100_weekly_2024_2025.parquet`
  - columns: `week_start, gx, gy, grid_id, visits, visitors, cell_lon, cell_lat`
  - optional columns (if you enable weekend features during aggregation): `weekday_visits, weekend_visits, weekend_share`

### Step 1. Build panel CSV (Dataset_Custom style)

This creates a training table and a manifest:

```powershell
cd "E:\Urban Computing Final Project\Try_0412"
.\.venv\Scripts\python.exe .\panel_training_0426\build_panel_weekly_dataset.py --top-k 100 --date-start 2024-01-01 --date-end 2025-12-31
```

Default build enables a conservative spatial lag feature (**`nbr8_std_visits_lag1`**) and keeps weekend covariates off.

Optional: attach **static POI supply indices** per grid (built from the raw POI-week parquet).
This is useful for studying crowd vs infrastructure alignment (underserved grids).

1) Build grid-level POI indices (recommended: ref-year=2024 for non-leakage):

```powershell
.\.venv\Scripts\python.exe .\scripts\build_grid_poi_static.py --ref-year 2024 --cell-meters 100 --output data\grid100_poi_static_2024.parquet
```

2) Build panel and merge POI indices by `grid_id`:

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\build_panel_weekly_dataset.py --poi-static data\grid100_poi_static_2024.parquet --top-k 100 --date-start 2024-01-01 --date-end 2025-12-31
```

3) (QGIS) Export POI indices as a grid layer (GPKG) with an attribute table:

```powershell
.\.venv\Scripts\python.exe .\scripts\export_grid_poi_static_gpkg.py --grid-weekly data\grid100_weekly_2024_2025.parquet --poi-static data\grid100_poi_static_2024.parquet --output data\grid100_poi_static_2024.gpkg
```

Optional: add **past-only** weekend signal `weekend_share_lag1` as a covariate channel.

1) Rebuild grid-weekly parquet with weekend columns:

```powershell
.\.venv\Scripts\python.exe .\scripts\aggregate_grid_weekly.py --date-start 2024-01-01 --date-end 2025-12-31 --cell-meters 100 --add-weekend-share --output data\grid100_weekly_2024_2025.parquet
```

2) Build panel with lagged weekend covariate:

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\build_panel_weekly_dataset.py --weekend-cov share_lag1 --top-k 100 --date-start 2024-01-01 --date-end 2025-12-31
```

Optional: add **past-only** spatial lag covariates from 8-neighborhood (more robust for local anomalies):

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\build_panel_weekly_dataset.py --spatial-cov nbr8_meanstd_lag1 --top-k 100 --date-start 2024-01-01 --date-end 2025-12-31
```

More conservative option (recommended if you worry about over-smoothing): only use neighbor **std** (lag1):

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\build_panel_weekly_dataset.py --spatial-cov nbr8_std_lag1 --top-k 100 --date-start 2024-01-01 --date-end 2025-12-31
```

Outputs:
- `panel_training_0426/outputs/panel_weekly_top100_2024_2025_topk2024_city_lag1_log1p.csv`
- `panel_training_0426/outputs/panel_weekly_top100_manifest_topk2024_city_lag1_log1p.csv`

### Step 2. Train one shared Autoformer

By default training runs the full `--epochs` with **no early stopping**. Pass **`--early-stop`** if you want to stop after validation loss fails to improve for **`--patience`** epochs.

Split policy:
- default **`--split-mode year`**: train targets are in 2024 (<= `--train-end`), test targets are in 2025 (>= `--test-start`).
- `--split-mode ratio`: legacy 70/10/20 on the mixed 2024–2025 axis.

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\train_panel_autoformer.py --autoformer-root "E:\Urban Computing Final Project\autoformer_spatial_0425\Autoformer"
```

Default training loss is **Huber** on the scaled OT head (`--huber-delta 1.0`). Checkpoints go under a folder suffix like `..._log1p_huber1`. For a plain MSE baseline, pass **`--loss mse`** (folder name `..._log1p` only).

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\train_panel_autoformer.py --autoformer-root "<Autoformer_repo>" --loss mse
```

This writes:
- `panel_training_0426/checkpoints/<setting>/checkpoint.pth`

### Daily variant (Top-K grids, shared model)

If you have daily grid parquet (`data/grid100_daily_2024_2025.parquet`), you can build a daily panel CSV:

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\build_panel_daily_dataset.py --top-k 100 --date-start 2024-01-01 --date-end 2025-12-31
```

Train a daily model with **ratio split** (train/val/test = 0.7/0.15/0.15) using the same Autoformer architecture as weekly:

```powershell
.\.venv\Scripts\python.exe -u .\panel_training_0426\train_panel_autoformer_daily_ratio.py --autoformer-root "E:\Urban Computing Final Project\autoformer_spatial_0425\Autoformer"
```

Defaults (overrideable): `freq=d`, `split-mode=ratio`, `seq_len=84`, `label_len=42`, `pred_len=14`, `lr=5e-5`, `batch_size=24`, `epochs=40`, `early-stop` with `patience=12`, `d_model=192`, `d_ff=768`, `dropout=0.08`, `moving_avg=7`, `weight_decay=1e-4`, `grad_clip_norm=1.0`. Checkpoints save under a folder name that includes `dm…_el…_dl…_ma…` (must match when exporting).

Export **2025 full-year** daily predictions (scope=test by default):

```powershell
.\.venv\Scripts\python.exe -u .\panel_training_0426\export_panel_predictions_daily_ratio.py --autoformer-root "E:\Urban Computing Final Project\autoformer_spatial_0425\Autoformer"
```

### Step 3. Export predictions (2025) and validate

Export rolling forecasts or strict test:

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\export_panel_predictions.py --scope test --target-year 2025
python .\use_official_autoformer_grid\validate_grid_predictions.py --pred-by-date panel_training_0426/outputs/panel_pred_test_2025_by_date.csv
```

