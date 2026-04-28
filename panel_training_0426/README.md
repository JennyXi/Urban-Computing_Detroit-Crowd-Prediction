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

### Step 1. Build panel CSV (Dataset_Custom style)

This creates a training table and a manifest:

```powershell
cd "E:\Urban Computing Final Project\Try_0412"
.\.venv\Scripts\python.exe .\panel_training_0426\build_panel_weekly_dataset.py --top-k 100 --date-start 2024-01-01 --date-end 2025-12-31
```

Outputs:
- `panel_training_0426/outputs/panel_weekly_top100_2024_2025_topk2024_city_lag1_log1p.csv`
- `panel_training_0426/outputs/panel_weekly_top100_manifest_topk2024_city_lag1_log1p.csv`

### Step 2. Train one shared Autoformer

By default training runs the full `--epochs` with **no early stopping**. Pass **`--early-stop`** if you want to stop after validation loss fails to improve for **`--patience`** epochs.

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\train_panel_autoformer.py --autoformer-root "E:\Urban Computing Final Project\autoformer_spatial_0425\Autoformer"
```

Default training loss is **Huber** on the scaled OT head (`--huber-delta 1.0`). Checkpoints go under a folder suffix like `..._log1p_huber1`. For a plain MSE baseline, pass **`--loss mse`** (folder name `..._log1p` only).

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\train_panel_autoformer.py --autoformer-root "<Autoformer_repo>" --loss mse
```

This writes:
- `panel_training_0426/checkpoints/<setting>/checkpoint.pth`

### Step 3. Export predictions (2025) and validate

Export rolling forecasts or strict test:

```powershell
.\.venv\Scripts\python.exe .\panel_training_0426\export_panel_predictions.py --scope test --target-year 2025
python .\use_official_autoformer_grid\validate_grid_predictions.py --pred-by-date panel_training_0426/outputs/panel_pred_test_2025_by_date.csv
```

