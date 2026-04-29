## POI_Alignment_0429

This folder contains **grid-aligned POI static indices** for QGIS visualization and downstream alignment analysis.

### Files

- `grid100_poi_static_2024.gpkg`
  - 100m x 100m grid polygons (EPSG:32617) aligned to `data/grid100_weekly_2024_2025.parquet`
  - Attribute table includes:
    - `grid_id`, `gx`, `gy`
    - 4D POI supply vector (counts): `poi_cnt_life`, `poi_cnt_transport`, `poi_cnt_economy`, `poi_cnt_public_service`
    - `poi_cnt_total`, shares (`poi_share_*`), and `poi_supercat_entropy`

- `grid100_poi_static_2024.parquet`
  - Same attributes as above (tabular), keyed by `grid_id`.

- `compute_alignment.py`
  - Builds crowd-POI alignment from prediction CSV + static POI parquet.
  - Current default (recommended main run): `--ridge-alpha 0.1 --target-log1p`.
  - Input features are the 4 POI counts (`poi_cnt_life`, `poi_cnt_transport`, `poi_cnt_economy`, `poi_cnt_public_service`), with `log1p` feature transform enabled by default.
  - Writes per-grid `c_bar`, `c_hat`, `r_alignment` and fit metrics JSON (RMSE/MAE/MedAE/SMAPE/R2).

- `summarize_alignment.py`
  - Summarizes alignment outputs and writes top positive/negative mismatch tables.
  - For high-mismatch candidates, exports both:
    - `priority_abs_*`: absolute-count scarcity priority
    - `priority_q_*`: within-high-mismatch-group quantile scarcity priority (**recommended**)

### How it was built (repro)

```powershell
cd "E:\Urban Computing Final Project\Try_0412"

# 1) Build static POI indices (reference year = 2024)
.\.venv\Scripts\python.exe .\scripts\build_grid_poi_static.py --ref-year 2024 --cell-meters 100 --output data\grid100_poi_static_2024.parquet

# 2) Export QGIS layer (GPKG)
.\.venv\Scripts\python.exe .\scripts\export_grid_poi_static_gpkg.py --grid-weekly data\grid100_weekly_2024_2025.parquet --poi-static data\grid100_poi_static_2024.parquet --output data\grid100_poi_static_2024.gpkg
```

### Alignment run (Oct-Dec 2025 example)

```powershell
cd "E:\Urban Computing Final Project\Try_0412"

# Main model (recommended): robust ranking with target log1p + alpha=0.1
.\.venv\Scripts\python.exe .\POI_Alignment_0429\compute_alignment.py

# Summaries + planning priority tables
.\.venv\Scripts\python.exe .\POI_Alignment_0429\summarize_alignment.py
```

### Sensitivity reference

- Main model (recommended for ranking stability):
  - `--target-log1p --ridge-alpha 0.1`
- Robustness check (higher variance explanation baseline):
  - `--no-target-log1p --ridge-alpha 1`

