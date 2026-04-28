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

### How it was built (repro)

```powershell
cd "E:\Urban Computing Final Project\Try_0412"

# 1) Build static POI indices (reference year = 2024)
.\.venv\Scripts\python.exe .\scripts\build_grid_poi_static.py --ref-year 2024 --cell-meters 100 --output data\grid100_poi_static_2024.parquet

# 2) Export QGIS layer (GPKG)
.\.venv\Scripts\python.exe .\scripts\export_grid_poi_static_gpkg.py --grid-weekly data\grid100_weekly_2024_2025.parquet --poi-static data\grid100_poi_static_2024.parquet --output data\grid100_poi_static_2024.gpkg
```

