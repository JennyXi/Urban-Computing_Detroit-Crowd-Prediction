"""
Export grid-level static POI indices to a GIS layer (GPKG or Shapefile).

This creates a polygon grid (e.g., 100m x 100m) aligned to the same (gx, gy) indices
used by `scripts/aggregate_grid_weekly.py`, then attaches POI index attributes.

Why this exists:
- QGIS needs geometry + attribute table.
- Our POI indices are keyed by `grid_id = f"{gx}_{gy}"`.

Implementation notes:
- Geometry is generated in a projected CRS (default EPSG:32617, UTM 17N) and written
  with that CRS. QGIS can reproject on the fly.
- Writing uses `pyogrio` (does NOT require fiona wheels).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
from shapely.geometry import box

try:
    import geopandas as gpd
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing geopandas. Install with: python -m pip install geopandas"
    ) from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Export grid POI static indices as a GPKG/SHP layer for QGIS.")
    parser.add_argument(
        "--grid-weekly",
        default="data/grid100_weekly_2024_2025.parquet",
        help="Grid-weekly parquet (used to enumerate grids and carry gx/gy/grid_id).",
    )
    parser.add_argument(
        "--poi-static",
        default="data/grid100_poi_static_2024.parquet",
        help="Static POI indices parquet produced by scripts/build_grid_poi_static.py",
    )
    parser.add_argument(
        "--output",
        default="data/grid100_poi_static_2024.gpkg",
        help="Output path (.gpkg recommended; .shp also supported).",
    )
    parser.add_argument("--layer", default="grid_poi_static", help="Layer name for GPKG.")
    parser.add_argument("--cell-meters", type=float, default=100.0, help="Grid cell size (must match aggregation).")
    parser.add_argument("--epsg", type=int, default=32617, help="CRS EPSG for the grid (must match aggregation).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    grid_path = (repo_root / str(args.grid_weekly)).resolve()
    poi_path = (repo_root / str(args.poi_static)).resolve()
    out_path = (repo_root / str(args.output)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not grid_path.exists():
        raise SystemExit(f"Missing --grid-weekly parquet: {grid_path}")
    if not poi_path.exists():
        raise SystemExit(f"Missing --poi-static parquet: {poi_path}")

    cell = float(args.cell_meters)
    epsg = int(args.epsg)

    # Enumerate grids from the weekly aggregated parquet (authoritative gx/gy/grid_id set)
    g = pq.read_table(grid_path, columns=["grid_id", "gx", "gy", "cell_lon", "cell_lat"]).to_pandas()
    g["grid_id"] = g["grid_id"].astype(str)
    g["gx"] = pd.to_numeric(g["gx"], errors="coerce").astype("Int64")
    g["gy"] = pd.to_numeric(g["gy"], errors="coerce").astype("Int64")
    g = g.dropna(subset=["gx", "gy"]).copy()
    g["gx"] = g["gx"].astype(int)
    g["gy"] = g["gy"].astype(int)
    grids = g.drop_duplicates(subset=["grid_id", "gx", "gy"])[["grid_id", "gx", "gy", "cell_lon", "cell_lat"]].copy()

    # Load POI static indices
    poi = pq.read_table(poi_path).to_pandas()
    if "grid_id" not in poi.columns:
        raise SystemExit("--poi-static must contain column grid_id.")
    poi["grid_id"] = poi["grid_id"].astype(str)

    # merge (left: all grids from weekly parquet)
    df = grids.merge(poi.drop(columns=[c for c in ["gx", "gy"] if c in poi.columns]), on="grid_id", how="left")

    # fill numeric NaNs with 0
    for c in df.columns:
        if c in {"grid_id"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or c.startswith("poi_"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # build polygon geometry in projected CRS (EPSG:xxxx)
    # cell bounds: [gx*cell, (gx+1)*cell] x [gy*cell, (gy+1)*cell]
    geom = [box(gx * cell, gy * cell, (gx + 1) * cell, (gy + 1) * cell) for gx, gy in zip(df["gx"], df["gy"])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=f"EPSG:{epsg}")

    # Write using pyogrio (no fiona dependency)
    # For SHP output, `layer` is ignored by GDAL driver.
    pyogrio.write_dataframe(gdf, out_path, layer=str(args.layer), overwrite=True)
    print(f"Wrote: {out_path}  grids={len(gdf)}  crs=EPSG:{epsg}  cell_m={cell}")


if __name__ == "__main__":
    main()

