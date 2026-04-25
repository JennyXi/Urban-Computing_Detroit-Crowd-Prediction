from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Top-100 100m grid polygons to a GeoPackage for QGIS.")
    parser.add_argument(
        "--manifest",
        default="use_official_autoformer_grid/data/grid_weekly_top100_visits/grid_topk_manifest.csv",
        help="Manifest with grid_id,gx,gy,cell_lon,cell_lat,...",
    )
    parser.add_argument(
        "--pred-by-date",
        default="use_official_autoformer_grid/outputs/grid_top100_weekly_visits_all_2025_pred_by_date.csv",
        help="Optional predictions table to join (grid_id,date,y_pred_last,...)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="If provided (YYYY-MM-DD), join predictions for that week into the grid layer.",
    )
    parser.add_argument("--cell-meters", type=float, default=100.0)
    parser.add_argument("--epsg-grid", type=int, default=32617, help="Projected CRS used for gx/gy grid indexing.")
    parser.add_argument("--out", default="use_official_autoformer_grid/outputs/grid_top100_100m.gpkg")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    pred_path = (repo_root / args.pred_by_date).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    m = pd.read_csv(manifest_path)
    need = {"grid_id", "gx", "gy", "cell_lon", "cell_lat"}
    missing = need - set(m.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")

    cell = float(args.cell_meters)

    # gx/gy are cell indices in a projected CRS (UTM 17N by default).
    # Build polygons in that CRS: [gx*cell, (gx+1)*cell] x [gy*cell, (gy+1)*cell]
    geoms = [
        box(gx * cell, gy * cell, (gx + 1) * cell, (gy + 1) * cell)
        for gx, gy in zip(m["gx"].astype(int).to_list(), m["gy"].astype(int).to_list())
    ]

    gdf = gpd.GeoDataFrame(m.copy(), geometry=geoms, crs=f"EPSG:{int(args.epsg_grid)}")

    if args.date is not None:
        if not pred_path.exists():
            raise SystemExit(f"Missing predictions table: {pred_path}")
        p = pd.read_csv(pred_path, parse_dates=["date"])
        dt = pd.Timestamp(args.date).normalize()
        p = p[p["date"].dt.normalize() == dt].copy()
        if p.empty:
            raise SystemExit(f"No prediction rows for date={dt.date()} in {pred_path}")
        # Keep only useful columns for mapping
        keep = [c for c in ["grid_id", "y_true", "y_pred_last", "y_pred_mean", "n_preds"] if c in p.columns]
        gdf = gdf.merge(p[keep], on="grid_id", how="left")
        gdf["pred_date"] = dt.strftime("%Y-%m-%d")

    # Write layer to GeoPackage (pyogrio engine via geopandas)
    layer = "grid_top100_100m"
    gdf.to_file(out_path, layer=layer, driver="GPKG")

    print(f"Wrote: {out_path}")
    print(f"Layer: {layer}  rows={len(gdf)}  crs={gdf.crs}")
    if args.date is not None:
        print(f"Joined predictions for week_start={args.date} from {pred_path.name}")


if __name__ == "__main__":
    main()

