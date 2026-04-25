from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-grid weekly CSVs (date, OT=visits) for Autoformer.")
    parser.add_argument(
        "--input",
        default=r"data/grid100_weekly_2025.parquet",
        help="Input Parquet produced by scripts/aggregate_grid_weekly.py (grid x week).",
    )
    parser.add_argument(
        "--out-dir",
        default=r"use_official_autoformer_grid/data/grid_weekly_top100_visits",
        help="Output directory for per-grid CSVs.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--date-start", default="2025-01-01")
    parser.add_argument("--date-end", default="2025-12-31")
    args = parser.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(inp, columns=["week_start", "grid_id", "visits", "visitors", "cell_lon", "cell_lat", "gx", "gy"])
    df = table.to_pandas()
    df["week_start"] = pd.to_datetime(df["week_start"])
    d0 = pd.Timestamp(args.date_start).normalize()
    d1 = pd.Timestamp(args.date_end).normalize()
    df = df[(df["week_start"].dt.normalize() >= d0) & (df["week_start"].dt.normalize() <= d1)]

    # Rank grids by total visits (2025)
    totals = df.groupby("grid_id", as_index=False)["visits"].sum().rename(columns={"visits": "visits_2025"})
    totals = totals.sort_values("visits_2025", ascending=False).head(int(args.top_k))
    top_ids = totals["grid_id"].tolist()

    # Write a manifest with static grid attributes (good for QGIS join)
    static = (
        df[df["grid_id"].isin(top_ids)]
        .groupby("grid_id", as_index=False)
        .agg(
            gx=("gx", "first"),
            gy=("gy", "first"),
            cell_lon=("cell_lon", "first"),
            cell_lat=("cell_lat", "first"),
            visits_2025=("visits", "sum"),
            visitors_2025=("visitors", "sum"),
        )
        .sort_values("visits_2025", ascending=False)
    )
    static.to_csv(out_dir / "grid_topk_manifest.csv", index=False)

    # Write per-grid Autoformer CSVs: date, OT (univariate)
    for grid_id in top_ids:
        g = df[df["grid_id"] == grid_id].sort_values("week_start")
        out = pd.DataFrame(
            {
                "date": g["week_start"].dt.strftime("%Y-%m-%d"),
                "OT": g["visits"].astype(np.float64).to_numpy(),
            }
        )
        out.to_csv(out_dir / f"grid_{grid_id}.csv", index=False)

    print(f"Wrote {len(top_ids)} grid CSVs to {out_dir}")
    print(f"Manifest: {out_dir / 'grid_topk_manifest.csv'}")


if __name__ == "__main__":
    main()

