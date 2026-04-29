"""
Aggregate Detroit POI Parquet to a regular grid (default 100m x 100m) x day.

- Projects WGS84 lat/lon to UTM zone 17N (EPSG:32617), suitable for Detroit area.
- Assigns each row to cell indices (gx, gy); sums VISIT_COUNTS / VISITOR_COUNTS per (date, cell).
- Writes a long table: date, gx, gy, grid_id, visits, visitors, cell_lon, cell_lat, is_weekend.

Requires: pyproj, pandas, pyarrow (geopandas not required).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="POI Parquet -> grid x day (UTM 17N).")
    parser.add_argument(
        "--input",
        default=r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet",
    )
    parser.add_argument(
        "--output",
        default=r"E:\Urban Computing Final Project\Try_0412\data\grid100_daily_2024_2025.parquet",
        help="Output Parquet (long: date, gx, gy, visits, visitors, cell center lon/lat).",
    )
    parser.add_argument("--date-start", default="2024-01-01")
    parser.add_argument("--date-end", default="2025-12-31")
    parser.add_argument(
        "--cell-meters",
        type=float,
        default=100.0,
        help="Square cell side length in meters (projected CRS).",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=32617,
        help="Projected CRS for grid (default 32617 = WGS84 UTM 17N, good for Detroit).",
    )
    parser.add_argument("--batch-rows", type=int, default=100_000)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cell = float(args.cell_meters)
    t0 = pd.Timestamp(args.date_start).normalize()
    t1 = pd.Timestamp(args.date_end).normalize()

    to_proj = Transformer.from_crs("EPSG:4326", f"EPSG:{args.epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{args.epsg}", "EPSG:4326", always_xy=True)

    # Partial sums per (date, gx, gy): visits, visitors
    acc: dict[tuple[pd.Timestamp, int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    cols = ["DATE_RANGE_START", "LATITUDE", "LONGITUDE", "VISIT_COUNTS", "VISITOR_COUNTS"]
    pf = pq.ParquetFile(inp)

    for batch in pf.iter_batches(columns=cols, batch_size=args.batch_rows):
        sub = batch.to_pandas()
        sub["DATE_RANGE_START"] = pd.to_datetime(sub["DATE_RANGE_START"])
        day = sub["DATE_RANGE_START"].dt.normalize()
        sub = sub[(day >= t0) & (day <= t1)]
        if sub.empty:
            continue

        lat = sub["LATITUDE"].to_numpy(dtype=np.float64)
        lon = sub["LONGITUDE"].to_numpy(dtype=np.float64)
        ok = np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            continue

        sub = sub.loc[ok].copy()
        lat = lat[ok]
        lon = lon[ok]
        day = sub["DATE_RANGE_START"].dt.normalize()

        x, y = to_proj.transform(lon, lat)
        gx = np.floor(np.asarray(x, dtype=np.float64) / cell).astype(np.int64)
        gy = np.floor(np.asarray(y, dtype=np.float64) / cell).astype(np.int64)

        sub = sub.assign(_day=day.values, _gx=gx, _gy=gy)
        sub["VISIT_COUNTS"] = sub["VISIT_COUNTS"].fillna(0).astype(np.float64)
        sub["VISITOR_COUNTS"] = sub["VISITOR_COUNTS"].fillna(0).astype(np.float64)
        part = sub.groupby(["_day", "_gx", "_gy"], sort=False)[["VISIT_COUNTS", "VISITOR_COUNTS"]].sum().reset_index()

        for _, r in part.iterrows():
            key = (pd.Timestamp(r["_day"]), int(r["_gx"]), int(r["_gy"]))
            a = acc[key]
            a[0] += float(r["VISIT_COUNTS"])
            a[1] += float(r["VISITOR_COUNTS"])

    if not acc:
        raise SystemExit("No rows aggregated. Check date range, lat/lon, and input path.")

    rows = []
    for (day, gx, gy), vals in sorted(acc.items()):
        visits = float(vals[0])
        visitors = float(vals[1])
        cx = (gx + 0.5) * cell
        cy = (gy + 0.5) * cell
        clon, clat = to_wgs.transform(cx, cy)
        rows.append(
            {
                "date": day,
                "gx": gx,
                "gy": gy,
                "grid_id": f"{gx}_{gy}",
                "visits": visits,
                "visitors": visitors,
                "cell_lon": clon,
                "cell_lat": clat,
                "is_weekend": int(pd.Timestamp(day).dayofweek >= 5),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out, index=False)
    print(
        f"Wrote {out}  rows={len(out_df)}  cell_m={cell}  epsg={args.epsg}  "
        f"date=[{args.date_start}, {args.date_end}]"
    )


if __name__ == "__main__":
    main()
