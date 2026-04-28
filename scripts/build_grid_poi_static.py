"""
Build grid-level *static* POI indices from SafeGraph-style POI-week parquet.

Goal: turn POI attributes (category / NAICS / dwell) into per-grid supply features that
can be merged into the weekly grid forecasting pipeline.

Notes / assumptions:
- Input is a POI-week table (many rows per POI over weeks).
- We de-duplicate POIs by `PERSISTENT_ID` (fallback to `FOOTPRINT_ID` if missing).
- For non-leakage baselines, you can build indices using only a reference year (default 2024).
- "Natural" and "Residential" are typically NOT well-covered by POI datasets; you may
  want to complement them with GIS layers (OSM landuse / parks / water, census, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer


# Keep only the 4 POI super categories that are reliably represented in POI datasets.
# (Residential and Nature are better sourced from landuse / census / OSM layers.)
SUPER_CATS = ["life", "transport", "economy", "public_service"]


def _naics2(x: object) -> str | None:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    # common: "445110", sometimes "4451.0"
    s = "".join(ch for ch in s if ch.isdigit())
    if len(s) < 2:
        return None
    return s[:2]


def _norm_str(x: object) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return str(x).strip().lower()


def map_supercat(naics_code: object, top_category: object, sub_category: object, location_name: object) -> str:
    """
    Map a POI to one of the super categories:
      - life (retail / food / entertainment)
      - transport (transit / gas / parking / towing ...)
      - economy (finance / companies / business services)
      - public_service (hospital / police / school / fire ...)
    """
    na2 = _naics2(naics_code)
    top = _norm_str(top_category)
    sub = _norm_str(sub_category)
    name = _norm_str(location_name)

    # --- NAICS-first mapping (stable) ---
    if na2 is not None:
        if na2 in {"44", "45", "72", "71"}:
            return "life"
        if na2 in {"48", "49"}:
            return "transport"
        if na2 in {"52", "54", "55", "56", "42"}:
            # finance, professional services, management, admin/support, wholesale
            return "economy"
        if na2 in {"61", "62", "92"}:
            return "public_service"
        if na2 in {"11", "21", "22", "23", "31", "32", "33"}:
            # agriculture/mining/utilities/construction/manufacturing -> "economy" (industrial supply)
            return "economy"
        # 53 real estate can be treated as economy
        if na2 in {"53"}:
            return "economy"

    # --- TOP/SUB fallback mapping (covers common POI strings) ---
    # transport
    if (
        "gasoline station" in top
        or "parking" in sub
        or "road transportation" in top
        or "towing" in sub
        or "transit" in top
        or "station" in sub
        or "station" in top
    ):
        return "transport"

    # public service
    if any(k in top for k in ["hospital", "police", "fire", "school", "government"]) or any(
        k in sub for k in ["hospital", "police", "fire", "school", "government"]
    ):
        return "public_service"

    # life
    if any(k in top for k in ["restaurant", "eating", "grocery", "drinking", "clothing", "merchandise", "fitness", "recreation"]):
        return "life"

    # economy
    if any(k in top for k in ["credit intermediation", "bank", "accounting", "telecommunications"]) or any(
        k in sub for k in ["bank", "tax preparation", "accounting"]
    ):
        return "economy"

    # Parks/green spaces are typically better captured by GIS landuse layers; treat them as "life" here.
    # default to life: most POIs are "places" people visit.
    return "life"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grid-level static POI indices from detroit_filtered.parquet.")
    parser.add_argument("--input", default="data/detroit_filtered.parquet")
    parser.add_argument("--output", default="data/grid100_poi_static_2024.parquet")
    parser.add_argument("--ref-year", type=int, default=2024, help="Use only rows whose DATE_RANGE_START is within this year.")
    parser.add_argument("--cell-meters", type=float, default=100.0)
    parser.add_argument("--epsg", type=int, default=32617, help="Projected CRS for grid (default 32617 = UTM 17N).")
    parser.add_argument("--batch-rows", type=int, default=200_000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inp = (repo_root / args.input).resolve()
    out = (repo_root / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Missing input parquet: {inp}")

    year = int(args.ref_year)
    y0 = pd.Timestamp(f"{year}-01-01").normalize()
    y1 = pd.Timestamp(f"{year}-12-31").normalize()

    cell = float(args.cell_meters)
    to_proj = Transformer.from_crs("EPSG:4326", f"EPSG:{int(args.epsg)}", always_xy=True)

    # accumulate unique POIs by id -> attributes
    # keep first seen location/cat and median dwell median across the year
    poi_rows = []
    pf = pq.ParquetFile(inp)
    cols = [
        "DATE_RANGE_START",
        "LATITUDE",
        "LONGITUDE",
        "PERSISTENT_ID",
        "FOOTPRINT_ID",
        "TOP_CATEGORY",
        "SUB_CATEGORY",
        "NAICS_CODE",
        "LOCATION_NAME",
        "MEDIAN_DWELL",
        "BRAND",
    ]
    schema = set(pf.schema.names)
    cols = [c for c in cols if c in schema]

    for batch in pf.iter_batches(columns=cols, batch_size=int(args.batch_rows)):
        sub = batch.to_pandas()
        sub["DATE_RANGE_START"] = pd.to_datetime(sub["DATE_RANGE_START"], errors="coerce")
        wk = sub["DATE_RANGE_START"].dt.normalize()
        sub = sub[(wk >= y0) & (wk <= y1)]
        if sub.empty:
            continue

        lat = sub.get("LATITUDE")
        lon = sub.get("LONGITUDE")
        if lat is None or lon is None:
            continue
        latv = pd.to_numeric(lat, errors="coerce").to_numpy(dtype=np.float64)
        lonv = pd.to_numeric(lon, errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(latv) & np.isfinite(lonv)
        if not ok.any():
            continue
        sub = sub.loc[ok].copy()
        latv = latv[ok]
        lonv = lonv[ok]

        # POI id
        if "PERSISTENT_ID" in sub.columns:
            pid = sub["PERSISTENT_ID"].astype("string")
        elif "FOOTPRINT_ID" in sub.columns:
            pid = sub["FOOTPRINT_ID"].astype("string")
        else:
            # cannot de-duplicate; fall back to row id (bad)
            pid = pd.Series(np.arange(len(sub)), dtype="int64").astype("string")
        sub["_poi_id"] = pid.fillna("").astype("string")
        sub = sub[sub["_poi_id"].str.len() > 0]
        if sub.empty:
            continue

        x, y = to_proj.transform(lonv, latv)
        gx = np.floor(np.asarray(x, dtype=np.float64) / cell).astype(np.int64)
        gy = np.floor(np.asarray(y, dtype=np.float64) / cell).astype(np.int64)
        sub["_gx"] = gx
        sub["_gy"] = gy

        # keep minimal attributes; we will aggregate by unique poi_id at the end
        keep = ["_poi_id", "_gx", "_gy"]
        for c in ["TOP_CATEGORY", "SUB_CATEGORY", "NAICS_CODE", "LOCATION_NAME", "MEDIAN_DWELL", "BRAND"]:
            if c in sub.columns:
                keep.append(c)
        poi_rows.append(sub[keep])

    if not poi_rows:
        raise SystemExit("No rows collected. Check --ref-year and parquet contents.")

    df = pd.concat(poi_rows, ignore_index=True)

    # Deduplicate POIs within the ref year: keep first categorical fields, median dwell as median
    agg = {"_gx": "first", "_gy": "first"}
    for c in ["TOP_CATEGORY", "SUB_CATEGORY", "NAICS_CODE", "LOCATION_NAME", "BRAND"]:
        if c in df.columns:
            agg[c] = "first"
    if "MEDIAN_DWELL" in df.columns:
        df["MEDIAN_DWELL"] = pd.to_numeric(df["MEDIAN_DWELL"], errors="coerce")
        agg["MEDIAN_DWELL"] = "median"

    poi = df.groupby("_poi_id", as_index=False).agg(agg)
    poi["grid_id"] = poi["_gx"].astype(str) + "_" + poi["_gy"].astype(str)

    # map super category
    poi["poi_supercat"] = poi.apply(
        lambda r: map_supercat(
            r.get("NAICS_CODE", None),
            r.get("TOP_CATEGORY", None),
            r.get("SUB_CATEGORY", None),
            r.get("LOCATION_NAME", None),
        ),
        axis=1,
    )

    # grid-level indices
    base = poi.groupby(["grid_id", "_gx", "_gy"], as_index=False)["_poi_id"].nunique().rename(columns={"_poi_id": "poi_cnt_total"})
    pivot = (
        poi.pivot_table(
            index=["grid_id", "_gx", "_gy"],
            columns="poi_supercat",
            values="_poi_id",
            aggfunc=pd.Series.nunique,
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    out_df = base.merge(pivot, on=["grid_id", "_gx", "_gy"], how="left")
    for c in SUPER_CATS:
        if c not in out_df.columns:
            out_df[c] = 0
        out_df[f"poi_cnt_{c}"] = out_df[c].astype(np.float64)
        out_df.drop(columns=[c], inplace=True)

    # shares + entropy
    eps = 1e-9
    for c in SUPER_CATS:
        out_df[f"poi_share_{c}"] = out_df[f"poi_cnt_{c}"] / (out_df["poi_cnt_total"].astype(np.float64) + eps)
    shares = np.vstack([out_df[f"poi_share_{c}"].to_numpy(dtype=np.float64) for c in SUPER_CATS]).T
    out_df["poi_supercat_entropy"] = (-np.where(shares > 0, shares * np.log(shares + eps), 0.0)).sum(axis=1).astype(np.float64)

    # dwell proxy (optional)
    if "MEDIAN_DWELL" in poi.columns:
        dwell = poi.groupby(["grid_id"], as_index=False)["MEDIAN_DWELL"].median().rename(columns={"MEDIAN_DWELL": "poi_median_dwell_med"})
        out_df = out_df.merge(dwell, on="grid_id", how="left")
        out_df["poi_median_dwell_med"] = pd.to_numeric(out_df["poi_median_dwell_med"], errors="coerce").fillna(0.0).astype(np.float64)

    out_df = out_df.rename(columns={"_gx": "gx", "_gy": "gy"})
    out_df.to_parquet(out, index=False)
    print(f"Wrote: {out}  grids={len(out_df)}  ref_year={year}  cell_m={cell}  epsg={args.epsg}")


if __name__ == "__main__":
    main()

