from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _make_city_cov(
    city_total: pd.DataFrame, all_weeks: pd.DatetimeIndex, mode: str
) -> tuple[pd.DataFrame, str | None]:
    """
    Build a *future-available* city-level covariate table indexed by week_start.

    mode:
      - none: no covariate
      - lag1: last week's city visits
      - lag4: 4 weeks ago city visits
      - roll4_mean_lag1: rolling mean over last 4 weeks, shifted by 1 (uses only past)

    Returns: (df with columns [week_start, <feature>], feature_name or None)
    """
    base = (
        city_total.set_index("week_start")
        .reindex(all_weeks, fill_value=0.0)
        .reset_index()
        .rename(columns={"index": "week_start"})
    )
    base["city_visits"] = base["city_visits"].astype(np.float64)

    mode = str(mode).lower().strip()
    if mode == "none":
        return base[["week_start"]].copy(), None
    if mode == "lag1":
        base["city_visits_lag1"] = base["city_visits"].shift(1).fillna(0.0)
        return base[["week_start", "city_visits_lag1"]].copy(), "city_visits_lag1"
    if mode == "lag4":
        base["city_visits_lag4"] = base["city_visits"].shift(4).fillna(0.0)
        return base[["week_start", "city_visits_lag4"]].copy(), "city_visits_lag4"
    if mode == "roll4_mean_lag1":
        # mean of last 4 weeks, excluding current week (shift by 1)
        base["city_visits_roll4_mean_lag1"] = (
            base["city_visits"].rolling(window=4, min_periods=1).mean().shift(1).fillna(0.0)
        )
        return (
            base[["week_start", "city_visits_roll4_mean_lag1"]].copy(),
            "city_visits_roll4_mean_lag1",
        )

    raise SystemExit(
        f"Unknown --city-cov={mode!r}. Choose from: none, lag1, lag4, roll4_mean_lag1"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a panel weekly dataset (Top-K grids) for shared Autoformer training.")
    parser.add_argument("--input", default="data/grid100_weekly_2024_2025.parquet", help="Grid weekly parquet (long).")
    parser.add_argument("--out-dir", default="panel_training_0426/outputs", help="Output directory.")
    parser.add_argument(
        "--poi-static",
        default="",
        help="Optional grid-level static POI indices parquet to merge by grid_id. "
        "Example: data/grid100_poi_static_2024.parquet (built from scripts/build_grid_poi_static.py).",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--date-start", default="2024-01-01")
    parser.add_argument("--date-end", default="2025-12-31")
    parser.add_argument(
        "--topk-year",
        type=int,
        default=2024,
        help="Year used to rank grids by total visits for selecting Top-K. "
        "For strict non-leakage baselines, use 2024 (train-only year).",
    )
    parser.add_argument(
        "--city-cov",
        default="lag1",
        choices=["none", "lag1", "lag4", "roll4_mean_lag1"],
        help="City-level weekly covariate to include. "
        "Use lag/rolling variants to avoid using future/unavailable contemporaneous totals.",
    )
    parser.add_argument(
        "--weekend-cov",
        default="none",
        choices=["none", "share_lag1", "components_lag1"],
        help="Optional past-only weekend/weekday covariates derived from VISITS_BY_DAY at aggregation time. "
        "IMPORTANT: do NOT use contemporaneous weekend_share as a same-week covariate (it leaks target). "
        "This option uses lagged features only.",
    )
    parser.add_argument(
        "--spatial-cov",
        default="nbr8_std_lag1",
        choices=["none", "nbr8_meanstd_lag1", "nbr8_std_lag1"],
        help="Optional past-only spatial covariates using 8-neighborhood (Moore) aggregation. "
        "nbr8_meanstd_lag1 adds mean/std of neighbors' visits from last week. "
        "nbr8_std_lag1 adds only the neighbors' std from last week (more conservative).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional experiment tag appended to output filenames (e.g. 'abl_city_lag1_topk2024').",
    )
    parser.add_argument(
        "--target-transform",
        default="log1p",
        choices=["none", "log1p"],
        help="Optional target transform applied to OT (visits) for training stability.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inp = (repo_root / args.input).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Missing input parquet: {inp}")

    base_cols = ["week_start", "grid_id", "visits", "visitors", "cell_lon", "cell_lat", "gx", "gy"]
    weekend_mode = str(args.weekend_cov).lower().strip()
    want_weekend = weekend_mode != "none"
    extra_cols = ["weekday_visits", "weekend_visits", "weekend_share"] if want_weekend else []
    schema_cols = set(pq.ParquetFile(inp).schema.names)
    cols = base_cols + [c for c in extra_cols if c in schema_cols]
    if want_weekend and not set(extra_cols).issubset(schema_cols):
        raise SystemExit(
            "Requested --weekend-cov but input parquet is missing weekend columns. "
            "Rebuild grid-weekly parquet with: scripts/aggregate_grid_weekly.py --add-weekend-share"
        )
    df = pq.read_table(inp, columns=cols).to_pandas()
    df["week_start"] = pd.to_datetime(df["week_start"])
    d0 = pd.Timestamp(args.date_start).normalize()
    d1 = pd.Timestamp(args.date_end).normalize()
    df = df[(df["week_start"].dt.normalize() >= d0) & (df["week_start"].dt.normalize() <= d1)].copy()
    if df.empty:
        raise SystemExit("No rows after date filter.")

    # Optional: merge static POI indices by grid_id (supply-side features)
    poi_path_str = str(args.poi_static).strip()
    poi_static: pd.DataFrame | None = None
    poi_cols: list[str] = []
    if poi_path_str:
        poi_path = (repo_root / poi_path_str).resolve()
        if not poi_path.exists():
            raise SystemExit(f"Missing --poi-static parquet: {poi_path}")
        poi_static = pq.read_table(poi_path).to_pandas()
        if "grid_id" not in poi_static.columns:
            raise SystemExit("--poi-static must contain column grid_id.")
        poi_static["grid_id"] = poi_static["grid_id"].astype(str)
        # Keep numeric columns only (besides identifiers)
        drop = {"grid_id", "gx", "gy"}
        poi_cols = [c for c in poi_static.columns if c not in drop]
        if not poi_cols:
            raise SystemExit("--poi-static has no feature columns besides grid_id/gx/gy.")
        for c in poi_cols:
            poi_static[c] = pd.to_numeric(poi_static[c], errors="coerce").fillna(0.0).astype(np.float64)

    # Rank Top-K grids by total visits in a specified year
    rank_year = int(args.topk_year)
    y0 = pd.Timestamp(f"{rank_year}-01-01")
    y1 = pd.Timestamp(f"{rank_year}-12-31")
    df_rank = df[(df["week_start"] >= y0) & (df["week_start"] <= y1)].copy()
    if df_rank.empty:
        raise SystemExit(
            f"No rows found for --topk-year={rank_year} within date range [{d0.date()}, {d1.date()}]. "
            f"Adjust --date-start/--date-end or pick a different --topk-year."
        )
    totals_rank = (
        df_rank.groupby("grid_id", as_index=False)["visits"]
        .sum()
        .rename(columns={"visits": f"visits_{rank_year}"})
        .sort_values(f"visits_{rank_year}", ascending=False)
    )
    top_ids = totals_rank.head(int(args.top_k))["grid_id"].astype(str).tolist()
    df = df[df["grid_id"].astype(str).isin(top_ids)].copy()

    # Ensure complete weekly index per grid (fill missing weeks with 0 visits/visitors)
    all_weeks = pd.date_range(df["week_start"].min(), df["week_start"].max(), freq="W-MON")
    out_rows = []
    static_rows = []

    # Pre-compute citywide total visits per week (used to derive a *past-only* covariate)
    city_total = df.groupby("week_start", as_index=False)["visits"].sum().rename(columns={"visits": "city_visits"})
    city_cov_df, city_cov_name = _make_city_cov(city_total, all_weeks, mode=str(args.city_cov))

    spatial_mode = str(args.spatial_cov).lower().strip()
    want_spatial = spatial_mode != "none"
    if want_spatial and spatial_mode not in {"nbr8_meanstd_lag1", "nbr8_std_lag1"}:
        raise SystemExit(f"Unknown --spatial-cov={spatial_mode!r}. Choose from: none, nbr8_meanstd_lag1, nbr8_std_lag1")

    # --- Prebuild complete weekly series per grid (needed for spatial neighbor features) ---
    grid_series: dict[str, pd.DataFrame] = {}
    for grid_id in top_ids:
        g0 = df[df["grid_id"].astype(str) == str(grid_id)].sort_values("week_start").copy()
        if g0.empty:
            continue
        gx = int(g0["gx"].iloc[0])
        gy = int(g0["gy"].iloc[0])
        cell_lon = float(g0["cell_lon"].iloc[0])
        cell_lat = float(g0["cell_lat"].iloc[0])

        g = g0.set_index("week_start").reindex(all_weeks)
        g["grid_id"] = str(grid_id)
        g["gx"] = gx
        g["gy"] = gy
        g["cell_lon"] = cell_lon
        g["cell_lat"] = cell_lat
        g["visits"] = g["visits"].fillna(0.0).astype(np.float64)
        g["visitors"] = g["visitors"].fillna(0.0).astype(np.float64)
        if want_weekend:
            g["weekday_visits"] = g.get("weekday_visits", 0.0)
            g["weekend_visits"] = g.get("weekend_visits", 0.0)
            g["weekend_share"] = g.get("weekend_share", 0.0)
            g["weekday_visits"] = g["weekday_visits"].fillna(0.0).astype(np.float64)
            g["weekend_visits"] = g["weekend_visits"].fillna(0.0).astype(np.float64)
            g["weekend_share"] = g["weekend_share"].fillna(0.0).astype(np.float64)

        g = g.reset_index().rename(columns={"index": "week_start"})
        grid_series[str(grid_id)] = g

    # --- Spatial neighbor covariates (8-neighborhood) ---
    if want_spatial:
        # map (week_start, gx, gy) -> visits for the Top-K set
        vmap: dict[tuple[pd.Timestamp, int, int], float] = {}
        for gid, g in grid_series.items():
            gx = int(g["gx"].iloc[0])
            gy = int(g["gy"].iloc[0])
            for wk, v in zip(pd.to_datetime(g["week_start"]).tolist(), g["visits"].tolist()):
                vmap[(pd.Timestamp(wk).normalize(), gx, gy)] = float(v)

        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        for gid, g in grid_series.items():
            gx = int(g["gx"].iloc[0])
            gy = int(g["gy"].iloc[0])
            means = []
            stds = []
            for wk in pd.to_datetime(g["week_start"]).tolist():
                key_w = pd.Timestamp(wk).normalize()
                nbr_vals = []
                for dx, dy in offsets:
                    k = (key_w, gx + dx, gy + dy)
                    if k in vmap:
                        nbr_vals.append(vmap[k])
                if len(nbr_vals) == 0:
                    means.append(0.0)
                    stds.append(0.0)
                else:
                    arr = np.asarray(nbr_vals, dtype=np.float64)
                    means.append(float(arr.mean()))
                    stds.append(float(arr.std(ddof=0)))

            g["nbr8_mean_visits"] = np.asarray(means, dtype=np.float64)
            g["nbr8_std_visits"] = np.asarray(stds, dtype=np.float64)
            # past-only lag1
            g["nbr8_mean_visits_lag1"] = g["nbr8_mean_visits"].shift(1).fillna(0.0).astype(np.float64)
            g["nbr8_std_visits_lag1"] = g["nbr8_std_visits"].shift(1).fillna(0.0).astype(np.float64)

            grid_series[gid] = g

    for grid_id in top_ids:
        if str(grid_id) not in grid_series:
            continue
        g = grid_series[str(grid_id)].copy()
        gx = int(g["gx"].iloc[0])
        gy = int(g["gy"].iloc[0])
        cell_lon = float(g["cell_lon"].iloc[0])
        cell_lat = float(g["cell_lat"].iloc[0])

        # attach static POI indices (if provided)
        poi_feat: dict[str, float] = {}
        if poi_static is not None and poi_cols:
            hit = poi_static[poi_static["grid_id"] == str(grid_id)]
            if len(hit) > 0:
                r0 = hit.iloc[0]
                poi_feat = {c: float(r0[c]) for c in poi_cols}
            else:
                poi_feat = {c: 0.0 for c in poi_cols}

        # static stats from 2024 as covariates (interpretable heterogeneity features)
        g_2024 = g[(g["week_start"] >= pd.Timestamp("2024-01-01")) & (g["week_start"] <= pd.Timestamp("2024-12-31"))]
        mean_2024 = float(g_2024["visits"].mean()) if len(g_2024) else 0.0
        std_2024 = float(g_2024["visits"].std(ddof=0)) if len(g_2024) else 0.0
        total_2024 = float(g_2024["visits"].sum()) if len(g_2024) else 0.0

        # merge global (past-only) covariate, if enabled
        if city_cov_name is not None:
            g = g.merge(city_cov_df, on="week_start", how="left")
            g[city_cov_name] = g[city_cov_name].fillna(0.0).astype(np.float64)

        # grid-level weekend covariates (past-only)
        weekend_cov_names: list[str] = []
        if want_weekend:
            if weekend_mode == "share_lag1":
                g["weekend_share_lag1"] = g["weekend_share"].shift(1).fillna(0.0).astype(np.float64)
                weekend_cov_names = ["weekend_share_lag1"]
            elif weekend_mode == "components_lag1":
                g["weekday_visits_lag1"] = g["weekday_visits"].shift(1).fillna(0.0).astype(np.float64)
                g["weekend_visits_lag1"] = g["weekend_visits"].shift(1).fillna(0.0).astype(np.float64)
                g["weekend_share_lag1"] = g["weekend_share"].shift(1).fillna(0.0).astype(np.float64)
                weekend_cov_names = ["weekday_visits_lag1", "weekend_visits_lag1", "weekend_share_lag1"]
            else:
                raise SystemExit(f"Unknown --weekend-cov={weekend_mode!r}. Choose from: none, share_lag1, components_lag1")

        spatial_cov_names: list[str] = []
        if want_spatial:
            if spatial_mode == "nbr8_meanstd_lag1":
                spatial_cov_names = ["nbr8_mean_visits_lag1", "nbr8_std_visits_lag1"]
            else:
                spatial_cov_names = ["nbr8_std_visits_lag1"]

        # target
        ot = g["visits"].to_numpy(dtype=np.float64)
        if args.target_transform == "log1p":
            ot = np.log1p(np.maximum(ot, 0.0))

        # Build rows in Dataset_Custom style: first col `date`, last col `OT`
        # Middle columns are numeric covariates:
        # - dynamic: city-level covariate (past-only): city_visits_lag* / rolling mean lagged
        # - static repeated per row: gx, gy, cell_lon, cell_lat, mean_2024, std_2024, total_2024
        for i, wk in enumerate(g["week_start"].to_list()):
            row = {
                "grid_id": str(grid_id),
                "date": pd.Timestamp(wk).strftime("%Y-%m-%d"),
                "gx": float(gx),
                "gy": float(gy),
                "cell_lon": float(cell_lon),
                "cell_lat": float(cell_lat),
                "mean_visits_2024": float(mean_2024),
                "std_visits_2024": float(std_2024),
                "total_visits_2024": float(total_2024),
            }
            for k, v in poi_feat.items():
                row[k] = float(v)
            if city_cov_name is not None:
                row[city_cov_name] = float(g[city_cov_name].iloc[i])
            for nm in weekend_cov_names:
                row[nm] = float(g[nm].iloc[i])
            for nm in spatial_cov_names:
                row[nm] = float(g[nm].iloc[i])
            out_rows.append(row)

        static_rows.append(
            {
                "grid_id": str(grid_id),
                "gx": gx,
                "gy": gy,
                "cell_lon": cell_lon,
                "cell_lat": cell_lat,
                "mean_visits_2024": mean_2024,
                "std_visits_2024": std_2024,
                "total_visits_2024": total_2024,
                f"visits_{rank_year}": float(df_rank[df_rank["grid_id"].astype(str) == str(grid_id)]["visits"].sum()),
            }
        )

        # attach OT values (vector) after building rows
        start = len(out_rows) - len(g)
        for i in range(len(g)):
            out_rows[start + i]["OT"] = float(ot[i])

    panel = pd.DataFrame(out_rows)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["grid_id", "date"]).reset_index(drop=True)

    # For panel training we keep the long table; the trainer will sample windows per-grid.
    tag = str(args.tag).strip()
    if not tag:
        city_tag = f"city_{args.city_cov}"
        wk_tag = f"wk_{args.weekend_cov}"
        sp_tag = f"sp_{args.spatial_cov}"
        tag = f"topk{rank_year}_{city_tag}_{wk_tag}_{sp_tag}_{args.target_transform}"
    out_csv = out_dir / f"panel_weekly_top{int(args.top_k)}_{d0.year}_{d1.year}_{tag}.csv"
    panel.to_csv(out_csv, index=False)

    manifest = pd.DataFrame(static_rows).sort_values(f"visits_{rank_year}", ascending=False).reset_index(drop=True)
    man_csv = out_dir / f"panel_weekly_top{int(args.top_k)}_manifest_{tag}.csv"
    manifest.to_csv(man_csv, index=False)

    print(f"Wrote: {out_csv}  rows={len(panel)}  grids={panel['grid_id'].nunique()}")
    print(f"Wrote: {man_csv}")
    print(f"Target transform: {args.target_transform}")
    print(f"Top-K ranking year: {rank_year}")
    print(f"City covariate: {args.city_cov}  (feature={city_cov_name})")


if __name__ == "__main__":
    main()

