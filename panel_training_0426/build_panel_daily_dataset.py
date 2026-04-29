from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _make_city_cov(city_total: pd.DataFrame, all_days: pd.DatetimeIndex, mode: str) -> tuple[pd.DataFrame, str | None]:
    """
    Build a future-available city-level covariate table indexed by date.

    mode:
      - none: no covariate
      - lag1: yesterday city visits
      - lag7: 7 days ago city visits
      - roll7_mean_lag1: rolling mean over last 7 days, shifted by 1 (uses only past)
    """
    base = (
        city_total.set_index("date")
        .reindex(all_days, fill_value=0.0)
        .reset_index()
        .rename(columns={"index": "date"})
    )
    base["city_visits"] = base["city_visits"].astype(np.float64)

    mode = str(mode).lower().strip()
    if mode == "none":
        return base[["date"]].copy(), None
    if mode == "lag1":
        base["city_visits_lag1"] = base["city_visits"].shift(1).fillna(0.0)
        return base[["date", "city_visits_lag1"]].copy(), "city_visits_lag1"
    if mode == "lag7":
        base["city_visits_lag7"] = base["city_visits"].shift(7).fillna(0.0)
        return base[["date", "city_visits_lag7"]].copy(), "city_visits_lag7"
    if mode == "roll7_mean_lag1":
        base["city_visits_roll7_mean_lag1"] = (
            base["city_visits"].rolling(window=7, min_periods=1).mean().shift(1).fillna(0.0)
        )
        return (
            base[["date", "city_visits_roll7_mean_lag1"]].copy(),
            "city_visits_roll7_mean_lag1",
        )

    raise SystemExit(f"Unknown --city-cov={mode!r}. Choose from: none, lag1, lag7, roll7_mean_lag1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a panel daily dataset (Top-K grids) for shared Autoformer training.")
    parser.add_argument("--input", default="data/grid100_daily_2024_2025.parquet", help="Grid daily parquet (long).")
    parser.add_argument("--out-dir", default="panel_training_0426/outputs", help="Output directory.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--date-start", default="2024-01-01")
    parser.add_argument("--date-end", default="2025-12-31")
    parser.add_argument(
        "--topk-year",
        type=int,
        default=2024,
        help="Year used to rank grids by total visits for selecting Top-K.",
    )
    parser.add_argument(
        "--city-cov",
        default="lag1",
        choices=["none", "lag1", "lag7", "roll7_mean_lag1"],
        help="City-level daily covariate to include (past-only variants).",
    )
    parser.add_argument(
        "--weekend-cov",
        default="is_weekend",
        choices=["none", "is_weekend"],
        help="Optional known-in-advance calendar covariate.",
    )
    parser.add_argument(
        "--spatial-cov",
        default="nbr8_std_lag1",
        choices=["none", "nbr8_meanstd_lag1", "nbr8_std_lag1"],
        help="Optional past-only spatial covariates using 8-neighborhood aggregation.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional experiment tag appended to output filenames.",
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

    base_cols = ["date", "grid_id", "visits", "visitors", "cell_lon", "cell_lat", "gx", "gy"]
    schema_cols = set(pq.ParquetFile(inp).schema.names)
    if "is_weekend" in schema_cols:
        base_cols.append("is_weekend")
    df = pq.read_table(inp, columns=base_cols).to_pandas()

    df["date"] = pd.to_datetime(df["date"])
    d0 = pd.Timestamp(args.date_start).normalize()
    d1 = pd.Timestamp(args.date_end).normalize()
    df = df[(df["date"].dt.normalize() >= d0) & (df["date"].dt.normalize() <= d1)].copy()
    if df.empty:
        raise SystemExit("No rows after date filter.")

    # Rank Top-K by total visits in specified year
    rank_year = int(args.topk_year)
    y0 = pd.Timestamp(f"{rank_year}-01-01")
    y1 = pd.Timestamp(f"{rank_year}-12-31")
    df_rank = df[(df["date"] >= y0) & (df["date"] <= y1)].copy()
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

    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    out_rows = []
    static_rows = []

    city_total = df.groupby("date", as_index=False)["visits"].sum().rename(columns={"visits": "city_visits"})
    city_cov_df, city_cov_name = _make_city_cov(city_total, all_days, mode=str(args.city_cov))

    spatial_mode = str(args.spatial_cov).lower().strip()
    want_spatial = spatial_mode != "none"
    if want_spatial and spatial_mode not in {"nbr8_meanstd_lag1", "nbr8_std_lag1"}:
        raise SystemExit(f"Unknown --spatial-cov={spatial_mode!r}. Choose from: none, nbr8_meanstd_lag1, nbr8_std_lag1")

    # Build complete daily series per grid
    grid_series: dict[str, pd.DataFrame] = {}
    for grid_id in top_ids:
        g0 = df[df["grid_id"].astype(str) == str(grid_id)].sort_values("date").copy()
        if g0.empty:
            continue
        gx = int(g0["gx"].iloc[0])
        gy = int(g0["gy"].iloc[0])
        cell_lon = float(g0["cell_lon"].iloc[0])
        cell_lat = float(g0["cell_lat"].iloc[0])

        g = g0.set_index("date").reindex(all_days)
        g["grid_id"] = str(grid_id)
        g["gx"] = gx
        g["gy"] = gy
        g["cell_lon"] = cell_lon
        g["cell_lat"] = cell_lat
        g["visits"] = g["visits"].fillna(0.0).astype(np.float64)
        g["visitors"] = g["visitors"].fillna(0.0).astype(np.float64)
        weekend_series = pd.Series((g.index.dayofweek >= 5).astype(np.float64), index=g.index)
        if "is_weekend" in g.columns:
            g["is_weekend"] = g["is_weekend"].fillna(weekend_series).astype(np.float64)
        else:
            g["is_weekend"] = weekend_series.astype(np.float64)

        g = g.reset_index().rename(columns={"index": "date"})
        grid_series[str(grid_id)] = g

    # Spatial neighbor covariates (8-neighborhood)
    if want_spatial:
        vmap: dict[tuple[pd.Timestamp, int, int], float] = {}
        for _gid, g in grid_series.items():
            gx = int(g["gx"].iloc[0])
            gy = int(g["gy"].iloc[0])
            for day, v in zip(pd.to_datetime(g["date"]).tolist(), g["visits"].tolist()):
                vmap[(pd.Timestamp(day).normalize(), gx, gy)] = float(v)

        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for gid, g in grid_series.items():
            gx = int(g["gx"].iloc[0])
            gy = int(g["gy"].iloc[0])
            means = []
            stds = []
            for day in pd.to_datetime(g["date"]).tolist():
                key_d = pd.Timestamp(day).normalize()
                nbr_vals = []
                for dx, dy in offsets:
                    k = (key_d, gx + dx, gy + dy)
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
            g["nbr8_mean_visits_lag1"] = g["nbr8_mean_visits"].shift(1).fillna(0.0).astype(np.float64)
            g["nbr8_std_visits_lag1"] = g["nbr8_std_visits"].shift(1).fillna(0.0).astype(np.float64)
            grid_series[gid] = g

    weekend_mode = str(args.weekend_cov).lower().strip()

    for grid_id in top_ids:
        if str(grid_id) not in grid_series:
            continue
        g = grid_series[str(grid_id)].copy()
        gx = int(g["gx"].iloc[0])
        gy = int(g["gy"].iloc[0])
        cell_lon = float(g["cell_lon"].iloc[0])
        cell_lat = float(g["cell_lat"].iloc[0])

        # static stats from 2024 as covariates
        g_2024 = g[(g["date"] >= pd.Timestamp("2024-01-01")) & (g["date"] <= pd.Timestamp("2024-12-31"))]
        mean_2024 = float(g_2024["visits"].mean()) if len(g_2024) else 0.0
        std_2024 = float(g_2024["visits"].std(ddof=0)) if len(g_2024) else 0.0
        total_2024 = float(g_2024["visits"].sum()) if len(g_2024) else 0.0

        if city_cov_name is not None:
            g = g.merge(city_cov_df, on="date", how="left")
            g[city_cov_name] = g[city_cov_name].fillna(0.0).astype(np.float64)

        weekend_cov_names: list[str] = []
        if weekend_mode == "is_weekend":
            weekend_cov_names = ["is_weekend"]
        elif weekend_mode != "none":
            raise SystemExit(f"Unknown --weekend-cov={weekend_mode!r}. Choose from: none, is_weekend")

        spatial_cov_names: list[str] = []
        if want_spatial:
            if spatial_mode == "nbr8_meanstd_lag1":
                spatial_cov_names = ["nbr8_mean_visits_lag1", "nbr8_std_visits_lag1"]
            else:
                spatial_cov_names = ["nbr8_std_visits_lag1"]

        ot = g["visits"].to_numpy(dtype=np.float64)
        if args.target_transform == "log1p":
            ot = np.log1p(np.maximum(ot, 0.0))

        for i, day in enumerate(g["date"].to_list()):
            row = {
                "grid_id": str(grid_id),
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "gx": float(gx),
                "gy": float(gy),
                "cell_lon": float(cell_lon),
                "cell_lat": float(cell_lat),
                "mean_visits_2024": float(mean_2024),
                "std_visits_2024": float(std_2024),
                "total_visits_2024": float(total_2024),
            }
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

        start = len(out_rows) - len(g)
        for i in range(len(g)):
            out_rows[start + i]["OT"] = float(ot[i])

    panel = pd.DataFrame(out_rows)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["grid_id", "date"]).reset_index(drop=True)

    tag = str(args.tag).strip()
    if not tag:
        city_tag = f"city_{args.city_cov}"
        wk_tag = f"wk_{args.weekend_cov}"
        sp_tag = f"sp_{args.spatial_cov}"
        tag = f"topk{rank_year}_{city_tag}_{wk_tag}_{sp_tag}_{args.target_transform}"

    out_csv = out_dir / f"panel_daily_top{int(args.top_k)}_{d0.year}_{d1.year}_{tag}.csv"
    panel.to_csv(out_csv, index=False)

    manifest = pd.DataFrame(static_rows).sort_values(f"visits_{rank_year}", ascending=False).reset_index(drop=True)
    man_csv = out_dir / f"panel_daily_top{int(args.top_k)}_manifest_{tag}.csv"
    manifest.to_csv(man_csv, index=False)

    print(f"Wrote: {out_csv}  rows={len(panel)}  grids={panel['grid_id'].nunique()}")
    print(f"Wrote: {man_csv}")
    print(f"Target transform: {args.target_transform}")
    print(f"Top-K ranking year: {rank_year}")
    print(f"City covariate: {args.city_cov}  (feature={city_cov_name})")


if __name__ == "__main__":
    main()
