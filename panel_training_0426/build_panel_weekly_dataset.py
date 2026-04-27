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

    cols = ["week_start", "grid_id", "visits", "visitors", "cell_lon", "cell_lat", "gx", "gy"]
    df = pq.read_table(inp, columns=cols).to_pandas()
    df["week_start"] = pd.to_datetime(df["week_start"])
    d0 = pd.Timestamp(args.date_start).normalize()
    d1 = pd.Timestamp(args.date_end).normalize()
    df = df[(df["week_start"].dt.normalize() >= d0) & (df["week_start"].dt.normalize() <= d1)].copy()
    if df.empty:
        raise SystemExit("No rows after date filter.")

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

    for grid_id in top_ids:
        g = df[df["grid_id"].astype(str) == str(grid_id)].sort_values("week_start").copy()
        if g.empty:
            continue
        # static attrs
        gx = int(g["gx"].iloc[0])
        gy = int(g["gy"].iloc[0])
        cell_lon = float(g["cell_lon"].iloc[0])
        cell_lat = float(g["cell_lat"].iloc[0])

        # reindex weekly
        g = g.set_index("week_start").reindex(all_weeks)
        g["grid_id"] = str(grid_id)
        g["gx"] = gx
        g["gy"] = gy
        g["cell_lon"] = cell_lon
        g["cell_lat"] = cell_lat
        g["visits"] = g["visits"].fillna(0.0).astype(np.float64)
        g["visitors"] = g["visitors"].fillna(0.0).astype(np.float64)
        g = g.reset_index().rename(columns={"index": "week_start"})

        # static stats from 2024 as covariates (interpretable heterogeneity features)
        g_2024 = g[(g["week_start"] >= pd.Timestamp("2024-01-01")) & (g["week_start"] <= pd.Timestamp("2024-12-31"))]
        mean_2024 = float(g_2024["visits"].mean()) if len(g_2024) else 0.0
        std_2024 = float(g_2024["visits"].std(ddof=0)) if len(g_2024) else 0.0
        total_2024 = float(g_2024["visits"].sum()) if len(g_2024) else 0.0

        # merge global (past-only) covariate, if enabled
        if city_cov_name is not None:
            g = g.merge(city_cov_df, on="week_start", how="left")
            g[city_cov_name] = g[city_cov_name].fillna(0.0).astype(np.float64)

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
            if city_cov_name is not None:
                row[city_cov_name] = float(g[city_cov_name].iloc[i])
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
        tag = f"topk{rank_year}_{city_tag}_{args.target_transform}"
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

