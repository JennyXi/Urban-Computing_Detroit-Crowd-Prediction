"""
Build weekly multivariate CSV for Autoformer (Dataset_Custom).

- Aggregates POI-week rows to Detroit city-wide weekly totals.
- Adds explicit calendar / seasonality / volatility channels you asked for:
  - Monthly cycle: sin/cos of month angle (long-horizon seasonality).
  - Within-year weekly pattern: sin/cos of ISO week-of-year (52-ish rhythm).
  - Fluctuation: rolling std & optional log-diff of visits (short-horizon volatility).

Time grain: one row per DATE_RANGE_START (weekly). Use Autoformer --freq W.

Usage:
  python scripts/build_autoformer_input.py
  python scripts/build_autoformer_input.py
  python scripts/build_autoformer_input.py --date-start 2025-01-01 --date-end 2025-12-31
  python scripts/build_autoformer_input.py --input path/to/detroit_filtered.parquet --output data/autoformer_weekly_2025.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet",
        help="Filtered Detroit Parquet (POI x week).",
    )
    parser.add_argument(
        "--output",
        default=r"E:\Urban Computing Final Project\Try_0412\data\autoformer_weekly_2025_sample.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--date-start",
        default="2025-01-01",
        help="Inclusive lower bound on week-start date (DATE_RANGE_START, by day).",
    )
    parser.add_argument(
        "--date-end",
        default="2025-12-31",
        help="Inclusive upper bound on week-start date.",
    )
    parser.add_argument("--roll", type=int, default=4, help="Rolling window in weeks (~1 month).")
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    path_sql = str(src).replace("\\", "/")
    d0 = args.date_start
    d1 = args.date_end
    # Aggregate inside DuckDB; filter week-starts to [date-start, date-end] (e.g. 2025 Q2).
    g = duckdb.sql(
        f"""
        SELECT
          cast(date_trunc('day', "DATE_RANGE_START") AS DATE) AS date,
          sum(coalesce("VISIT_COUNTS", 0))::DOUBLE AS visits,
          sum(coalesce("VISITOR_COUNTS", 0))::DOUBLE AS visitors
        FROM read_parquet('{path_sql}')
        WHERE cast(date_trunc('day', "DATE_RANGE_START") AS DATE) >= DATE '{d0}'
          AND cast(date_trunc('day', "DATE_RANGE_START") AS DATE) <= DATE '{d1}'
        GROUP BY 1
        ORDER BY 1
        """
    ).df()

    g["date"] = pd.to_datetime(g["date"])

    # Monthly harmonic (12-month phase on calendar month)
    month = g["date"].dt.month.astype(float)
    g["m_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    g["m_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    # Within-year weekly rhythm (ISO week-of-year)
    woy = g["date"].dt.isocalendar().week.astype(float)
    g["woy_sin"] = np.sin(2 * np.pi * (woy - 1) / 52.0)
    g["woy_cos"] = np.cos(2 * np.pi * (woy - 1) / 52.0)

    # Week-start weekday: 0=Mon ... (mostly constant if data always Monday; still useful if mixed)
    g["week_start_dow"] = g["date"].dt.dayofweek / 6.0 - 0.5

    # Volatility / fluctuation (weekly series)
    w = args.roll
    g["visits_roll_mean"] = g["visits"].rolling(window=w, min_periods=1).mean()
    g["visits_roll_std"] = g["visits"].rolling(window=w, min_periods=2).std()
    g["visits_logdiff"] = np.log1p(g["visits"]).diff()
    # Short windows (e.g. one quarter): keep all weeks; impute missing rolling/logdiff at start.
    g["visits_roll_std"] = g["visits_roll_std"].fillna(0.0)
    g["visits_logdiff"] = g["visits_logdiff"].fillna(0.0)
    denom = g["visits_roll_std"].replace(0.0, np.nan).fillna(1e-6)
    g["visits_z"] = (g["visits"] - g["visits_roll_mean"]) / denom

    # Autoformer Dataset_Custom: last column is target for MS; use 'OT' as in ETT convention
    g["OT"] = g["visits"]

    out_df = g[
        [
            "date",
            "m_sin",
            "m_cos",
            "woy_sin",
            "woy_cos",
            "week_start_dow",
            "visitors",
            "visits_roll_mean",
            "visits_roll_std",
            "visits_logdiff",
            "visits_z",
            "OT",
        ]
    ].copy()
    out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d")

    out_df.to_csv(out, index=False)
    print(
        f"Wrote {out}  date_range=[{d0}, {d1}]  rows={len(out_df)}  cols={list(out_df.columns)}"
    )


if __name__ == "__main__":
    main()
