"""
Weekly-scale preprocessing for Autoformer (Dataset_Custom), combining:

1) City-wide week totals: VISIT_COUNTS / VISITOR_COUNTS (DuckDB).
2) Intra-week short-term shape: VISITS_BY_EACH_HOUR parsed as a numeric vector per POI,
   summed across all POIs by week → columns h_0..h_{D-1} (often D=168 for 7×24).
3) Calendar & volatility on weekly total visits: m_sin/m_cos, woy_sin/woy_cos,
   rolling mean/std, log-diff, z-score.

Output CSV: one row per week; use Autoformer --freq W.

Hourly-scale detail is encoded as extra channels (h_*), not as separate timestamps.

Usage:
  python scripts/preprocess_weekly_for_autoformer.py
  python scripts/preprocess_weekly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
  python scripts/preprocess_weekly_for_autoformer.py --output data/autoformer_weekly_merged_2025.csv

Requires: duckdb, pandas, numpy, pyarrow (for batched Parquet read).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def _parse_hour_vec(val) -> np.ndarray | None:
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.asarray(val, dtype=np.float64).ravel()
    s = str(val).strip()
    if not s or s.lower() == "none":
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = s.strip()
    try:
        if s.startswith("["):
            return np.asarray(json.loads(s), dtype=np.float64).ravel()
    except json.JSONDecodeError:
        pass
    try:
        return np.asarray(ast.literal_eval(s), dtype=np.float64).ravel()
    except (SyntaxError, ValueError, TypeError):
        return None


def _accumulate_hourly_by_week_parquet(
    path: Path,
    date_start: str,
    date_end: str,
    hour_col: str,
    batch_rows: int,
) -> tuple[dict[pd.Timestamp, np.ndarray], int]:
    import pyarrow.parquet as pq

    acc: dict[pd.Timestamp, np.ndarray] = {}
    n_bad = 0
    t0 = pd.Timestamp(date_start).normalize()
    t1 = pd.Timestamp(date_end).normalize()

    pf = pq.ParquetFile(path)
    cols = ["DATE_RANGE_START", hour_col]
    for batch in pf.iter_batches(columns=cols, batch_size=batch_rows):
        sub = batch.to_pandas()
        sub["DATE_RANGE_START"] = pd.to_datetime(sub["DATE_RANGE_START"])
        sub = sub[(sub["DATE_RANGE_START"].dt.normalize() >= t0) & (sub["DATE_RANGE_START"].dt.normalize() <= t1)]
        for w, v in zip(sub["DATE_RANGE_START"].dt.normalize(), sub[hour_col]):
            arr = _parse_hour_vec(v)
            if arr is None or arr.size == 0:
                n_bad += 1
                continue
            key = pd.Timestamp(w).normalize()
            if key not in acc:
                acc[key] = arr.astype(np.float64).copy()
                continue
            a, b = acc[key], arr.astype(np.float64)
            if a.size == b.size:
                acc[key] = a + b
            else:
                m = max(a.size, b.size)
                tmp = np.zeros(m, dtype=np.float64)
                tmp[: a.size] = a
                tmp[: b.size] += b
                acc[key] = tmp

    return acc, n_bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet",
    )
    parser.add_argument(
        "--output",
        default=r"E:\Urban Computing Final Project\Try_0412\data\autoformer_weekly_preprocessed.csv",
    )
    parser.add_argument("--date-start", default="2025-01-01")
    parser.add_argument("--date-end", default="2025-12-31")
    parser.add_argument("--hour-col", default="VISITS_BY_EACH_HOUR")
    parser.add_argument("--roll", type=int, default=4)
    parser.add_argument("--batch-rows", type=int, default=50_000, help="PyArrow batch size.")
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    path_sql = str(src).replace("\\", "/")
    d0, d1 = args.date_start, args.date_end

    # --- 1) Weekly totals (long-term / summary targets) ---
    totals = duckdb.sql(
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
    totals["date"] = pd.to_datetime(totals["date"])

    # --- 2) Hourly-within-week profile (short-term shape), city sum by week ---
    acc, n_bad = _accumulate_hourly_by_week_parquet(
        src, d0, d1, args.hour_col, args.batch_rows
    )
    if not acc:
        raise SystemExit(
            f"No hourly vectors parsed (--hour-col={args.hour_col!r}). "
            f"Skipped/empty rows in batches: {n_bad}. Check column name and format."
        )

    rows = sorted(acc.items(), key=lambda x: x[0])
    d_max = max(a.size for _, a in rows)
    mat = np.zeros((len(rows), d_max), dtype=np.float64)
    for i, (_, a) in enumerate(rows):
        mat[i, : a.size] = a
    dates = pd.to_datetime([t for t, _ in rows])
    hour_df = pd.concat(
        [
            pd.DataFrame({"date": dates}),
            pd.DataFrame(mat, columns=[f"h_{j}" for j in range(d_max)]),
        ],
        axis=1,
    )

    # --- 3) Merge weekly totals + hourly wide ---
    merged = totals.merge(hour_df, on="date", how="outer").sort_values("date")
    merged = merged.fillna({c: 0.0 for c in merged.columns if c.startswith("h_")})
    merged["visits"] = merged["visits"].fillna(0.0)
    merged["visitors"] = merged["visitors"].fillna(0.0)

    # --- 4) Calendar + volatility on weekly total visits (one concat avoids fragmentation) ---
    g = merged.copy()
    dt = g["date"]
    month = dt.dt.month.astype(float)
    woy = dt.dt.isocalendar().week.astype(float)
    visits = g["visits"]

    w = args.roll
    roll_mean = visits.rolling(window=w, min_periods=1).mean()
    roll_std = visits.rolling(window=w, min_periods=2).std().fillna(0.0)
    logdiff = np.log1p(visits).diff().fillna(0.0)
    denom = roll_std.replace(0.0, np.nan).fillna(1e-6)
    z = (visits - roll_mean) / denom

    extra = pd.DataFrame(
        {
            "m_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
            "m_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
            "woy_sin": np.sin(2 * np.pi * (woy - 1) / 52.0),
            "woy_cos": np.cos(2 * np.pi * (woy - 1) / 52.0),
            "week_start_dow": dt.dt.dayofweek / 6.0 - 0.5,
            "visits_roll_mean": roll_mean,
            "visits_roll_std": roll_std,
            "visits_logdiff": logdiff,
            "visits_z": z,
            "OT": visits,
        }
    )
    g = pd.concat([g, extra], axis=1)

    h_cols = [c for c in g.columns if c.startswith("h_")]
    feat_cols = (
        h_cols
        + ["m_sin", "m_cos", "woy_sin", "woy_cos", "week_start_dow", "visitors"]
        + ["visits_roll_mean", "visits_roll_std", "visits_logdiff", "visits_z"]
    )
    out_df = g[["date"] + feat_cols + ["OT"]].copy()
    out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d")
    out_df.to_csv(out, index=False)

    n_channels = len([c for c in out_df.columns if c != "date"])
    print(
        f"Wrote {out}\n"
        f"  weeks={len(out_df)}  hour_slots={d_max}  skipped_empty_hour_fields~={n_bad}\n"
        f"  channels (non-date columns) = {n_channels}  → set enc_in/dec_in/c_out per --features and --target\n"
        f"  date_range=[{d0}, {d1}]"
    )


if __name__ == "__main__":
    main()
