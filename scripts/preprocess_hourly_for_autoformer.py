"""
Hourly-scale preprocessing for Autoformer / Transformer models.

Builds an hourly time series (one row per hour) by expanding SafeGraph-style
weekly vectors `VISITS_BY_EACH_HOUR` (length often 168) into true hourly
timestamps and summing across all POIs for each hour.

Output CSV format (Dataset_Custom-compatible):
  - first column: `date` (hourly timestamp)
  - last column:  `OT`   (target; here OT = hourly visits)
  - middle columns: time features + rolling stats

Usage:
  python scripts/preprocess_hourly_for_autoformer.py
  python scripts/preprocess_hourly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31

Requires: pandas, numpy, pyarrow (for batched Parquet read).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

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


def _accumulate_hourly_vectors_by_week(
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


def _expand_week_vectors_to_hourly(acc: dict[pd.Timestamp, np.ndarray]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for week_start, vec in sorted(acc.items(), key=lambda x: x[0]):
        d = int(vec.size)
        dt = week_start + pd.to_timedelta(np.arange(d, dtype=np.int64), unit="h")
        rows.append(pd.DataFrame({"date": dt, "visits": vec.astype(np.float64)}))
    if not rows:
        return pd.DataFrame(columns=["date", "visits"])
    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet",
    )
    parser.add_argument(
        "--output",
        default=r"E:\Urban Computing Final Project\Try_0412\data\autoformer_hourly_preprocessed.csv",
    )
    parser.add_argument("--date-start", default="2025-01-01", help="Inclusive lower bound on week-start date.")
    parser.add_argument("--date-end", default="2025-12-31", help="Inclusive upper bound on week-start date.")
    parser.add_argument("--hour-col", default="VISITS_BY_EACH_HOUR")
    parser.add_argument("--roll-hours", type=int, default=168, help="Rolling window in hours (default: 7 days).")
    parser.add_argument("--batch-rows", type=int, default=50_000, help="PyArrow batch size.")
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Accumulate city-wide weekly hour vectors
    acc, n_bad = _accumulate_hourly_vectors_by_week(
        src, args.date_start, args.date_end, args.hour_col, args.batch_rows
    )
    if not acc:
        raise SystemExit(
            f"No hourly vectors parsed (--hour-col={args.hour_col!r}). "
            f"Skipped/empty rows in batches: {n_bad}. Check column name and format."
        )

    # 2) Expand to hourly timestamps
    base = _expand_week_vectors_to_hourly(acc)

    # 3) Time features + rolling stats (build in one concat to avoid fragmentation)
    dt = pd.to_datetime(base["date"])
    hour = dt.dt.hour.astype(float)
    dow = dt.dt.dayofweek.astype(float)  # 0=Mon..6=Sun
    month = dt.dt.month.astype(float)

    visits = base["visits"].astype(np.float64)
    w = int(args.roll_hours)
    roll_mean = visits.rolling(window=w, min_periods=1).mean()
    roll_std = visits.rolling(window=w, min_periods=2).std().fillna(0.0)
    logdiff = np.log1p(visits).diff().fillna(0.0)
    denom = roll_std.replace(0.0, np.nan).fillna(1e-6)
    z = (visits - roll_mean) / denom

    extra = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "m_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
            "m_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
            "visits_roll_mean": roll_mean,
            "visits_roll_std": roll_std,
            "visits_logdiff": logdiff,
            "visits_z": z,
            "OT": visits,
        }
    )

    out_df = pd.concat([pd.DataFrame({"date": dt}), extra], axis=1)
    out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out_df.to_csv(out, index=False)

    print(
        f"Wrote {out}\n"
        f"  hours={len(out_df)}  skipped_empty_hour_fields~={n_bad}\n"
        f"  date_range_week_starts=[{args.date_start}, {args.date_end}]\n"
        f"  hour_slots_per_week~=max({max(v.size for v in acc.values())}, ...)"
    )


if __name__ == "__main__":
    main()

