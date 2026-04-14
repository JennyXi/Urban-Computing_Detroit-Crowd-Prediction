"""
Parse list-of-hours strings (e.g. popularity_by_each_hour / VISITS_BY_EACH_HOUR),
sum vectors city-wide by week, export a WIDE CSV: one row per week, h_0..h_{D-1}.

Use as multivariate input for Autoformer (--freq W): each hour slot is one channel.

Usage:
  python scripts/build_hourly_profile_weekly.py --hour-col popularity_by_each_hour
  python scripts/build_hourly_profile_weekly.py --hour-col VISITS_BY_EACH_HOUR --date-start 2025-01-01 --date-end 2025-12-31
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_vec(val) -> np.ndarray | None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"E:\Urban Computing Final Project\Try_0412\data\detroit_filtered.parquet",
    )
    parser.add_argument(
        "--output",
        default=r"E:\Urban Computing Final Project\Try_0412\data\autoformer_weekly_2025_hourly24_sample.csv",
    )
    parser.add_argument(
        "--hour-col",
        default="VISITS_BY_EACH_HOUR",
        help="Column with a list/JSON string of counts per hour (or 7x24=168 values).",
    )
    parser.add_argument("--date-start", default="2025-01-01")
    parser.add_argument("--date-end", default="2025-12-31")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    usecols = ["DATE_RANGE_START", args.hour_col]
    try:
        df = pd.read_parquet(inp, columns=usecols)
    except Exception as e:
        raise SystemExit(
            f"Failed to read columns {usecols} from {inp}: {e}\n"
            f"Install pyarrow if needed: pip install pyarrow"
        ) from e

    df["DATE_RANGE_START"] = pd.to_datetime(df["DATE_RANGE_START"])
    d0, d1 = pd.Timestamp(args.date_start), pd.Timestamp(args.date_end)
    df = df[(df["DATE_RANGE_START"].dt.normalize() >= d0.normalize()) & (df["DATE_RANGE_START"].dt.normalize() <= d1.normalize())]

    week = df["DATE_RANGE_START"].dt.normalize()
    col = df[args.hour_col]

    acc: dict[pd.Timestamp, np.ndarray] = {}
    lengths: list[int] = []

    for w, v in zip(week, col):
        arr = _parse_vec(v)
        if arr is None or arr.size == 0:
            continue
        lengths.append(int(arr.size))
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

    if not acc:
        raise SystemExit(f"No rows parsed for --hour-col {args.hour_col!r}. Check column name and string format.")

    uniq_lens = sorted(set(lengths))
    print("Parsed vector lengths (sample of rows):", uniq_lens[:10], "...", "n_lengths=", len(uniq_lens))

    rows = sorted(acc.items(), key=lambda x: x[0])
    D = max(a.size for _, a in rows)
    # pad shorter weeks with zeros (rare length mismatch)
    mat = np.zeros((len(rows), D), dtype=np.float64)
    for i, (_, a) in enumerate(rows):
        mat[i, : a.size] = a

    out_df = pd.DataFrame(mat, columns=[f"h_{j}" for j in range(D)])
    out_df.insert(0, "date", [t.strftime("%Y-%m-%d") for t, _ in rows])

    # Target: total hourly mass (sum of slots) — place last for MS as OT
    out_df["OT"] = mat.sum(axis=1)

    out_df.to_csv(out, index=False)
    print(f"Wrote {out}  weeks={len(out_df)}  hour_slots={D}  hour_col={args.hour_col!r}")


if __name__ == "__main__":
    main()
