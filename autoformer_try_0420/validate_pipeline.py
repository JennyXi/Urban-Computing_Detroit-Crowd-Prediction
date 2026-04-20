"""
Validate preprocessing logic and data correctness for the Autoformer datasets.

Checks:
- CSV schema: date column exists, OT exists, numeric columns parse
- No NaN/inf in numeric columns (after parsing)
- Time column monotonic increasing
- Hourly -> weekly consistency: sum(hourly OT in each week) ~= weekly OT

Usage:
  python autoformer_try_0420/validate_pipeline.py
  python autoformer_try_0420/validate_pipeline.py --hourly data/autoformer_hourly_preprocessed.csv --weekly data/autoformer_weekly_preprocessed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _finite_report(df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    problems: list[str] = []
    for c in numeric_cols:
        a = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        n_nan = int(np.isnan(a).sum())
        n_inf = int(np.isinf(a).sum())
        if n_nan or n_inf:
            problems.append(f"{c}: nan={n_nan} inf={n_inf}")
    return problems


def _week_start_monday(ts: pd.Series) -> pd.Series:
    # Monday-based week start. This matches the typical weekly patterns DATE_RANGE_START convention.
    dt = pd.to_datetime(ts)
    return (dt.dt.normalize() - pd.to_timedelta(dt.dt.dayofweek, unit="D")).dt.normalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Autoformer preprocessing outputs.")
    parser.add_argument(
        "--hourly",
        default=r"E:\Urban Computing Final Project\Try_0412\data\autoformer_hourly_preprocessed.csv",
        help="Hourly CSV path (date ... OT).",
    )
    parser.add_argument(
        "--weekly",
        default=r"E:\Urban Computing Final Project\Try_0412\data\autoformer_weekly_preprocessed.csv",
        help="Weekly CSV path (date ... OT).",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for consistency checks.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for consistency checks.")
    args = parser.parse_args()

    hourly_path = Path(args.hourly)
    weekly_path = Path(args.weekly)

    if not hourly_path.exists():
        raise SystemExit(f"Missing hourly CSV: {hourly_path}")
    if not weekly_path.exists():
        raise SystemExit(f"Missing weekly CSV: {weekly_path}")

    print("== Paths ==")
    print("hourly:", hourly_path)
    print("weekly:", weekly_path)

    # --- Load ---
    h = pd.read_csv(hourly_path)
    w = pd.read_csv(weekly_path)

    # --- Basic schema checks ---
    for name, df in [("hourly", h), ("weekly", w)]:
        if "date" not in df.columns:
            raise SystemExit(f"[FAIL] {name}: missing required column 'date'")
        if "OT" not in df.columns:
            raise SystemExit(f"[FAIL] {name}: missing required column 'OT'")
        print(f"\n== {name.upper()} summary ==")
        print("rows:", len(df), "cols:", len(df.columns))
        print("first cols:", list(df.columns[:8]), "..." if len(df.columns) > 8 else "")
        print("last cols:", list(df.columns[-8:]) if len(df.columns) >= 8 else list(df.columns))

    # --- Time parsing / monotonic ---
    h_dt = pd.to_datetime(h["date"], errors="coerce")
    w_dt = pd.to_datetime(w["date"], errors="coerce")
    if h_dt.isna().any():
        raise SystemExit(f"[FAIL] hourly: date parse failed for {int(h_dt.isna().sum())} rows")
    if w_dt.isna().any():
        raise SystemExit(f"[FAIL] weekly: date parse failed for {int(w_dt.isna().sum())} rows")

    if not h_dt.is_monotonic_increasing:
        print("[WARN] hourly: date is not monotonic increasing; sorting for downstream checks.")
    if not w_dt.is_monotonic_increasing:
        print("[WARN] weekly: date is not monotonic increasing; sorting for downstream checks.")

    # Avoid pandas fragmentation warnings on very wide frames by assigning once on a copy.
    h = h.copy()
    h["_dt"] = h_dt
    h = h.sort_values("_dt").reset_index(drop=True)

    w = w.copy()
    w["_dt"] = w_dt
    w = w.sort_values("_dt").reset_index(drop=True)

    # --- Finite checks for numeric columns ---
    h_num_cols = [c for c in h.columns if c not in ("date", "_dt")]
    w_num_cols = [c for c in w.columns if c not in ("date", "_dt")]

    h_bad = _finite_report(h, h_num_cols)
    w_bad = _finite_report(w, w_num_cols)

    if h_bad:
        print("\n[FAIL] hourly: non-finite values found:")
        for s in h_bad[:25]:
            print(" -", s)
        if len(h_bad) > 25:
            print(f" ... {len(h_bad) - 25} more columns")
        raise SystemExit(2)
    if w_bad:
        print("\n[FAIL] weekly: non-finite values found:")
        for s in w_bad[:25]:
            print(" -", s)
        if len(w_bad) > 25:
            print(f" ... {len(w_bad) - 25} more columns")
        raise SystemExit(2)

    print("\n[OK] numeric columns are finite (no NaN/inf).")

    # --- Hourly -> weekly consistency ---
    #
    # IMPORTANT:
    # - hourly `OT` is derived from VISITS_BY_EACH_HOUR (sum across POIs per hour)
    # - weekly `OT` is derived from VISIT_COUNTS (weekly totals)
    # These two totals are not guaranteed to match.
    #
    # Therefore we validate the "same-source" consistency:
    #   sum(hourly OT within week)  ~=  sum(weekly h_0..h_167)
    h_week = _week_start_monday(h["_dt"])
    h_weekly = (
        pd.DataFrame({"week_start": h_week, "OT": pd.to_numeric(h["OT"], errors="coerce").astype(np.float64)})
        .groupby("week_start", as_index=False)["OT"]
        .sum()
        .rename(columns={"OT": "OT_hourly_sum"})
    )

    w_week = w["_dt"].dt.normalize()
    # weekly hourly-mass from h_* columns (same source as hourly OT)
    h_cols = [c for c in w.columns if c.startswith("h_")]
    if not h_cols:
        raise SystemExit("[FAIL] weekly: no h_* columns found; cannot validate hourly-vector consistency.")
    weekly_hourly_mass = (
        w[h_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64, copy=False)
        .sum(axis=1)
    )
    w_weekly = pd.DataFrame(
        {
            "week_start": w_week,
            "hourly_mass_weekly": weekly_hourly_mass,
            "OT_weekly": pd.to_numeric(w["OT"], errors="coerce").astype(np.float64),
        }
    )

    merged = h_weekly.merge(w_weekly, on="week_start", how="inner")
    if merged.empty:
        raise SystemExit(
            "[FAIL] No overlapping weeks between hourly-expanded data and weekly data. "
            "Check date ranges and week-start conventions."
        )

    diff = merged["OT_hourly_sum"] - merged["hourly_mass_weekly"]
    abs_diff = diff.abs()
    rel_denom = merged["hourly_mass_weekly"].abs().replace(0.0, np.nan)
    rel_diff = (abs_diff / rel_denom).fillna(0.0)

    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())
    worst_idx = int(abs_diff.idxmax())
    worst = merged.loc[worst_idx]

    print("\n== Consistency: sum(hourly OT) vs sum(weekly h_*) ==")
    print("overlapping weeks:", len(merged))
    print("max_abs_diff:", max_abs)
    print("max_rel_diff:", max_rel)
    print(
        "worst_week:",
        str(pd.Timestamp(worst["week_start"]).date()),
        "hourly_sum=",
        float(worst["OT_hourly_sum"]),
        "weekly_h_sum=",
        float(worst["hourly_mass_weekly"]),
        "abs_diff=",
        float(abs(worst["OT_hourly_sum"] - worst["hourly_mass_weekly"])),
    )

    ok = bool((abs_diff <= float(args.atol) + float(args.rtol) * merged["OT_weekly"].abs()).all())
    if ok:
        print("\n[OK] Hourly->weekly totals match within tolerances.")
    else:
        print("\n[WARN] Hourly->weekly totals do NOT match within tolerances.")
        print("This likely means a week-start alignment issue or inconsistent hour-slot length.")

    # Secondary report: weekly OT (VISIT_COUNTS) vs hourly-mass (sum of h_*) are not guaranteed to match.
    delta_ot = (merged["OT_weekly"] - merged["hourly_mass_weekly"]).abs()
    rel_ot = (delta_ot / merged["OT_weekly"].abs().replace(0.0, np.nan)).fillna(0.0)
    print("\n== Additional context: weekly OT vs sum(weekly h_*) ==")
    print("note: these can differ because they come from different source columns (VISIT_COUNTS vs VISITS_BY_EACH_HOUR).")
    print("max_abs_diff:", float(delta_ot.max()))
    print("max_rel_diff:", float(rel_ot.max()))

    print("\n== Done ==")


if __name__ == "__main__":
    main()

