from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _slope_per_grid(df: pd.DataFrame, value_col: str) -> pd.Series:
    """
    Fit a simple linear slope (value ~ t) per grid_id using ordinary least squares.
    Returns slope in 'value units per day' (since t is days from min date).
    """
    out = {}
    for grid_id, g in df.groupby("grid_id", sort=False):
        g = g.sort_values("date")
        y = g[value_col].to_numpy(dtype=float)
        if len(y) < 2 or not np.isfinite(y).all():
            out[grid_id] = np.nan
            continue
        t = (g["date"] - g["date"].min()).dt.days.to_numpy(dtype=float)
        if np.allclose(t, t[0]):
            out[grid_id] = np.nan
            continue
        # slope from polyfit degree 1: y = a*t + b
        a = float(np.polyfit(t, y, 1)[0])
        out[grid_id] = a
    return pd.Series(out, name=f"slope_{value_col}_per_day")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Top-100 grid weekly predictions into QGIS-friendly static metrics.")
    parser.add_argument(
        "--pred-by-date",
        default="use_official_autoformer_grid/outputs/grid_top100_weekly_visits_all_2025_pred_by_date.csv",
        help="Predictions table (one row per grid_id,date).",
    )
    parser.add_argument(
        "--manifest",
        default="use_official_autoformer_grid/data/grid_weekly_top100_visits/grid_topk_manifest.csv",
        help="Grid manifest with coordinates.",
    )
    parser.add_argument(
        "--out",
        default="use_official_autoformer_grid/outputs/grid_top100_summary_for_qgis.csv",
        help="Output summary CSV (one row per grid_id).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pred_path = (repo_root / args.pred_by_date).resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise SystemExit(f"Missing predictions: {pred_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    pred = pd.read_csv(pred_path, parse_dates=["date"])
    need = {"grid_id", "date", "y_true", "y_pred_last", "y_pred_mean"}
    miss = need - set(pred.columns)
    if miss:
        raise SystemExit(f"Predictions missing columns: {sorted(miss)}")

    pred["abs_err_last"] = (pred["y_pred_last"] - pred["y_true"]).abs()
    pred["sq_err_last"] = (pred["y_pred_last"] - pred["y_true"]) ** 2

    # Static summaries
    summary = (
        pred.groupby("grid_id", as_index=False)
        .agg(
            n_weeks=("date", "nunique"),
            mean_true=("y_true", "mean"),
            mean_pred_last=("y_pred_last", "mean"),
            mean_pred_mean=("y_pred_mean", "mean"),
            std_true=("y_true", "std"),
            std_pred_last=("y_pred_last", "std"),
            mae_last=("abs_err_last", "mean"),
            rmse_last=("sq_err_last", lambda x: float(np.sqrt(np.mean(x)))),
        )
        .sort_values("mean_true", ascending=False)
    )

    # Trend slopes (per day). For "per week" slope, multiply by 7 in QGIS or Excel.
    slope_true = _slope_per_grid(pred, "y_true").reset_index().rename(columns={"index": "grid_id"})
    slope_pred = _slope_per_grid(pred, "y_pred_last").reset_index().rename(columns={"index": "grid_id"})
    summary = summary.merge(slope_true, on="grid_id", how="left").merge(slope_pred, on="grid_id", how="left")

    # Attach coordinates for easy mapping (point) + join key.
    man = pd.read_csv(manifest_path)
    keep_man = [c for c in ["grid_id", "gx", "gy", "cell_lon", "cell_lat", "visits_2025", "visitors_2025"] if c in man.columns]
    summary = summary.merge(man[keep_man], on="grid_id", how="left")

    # Reorder columns (QGIS friendly)
    front = ["grid_id", "cell_lon", "cell_lat", "gx", "gy"]
    cols = front + [c for c in summary.columns if c not in front]
    summary = summary[cols]

    summary.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print("Preview:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

