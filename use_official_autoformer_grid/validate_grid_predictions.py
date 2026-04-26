from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


PredCol = Literal["y_pred_last", "y_pred_mean"]


@dataclass(frozen=True)
class Metrics:
    n: int
    mae: float
    rmse: float
    mape_pct: float
    smape_pct: float
    r2: float
    corr: float
    bias: float


def _safe_div(numer: np.ndarray, denom: np.ndarray, eps: float) -> np.ndarray:
    return numer / np.maximum(np.abs(denom), eps)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> Metrics:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask].astype(float, copy=False)
    yp = y_pred[mask].astype(float, copy=False)
    n = int(yt.size)
    if n == 0:
        return Metrics(n=0, mae=np.nan, rmse=np.nan, mape_pct=np.nan, smape_pct=np.nan, r2=np.nan, corr=np.nan, bias=np.nan)

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(_safe_div(err, yt, eps))) * 100.0)
    smape = float(np.mean(2.0 * np.abs(err) / np.maximum(np.abs(yt) + np.abs(yp), eps)) * 100.0)

    yt_mean = float(np.mean(yt))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    corr = float(np.corrcoef(yt, yp)[0, 1]) if n >= 2 else np.nan
    bias = float(np.mean(err))

    return Metrics(n=n, mae=mae, rmse=rmse, mape_pct=mape, smape_pct=smape, r2=r2, corr=corr, bias=bias)


def _row_level_fields(df: pd.DataFrame, eps: float, ytrue_min_for_ape: float) -> pd.DataFrame:
    out = df.copy()
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_err"] = out["error"].abs()
    denom = out["y_true"].abs().clip(lower=float(eps))
    out["ape_pct"] = (out["abs_err"] / denom) * 100.0
    # mask APE where y_true is too small (so we don't call tiny denominators "anomalies")
    out.loc[out["y_true"].abs() < float(ytrue_min_for_ape), "ape_pct"] = np.nan
    denom2 = (out["y_true"].abs() + out["y_pred"].abs()).clip(lower=float(eps))
    out["smape_pct"] = (2.0 * out["abs_err"] / denom2) * 100.0
    return out


def _weekly_gap_diagnostics(dates: pd.Series) -> dict:
    """
    For weekly data, check spacing between consecutive dates.
    Returns gap stats in days. If dates are irregular or missing weeks, max_gap_days will exceed 7.
    """
    d = pd.to_datetime(dates).dropna().sort_values().unique()
    if len(d) <= 1:
        return {
            "n_dates": int(len(d)),
            "min_date": (pd.Timestamp(d[0]).strftime("%Y-%m-%d") if len(d) == 1 else None),
            "max_date": (pd.Timestamp(d[0]).strftime("%Y-%m-%d") if len(d) == 1 else None),
            "max_gap_days": np.nan,
            "n_gaps_gt_7": 0,
            "expected_n_weeks": (int(len(d)) if len(d) == 1 else np.nan),
            "missing_weeks_est": 0,
        }
    gaps = np.diff(d).astype("timedelta64[D]").astype(int)
    max_gap = int(np.max(gaps)) if len(gaps) else 0
    n_gt7 = int(np.sum(gaps > 7))
    min_date = pd.Timestamp(d[0])
    max_date = pd.Timestamp(d[-1])
    expected = int(((max_date - min_date).days // 7) + 1)
    missing = int(max(expected - len(d), 0))
    return {
        "n_dates": int(len(d)),
        "min_date": min_date.strftime("%Y-%m-%d"),
        "max_date": max_date.strftime("%Y-%m-%d"),
        "max_gap_days": float(max_gap),
        "n_gaps_gt_7": int(n_gt7),
        "expected_n_weeks": int(expected),
        "missing_weeks_est": int(missing),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate exported Top-100 grid predictions by comparing y_pred_* to y_true and writing metrics tables."
    )
    parser.add_argument(
        "--pred-by-date",
        default="use_official_autoformer_grid/outputs/grid_top100_weekly_visits_all_2025_pred_by_date.csv",
        help="Predictions table with columns grid_id,date,y_true,y_pred_last,y_pred_mean.",
    )
    parser.add_argument(
        "--pred-col",
        default="y_pred_last",
        choices=["y_pred_last", "y_pred_mean"],
        help="Which prediction column to validate.",
    )
    parser.add_argument(
        "--out-dir",
        default="use_official_autoformer_grid/outputs/validation",
        help="Output directory for metrics artifacts.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="Small constant to stabilize (s)MAPE when y_true is near zero.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many worst/best grids and dates to include in anomaly summaries.",
    )
    parser.add_argument(
        "--ytrue-min-for-ape",
        type=float,
        default=1000.0,
        help="Only compute APE/MAPE-style %% errors when |y_true| >= this threshold.",
    )
    parser.add_argument(
        "--good-nrmse",
        type=float,
        default=0.10,
        help="NRMSE threshold for labeling a grid as 'good'.",
    )
    parser.add_argument(
        "--bad-nrmse",
        type=float,
        default=0.20,
        help="NRMSE threshold for labeling a grid as 'bad' (above this is bad).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pred_path = (repo_root / args.pred_by_date).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise SystemExit(f"Missing predictions table: {pred_path}")

    df = pd.read_csv(pred_path, parse_dates=["date"])
    need = {"grid_id", "date", "y_true", "y_pred_last", "y_pred_mean"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Predictions missing columns: {sorted(miss)}")

    pred_col: PredCol = args.pred_col
    df = df[["grid_id", "date", "y_true", pred_col]].rename(columns={pred_col: "y_pred"}).copy()
    df["grid_id"] = df["grid_id"].astype(str)
    df = _row_level_fields(df, eps=float(args.eps), ytrue_min_for_ape=float(args.ytrue_min_for_ape))

    # --- Overall metrics (all rows) ---
    overall = _metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), eps=float(args.eps))

    # --- Per-grid metrics ---
    per_grid_rows: list[dict] = []
    for grid_id, g in df.groupby("grid_id", sort=False):
        m = _metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), eps=float(args.eps))
        per_grid_rows.append({"grid_id": grid_id, **asdict(m)})
    per_grid = pd.DataFrame(per_grid_rows).sort_values(["rmse", "mae"], ascending=[True, True])
    mean_true_by_grid = df.groupby("grid_id", as_index=False).agg(mean_true=("y_true", "mean"))

    # --- Per-date metrics (aggregate across grids per week) ---
    per_date_rows: list[dict] = []
    for dt, g in df.groupby("date", sort=True):
        m = _metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), eps=float(args.eps))
        per_date_rows.append({"date": pd.Timestamp(dt).strftime("%Y-%m-%d"), **asdict(m)})
    per_date = pd.DataFrame(per_date_rows).sort_values("date")

    # --- Optional: simple quarterly breakdown (Q1..Q4) ---
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    per_quarter_rows: list[dict] = []
    for q, g in df.groupby("quarter", sort=True):
        m = _metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), eps=float(args.eps))
        per_quarter_rows.append({"quarter": q, **asdict(m)})
    per_quarter = pd.DataFrame(per_quarter_rows).sort_values("quarter")

    # --- Diagnostics: data gaps / missing weeks per grid (weekly assumption) ---
    gap_rows: list[dict] = []
    for grid_id, g in df.groupby("grid_id", sort=False):
        gap = _weekly_gap_diagnostics(g["date"])
        gap_rows.append({"grid_id": grid_id, **gap})
    gaps = pd.DataFrame(gap_rows)

    # --- Anomaly cases table (row-level) ---
    # Pick the worst rows by absolute error and by smape (robust to scale).
    top_n = int(max(args.top_n, 1))
    worst_abs = df.sort_values("abs_err", ascending=False).head(top_n).copy()
    worst_abs["rank_by"] = "abs_err"
    worst_smape = df.sort_values("smape_pct", ascending=False).head(top_n).copy()
    worst_smape["rank_by"] = "smape_pct"
    anomaly_cases = (
        pd.concat([worst_abs, worst_smape], ignore_index=True)
        .drop_duplicates(subset=["grid_id", "date", "rank_by"])
        .sort_values(["rank_by", "abs_err"], ascending=[True, False])
    )

    # --- Top-N worst grids and worst dates for quick inspection ---
    worst_grids = per_grid.sort_values("rmse", ascending=False).head(top_n).copy()
    best_grids = per_grid.sort_values("rmse", ascending=True).head(top_n).copy()
    worst_dates = per_date.sort_values("rmse", ascending=False).head(top_n).copy()

    # --- Merge per-grid metrics + gaps so you can map "badness" + data quality together ---
    per_grid_diag = per_grid.merge(mean_true_by_grid, on="grid_id", how="left").merge(gaps, on="grid_id", how="left")
    # Add scale-normalized RMSE for cross-grid comparability
    per_grid_diag["nrmse"] = per_grid_diag["rmse"] / per_grid_diag["mean_true"].replace(0, np.nan)
    good_thr = float(args.good_nrmse)
    bad_thr = float(args.bad_nrmse)
    if not (0.0 < good_thr < bad_thr):
        raise SystemExit("--good-nrmse must be > 0 and < --bad-nrmse")

    def _band(x: float) -> str:
        if not np.isfinite(x):
            return "unknown"
        if x <= good_thr:
            return "good"
        if x >= bad_thr:
            return "bad"
        return "ok"

    per_grid_diag["quality_band"] = per_grid_diag["nrmse"].map(_band)

    # --- Write outputs ---
    (out_dir / "overall_metrics.json").write_text(json.dumps(asdict(overall), indent=2), encoding="utf-8")
    per_grid.to_csv(out_dir / "per_grid_metrics.csv", index=False)
    per_date.to_csv(out_dir / "per_date_metrics.csv", index=False)
    per_quarter.to_csv(out_dir / "per_quarter_metrics.csv", index=False)
    per_grid_diag.to_csv(out_dir / "per_grid_diagnostics.csv", index=False)
    anomaly_cases.to_csv(out_dir / "anomaly_cases.csv", index=False)
    worst_grids.to_csv(out_dir / "worst_grids_topN.csv", index=False)
    best_grids.to_csv(out_dir / "best_grids_topN.csv", index=False)
    worst_dates.to_csv(out_dir / "worst_dates_topN.csv", index=False)

    # Print a quick summary for terminal use
    print("Validation complete.")
    print(f"Input: {pred_path}")
    print(f"Using pred column: {pred_col}")
    print(f"Output dir: {out_dir}")
    print("Overall metrics:")
    print(json.dumps(asdict(overall), indent=2))
    print("")
    print("Best 5 grids by RMSE:")
    print(per_grid.head(5).to_string(index=False))
    print("")
    print("Worst 5 grids by RMSE:")
    print(per_grid.tail(5).to_string(index=False))
    print("")
    print(f"Wrote anomaly artifacts (top_n={top_n}):")
    print(f"- {out_dir / 'per_grid_diagnostics.csv'}")
    print(f"- {out_dir / 'anomaly_cases.csv'}")
    print(f"- {out_dir / 'worst_grids_topN.csv'}")
    print(f"- {out_dir / 'worst_dates_topN.csv'}")


if __name__ == "__main__":
    main()

