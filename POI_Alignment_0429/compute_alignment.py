"""
Crowd–POI alignment (Ridge): multi-week mean of y_pred_mean vs supply-only fit.

Defaults match POI_Alignment_0429 layout: predictions and POI parquet in this folder,
time window 2025-10-01 .. 2025-12-31, four super-category POI counts as features.
Default model uses alpha=0.1 with target log1p for robust alignment ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

POI_COUNT_COLS = [
    "poi_cnt_life",
    "poi_cnt_transport",
    "poi_cnt_economy",
    "poi_cnt_public_service",
]


def _fit_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    mape = float(np.mean(abs_err / np.maximum(np.abs(y_true), eps)))
    smape = float(np.mean(2.0 * abs_err / np.maximum(np.abs(y_true) + np.abs(y_pred), eps)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "medae": medae,
        "mape": mape,
        "smape": smape,
        "r2": r2,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Ridge alignment: mean y_pred vs POI counts.")
    parser.add_argument(
        "--pred-csv",
        type=str,
        default=str(here / "panel_pred_test_2025_by_date.csv"),
        help="Long table: grid_id, date, y_pred_mean (and optional y_true).",
    )
    parser.add_argument(
        "--poi-parquet",
        type=str,
        default=str(here / "grid100_poi_static_2024.parquet"),
        help="Static POI features keyed by grid_id.",
    )
    parser.add_argument("--date-start", type=str, default="2025-10-01")
    parser.add_argument("--date-end", type=str, default="2025-12-31")
    parser.add_argument(
        "--pred-col",
        type=str,
        default="y_pred_mean",
        choices=["y_pred_mean", "y_pred_last"],
    )
    parser.add_argument("--ridge-alpha", type=float, default=0.1, help="Ridge penalty (sklearn Ridge alpha).")
    parser.add_argument(
        "--log1p-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply log1p to the four POI count features (recommended).",
    )
    parser.add_argument(
        "--target-log1p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit Ridge on log1p(c_bar) and invert with expm1 for c_hat.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(here / "alignment_oct_dec_2025.csv"),
        help="Per-grid alignment table.",
    )
    parser.add_argument(
        "--out-coefs-json",
        type=str,
        default=str(here / "alignment_ridge_coefs_oct_dec_2025.json"),
        help="Ridge coefficients (on standardized features) + intercept.",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred_csv)
    if not pred_path.is_absolute():
        pred_path = (repo_root / pred_path).resolve()
    poi_path = Path(args.poi_parquet)
    if not poi_path.is_absolute():
        poi_path = (repo_root / poi_path).resolve()
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = (repo_root / out_csv).resolve()
    out_coef = Path(args.out_coefs_json)
    if not out_coef.is_absolute():
        out_coef = (repo_root / out_coef).resolve()

    if not pred_path.exists():
        raise SystemExit(
            f"Missing predictions CSV: {pred_path}\n"
            "Export with panel_training_0426/export_panel_predictions.py then copy or symlink "
            "panel_pred_test_2025_by_date.csv into POI_Alignment_0429, or pass --pred-csv."
        )
    if not poi_path.exists():
        raise SystemExit(f"Missing POI parquet: {poi_path}")

    d0 = pd.Timestamp(args.date_start).normalize()
    d1 = pd.Timestamp(args.date_end).normalize()

    pred = pd.read_csv(pred_path, parse_dates=["date"])
    need = {"grid_id", "date", str(args.pred_col)}
    miss = need - set(pred.columns)
    if miss:
        raise SystemExit(f"pred csv missing columns: {miss}")

    pred["grid_id"] = pred["grid_id"].astype(str)
    pred = pred[(pred["date"] >= d0) & (pred["date"] <= d1)].copy()
    if pred.empty:
        raise SystemExit(f"No prediction rows in [{d0.date()}, {d1.date()}]. Check dates and file.")

    cbar = (
        pred.groupby("grid_id", as_index=False)
        .agg(
            c_bar=(str(args.pred_col), "mean"),
            n_weeks=("date", "nunique"),
        )
        .rename(columns={"grid_id": "grid_id"})
    )
    if "y_true" in pred.columns:
        yt = pred.groupby("grid_id", as_index=False).agg(y_true_bar=("y_true", "mean"))
        cbar = cbar.merge(yt, on="grid_id", how="left")

    poi = pq.read_table(poi_path).to_pandas()
    if "grid_id" not in poi.columns:
        raise SystemExit("POI parquet must contain grid_id.")
    poi["grid_id"] = poi["grid_id"].astype(str)
    miss_poi = [c for c in POI_COUNT_COLS if c not in poi.columns]
    if miss_poi:
        raise SystemExit(f"POI parquet missing columns: {miss_poi}")

    poi_sub = poi[["grid_id"] + POI_COUNT_COLS].copy()
    for c in POI_COUNT_COLS:
        poi_sub[c] = pd.to_numeric(poi_sub[c], errors="coerce").fillna(0.0).astype(np.float64)

    m = cbar.merge(poi_sub, on="grid_id", how="left")
    for c in POI_COUNT_COLS:
        m[c] = m[c].fillna(0.0)

    X_raw = m[POI_COUNT_COLS].to_numpy(dtype=np.float64)
    if args.log1p_features:
        X = np.log1p(np.clip(X_raw, 0.0, None))
        feat_names = [f"log1p({c})" for c in POI_COUNT_COLS]
    else:
        X = X_raw
        feat_names = list(POI_COUNT_COLS)

    y = m["c_bar"].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise SystemExit("c_bar contains non-finite values.")
    y_fit = np.log1p(np.clip(y, 0.0, None)) if args.target_log1p else y

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(args.ridge_alpha), random_state=0)),
        ]
    )
    model.fit(X, y_fit)
    y_fit_hat = model.predict(X).astype(np.float64)
    c_hat = np.expm1(y_fit_hat) if args.target_log1p else y_fit_hat
    r = y - c_hat
    metrics = _fit_metrics(y_true=y, y_pred=c_hat)

    out = m[["grid_id", "c_bar", "n_weeks"] + POI_COUNT_COLS].copy()
    if "y_true_bar" in m.columns:
        out["y_true_bar"] = m["y_true_bar"]
    out["c_hat"] = c_hat
    out["r_alignment"] = r

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    ridge: Ridge = model.named_steps["ridge"]
    scaler: StandardScaler = model.named_steps["scaler"]
    coef_payload = {
        "date_start": str(d0.date()),
        "date_end": str(d1.date()),
        "pred_col": str(args.pred_col),
        "ridge_alpha": float(args.ridge_alpha),
        "log1p_features": bool(args.log1p_features),
        "target_log1p": bool(args.target_log1p),
        "feature_names": feat_names,
        "coef_scaled_input": [float(x) for x in ridge.coef_.ravel()],
        "intercept": float(ridge.intercept_),
        "scaler_mean": [float(x) for x in scaler.mean_.ravel()],
        "scaler_scale": [float(x) for x in scaler.scale_.ravel()],
        "fit_metrics": metrics,
        "n_grids": int(len(out)),
        "pred_csv": str(pred_path),
        "poi_parquet": str(poi_path),
    }
    out_coef.parent.mkdir(parents=True, exist_ok=True)
    out_coef.write_text(json.dumps(coef_payload, indent=2), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_coef}")
    print(f"Grids: {len(out)}  window: {d0.date()} .. {d1.date()}  alpha={args.ridge_alpha}")
    print(
        "Fit metrics: "
        f"RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, "
        f"MedAE={metrics['medae']:.3f}, SMAPE={metrics['smape']:.4f}, R2={metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
