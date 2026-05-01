"""
Evaluate exported panel predictions (daily or weekly long format).

Reads CSV with columns at least: grid_id, y_true, y_pred.
Optional: horizon (rolling 7d export) or stitched_day (28d stitch).

Writes:
  - eval_summary.txt       — human-readable global + by-horizon / by-week
  - eval_per_grid.csv      — one row per grid_id: MAE, RMSE, bias, n, MAPE-safe
  - eval_grids_best.csv    — top-K grids by lowest MAE (best predicted)
  - eval_grids_worst.csv   — top-K grids by highest MAE (hardest)

Example (CMD):

  .venv\\Scripts\\python.exe panel_training_0426\\evaluate_panel_predictions.py ^
    --pred-csv daily_training_0430/panel_pred_test_2025_long.csv ^
    --out-dir daily_training_0430/eval --top-k 15 --exclude-worst-grids-quantile 0.05
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.abs(y_true) > 1e-6
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs((y_pred[m] - y_true[m]) / y_true[m])) * 100.0)


def _r2_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    num = float(np.sum(np.abs(y_pred - y_true)))
    den = float(np.sum(np.abs(y_true)))
    if den <= 0:
        return float("nan")
    return num / den * 100.0


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    e = y_pred - y_true
    ae = np.abs(e)
    return {
        "n": float(len(y_true)),
        "MAE": float(np.mean(ae)),
        "RMSE": float(math.sqrt(np.mean(e**2))),
        "median_AE": float(np.median(ae)),
        "p90_AE": float(np.percentile(ae, 90)),
        "p95_AE": float(np.percentile(ae, 95)),
        "bias_mean_pred_minus_true": float(np.mean(e)),
        "MAPE_pct_y_true_gt0": _safe_mape(y_true, y_pred),
        "WAPE_pct": _wape(y_true, y_pred),
        "R2_pooled": _r2_pooled(y_true, y_pred),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-csv", required=True, help="Long or stitched predictions CSV")
    p.add_argument("--out-dir", required=True, help="Directory for eval outputs")
    p.add_argument("--top-k", type=int, default=10, help="How many best/worst grids to list")
    p.add_argument(
        "--exclude-worst-grids-quantile",
        type=float,
        default=None,
        help="If set (e.g. 0.05), recompute global metrics after removing that fraction of grids with highest per-grid MAE.",
    )
    args = p.parse_args()

    pred_path = Path(args.pred_csv)
    if not pred_path.exists():
        raise SystemExit(f"Missing --pred-csv: {pred_path}")

    df = pd.read_csv(pred_path)
    need = {"grid_id", "y_true", "y_pred"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Missing columns: {sorted(miss)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = df["y_true"].to_numpy(dtype=np.float64)
    y_pred = df["y_pred"].to_numpy(dtype=np.float64)
    global_m = _metrics_block(y_true, y_pred)

    lines: list[str] = []
    lines.append(f"Input: {pred_path.resolve()}")
    lines.append(f"Rows: {len(df)}  Grids: {df['grid_id'].nunique()}")
    lines.append("")
    lines.append("=== Global (all rows pooled) ===")
    for k, v in global_m.items():
        if k == "n":
            lines.append(f"  {k}: {int(v)}")
        else:
            lines.append(f"  {k}: {v:.6g}" if not math.isnan(v) else f"  {k}: nan")

    if "horizon" in df.columns:
        lines.append("")
        lines.append("=== By horizon (if present) ===")
        for h, sub in df.groupby("horizon", sort=True):
            m = _metrics_block(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
            lines.append(f"  horizon={h}: MAE={m['MAE']:.6g}  RMSE={m['RMSE']:.6g}  n={int(m['n'])}")

    if "stitched_day" in df.columns:
        lines.append("")
        lines.append("=== By stitched_day (if present) ===")
        for d, sub in df.groupby("stitched_day", sort=True):
            m = _metrics_block(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
            lines.append(f"  day={d}: MAE={m['MAE']:.6g}  n={int(m['n'])}")
        df["_week"] = ((df["stitched_day"].astype(int) - 1) // 7) + 1
        lines.append("")
        lines.append("=== By week index (stitched 1–7 => week 1, etc.) ===")
        for w, sub in df.groupby("_week", sort=True):
            m = _metrics_block(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
            lines.append(f"  week={int(w)}: MAE={m['MAE']:.6g}  RMSE={m['RMSE']:.6g}  n={int(m['n'])}")

    # Per-grid
    rows = []
    for gid, sub in df.groupby("grid_id", sort=False):
        yt = sub["y_true"].to_numpy(dtype=np.float64)
        yp = sub["y_pred"].to_numpy(dtype=np.float64)
        m = _metrics_block(yt, yp)
        rows.append({"grid_id": gid, **m})

    per_grid = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    per_grid_path = out_dir / "eval_per_grid.csv"
    per_grid.to_csv(per_grid_path, index=False)

    if args.exclude_worst_grids_quantile is not None and float(args.exclude_worst_grids_quantile) > 0:
        q = float(args.exclude_worst_grids_quantile)
        n_cut = max(1, int(round(len(per_grid) * q)))
        drop_ids = set(per_grid.nlargest(n_cut, "MAE")["grid_id"].astype(str))
        mask = ~df["grid_id"].astype(str).isin(drop_ids)
        df_core = df.loc[mask]
        m_core = _metrics_block(df_core["y_true"].to_numpy(), df_core["y_pred"].to_numpy())
        lines.append("")
        lines.append(
            f"=== Global excluding worst {q * 100:.1f}% grids by MAE ({n_cut} grids, ~{100 * (1 - q):.0f}% grids kept) ==="
        )
        for k2, v in m_core.items():
            if k2 == "n":
                lines.append(f"  {k2}: {int(v)}")
            else:
                lines.append(f"  {k2}: {v:.6g}" if not math.isnan(v) else f"  {k2}: nan")
        lines.append(f"  dropped_grid_ids: {sorted(drop_ids)}")

    k = max(1, int(args.top_k))
    best = per_grid.head(k).copy()
    worst = per_grid.sort_values("MAE", ascending=False).head(k).copy()
    best.to_csv(out_dir / "eval_grids_best.csv", index=False)
    worst.to_csv(out_dir / "eval_grids_worst.csv", index=False)

    lines.append("")
    lines.append(f"=== Best {k} grids (lowest MAE) ===")
    for _, r in best.iterrows():
        lines.append(f"  {r['grid_id']!s}  MAE={r['MAE']:.6g}  RMSE={r['RMSE']:.6g}  n={int(r['n'])}")

    lines.append("")
    lines.append(f"=== Worst {k} grids (highest MAE) ===")
    for _, r in worst.iterrows():
        lines.append(f"  {r['grid_id']!s}  MAE={r['MAE']:.6g}  RMSE={r['RMSE']:.6g}  n={int(r['n'])}")

    summary_path = out_dir / "eval_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {per_grid_path}")
    print(f"Wrote: {out_dir / 'eval_grids_best.csv'}")
    print(f"Wrote: {out_dir / 'eval_grids_worst.csv'}")


if __name__ == "__main__":
    main()
