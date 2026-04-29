"""
Build a performance matrix for weekly crowd prediction + POI alignment.

Outputs:
1) performance_matrix.csv: combined table (crowd rows + one alignment row)
2) crowd_model_ranking.csv: crowd-only ranking by rmse (asc)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_crowd_rows(repo_root: Path, crowd_metrics_glob: str) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(repo_root.glob(crowd_metrics_glob)):
        d = _read_json(p)
        exp_name = p.parent.parent.name  # .../<exp>/validation/overall_metrics.json
        rows.append(
            {
                "component": "crowd_prediction",
                "experiment": exp_name,
                "source_file": str(p.resolve()),
                "n": d.get("n"),
                "mae": d.get("mae"),
                "rmse": d.get("rmse"),
                "mape_pct": d.get("mape_pct"),
                "smape_pct": d.get("smape_pct"),
                "r2": d.get("r2"),
                "corr": d.get("corr"),
                "bias": d.get("bias"),
            }
        )
    return rows


def _collect_alignment_row(repo_root: Path, coefs_json: Path, summary_json: Path) -> dict:
    c = _read_json(coefs_json)
    s = _read_json(summary_json)
    fm = c.get("fit_metrics", {})
    ac = s.get("action_counts", {})
    return {
        "component": "poi_alignment",
        "experiment": "alignment_main",
        "source_file": str(coefs_json.resolve()),
        "date_start": c.get("date_start"),
        "date_end": c.get("date_end"),
        "n": c.get("n_grids"),
        "mae": fm.get("mae"),
        "rmse": fm.get("rmse"),
        "mape_pct": (fm.get("mape") * 100.0) if fm.get("mape") is not None else None,
        "smape_pct": (fm.get("smape") * 100.0) if fm.get("smape") is not None else None,
        "r2": fm.get("r2"),
        "corr": None,
        "bias": None,
        "medae": fm.get("medae"),
        "ridge_alpha": c.get("ridge_alpha"),
        "target_log1p": c.get("target_log1p"),
        "log1p_features": c.get("log1p_features"),
        "pred_col": c.get("pred_col"),
        "tier1_actionable": ac.get("tier1_actionable"),
        "tier2_watchlist": ac.get("tier2_watchlist"),
        "high_positive_q": s.get("high_positive_quantile"),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    parser = argparse.ArgumentParser(description="Build performance matrix for crowd+alignment pipeline.")
    parser.add_argument(
        "--crowd-metrics-glob",
        default="panel_training_0426/**/validation/overall_metrics.json",
        help="Glob (relative to repo root) for crowd model overall_metrics.json files.",
    )
    parser.add_argument(
        "--alignment-coefs-json",
        default=str(here / "alignment_ridge_coefs_oct_dec_2025.json"),
    )
    parser.add_argument(
        "--alignment-summary-json",
        default=str(here / "alignment_summary_oct_dec_2025.json"),
    )
    parser.add_argument(
        "--out-matrix-csv",
        default=str(here / "performance_matrix.csv"),
    )
    parser.add_argument(
        "--out-crowd-ranking-csv",
        default=str(here / "crowd_model_ranking.csv"),
    )
    args = parser.parse_args()

    coefs_path = Path(args.alignment_coefs_json)
    if not coefs_path.is_absolute():
        coefs_path = (repo_root / coefs_path).resolve()
    summary_path = Path(args.alignment_summary_json)
    if not summary_path.is_absolute():
        summary_path = (repo_root / summary_path).resolve()
    out_matrix = Path(args.out_matrix_csv)
    if not out_matrix.is_absolute():
        out_matrix = (repo_root / out_matrix).resolve()
    out_rank = Path(args.out_crowd_ranking_csv)
    if not out_rank.is_absolute():
        out_rank = (repo_root / out_rank).resolve()

    crowd_rows = _collect_crowd_rows(repo_root, args.crowd_metrics_glob)
    if not crowd_rows:
        raise SystemExit("No crowd metrics found. Check --crowd-metrics-glob.")
    align_row = _collect_alignment_row(repo_root, coefs_path, summary_path)

    matrix = pd.DataFrame(crowd_rows + [align_row])
    crowd_df = pd.DataFrame(crowd_rows).sort_values(["rmse", "mae"], ascending=[True, True]).reset_index(drop=True)
    crowd_df.insert(0, "rank_rmse", range(1, len(crowd_df) + 1))

    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    out_rank.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_matrix, index=False)
    crowd_df.to_csv(out_rank, index=False)

    print(f"Wrote: {out_matrix}")
    print(f"Wrote: {out_rank}")
    print(f"Crowd models: {len(crowd_rows)}  + alignment row: 1")


if __name__ == "__main__":
    main()

