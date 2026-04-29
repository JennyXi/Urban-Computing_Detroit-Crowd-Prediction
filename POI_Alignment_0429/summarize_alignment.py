"""
Summarize alignment outputs for planning interpretation.

Reads per-grid alignment CSV from compute_alignment.py and writes:
1) overall summary stats JSON
2) top positive/negative mismatch CSVs
3) high-positive candidates with category priority suggestions
   (both absolute-count scarcity and within-high-group quantile scarcity)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POI_COUNT_COLS = [
    "poi_cnt_life",
    "poi_cnt_transport",
    "poi_cnt_economy",
    "poi_cnt_public_service",
]

COL2LABEL = {
    "poi_cnt_life": "life",
    "poi_cnt_transport": "transport",
    "poi_cnt_economy": "economy",
    "poi_cnt_public_service": "public_service",
}


def _safe_quantile(s: pd.Series, q: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    return float(x.quantile(q))


def _priority_from_scores(score_df: pd.DataFrame) -> tuple[list[int], list[int]]:
    arr = score_df.to_numpy(dtype=float)
    best_idx = np.argmax(arr, axis=1)
    second_idx = np.argsort(-arr, axis=1)[:, 1]
    return best_idx.tolist(), second_idx.tolist()


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Summarize alignment results and suggest POI category priorities.")
    parser.add_argument(
        "--alignment-csv",
        type=str,
        default=str(here / "alignment_oct_dec_2025.csv"),
        help="Output CSV from compute_alignment.py",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Rows to keep for top positive/negative mismatch.")
    parser.add_argument(
        "--high-positive-quantile",
        type=float,
        default=0.8,
        help="Quantile threshold for high-positive mismatch candidates.",
    )
    parser.add_argument(
        "--out-summary-json",
        type=str,
        default=str(here / "alignment_summary_oct_dec_2025.json"),
    )
    parser.add_argument(
        "--out-top-positive-csv",
        type=str,
        default=str(here / "alignment_top_positive_oct_dec_2025.csv"),
    )
    parser.add_argument(
        "--out-top-negative-csv",
        type=str,
        default=str(here / "alignment_top_negative_oct_dec_2025.csv"),
    )
    parser.add_argument(
        "--out-priority-csv",
        type=str,
        default=str(here / "alignment_priority_candidates_oct_dec_2025.csv"),
    )
    parser.add_argument("--gate-r-quantile", type=float, default=0.8, help="Gate: r_alignment must be >= this quantile.")
    parser.add_argument("--gate-min-weeks", type=int, default=12, help="Gate: minimum n_weeks.")
    parser.add_argument("--gate-cbar-quantile", type=float, default=0.5, help="Gate: c_bar must be >= this quantile.")
    parser.add_argument(
        "--gate-scarcity-q-min",
        type=float,
        default=0.7,
        help="Gate: priority_q_1 scarcity score must be >= this threshold.",
    )
    args = parser.parse_args()

    in_path = Path(args.alignment_csv)
    if not in_path.is_absolute():
        in_path = (here.parent / in_path).resolve()
    out_summary = Path(args.out_summary_json)
    out_top_pos = Path(args.out_top_positive_csv)
    out_top_neg = Path(args.out_top_negative_csv)
    out_priority = Path(args.out_priority_csv)
    if not out_summary.is_absolute():
        out_summary = (here.parent / out_summary).resolve()
    if not out_top_pos.is_absolute():
        out_top_pos = (here.parent / out_top_pos).resolve()
    if not out_top_neg.is_absolute():
        out_top_neg = (here.parent / out_top_neg).resolve()
    if not out_priority.is_absolute():
        out_priority = (here.parent / out_priority).resolve()

    if not in_path.exists():
        raise SystemExit(f"Missing --alignment-csv: {in_path}")

    df = pd.read_csv(in_path)
    need = {"grid_id", "c_bar", "c_hat", "r_alignment"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"alignment csv missing columns: {miss}")

    for c in ["c_bar", "c_hat", "r_alignment"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in POI_COUNT_COLS:
        if c not in df.columns:
            raise SystemExit(f"alignment csv missing required POI feature column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["r_alignment"]).copy()
    if df.empty:
        raise SystemExit("No valid rows after filtering non-finite r_alignment.")

    q_hi = _safe_quantile(df["r_alignment"], float(args.high_positive_quantile))
    q_lo = _safe_quantile(df["r_alignment"], 1.0 - float(args.high_positive_quantile))

    summary = {
        "n_grids": int(len(df)),
        "top_n": int(args.top_n),
        "high_positive_quantile": float(args.high_positive_quantile),
        "r_alignment": {
            "mean": float(df["r_alignment"].mean()),
            "std": float(df["r_alignment"].std(ddof=1)),
            "min": float(df["r_alignment"].min()),
            "p10": _safe_quantile(df["r_alignment"], 0.10),
            "p25": _safe_quantile(df["r_alignment"], 0.25),
            "p50": _safe_quantile(df["r_alignment"], 0.50),
            "p75": _safe_quantile(df["r_alignment"], 0.75),
            "p90": _safe_quantile(df["r_alignment"], 0.90),
            "max": float(df["r_alignment"].max()),
        },
        "thresholds": {"high_positive_ge": q_hi, "high_negative_le": q_lo},
        "action_gates": {
            "r_alignment_quantile": float(args.gate_r_quantile),
            "min_weeks": int(args.gate_min_weeks),
            "c_bar_quantile": float(args.gate_cbar_quantile),
            "scarcity_q_min": float(args.gate_scarcity_q_min),
        },
    }

    # Top mismatch tables
    k = int(max(1, args.top_n))
    top_pos = df.sort_values("r_alignment", ascending=False).head(k).copy()
    top_neg = df.sort_values("r_alignment", ascending=True).head(k).copy()
    top_pos["rank_positive"] = np.arange(1, len(top_pos) + 1)
    top_neg["rank_negative"] = np.arange(1, len(top_neg) + 1)

    # Priority suggestions for high-positive mismatch
    high = df[df["r_alignment"] >= q_hi].copy()
    if not high.empty:
        # Method A (absolute scarcity): smaller count => higher priority
        for c in POI_COUNT_COLS:
            high[f"{c}_scarcity_abs"] = high[c].rank(method="average", ascending=True, pct=True)

        abs_cols = [f"{c}_scarcity_abs" for c in POI_COUNT_COLS]
        abs_best, abs_second = _priority_from_scores(high[abs_cols])
        high["priority_abs_1"] = [COL2LABEL[POI_COUNT_COLS[i]] for i in abs_best]
        high["priority_abs_2"] = [COL2LABEL[POI_COUNT_COLS[i]] for i in abs_second]

        # Method B (recommended): scarcity vs high-mismatch group distribution
        # 1 - percentile rank within high-mismatch group (lower count -> higher scarcity)
        for c in POI_COUNT_COLS:
            high[f"{c}_pct_in_high"] = high[c].rank(method="average", ascending=True, pct=True)
            high[f"{c}_scarcity_q"] = 1.0 - high[f"{c}_pct_in_high"]

        q_cols = [f"{c}_scarcity_q" for c in POI_COUNT_COLS]
        q_best, q_second = _priority_from_scores(high[q_cols])
        high["priority_q_1"] = [COL2LABEL[POI_COUNT_COLS[i]] for i in q_best]
        high["priority_q_2"] = [COL2LABEL[POI_COUNT_COLS[i]] for i in q_second]
        high["priority_reason"] = "high mismatch + within-high-group quantile scarcity (recommended)"

        # Constraint-based actionability gates
        gate_r_th = _safe_quantile(df["r_alignment"], float(args.gate_r_quantile))
        gate_cbar_th = _safe_quantile(df["c_bar"], float(args.gate_cbar_quantile))
        q1_col_map = {COL2LABEL[c]: f"{c}_scarcity_q" for c in POI_COUNT_COLS}
        high["priority_q_1_scarcity"] = high["priority_q_1"].map(q1_col_map)
        high["priority_q_1_scarcity_score"] = high.apply(
            lambda r: float(r.get(str(r["priority_q_1_scarcity"]), np.nan)),
            axis=1,
        )

        high["gate_r_ok"] = high["r_alignment"] >= gate_r_th
        high["gate_weeks_ok"] = high.get("n_weeks", 0) >= int(args.gate_min_weeks)
        high["gate_cbar_ok"] = high["c_bar"] >= gate_cbar_th
        high["gate_scarcity_ok"] = high["priority_q_1_scarcity_score"] >= float(args.gate_scarcity_q_min)
        high["is_actionable"] = high[["gate_r_ok", "gate_weeks_ok", "gate_cbar_ok", "gate_scarcity_ok"]].all(axis=1)
        high["action_tier"] = np.where(high["is_actionable"], "tier1_actionable", "tier2_watchlist")
        summary["action_threshold_values"] = {
            "r_alignment_ge": float(gate_r_th),
            "c_bar_ge": float(gate_cbar_th),
            "min_weeks": int(args.gate_min_weeks),
            "priority_q_1_scarcity_ge": float(args.gate_scarcity_q_min),
        }
        summary["action_counts"] = {
            "tier1_actionable": int((high["action_tier"] == "tier1_actionable").sum()),
            "tier2_watchlist": int((high["action_tier"] == "tier2_watchlist").sum()),
        }

        keep_cols = [
            "grid_id",
            "c_bar",
            "c_hat",
            "r_alignment",
            "n_weeks",
            *POI_COUNT_COLS,
            "priority_abs_1",
            "priority_abs_2",
            "priority_q_1",
            "priority_q_2",
            "priority_q_1_scarcity_score",
            "gate_r_ok",
            "gate_weeks_ok",
            "gate_cbar_ok",
            "gate_scarcity_ok",
            "is_actionable",
            "action_tier",
            "priority_reason",
        ]
        keep_cols = [c for c in keep_cols if c in high.columns]
        priority = high.sort_values("r_alignment", ascending=False)[keep_cols].copy()
    else:
        priority = pd.DataFrame(
            columns=[
                "grid_id",
                "c_bar",
                "c_hat",
                "r_alignment",
                "n_weeks",
                *POI_COUNT_COLS,
                "priority_abs_1",
                "priority_abs_2",
                "priority_q_1",
                "priority_q_2",
                "priority_reason",
            ]
        )

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_top_pos.parent.mkdir(parents=True, exist_ok=True)
    out_top_neg.parent.mkdir(parents=True, exist_ok=True)
    out_priority.parent.mkdir(parents=True, exist_ok=True)

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    top_pos.to_csv(out_top_pos, index=False)
    top_neg.to_csv(out_top_neg, index=False)
    priority.to_csv(out_priority, index=False)

    print(f"Wrote: {out_summary}")
    print(f"Wrote: {out_top_pos}")
    print(f"Wrote: {out_top_neg}")
    print(f"Wrote: {out_priority}")
    print(f"Rows: {len(df)}  high-positive candidates: {len(priority)}  q={args.high_positive_quantile}")


if __name__ == "__main__":
    main()
