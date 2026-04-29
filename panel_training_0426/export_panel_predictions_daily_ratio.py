from __future__ import annotations

"""
Daily prediction export launcher matching the daily ratio training defaults.

This reuses `export_panel_predictions.py` but sets daily-appropriate defaults:
- freq = "d"
- split-mode = "ratio" with 0.7/0.15/0.15
- seq/label/pred match `train_panel_autoformer_daily_ratio.py`
- target-year defaults to 2025

You can override any value by passing the corresponding CLI flags.
"""

import os
import sys
from pathlib import Path
import subprocess


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    export_py = (repo_root / "panel_training_0426" / "export_panel_predictions.py").resolve()
    if not export_py.exists():
        raise SystemExit(f"Missing export entrypoint: {export_py}")

    default_args = [
        str(export_py),
        "--panel-csv",
        "panel_training_0426/outputs/panel_daily_top100_2024_2025_topk2024_city_lag1_wk_is_weekend_sp_nbr8_std_lag1_log1p.csv",
        "--freq",
        "d",
        "--split-mode",
        "ratio",
        "--train-ratio",
        "0.7",
        "--val-ratio",
        "0.15",
        "--test-ratio",
        "0.15",
        "--seq-len",
        "84",
        "--label-len",
        "42",
        "--pred-len",
        "14",
        "--target-year",
        "2025",
        "--scope",
        "test",
    ]

    argv = [sys.executable] + default_args + sys.argv[1:]
    p = subprocess.run(argv, cwd=str(repo_root))
    raise SystemExit(int(p.returncode))


if __name__ == "__main__":
    main()

