from __future__ import annotations

"""
Daily prediction export launcher matching `train_panel_autoformer_daily_ratio.py`.

Defaults: year split (2024 train/val, 2025 test), same seq/label/pred and model dims as training.
Override with CLI flags if you trained with ratio split or different panel.
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
        "--checkpoints-dir",
        "daily_training_0430/checkpoints",
        "--out-dir",
        "daily_training_0430",
        "--panel-csv",
        "panel_training_0426/outputs/panel_daily_top100_2024_2025_topk2024_city_lag1_wk_is_weekend_sp_nbr8_std_lag1_log1p.csv",
        "--freq",
        "d",
        "--split-mode",
        "year",
        "--train-end",
        "2024-12-31",
        "--test-start",
        "2025-01-01",
        "--val-weeks",
        "10",
        "--seq-len",
        "84",
        "--label-len",
        "42",
        "--pred-len",
        "7",
        "--d-model",
        "256",
        "--d-ff",
        "1024",
        "--dropout",
        "0.12",
        "--moving-avg",
        "7",
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

