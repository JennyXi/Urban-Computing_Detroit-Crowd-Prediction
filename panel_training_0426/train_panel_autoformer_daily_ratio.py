from __future__ import annotations

"""
Daily training launcher with ratio split defaults.

This keeps the Autoformer architecture identical to weekly training by reusing
the same underlying `train_panel_autoformer.py` entrypoint, but sets daily-
appropriate defaults:
- freq = "d"
- split-mode = "ratio" with 0.7/0.15/0.15
- a longer seq_len than weekly to capture multiple weekly cycles

You can still override any value by passing the corresponding CLI flags.
"""

import os
import sys
from pathlib import Path
import subprocess


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train_py = (repo_root / "panel_training_0426" / "train_panel_autoformer.py").resolve()
    if not train_py.exists():
        raise SystemExit(f"Missing training entrypoint: {train_py}")

    # Defaults tuned for daily series (strong baseline):
    # - seq_len=84 (12 weeks), label_len=42 (6 weeks), pred_len=14 (2 weeks)
    #   captures multiple weekly cycles + mid-term trend while staying tractable.
    default_args = [
        str(train_py),
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
    ]

    # Let user override defaults by passing flags after this launcher.
    argv = [sys.executable] + default_args + sys.argv[1:]
    # Use subprocess to avoid Windows path-with-spaces issues observed with execv.
    p = subprocess.run(argv, cwd=str(repo_root))
    raise SystemExit(int(p.returncode))


if __name__ == "__main__":
    main()

