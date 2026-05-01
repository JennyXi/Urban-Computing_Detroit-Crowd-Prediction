from __future__ import annotations

"""
Daily training launcher (aligned with weekly *evaluation philosophy*, not same numbers).

Defaults (checkpoints under `daily_training_0430/checkpoints/`):
- freq = "d"
- split-mode = **year**: train/val targets in 2024 (<= --train-end), test targets in 2025 (>= --test-start).
  Requires a **2024–2025** panel CSV (see default --panel-csv). Same idea as weekly README.
- If you only have a **2025-only** panel, override explicitly, e.g.:
    --split-mode ratio --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

Daily-specific tuning:
- d_model=256, d_ff=1024; moving_avg=7; Huber delta=1.5
- Slightly stronger regularization (dropout 0.12, weight_decay 2e-4, lr 2.5e-5) to reduce val overfitting on noisy days.
"""

import sys
from pathlib import Path
import subprocess


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train_py = (repo_root / "panel_training_0426" / "train_panel_autoformer.py").resolve()
    if not train_py.exists():
        raise SystemExit(f"Missing training entrypoint: {train_py}")

    # Year split + 2024–2025 panel: test on true future year (like weekly), not ratio on a single year.
    default_args = [
        str(train_py),
        "--checkpoints-dir",
        "daily_training_0430/checkpoints",
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
        "--lr",
        "2.5e-5",
        "--batch-size",
        "20",
        "--epochs",
        "60",
        "--early-stop",
        "--patience",
        "18",
        "--d-model",
        "256",
        "--d-ff",
        "1024",
        "--dropout",
        "0.12",
        "--moving-avg",
        "7",
        "--huber-delta",
        "1.5",
        "--weight-decay",
        "2e-4",
        "--grad-clip-norm",
        "1.0",
    ]

    # Let user override defaults by passing flags after this launcher.
    argv = [sys.executable] + default_args + sys.argv[1:]
    # Use subprocess to avoid Windows path-with-spaces issues observed with execv.
    p = subprocess.run(argv, cwd=str(repo_root))
    raise SystemExit(int(p.returncode))


if __name__ == "__main__":
    main()

