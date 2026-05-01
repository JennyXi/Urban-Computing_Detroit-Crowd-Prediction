"""
Hyperparameter sweep for daily training (not weight-space interpolation).

Calls `train_panel_autoformer_daily_ratio.py` once per configuration, passes
`--metrics-json` into the underlying trainer so each run records best val loss.

Edit SWEEP_RUNS below (each entry = extra argv tokens appended after the launcher).
Or use --config-json path to a JSON array of string arrays.

Example (CMD, repo root):

  .venv\\Scripts\\python.exe scripts\\sweep_daily_hyperparams.py ^
    --autoformer-root "E:\\...\\Autoformer" ^
    --panel-csv panel_training_0426/outputs/panel_daily_top100_2025_2025_poi2024.csv

After all runs, open sweep_summary.csv and sort by best_val_loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# Each list is extra CLI args (override daily launcher defaults). First run = [] uses launcher defaults only.
SWEEP_RUNS: list[list[str]] = [
    [],
    ["--dropout", "0.12", "--weight-decay", "2e-4"],
    ["--dropout", "0.15", "--lr", "2.5e-5"],
    ["--lr", "2e-5", "--patience", "25"],
    ["--d-model", "192", "--d-ff", "768", "--lr", "3e-5"],
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--autoformer-root", required=True)
    ap.add_argument("--panel-csv", required=True)
    ap.add_argument(
        "--config-json",
        default=None,
        help='JSON file: [["--lr","2e-5"], ["--dropout","0.12"]] overrides SWEEP_RUNS',
    )
    ap.add_argument(
        "--out-summary",
        default="daily_training_0430/sweep_summary.csv",
        help="CSV path (relative to repo root) aggregating metrics from each run.",
    )
    ap.add_argument(
        "--metrics-dir",
        default="daily_training_0430/sweep_metrics",
        help="Directory for per-run metrics JSON (relative to repo root).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands only.")
    ap.add_argument("--max-runs", type=int, default=None, help="Only first N sweep entries.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_launcher = repo_root / "panel_training_0426" / "train_panel_autoformer_daily_ratio.py"
    if not train_launcher.exists():
        raise SystemExit(f"Missing {train_launcher}")

    runs: list[list[str]] = SWEEP_RUNS
    if args.config_json:
        p = Path(args.config_json)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        runs = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(runs, list):
            raise SystemExit("--config-json must be a JSON array of argv lists")

    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]

    py = repo_root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    metrics_dir = (repo_root / args.metrics_dir).resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (repo_root / args.out_summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []

    for i, extra in enumerate(runs):
        tag = f"run_{i:03d}"
        metrics_rel = str(Path(args.metrics_dir) / f"{tag}.json")
        cmd = [
            str(py),
            "-u",
            str(train_launcher),
            "--autoformer-root",
            args.autoformer_root,
            "--panel-csv",
            args.panel_csv,
            "--metrics-json",
            metrics_rel,
        ] + list(extra)

        print("---")
        print("RUN", i, "extra:", extra if extra else "(defaults)")
        print("CMD:", " ".join(f'"{c}"' if " " in c else c for c in cmd))

        if args.dry_run:
            continue

        r = subprocess.run(cmd, cwd=str(repo_root))
        if r.returncode != 0:
            print(f"Run {i} failed with exit {r.returncode}", file=sys.stderr)
            rows_out.append({"run_id": tag, "status": "failed", "exit_code": r.returncode})
            continue

        mj = (repo_root / metrics_rel).resolve()
        if not mj.exists():
            rows_out.append({"run_id": tag, "status": "no_metrics"})
            continue
        data = json.loads(mj.read_text(encoding="utf-8"))
        rows_out.append(
            {
                "run_id": tag,
                "status": "ok",
                "best_val_loss": data.get("best_val_loss"),
                "best_epoch": data.get("best_epoch"),
                "epochs_run": data.get("epochs_run"),
                "setting": data.get("setting"),
                "extra_args": " ".join(extra),
            }
        )

    if args.dry_run:
        return

    fieldnames = ["run_id", "status", "best_val_loss", "best_epoch", "epochs_run", "setting", "extra_args", "exit_code"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
