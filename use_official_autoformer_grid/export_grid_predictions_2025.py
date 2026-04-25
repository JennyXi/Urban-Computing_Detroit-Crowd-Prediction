from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _latest_setting(checkpoints_dir: Path, prefix: str) -> str | None:
    cands = [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Top-100 grid weekly predictions (2025) into one CSV for QGIS.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repo root (Try_0412).",
    )
    parser.add_argument(
        "--manifest",
        default="use_official_autoformer_grid/data/grid_weekly_top100_visits/grid_topk_manifest.csv",
        help="Manifest CSV produced by prepare_top100_grid_weekly_csvs.py",
    )
    parser.add_argument(
        "--out-dir",
        default="use_official_autoformer_grid/outputs",
        help="Output directory for combined exports.",
    )
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument(
        "--autoformer-root",
        default=r"E:\Urban Computing Final Project\autoformer_spatial_0425\Autoformer",
        help="Path to official Autoformer repo clone (used by exporter for model code).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = repo_root / args.manifest
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = repo_root / "use_official_autoformer_grid" / "checkpoints"
    grid_data_root = repo_root / "use_official_autoformer_grid" / "data" / "grid_weekly_top100_visits"

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    man = pd.read_csv(manifest_path)
    if "grid_id" not in man.columns:
        raise SystemExit("Manifest missing grid_id column")

    rows = []
    # We re-use the city-level exporter to do aligned date exports per grid model.
    exporter = repo_root / "use_official_autoformer" / "export_predictions.py"
    if not exporter.exists():
        raise SystemExit(f"Missing exporter script: {exporter}")

    py = repo_root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        raise SystemExit(f"Missing venv python: {py}")

    for grid_id in man["grid_id"].astype(str).tolist():
        prefix = f"detroit_grid100_{grid_id}_"
        setting = _latest_setting(checkpoints_dir, prefix)
        if setting is None:
            # not trained yet
            continue

        data_path = f"grid_{grid_id}.csv"
        if not (grid_data_root / data_path).exists():
            continue

        out_path = out_dir / f"{setting}_all_{args.target_year}_pred.csv"
        cmd = [
            str(py),
            str(exporter),
            "--autoformer-root",
            str(Path(args.autoformer_root).resolve()),
            "--checkpoints-dir",
            str(checkpoints_dir),
            "--scope",
            "all",
            "--target-year",
            str(args.target_year),
            "--features",
            "S",
            "--freq",
            "w",
            "--seq-len",
            "12",
            "--label-len",
            "6",
            "--pred-len",
            "4",
            "--d-model",
            "128",
            "--d-ff",
            "512",
            "--n-heads",
            "8",
            "--e-layers",
            "2",
            "--d-layers",
            "1",
            "--data-path",
            data_path,
            "--data-root",
            str(grid_data_root),
            "--setting",
            setting,
            "--out",
            str(out_path),
            "--repo-root",
            str(repo_root),
        ]

        # run exporter
        import subprocess

        subprocess.run(cmd, check=True)
        df = pd.read_csv(out_path, parse_dates=["date", "window_start"])
        df.insert(0, "grid_id", grid_id)
        rows.append(df)

    if not rows:
        raise SystemExit("No grid predictions exported. Did you train any grid models yet?")

    all_long = pd.concat(rows, ignore_index=True)
    all_long = all_long.sort_values(["grid_id", "date", "horizon", "window_start"]).reset_index(drop=True)
    long_path = out_dir / f"grid_top100_weekly_visits_all_{args.target_year}_pred_long.csv"
    all_long.to_csv(long_path, index=False)

    # One row per (grid_id, date) for mapping.
    by_date = (
        all_long.sort_values(["grid_id", "date", "window_start"])
        .groupby(["grid_id", "date"], as_index=False)
        .agg(
            y_true=("y_true", "mean"),
            y_pred_mean=("y_pred", "mean"),
            y_pred_last=("y_pred", "last"),
            n_preds=("y_pred", "size"),
        )
        .sort_values(["date", "grid_id"])
    )
    by_date_path = out_dir / f"grid_top100_weekly_visits_all_{args.target_year}_pred_by_date.csv"
    by_date.to_csv(by_date_path, index=False)

    print(f"Wrote: {long_path}")
    print(f"Wrote: {by_date_path}")
    print("Preview (by_date):")
    print(by_date.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

