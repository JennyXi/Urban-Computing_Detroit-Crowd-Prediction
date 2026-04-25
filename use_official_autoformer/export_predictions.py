from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class CustomSplit:
    n: int
    seq_len: int
    pred_len: int
    num_train: int
    num_test: int
    num_val: int
    border1s: tuple[int, int, int]
    border2s: tuple[int, int, int]


def _custom_split(n: int, seq_len: int, pred_len: int) -> CustomSplit:
    """
    Match thuml/Autoformer `Dataset_Custom` split logic:
      num_train = int(n * 0.7)
      num_test  = int(n * 0.2)
      num_val   = n - num_train - num_test
      border1s = [0, num_train - seq_len, n - num_test - seq_len]
      border2s = [num_train, num_train + num_val, n]
    """
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test
    border1s = (0, num_train - seq_len, n - num_test - seq_len)
    border2s = (num_train, num_train + num_val, n)
    return CustomSplit(
        n=n,
        seq_len=seq_len,
        pred_len=pred_len,
        num_train=num_train,
        num_test=num_test,
        num_val=num_val,
        border1s=border1s,
        border2s=border2s,
    )


def _find_latest_setting(checkpoints_dir: Path, prefix: str) -> str:
    cands = [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not cands:
        raise SystemExit(f"No checkpoint folders under {checkpoints_dir} with prefix {prefix!r}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Autoformer predictions to CSV with dates aligned.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to Try_0412 repo root.",
    )
    parser.add_argument(
        "--autoformer-root",
        default=None,
        help="Path to official Autoformer repo. Defaults to use_official_autoformer/third_party/Autoformer",
    )
    parser.add_argument(
        "--data-path",
        default="autoformer_weekly_preprocessed.csv",
        help="CSV filename under <repo-root>/data/",
    )
    parser.add_argument("--freq", default="w", help="Autoformer time feature frequency, e.g. w/h.")
    parser.add_argument("--features", default="MS", choices=["S", "M", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--label-len", type=int, default=6)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument(
        "--scope",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which portion to export. 'test' matches Autoformer default evaluation. 'all' runs rolling forecasts across the whole year.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=None,
        help="If set (e.g. 2025), only keep rows whose forecast `date` is within that year.",
    )
    parser.add_argument(
        "--setting",
        default="latest",
        help="Checkpoint folder name under use_official_autoformer/checkpoints. Use 'latest' to auto-pick.",
    )
    parser.add_argument(
        "--setting-prefix",
        default="detroit_2025_weekly_",
        help="Used when --setting=latest.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: use_official_autoformer/outputs/<setting>_pred.csv",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    autoformer_root = (
        Path(args.autoformer_root).resolve()
        if args.autoformer_root
        else (repo_root / "use_official_autoformer" / "third_party" / "Autoformer").resolve()
    )
    data_root = repo_root / "data"
    csv_path = data_root / args.data_path
    checkpoints_dir = repo_root / "use_official_autoformer" / "checkpoints"

    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")
    if not autoformer_root.exists():
        raise SystemExit(f"Missing Autoformer repo: {autoformer_root}")
    if not checkpoints_dir.exists():
        raise SystemExit(f"Missing checkpoints dir: {checkpoints_dir}")

    setting = args.setting
    if setting == "latest":
        setting = _find_latest_setting(checkpoints_dir, args.setting_prefix)

    ckpt_path = checkpoints_dir / setting / "checkpoint.pth"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        suffix = f"{args.scope}"
        if args.target_year is not None:
            suffix += f"_{int(args.target_year)}"
        out_path = repo_root / "use_official_autoformer" / "outputs" / f"{setting}_{suffix}_pred.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load full raw CSV (unscaled domain) ---
    df_raw = pd.read_csv(csv_path)
    if "date" not in df_raw.columns:
        raise SystemExit("CSV must have a 'date' column")
    if args.target not in df_raw.columns:
        raise SystemExit(f"CSV missing target column: {args.target}")
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    n = len(df_raw)
    split = _custom_split(n=n, seq_len=args.seq_len, pred_len=args.pred_len)

    # --- Import official utilities (time features, model) ---
    sys.path.insert(0, str(autoformer_root))
    from models import Autoformer  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from utils.timefeatures import time_features  # type: ignore

    # Enc/dec/c_out in Autoformer args are "number of input channels".
    # For MS, Autoformer uses all columns except 'date' (including target).
    enc_in_val = len(pd.read_csv(csv_path, nrows=1).columns) - 1

    class _Args:
        # minimal namespace for the model
        seq_len = args.seq_len
        label_len = args.label_len
        pred_len = args.pred_len
        features = args.features
        enc_in = enc_in_val
        dec_in = enc_in_val
        c_out = enc_in_val
        d_model = 256
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 1024
        factor = 1
        dropout = 0.05
        embed = "timeF"
        freq = args.freq
        activation = "gelu"
        output_attention = False
        moving_avg = 25
        distil = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Autoformer.Model(_Args()).float().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Reorder columns like Dataset_Custom does: ['date'] + other + [target] ---
    cols = list(df_raw.columns)
    cols.remove(args.target)
    cols.remove("date")
    df_reordered = df_raw[["date"] + cols + [args.target]].copy()

    # --- Build df_data according to features setting ---
    if args.features in ("M", "MS"):
        cols_data = df_reordered.columns[1:]  # all numeric incl target
        df_data = df_reordered[cols_data]
        f_dim = -1 if args.features == "MS" else 0
    else:  # 'S'
        df_data = df_reordered[[args.target]]
        f_dim = 0

    # --- Fit scaler on train range, transform full data (matches Dataset_Custom) ---
    scaler = StandardScaler()
    train_data = df_data.iloc[split.border1s[0] : split.border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values).astype(np.float32)

    # --- Time features encoding (matches embed=timeF => timeenc=1) ---
    stamps = time_features(pd.to_datetime(df_reordered["date"].values), freq=args.freq).transpose(1, 0).astype(np.float32)

    # --- Choose scope borders ---
    if args.scope == "train":
        border1, border2 = split.border1s[0], split.border2s[0]
    elif args.scope == "val":
        border1, border2 = split.border1s[1], split.border2s[1]
    elif args.scope == "test":
        border1, border2 = split.border1s[2], split.border2s[2]
    else:  # all
        border1, border2 = 0, n

    n_samples = (border2 - border1) - args.seq_len - args.pred_len + 1
    if n_samples <= 0:
        raise SystemExit(f"Not enough data for scope={args.scope!r} with seq_len={args.seq_len}, pred_len={args.pred_len}.")

    # Run rolling forecasts
    preds = np.zeros((n_samples, args.pred_len), dtype=np.float32)
    trues = np.zeros((n_samples, args.pred_len), dtype=np.float32)
    forecast_dates = []
    window_start_dates = []

    with torch.no_grad():
        for i in range(n_samples):
            g0 = border1 + i
            window_start_dates.append(df_reordered["date"].iloc[g0])
            f_idx0 = g0 + args.seq_len
            forecast_dates.append(df_reordered["date"].iloc[f_idx0 : f_idx0 + args.pred_len].to_list())

            s_begin, s_end = g0, g0 + args.seq_len
            r_begin = s_end - args.label_len
            r_end = r_begin + args.label_len + args.pred_len

            batch_x = torch.from_numpy(data[s_begin:s_end]).unsqueeze(0).to(device)
            batch_y = torch.from_numpy(data[r_begin:r_end]).unsqueeze(0).to(device)
            batch_x_mark = torch.from_numpy(stamps[s_begin:s_end]).unsqueeze(0).to(device)
            batch_y_mark = torch.from_numpy(stamps[r_begin:r_end]).unsqueeze(0).to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            out = outputs[:, -args.pred_len :, f_dim:].squeeze(-1).detach().cpu().numpy()
            true = batch_y[:, -args.pred_len :, f_dim:].squeeze(-1).detach().cpu().numpy()

            # inverse transform target only
            if args.features == "S":
                inv_out = scaler.inverse_transform(out.reshape(-1, 1)).ravel()
                inv_true = scaler.inverse_transform(true.reshape(-1, 1)).ravel()
            else:
                dummy = np.zeros((args.pred_len, enc_in_val), dtype=np.float32)
                dummy[:, -1] = out[0]
                inv_out = scaler.inverse_transform(dummy)[:, -1]
                dummy[:, -1] = true[0]
                inv_true = scaler.inverse_transform(dummy)[:, -1]

            preds[i, :] = inv_out
            trues[i, :] = inv_true

    # Long-form export
    rows = []
    for i in range(n_samples):
        for h in range(args.pred_len):
            rows.append(
                {
                    "setting": setting,
                    "sample_id": i,
                    "window_start": window_start_dates[i],
                    "horizon": h + 1,
                    "date": forecast_dates[i][h],
                    "y_true": float(trues[i, h]),
                    "y_pred": float(preds[i, h]),
                    "scope": args.scope,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df["date"] = pd.to_datetime(out_df["date"])
    out_df["window_start"] = pd.to_datetime(out_df["window_start"])

    if args.target_year is not None:
        y = int(args.target_year)
        out_df = out_df[(out_df["date"] >= pd.Timestamp(f"{y}-01-01")) & (out_df["date"] <= pd.Timestamp(f"{y}-12-31"))].copy()

    # Sort by actual forecast date first (Excel-friendly)
    out_df = out_df.sort_values(["date", "horizon", "window_start"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)

    # A second "one row per date" view (aggregated), easier for charts
    by_date_path = out_path.with_name(out_path.stem + "_by_date.csv")
    agg = (
        out_df.sort_values(["date", "window_start"])
        .groupby("date", as_index=False)
        .agg(
            y_true=("y_true", "mean"),  # should be identical per date; mean is safe
            y_pred_mean=("y_pred", "mean"),
            y_pred_last=("y_pred", "last"),  # last window_start (latest information)
            n_preds=("y_pred", "size"),
        )
        .sort_values("date")
    )
    agg.to_csv(by_date_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {by_date_path}")
    print(f"Rows: {len(out_df)} (samples={n_samples}, pred_len={args.pred_len}, scope={args.scope})")
    print("Preview:")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

