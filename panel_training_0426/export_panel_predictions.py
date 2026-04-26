from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def _custom_split(n: int, seq_len: int, pred_len: int):
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test
    border1s = (0, num_train - seq_len, n - num_test - seq_len)
    border2s = (num_train, num_train + num_val, n)
    return border1s, border2s


def main() -> None:
    parser = argparse.ArgumentParser(description="Export predictions from panel Autoformer checkpoint (per grid, aligned dates).")
    parser.add_argument("--panel-csv", default="panel_training_0426/outputs/panel_weekly_top100_2024_2025.csv")
    parser.add_argument("--autoformer-root", required=True)
    parser.add_argument("--checkpoints-dir", default="panel_training_0426/checkpoints")
    parser.add_argument("--setting", default=None, help="Checkpoint setting folder name. If omitted, uses latest by mtime.")
    parser.add_argument("--out-dir", default="panel_training_0426/outputs")
    parser.add_argument("--scope", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--label-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--freq", default="w")
    parser.add_argument("--target-transform", default="log1p", choices=["none", "log1p"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    panel_path = (repo_root / args.panel_csv).resolve()
    if not panel_path.exists():
        raise SystemExit(f"Missing panel csv: {panel_path}")

    ckpt_root = (repo_root / args.checkpoints_dir).resolve()
    if not ckpt_root.exists():
        raise SystemExit(f"Missing checkpoints dir: {ckpt_root}")

    if args.setting:
        setting = args.setting
    else:
        cands = [p for p in ckpt_root.iterdir() if p.is_dir()]
        if not cands:
            raise SystemExit(f"No checkpoint folders under: {ckpt_root}")
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        setting = cands[0].name

    ckpt_path = ckpt_root / setting / "checkpoint.pth"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    autoformer_root = Path(args.autoformer_root).resolve()
    sys.path.insert(0, str(autoformer_root))
    from models import Autoformer  # type: ignore
    from utils.timefeatures import time_features  # type: ignore

    df = pd.read_csv(panel_path, parse_dates=["date"])
    df["grid_id"] = df["grid_id"].astype(str)
    cov_cols = [c for c in df.columns if c not in {"grid_id", "date", "OT"}]
    cols_data = cov_cols + ["OT"]
    enc_in_val = len(cols_data)
    f_dim = -1

    # Fit scaler on train portion across all grids
    mats = []
    for gid, g in df.groupby("grid_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        border1s, border2s = _custom_split(n, args.seq_len, args.pred_len)
        mats.append(g.loc[border1s[0] : border2s[0] - 1, cols_data])
    scaler = StandardScaler()
    scaler.fit(pd.concat(mats, ignore_index=True).to_numpy(dtype=np.float32))

    class _Args:
        seq_len = int(args.seq_len)
        label_len = int(args.label_len)
        pred_len = int(args.pred_len)
        features = "MS"
        enc_in = enc_in_val
        dec_in = enc_in_val
        c_out = enc_in_val
        d_model = 128
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 512
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
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for gid, g in df.groupby("grid_id", sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            n = len(g)
            border1s, border2s = _custom_split(n, args.seq_len, args.pred_len)
            if args.scope == "train":
                border1, border2 = border1s[0], border2s[0]
            elif args.scope == "val":
                border1, border2 = border1s[1], border2s[1]
            elif args.scope == "test":
                border1, border2 = border1s[2], border2s[2]
            else:
                border1, border2 = 0, n

            n_samples = (border2 - border1) - args.seq_len - args.pred_len + 1
            if n_samples <= 0:
                continue

            mat = scaler.transform(g.loc[:, cols_data].to_numpy(dtype=np.float32)).astype(np.float32)
            stamps = time_features(pd.to_datetime(g["date"].values), freq=args.freq).transpose(1, 0).astype(np.float32)

            for i in range(n_samples):
                g0 = border1 + i
                s_begin, s_end = g0, g0 + args.seq_len
                r_begin = s_end - args.label_len
                r_end = r_begin + args.label_len + args.pred_len

                batch_x = torch.from_numpy(mat[s_begin:s_end]).unsqueeze(0).to(device)
                batch_y = torch.from_numpy(mat[r_begin:r_end]).unsqueeze(0).to(device)
                batch_x_mark = torch.from_numpy(stamps[s_begin:s_end]).unsqueeze(0).to(device)
                batch_y_mark = torch.from_numpy(stamps[r_begin:r_end]).unsqueeze(0).to(device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                out = outputs[:, -args.pred_len :, f_dim:].squeeze(-1).detach().cpu().numpy().reshape(-1)
                true = batch_y[:, -args.pred_len :, f_dim:].squeeze(-1).detach().cpu().numpy().reshape(-1)

                # inverse transform OT only
                dummy = np.zeros((args.pred_len, enc_in_val), dtype=np.float32)
                dummy[:, -1] = out
                inv_out = scaler.inverse_transform(dummy)[:, -1]
                dummy[:, -1] = true
                inv_true = scaler.inverse_transform(dummy)[:, -1]

                if args.target_transform == "log1p":
                    inv_out = np.expm1(inv_out)
                    inv_true = np.expm1(inv_true)

                dates = g["date"].iloc[g0 + args.seq_len : g0 + args.seq_len + args.pred_len].to_list()
                for h in range(args.pred_len):
                    dt = pd.Timestamp(dates[h])
                    if args.target_year is not None:
                        y = int(args.target_year)
                        if not (pd.Timestamp(f"{y}-01-01") <= dt <= pd.Timestamp(f"{y}-12-31")):
                            continue
                    rows.append(
                        {
                            "grid_id": str(gid),
                            "setting": setting,
                            "sample_id": int(i),
                            "window_start": pd.Timestamp(g["date"].iloc[g0]),
                            "horizon": int(h + 1),
                            "date": dt,
                            "y_true": float(inv_true[h]),
                            "y_pred": float(inv_out[h]),
                            "scope": str(args.scope),
                        }
                    )

    out_long = pd.DataFrame(rows).sort_values(["grid_id", "date", "horizon", "window_start"]).reset_index(drop=True)
    long_path = out_dir / f"panel_pred_{args.scope}_{int(args.target_year)}_long.csv"
    out_long.to_csv(long_path, index=False)

    by_date = (
        out_long.sort_values(["grid_id", "date", "window_start"])
        .groupby(["grid_id", "date"], as_index=False)
        .agg(
            y_true=("y_true", "mean"),
            y_pred_mean=("y_pred", "mean"),
            y_pred_last=("y_pred", "last"),
            n_preds=("y_pred", "size"),
        )
        .sort_values(["date", "grid_id"])
    )
    by_date_path = out_dir / f"panel_pred_{args.scope}_{int(args.target_year)}_by_date.csv"
    by_date.to_csv(by_date_path, index=False)

    print(f"Wrote: {long_path}")
    print(f"Wrote: {by_date_path}")


if __name__ == "__main__":
    main()

