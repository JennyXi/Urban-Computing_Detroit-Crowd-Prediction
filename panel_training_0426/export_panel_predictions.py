from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def _custom_split(n: int, seq_len: int, pred_len: int, train_ratio: float, val_ratio: float, test_ratio: float):
    num_train = int(round(n * float(train_ratio)))
    num_val = int(round(n * float(val_ratio)))
    num_test = n - num_train - num_val
    border1s = (0, num_train - seq_len, num_train + num_val - seq_len)
    border2s = (num_train, num_train + num_val, n)
    return border1s, border2s


def _year_split_g0s(
    dates: pd.Series, seq_len: int, pred_len: int, train_end: str, test_start: str, val_weeks: int
) -> tuple[list[int], list[int], list[int], int]:
    d = pd.to_datetime(dates).reset_index(drop=True)
    n = len(d)
    max_g0 = n - seq_len - pred_len
    if max_g0 < 0:
        return [], [], [], 0

    te = pd.Timestamp(train_end).normalize()
    ts = pd.Timestamp(test_start).normalize()
    valw = int(val_weeks)
    val_target_start = te - pd.Timedelta(days=7 * (valw - 1))

    train_g0s: list[int] = []
    val_g0s: list[int] = []
    test_g0s: list[int] = []
    for g0 in range(0, max_g0 + 1):
        t = pd.to_datetime(d.iloc[g0 + seq_len : g0 + seq_len + pred_len])
        if t.empty:
            continue
        t0 = pd.Timestamp(t.min()).normalize()
        t1 = pd.Timestamp(t.max()).normalize()
        if t0 >= ts:
            test_g0s.append(g0)
            continue
        if t1 <= te:
            if t0 >= val_target_start:
                val_g0s.append(g0)
            else:
                train_g0s.append(g0)

    scaler_fit_end_excl = int((d.dt.normalize() <= te).sum())
    return train_g0s, val_g0s, test_g0s, scaler_fit_end_excl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export predictions from panel Autoformer checkpoint (per grid, aligned dates).")
    parser.add_argument(
        "--panel-csv",
        default="panel_training_0426/outputs/panel_weekly_top100_2024_2025_topk2024_city_lag1_log1p.csv",
    )
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
    parser.add_argument(
        "--split-mode",
        default="year",
        choices=["ratio", "year"],
        help="Must match training. year: strict train<=train_end, test>=test_start.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Used when --split-mode ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Used when --split-mode ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Used when --split-mode ratio.")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--val-weeks", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128, help="Must match training checkpoint.")
    parser.add_argument("--n-heads", type=int, default=8, help="Must match training.")
    parser.add_argument("--e-layers", type=int, default=2, help="Must match training.")
    parser.add_argument("--d-layers", type=int, default=1, help="Must match training.")
    parser.add_argument("--d-ff", type=int, default=512, help="Must match training.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Must match training.")
    parser.add_argument("--moving-avg", type=int, default=25, help="Must match training.")
    parser.add_argument(
        "--stamp",
        default=None,
        help="Optional filename stamp (e.g. 20260428_103000). If omitted, a local timestamp is used. "
        "Dated outputs are written in addition to the stable filenames.",
    )
    args = parser.parse_args()
    dm = int(args.d_model)
    nh = int(args.n_heads)
    if dm % nh != 0:
        raise SystemExit(f"--d-model ({dm}) must be divisible by --n-heads ({nh}).")
    ratio_sum = float(args.train_ratio) + float(args.val_ratio) + float(args.test_ratio)
    if abs(ratio_sum - 1.0) > 1e-8:
        raise SystemExit(
            f"train/val/test ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"({args.train_ratio}, {args.val_ratio}, {args.test_ratio})."
        )

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

    sm = str(args.split_mode)
    # Per-grid scaler fit on each grid's train slice only (must match train_panel_autoformer.PanelWindows).
    per_grid_scalers: dict[str, StandardScaler] = {}
    for gid, g in df.groupby("grid_id", sort=False):
        gid = str(gid)
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if sm == "ratio":
            border1s, border2s = _custom_split(
                n,
                args.seq_len,
                args.pred_len,
                float(args.train_ratio),
                float(args.val_ratio),
                float(args.test_ratio),
            )
            tr = g.loc[border1s[0] : border2s[0] - 1, cols_data]
        else:
            _tr, _va, _te, end_excl = _year_split_g0s(
                g["date"], int(args.seq_len), int(args.pred_len), str(args.train_end), str(args.test_start), int(args.val_weeks)
            )
            tr = g.loc[: end_excl - 1, cols_data]
        if len(tr) == 0:
            raise SystemExit(f"No train rows to fit scaler for grid_id={gid!r}.")
        s = StandardScaler()
        s.fit(tr.to_numpy(dtype=np.float32))
        per_grid_scalers[gid] = s

    class _Args:
        seq_len = int(args.seq_len)
        label_len = int(args.label_len)
        pred_len = int(args.pred_len)
        features = "MS"
        enc_in = enc_in_val
        dec_in = enc_in_val
        c_out = enc_in_val
        d_model = int(args.d_model)
        n_heads = int(args.n_heads)
        e_layers = int(args.e_layers)
        d_layers = int(args.d_layers)
        d_ff = int(args.d_ff)
        factor = 1
        dropout = float(args.dropout)
        embed = "timeF"
        freq = args.freq
        activation = "gelu"
        output_attention = False
        moving_avg = int(args.moving_avg)
        distil = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Autoformer.Model(_Args()).float().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = str(args.stamp).strip() if args.stamp is not None else ""
    if not stamp:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    with torch.no_grad():
        for gid, g in df.groupby("grid_id", sort=False):
            gid = str(gid)
            g = g.sort_values("date").reset_index(drop=True)
            n = len(g)
            if sm == "ratio":
                border1s, border2s = _custom_split(
                    n,
                    args.seq_len,
                    args.pred_len,
                    float(args.train_ratio),
                    float(args.val_ratio),
                    float(args.test_ratio),
                )
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
                g0_list = list(range(border1, border1 + n_samples))
            else:
                tr, va, te, _end_excl = _year_split_g0s(
                    g["date"], int(args.seq_len), int(args.pred_len), str(args.train_end), str(args.test_start), int(args.val_weeks)
                )
                g0_list = {"train": tr, "val": va, "test": te}.get(str(args.scope), [])
                if args.scope == "all":
                    g0_list = list(range(0, max(0, n - int(args.seq_len) - int(args.pred_len) + 1)))
                if not g0_list:
                    continue

            gs = per_grid_scalers[gid]
            mat = gs.transform(g.loc[:, cols_data].to_numpy(dtype=np.float32)).astype(np.float32)
            stamps = time_features(pd.to_datetime(g["date"].values), freq=args.freq).transpose(1, 0).astype(np.float32)

            for i, g0 in enumerate(g0_list):
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
                inv_out = gs.inverse_transform(dummy)[:, -1]
                dummy[:, -1] = true
                inv_true = gs.inverse_transform(dummy)[:, -1]

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
    long_path_dated = out_dir / f"panel_pred_{args.scope}_{int(args.target_year)}_long_{stamp}.csv"
    out_long.to_csv(long_path, index=False)
    out_long.to_csv(long_path_dated, index=False)

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
    by_date_path_dated = out_dir / f"panel_pred_{args.scope}_{int(args.target_year)}_by_date_{stamp}.csv"
    by_date.to_csv(by_date_path, index=False)
    by_date.to_csv(by_date_path_dated, index=False)

    print(f"Wrote: {long_path}")
    print(f"Wrote: {by_date_path}")
    print(f"Wrote (dated): {long_path_dated}")
    print(f"Wrote (dated): {by_date_path_dated}")


if __name__ == "__main__":
    main()

