from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class PanelConfig:
    seq_len: int
    label_len: int
    pred_len: int
    batch_size: int
    lr: float
    epochs: int
    patience: int
    features: str
    freq: str
    target_transform: str


class PanelWindows(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        grid_ids: list[str],
        split: str,
        cfg: PanelConfig,
        per_grid_scalers: dict[str, StandardScaler] | None = None,
        time_features_fn=None,
        *,
        split_mode: str = "ratio",
        train_end: str = "2024-12-31",
        test_start: str = "2025-01-01",
        val_weeks: int = 10,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        assert split in {"train", "val", "test"}
        assert split_mode in {"ratio", "year"}
        self.cfg = cfg
        self.split = split
        self.grid_ids = grid_ids
        self.time_features_fn = time_features_fn
        self.split_mode = split_mode

        # Build per-grid arrays (already sorted by date)
        self.series = {}
        for gid in grid_ids:
            g = df[df["grid_id"] == gid].sort_values("date")
            self.series[gid] = g.reset_index(drop=True)

        # columns: numeric covariates + OT last
        # We keep ordering like Dataset_Custom: date + covariates + OT
        self.cov_cols = [c for c in df.columns if c not in {"grid_id", "date", "OT"}]
        self.cols_data = self.cov_cols + ["OT"]

        def _targets_for_g0(dates: pd.Series, g0: int) -> pd.DatetimeIndex:
            # target timestamps for this forecast window
            return pd.to_datetime(dates.iloc[g0 + cfg.seq_len : g0 + cfg.seq_len + cfg.pred_len])

        def _g0s_year_split(dates: pd.Series) -> tuple[list[int], list[int], list[int]]:
            d = pd.to_datetime(dates).reset_index(drop=True)
            n = len(d)
            max_g0 = n - cfg.seq_len - cfg.pred_len
            if max_g0 < 0:
                return [], [], []

            te = pd.Timestamp(train_end).normalize()
            ts = pd.Timestamp(test_start).normalize()
            valw = int(val_weeks)
            # val targets are the last `val_weeks` weeks within 2024 (up to train_end)
            val_target_start = te - pd.Timedelta(days=7 * (valw - 1))

            train_g0s: list[int] = []
            val_g0s: list[int] = []
            test_g0s: list[int] = []
            for g0 in range(0, max_g0 + 1):
                t = _targets_for_g0(d, g0)
                if t.empty:
                    continue
                t0 = pd.Timestamp(t.min()).normalize()
                t1 = pd.Timestamp(t.max()).normalize()

                # test: all targets in [test_start, ...]
                if t0 >= ts:
                    test_g0s.append(g0)
                    continue

                # train/val: targets must be within <= train_end
                if t1 <= te:
                    if t0 >= val_target_start:
                        val_g0s.append(g0)
                    else:
                        train_g0s.append(g0)
            return train_g0s, val_g0s, test_g0s

        def _ratio_split_borders(n: int) -> tuple[list[int], list[int]]:
            num_train = int(round(n * float(train_ratio)))
            num_val = int(round(n * float(val_ratio)))
            num_test = n - num_train - num_val
            border1s = [0, num_train - cfg.seq_len, num_train + num_val - cfg.seq_len]
            border2s = [num_train, num_train + num_val, n]
            return border1s, border2s

        # Build window indices
        self.indices: list[tuple[str, int]] = []  # (gid, g0)
        if split_mode == "ratio":
            # Dataset_Custom-style borders per grid (configurable train/val/test ratios)
            for gid, g in self.series.items():
                n = len(g)
                border1s, border2s = _ratio_split_borders(n)
                if split == "train":
                    border1, border2 = border1s[0], border2s[0]
                elif split == "val":
                    border1, border2 = border1s[1], border2s[1]
                else:
                    border1, border2 = border1s[2], border2s[2]

                n_samples = (border2 - border1) - cfg.seq_len - cfg.pred_len + 1
                if n_samples <= 0:
                    continue
                for i in range(n_samples):
                    self.indices.append((gid, border1 + i))
        else:
            # Strict year split: train targets in 2024 (<= train_end), test targets in 2025 (>= test_start)
            for gid, g in self.series.items():
                tr, va, te = _g0s_year_split(g["date"])
                g0s = {"train": tr, "val": va, "test": te}[split]
                for g0 in g0s:
                    self.indices.append((gid, int(g0)))

        # Per-grid StandardScaler fit on that grid's train partition only (avoids high-traffic grids
        # dominating the global mean/std and crushing low-traffic series in one shared scaler).
        if per_grid_scalers is None:
            self.per_grid_scalers: dict[str, StandardScaler] = {}
            if split_mode == "ratio":
                for gid, g in self.series.items():
                    n = len(g)
                    num_train = int(round(n * float(train_ratio)))
                    if num_train <= 0:
                        num_train = min(1, n)
                    tr = g.loc[: num_train - 1, self.cols_data]
                    s = StandardScaler()
                    s.fit(tr.to_numpy(dtype=np.float32))
                    self.per_grid_scalers[gid] = s
            else:
                te = pd.Timestamp(train_end).normalize()
                for gid, g in self.series.items():
                    mask = pd.to_datetime(g["date"]).dt.normalize() <= te
                    tr = g.loc[mask, self.cols_data]
                    if len(tr) == 0:
                        raise SystemExit(f"No train rows (date<=train_end) for grid_id={gid!r}; cannot fit scaler.")
                    s = StandardScaler()
                    s.fit(tr.to_numpy(dtype=np.float32))
                    self.per_grid_scalers[gid] = s
        else:
            self.per_grid_scalers = per_grid_scalers

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        gid, g0 = self.indices[idx]
        g = self.series[gid]
        cfg = self.cfg
        s_begin, s_end = g0, g0 + cfg.seq_len
        r_begin = s_end - cfg.label_len
        r_end = r_begin + cfg.label_len + cfg.pred_len

        # scaled numeric data (covariates+OT)
        mat = g.loc[:, self.cols_data].to_numpy(dtype=np.float32)
        mat = self.per_grid_scalers[gid].transform(mat).astype(np.float32)

        batch_x = mat[s_begin:s_end]
        batch_y = mat[r_begin:r_end]

        # time features (timeF)
        dates = pd.to_datetime(g["date"].to_numpy())
        stamps = self.time_features_fn(dates, freq=cfg.freq).transpose(1, 0).astype(np.float32)
        batch_x_mark = stamps[s_begin:s_end]
        batch_y_mark = stamps[r_begin:r_end]

        return (
            torch.from_numpy(batch_x),
            torch.from_numpy(batch_y),
            torch.from_numpy(batch_x_mark),
            torch.from_numpy(batch_y_mark),
        )


def _inverse_target(arr: np.ndarray, cfg: PanelConfig) -> np.ndarray:
    if cfg.target_transform == "log1p":
        return np.expm1(arr)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one shared Autoformer on a grid weekly panel dataset.")
    parser.add_argument(
        "--panel-csv",
        default="panel_training_0426/outputs/panel_weekly_top100_2024_2025_topk2024_city_lag1_log1p.csv",
    )
    parser.add_argument("--autoformer-root", required=True, help="Path to official Autoformer repo (thuml/Autoformer).")
    parser.add_argument("--checkpoints-dir", default="panel_training_0426/checkpoints")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--label-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop when validation loss does not improve for --patience epochs (default: run all --epochs).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if present.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freq", default="w")
    parser.add_argument("--target-transform", default="log1p", choices=["none", "log1p"])
    parser.add_argument(
        "--split-mode",
        default="year",
        choices=["ratio", "year"],
        help="ratio: configurable train/val/test split per grid timeline. year: strict train<=train_end, test>=test_start.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Used when --split-mode ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Used when --split-mode ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Used when --split-mode ratio.")
    parser.add_argument("--train-end", default="2024-12-31", help="Used when --split-mode year.")
    parser.add_argument("--test-start", default="2025-01-01", help="Used when --split-mode year.")
    parser.add_argument(
        "--val-weeks",
        type=int,
        default=10,
        help="Year split only: use the last N weeks before --train-end as validation targets.",
    )
    parser.add_argument(
        "--loss",
        default="huber",
        choices=["mse", "huber"],
        help="Training loss on scaled OT head (default: huber). mse = plain squared error on OT head.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="delta for torch.nn.HuberLoss when --loss huber (tune in scaled residual units).",
    )
    parser.add_argument("--d-model", type=int, default=128, help="Autoformer d_model (hidden size).")
    parser.add_argument("--n-heads", type=int, default=8, help="Attention heads (d_model must be divisible by n_heads).")
    parser.add_argument("--e-layers", type=int, default=2, help="Encoder layers.")
    parser.add_argument("--d-layers", type=int, default=1, help="Decoder layers.")
    parser.add_argument("--d-ff", type=int, default=512, help="FFN hidden size.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout probability.")
    parser.add_argument(
        "--moving-avg",
        type=int,
        default=25,
        help="Series decomposition moving-average kernel length (use ~7 for daily weekly seasonality).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW-style L2 penalty on weights (0 disables).",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="If >0, clip gradient norm after backward (stabilizes training).",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path (relative to repo root) to write best val loss, setting, and epochs as JSON.",
    )
    args = parser.parse_args()
    ratio_sum = float(args.train_ratio) + float(args.val_ratio) + float(args.test_ratio)
    if abs(ratio_sum - 1.0) > 1e-8:
        raise SystemExit(
            f"train/val/test ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"({args.train_ratio}, {args.val_ratio}, {args.test_ratio})."
        )
    dm = int(args.d_model)
    nh = int(args.n_heads)
    if dm % nh != 0:
        raise SystemExit(f"--d-model ({dm}) must be divisible by --n-heads ({nh}).")
    print(f"Running: {Path(__file__).resolve()}")
    print(f"Parsed --loss: {args.loss!r}  --huber-delta: {args.huber_delta}")

    repo_root = Path(__file__).resolve().parents[1]
    panel_path = (repo_root / args.panel_csv).resolve()
    if not panel_path.exists():
        raise SystemExit(f"Missing panel csv: {panel_path}")

    autoformer_root = Path(args.autoformer_root).resolve()
    sys.path.insert(0, str(autoformer_root))
    from models import Autoformer  # type: ignore
    from utils.timefeatures import time_features  # type: ignore

    df = pd.read_csv(panel_path, parse_dates=["date"])
    need = {"grid_id", "date", "OT"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Panel CSV missing columns: {sorted(miss)}")

    df["grid_id"] = df["grid_id"].astype(str)
    grid_ids = sorted(df["grid_id"].unique().tolist())

    cfg = PanelConfig(
        seq_len=int(args.seq_len),
        label_len=int(args.label_len),
        pred_len=int(args.pred_len),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        epochs=int(args.epochs),
        patience=int(args.patience),
        features="MS",
        freq=str(args.freq),
        target_transform=str(args.target_transform),
    )

    sm = str(args.split_mode)
    # Create datasets/loaders (scaler fitted inside train dataset)
    train_ds = PanelWindows(
        df=df,
        grid_ids=grid_ids,
        split="train",
        cfg=cfg,
        per_grid_scalers=None,
        time_features_fn=time_features,
        split_mode=sm,
        train_end=str(args.train_end),
        test_start=str(args.test_start),
        val_weeks=int(args.val_weeks),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )
    per_grid_scalers = train_ds.per_grid_scalers
    val_ds = PanelWindows(
        df=df,
        grid_ids=grid_ids,
        split="val",
        cfg=cfg,
        per_grid_scalers=per_grid_scalers,
        time_features_fn=time_features,
        split_mode=sm,
        train_end=str(args.train_end),
        test_start=str(args.test_start),
        val_weeks=int(args.val_weeks),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )
    test_ds = PanelWindows(
        df=df,
        grid_ids=grid_ids,
        split="test",
        cfg=cfg,
        per_grid_scalers=per_grid_scalers,
        time_features_fn=time_features,
        split_mode=sm,
        train_end=str(args.train_end),
        test_start=str(args.test_start),
        val_weeks=int(args.val_weeks),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )
    if sm == "year":
        if len(train_ds) == 0:
            raise SystemExit("Year split produced no training windows. Check panel date range and --train-end.")
        if len(val_ds) == 0:
            raise SystemExit("Year split produced no validation windows. Try smaller --val-weeks or smaller seq_len/pred_len.")
        if len(test_ds) == 0:
            raise SystemExit("Year split produced no test windows. Check panel date range and --test-start.")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # Args namespace for Autoformer
    enc_in_val = len(train_ds.cols_data)
    f_dim = -1  # MS -> predict only last channel (OT)

    class _Args:
        seq_len = cfg.seq_len
        label_len = cfg.label_len
        pred_len = cfg.pred_len
        features = cfg.features
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
        freq = cfg.freq
        activation = "gelu"
        output_attention = False
        moving_avg = int(args.moving_avg)
        distil = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Autoformer.Model(_Args()).float().to(device)
    wd = float(args.weight_decay)
    if wd > 0:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=wd)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_name = str(args.loss).lower().strip()
    if loss_name == "mse":
        crit = torch.nn.MSELoss()
    elif loss_name == "huber":
        crit = torch.nn.HuberLoss(delta=float(args.huber_delta), reduction="mean")
    else:
        raise SystemExit(f"Unknown --loss={args.loss!r}")

    huber_suffix = ""
    if loss_name == "huber":
        d = float(args.huber_delta)
        huber_suffix = "_" + f"huber{d:g}".replace(".", "p")

    def _run_val(loader: DataLoader) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : cfg.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -cfg.pred_len :, f_dim:]
                true = batch_y[:, -cfg.pred_len :, f_dim:].to(device)
                loss = crit(outputs, true)
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else math.inf

    # Training loop; optional early stopping on val loss (--early-stop)
    best_val = math.inf
    best_epoch = 0
    bad = 0
    ckpt_dir = (repo_root / args.checkpoints_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    setting = (
        f"panel_Autoformer_ftMS_sl{cfg.seq_len}_ll{cfg.label_len}_pl{cfg.pred_len}_"
        f"dm{int(args.d_model)}_el{int(args.e_layers)}_dl{int(args.d_layers)}_ma{int(args.moving_avg)}_"
        f"{cfg.target_transform}{huber_suffix}_pgs"
    )
    out_path = ckpt_dir / setting
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "checkpoint.pth"

    device_name = str(device)
    print(f"Using device: {device_name}")
    print(
        f"Model: d_model={int(args.d_model)} n_heads={int(args.n_heads)} "
        f"e_layers={int(args.e_layers)} d_layers={int(args.d_layers)} d_ff={int(args.d_ff)} "
        f"dropout={float(args.dropout):g} moving_avg={int(args.moving_avg)}"
    )
    opt_name = "AdamW" if wd > 0 else "Adam"
    print(
        f"Optim: {opt_name} lr={cfg.lr:g} weight_decay={float(args.weight_decay):g} "
        f"grad_clip_norm={float(args.grad_clip_norm):g}"
    )
    print(f"Loss: {loss_name}" + (f" (delta={float(args.huber_delta):g})" if loss_name == "huber" else ""))
    print(f"Early stopping: {'on' if args.early_stop else 'off (full --epochs)'}")
    if sm == "year":
        print(f"Split mode: {sm}  train_end={args.train_end}  test_start={args.test_start}  val_weeks={args.val_weeks}")
    else:
        print(
            f"Split mode: {sm}  train_ratio={args.train_ratio}  "
            f"val_ratio={args.val_ratio}  test_ratio={args.test_ratio}"
        )

    if args.resume and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Resumed from: {ckpt_path}")

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len :, :]).float()
            dec_inp = torch.cat([batch_y[:, : cfg.label_len, :], dec_inp], dim=1).float().to(device)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -cfg.pred_len :, f_dim:]
            true = batch_y[:, -cfg.pred_len :, f_dim:].to(device)
            loss = crit(outputs, true)
            loss.backward()
            gcn = float(args.grad_clip_norm)
            if gcn > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gcn)
            optim.step()
            losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(losses)) if losses else math.inf
        val_loss = _run_val(val_loader)
        print(f"Epoch {epoch+1}/{cfg.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if args.early_stop and bad >= cfg.patience:
                print("Early stopping.")
                break

    epochs_run = epoch + 1
    print(f"Best val_loss: {best_val:.6f}  (epoch {best_epoch})  epochs_run: {epochs_run}")
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Setting: {setting}")
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}  Test samples: {len(test_ds)}")

    if args.metrics_json:
        mpath = (repo_root / str(args.metrics_json)).resolve()
        mpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "setting": setting,
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "epochs_run": int(epochs_run),
            "checkpoint": str(ckpt_path),
            "panel_csv": str(panel_path),
            "lr": float(cfg.lr),
            "d_model": int(args.d_model),
            "d_ff": int(args.d_ff),
            "dropout": float(args.dropout),
            "weight_decay": float(args.weight_decay),
            "loss": str(args.loss),
            "huber_delta": float(args.huber_delta),
        }
        mpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote metrics: {mpath}")


if __name__ == "__main__":
    main()

