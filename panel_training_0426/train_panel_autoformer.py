from __future__ import annotations

import argparse
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
        scaler: StandardScaler | None = None,
        time_features_fn=None,
    ):
        assert split in {"train", "val", "test"}
        self.cfg = cfg
        self.split = split
        self.grid_ids = grid_ids
        self.time_features_fn = time_features_fn

        # Build per-grid arrays (already sorted by date)
        self.series = {}
        for gid in grid_ids:
            g = df[df["grid_id"] == gid].sort_values("date")
            self.series[gid] = g.reset_index(drop=True)

        # columns: numeric covariates + OT last
        # We keep ordering like Dataset_Custom: date + covariates + OT
        self.cov_cols = [c for c in df.columns if c not in {"grid_id", "date", "OT"}]
        self.cols_data = self.cov_cols + ["OT"]

        # Compute Dataset_Custom borders per grid (70/10/20), then build window indices
        self.indices: list[tuple[str, int, int, int]] = []  # (gid, border1, border2, i)
        for gid, g in self.series.items():
            n = len(g)
            num_train = int(n * 0.7)
            num_test = int(n * 0.2)
            num_val = n - num_train - num_test
            border1s = [0, num_train - cfg.seq_len, n - num_test - cfg.seq_len]
            border2s = [num_train, num_train + num_val, n]
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
                self.indices.append((gid, border1, border2, i))

        # Fit/attach scaler on train partition only (matches Dataset_Custom spirit but across all grids)
        if scaler is None:
            scaler = StandardScaler()
            train_rows = []
            for gid, g in self.series.items():
                n = len(g)
                num_train = int(n * 0.7)
                # fit on [0:num_train]
                train_rows.append(g.loc[: num_train - 1, self.cols_data])
            train_mat = pd.concat(train_rows, ignore_index=True).to_numpy(dtype=np.float32)
            scaler.fit(train_mat)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        gid, border1, _border2, i = self.indices[idx]
        g = self.series[gid]
        cfg = self.cfg
        g0 = border1 + i
        s_begin, s_end = g0, g0 + cfg.seq_len
        r_begin = s_end - cfg.label_len
        r_end = r_begin + cfg.label_len + cfg.pred_len

        # scaled numeric data (covariates+OT)
        mat = g.loc[:, self.cols_data].to_numpy(dtype=np.float32)
        mat = self.scaler.transform(mat).astype(np.float32)

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
    parser.add_argument("--panel-csv", default="panel_training_0426/outputs/panel_weekly_top100_2024_2025.csv")
    parser.add_argument("--autoformer-root", required=True, help="Path to official Autoformer repo (thuml/Autoformer).")
    parser.add_argument("--checkpoints-dir", default="panel_training_0426/checkpoints")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--label-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping and always run all epochs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if present.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freq", default="w")
    parser.add_argument("--target-transform", default="log1p", choices=["none", "log1p"])
    args = parser.parse_args()

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

    # Create datasets/loaders (scaler fitted inside train dataset)
    train_ds = PanelWindows(df=df, grid_ids=grid_ids, split="train", cfg=cfg, scaler=None, time_features_fn=time_features)
    scaler = train_ds.scaler
    val_ds = PanelWindows(df=df, grid_ids=grid_ids, split="val", cfg=cfg, scaler=scaler, time_features_fn=time_features)
    test_ds = PanelWindows(df=df, grid_ids=grid_ids, split="test", cfg=cfg, scaler=scaler, time_features_fn=time_features)

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
        d_model = 128
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 512
        factor = 1
        dropout = 0.05
        embed = "timeF"
        freq = cfg.freq
        activation = "gelu"
        output_attention = False
        moving_avg = 25
        distil = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Autoformer.Model(_Args()).float().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = torch.nn.MSELoss()

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

    # Training loop with simple early stopping on val loss
    best_val = math.inf
    bad = 0
    ckpt_dir = (repo_root / args.checkpoints_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    setting = f"panel_Autoformer_ftMS_sl{cfg.seq_len}_ll{cfg.label_len}_pl{cfg.pred_len}_dm128_el2_dl1_{cfg.target_transform}"
    out_path = ckpt_dir / setting
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "checkpoint.pth"

    device_name = str(device)
    print(f"Using device: {device_name}")

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
            optim.step()
            losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(losses)) if losses else math.inf
        val_loss = _run_val(val_loader)
        print(f"Epoch {epoch+1}/{cfg.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if (not args.no_early_stop) and bad >= cfg.patience:
                print("Early stopping.")
                break

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Setting: {setting}")
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}  Test samples: {len(test_ds)}")


if __name__ == "__main__":
    main()

