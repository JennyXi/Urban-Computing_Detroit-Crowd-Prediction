"""
Stitch multiple 7-day Autoformer windows into one longer contiguous forecast.

Training stays seq_len=84, pred_len=7. Each inference window predicts the 7 days
starting at calendar row (g0 + seq_len). If the next window's window_start is
exactly 7 days later, its first predicted day continues the previous block with
no gap (same logic as rolling origin with step=7).

Input: panel_pred_*_long.csv from export_panel_predictions.py
Output: CSV with one row per (grid_id, stitched_day_index) covering passes*7 days.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="panel_pred_*_long.csv")
    p.add_argument("--output", required=True, help="Output stitched CSV path")
    p.add_argument("--passes", type=int, default=4, help="Number of 7-day chunks (4 => 28 days)")
    p.add_argument(
        "--target-days",
        type=int,
        default=None,
        help="Trim stitched series to this many days (e.g. 30). Default: passes * 7.",
    )
    p.add_argument(
        "--start-on-or-after",
        default=None,
        help="Only consider window_start on/after this date (YYYY-MM-DD). First valid chain per grid is kept.",
    )
    args = p.parse_args()

    inp = Path(args.input).resolve()
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    df = pd.read_csv(inp, parse_dates=["date", "window_start"])
    need = {"grid_id", "window_start", "horizon", "y_pred", "y_true"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Input missing columns: {sorted(miss)}")

    pred_len = int(df["horizon"].max())
    if pred_len != 7:
        raise SystemExit(
            f"This stitcher assumes pred_len=7 (horizon 1..7); got max horizon={pred_len}. "
            "Export with --pred-len 7."
        )

    start_floor = None
    if args.start_on_or_after:
        start_floor = pd.Timestamp(args.start_on_or_after).normalize()

    step = pd.Timedelta(days=7)
    n_chunk = int(args.passes)
    target_days = int(args.target_days) if args.target_days is not None else n_chunk * 7

    out_rows: list[dict] = []

    for gid, g in df.groupby("grid_id", sort=False):
        g = g.copy()
        g["window_start"] = pd.to_datetime(g["window_start"]).dt.normalize()

        by_ws: dict[pd.Timestamp, pd.DataFrame] = {}
        for ws, block in g.groupby("window_start", sort=False):
            ws = pd.Timestamp(ws).normalize()
            if start_floor is not None and ws < start_floor:
                continue
            b = block.sort_values("horizon")
            if len(b) != 7 or set(b["horizon"]) != set(range(1, 8)):
                continue
            by_ws[ws] = b

        if len(by_ws) < n_chunk:
            continue

        ws_sorted = sorted(by_ws.keys())
        chosen: list[pd.Timestamp] | None = None
        for ws0 in ws_sorted:
            chain = [ws0 + step * k for k in range(n_chunk)]
            if all(w in by_ws for w in chain):
                chosen = chain
                break

        if not chosen:
            continue

        stitched_day = 0
        stop_gid = False
        for ws in chosen:
            if stop_gid:
                break
            block = by_ws[ws].sort_values("horizon")
            for _, r in block.iterrows():
                stitched_day += 1
                if stitched_day > target_days:
                    stop_gid = True
                    break
                out_rows.append(
                    {
                        "grid_id": gid,
                        "stitched_day": stitched_day,
                        "date": r["date"],
                        "y_true": r["y_true"],
                        "y_pred": r["y_pred"],
                        "source_window_start": ws,
                        "source_horizon": int(r["horizon"]),
                    }
                )

    if not out_rows:
        raise SystemExit("No stitched chains found. Check window_start spacing (need +7d steps) and filters.")

    out = pd.DataFrame(out_rows).sort_values(["grid_id", "stitched_day"]).reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
