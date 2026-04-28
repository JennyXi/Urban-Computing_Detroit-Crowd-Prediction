from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _parse_list_str(s: object, expected_len: int) -> list[float] | None:
    if s is None:
        return None
    if isinstance(s, float) and not np.isfinite(s):
        return None
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s or s == "[]" or s.lower() == "none":
        return None
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != expected_len:
        return None
    try:
        return [float(p) for p in parts]
    except Exception:
        return None


def _weekend_share_from_168(vals_168: list[float], order: str) -> float | None:
    if len(vals_168) != 168:
        return None
    v = np.asarray(vals_168, dtype=float)
    if not np.isfinite(v).all():
        return None
    total = float(v.sum())
    if total <= 0:
        return 0.0

    # Split into 7 days x 24 hours
    days = v.reshape(7, 24).sum(axis=1)  # len=7
    if order == "mon_sun":
        weekend = float(days[5] + days[6])  # Sat, Sun
    elif order == "sun_sat":
        weekend = float(days[0] + days[6])  # Sun, Sat
    else:
        raise ValueError(order)
    return weekend / (total + 1e-12)


def _weekend_share_from_7(vals_7: list[float], order: str) -> float | None:
    if len(vals_7) != 7:
        return None
    v = np.asarray(vals_7, dtype=float)
    if not np.isfinite(v).all():
        return None
    total = float(v.sum())
    if total <= 0:
        return 0.0

    if order == "mon_sun":
        weekend = float(v[5] + v[6])  # Sat, Sun
    elif order == "sun_sat":
        weekend = float(v[0] + v[6])  # Sun, Sat
    else:
        raise ValueError(order)
    return weekend / (total + 1e-12)


def _corr(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-check VISITS_BY_DAY weekday/weekend ordering vs VISITS_BY_EACH_HOUR.")
    ap.add_argument("--input", default="data/detroit_filtered.parquet")
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=5000)
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    pf = pq.ParquetFile(inp)
    cols = ["VISITS_BY_DAY", "VISITS_BY_EACH_HOUR"]

    hour_share_mon: list[float] = []
    hour_share_sun: list[float] = []
    day_share_mon: list[float] = []
    day_share_sun: list[float] = []

    n_seen = 0
    n_used = 0
    for batch in pf.iter_batches(columns=cols, batch_size=int(args.batch_size)):
        df = batch.to_pandas()
        for v7_s, v168_s in zip(df["VISITS_BY_DAY"].tolist(), df["VISITS_BY_EACH_HOUR"].tolist()):
            n_seen += 1
            v7 = _parse_list_str(v7_s, 7)
            v168 = _parse_list_str(v168_s, 168)
            if v7 is None or v168 is None:
                continue

            h_mon = _weekend_share_from_168(v168, "mon_sun")
            h_sun = _weekend_share_from_168(v168, "sun_sat")
            d_mon = _weekend_share_from_7(v7, "mon_sun")
            d_sun = _weekend_share_from_7(v7, "sun_sat")
            if None in (h_mon, h_sun, d_mon, d_sun):
                continue

            hour_share_mon.append(float(h_mon))
            hour_share_sun.append(float(h_sun))
            day_share_mon.append(float(d_mon))
            day_share_sun.append(float(d_sun))
            n_used += 1
            if n_used >= int(args.max_rows):
                break
        if n_used >= int(args.max_rows):
            break

    if n_used == 0:
        raise SystemExit("No usable rows found (VISITS_BY_DAY / VISITS_BY_EACH_HOUR parsing failed).")

    # Compare which day-order matches hour-derived weekend share better
    corr_mon = _corr(day_share_mon, hour_share_mon)
    corr_sun = _corr(day_share_sun, hour_share_mon)  # mismatch on purpose
    corr_sun2 = _corr(day_share_sun, hour_share_sun)
    corr_mon2 = _corr(day_share_mon, hour_share_sun)  # mismatch on purpose

    mae_mon = float(np.mean(np.abs(np.asarray(day_share_mon) - np.asarray(hour_share_mon))))
    mae_sun = float(np.mean(np.abs(np.asarray(day_share_sun) - np.asarray(hour_share_sun))))

    print(f"Rows seen: {n_seen}  rows used: {n_used}")
    print("")
    print("Compare VISITS_BY_DAY against VISITS_BY_EACH_HOUR weekend_share:")
    print(f"- Assumption mon_sun: corr(day_mon, hour_mon)={corr_mon:.6f}  MAE={mae_mon:.6f}")
    print(f"- Assumption sun_sat: corr(day_sun, hour_sun)={corr_sun2:.6f}  MAE={mae_sun:.6f}")
    print("")
    print("Sanity (should be lower if assumptions differ):")
    print(f"- corr(day_sun, hour_mon)={corr_sun:.6f}")
    print(f"- corr(day_mon, hour_sun)={corr_mon2:.6f}")


if __name__ == "__main__":
    main()

