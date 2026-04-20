## Hourly dataset: structure and meaning (for report)

### Dataset name / file

- `data/autoformer_hourly_preprocessed.csv` (full year)
- `data/autoformer_hourly_4w_small.csv` (4-week subset used in the smoke test)

### One-line summary

This dataset converts Dewey/SafeGraph-style weekly vectors `VISITS_BY_EACH_HOUR` into a **true hourly time series**:

- **1 row = 1 hour**
- **target = `OT`** (Detroit-wide hourly total visits)
- other columns are covariates: **time cycles** + **recent volatility/state**

### Row / time axis

- `date`: hourly timestamp, format `YYYY-MM-DD HH:MM:SS`

### Column semantics (hourly)

#### `date`
- **Meaning**: the timestamp of the row (hourly).
- **Role in Autoformer**: used to build time encodings (month/day/weekday/hour) and align sequences.

#### `OT`
- **Meaning**: Detroit-wide **hourly total visits** (sum of visits across all POIs for that hour).
- **Role in modeling**: the prediction target (set `--target OT`).

#### Cyclical time features (help the model learn periodic patterns)

These encode periodic time variables as sine/cosine pairs (so that end/start of a cycle are close, e.g., hour 23 and hour 0).

- `hour_sin`, `hour_cos`
  - **Meaning**: hour-of-day cycle (period 24).
  - **Intuition**: daily rhythm (morning/evening peaks, nighttime trough).

- `dow_sin`, `dow_cos`
  - **Meaning**: day-of-week cycle (period 7).
  - **Intuition**: weekday vs weekend effects.

- `m_sin`, `m_cos`
  - **Meaning**: month-of-year cycle (period 12).
  - **Intuition**: seasonal shifts (winter/summer, school terms, holidays).

#### Rolling/volatility features (recent state of the series)

All are computed from the hourly visits series with a rolling window of **168 hours (7 days)** by default.

- `visits_roll_mean`
  - **Meaning**: rolling mean of hourly visits over the last 168 hours.
  - **Why useful**: gives the recent baseline level.

- `visits_roll_std`
  - **Meaning**: rolling standard deviation over the last 168 hours.
  - **Why useful**: captures how stable/volatile the recent week has been.

- `visits_logdiff`
  - **Meaning**: first difference of `log(1 + visits)` (a stabilized change-rate signal).
  - **Why useful**: indicates recent upward/downward movement while reducing sensitivity to extreme counts.

- `visits_z`
  - **Meaning**: z-score relative to the recent window:
    \[
    (visits - visits\_roll\_mean) / visits\_roll\_std
    \]
    (with safe handling when std is near zero).
  - **Why useful**: highlights anomalies/spikes relative to recent behavior.

### How Autoformer consumes it (MS task)

With `features=MS` in thuml/Autoformer `Dataset_Custom`:

- **inputs (`x`)**: all numeric columns except `date` (including `OT` history)
- **outputs (`y`)**: only `OT` (because you set `c_out=1` and `--target OT`)

In your smoke test:
- `seq_len=168` (use past 7 days)
- `pred_len=24` (predict next 24 hours)

