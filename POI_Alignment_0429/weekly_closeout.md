# Weekly Pipeline Closeout (Top-100 Grids)

## Scope Frozen

- Study scope: Detroit Top-100 crowd grids (weekly).
- Prediction source: `panel_pred_test_2025_by_date.csv` with `y_pred_mean`.
- POI alignment features: `poi_cnt_life`, `poi_cnt_transport`, `poi_cnt_economy`, `poi_cnt_public_service` (with `log1p`).
- Alignment model default: Ridge `alpha=0.1`, `target_log1p=true`.

## Main Window (Q4 2025: Oct-Dec)

- Files:
  - `alignment_oct_dec_2025.csv`
  - `alignment_ridge_coefs_oct_dec_2025.json`
  - `alignment_summary_oct_dec_2025.json`
  - `alignment_priority_candidates_oct_dec_2025.csv`
- Core fit metrics:
  - RMSE: 49252.057
  - MAE: 19537.799
  - MedAE: 6244.244
  - SMAPE: 0.3974
  - R2: 0.3147
- Action gates enabled in summary:
  - `r_alignment >= P80`
  - `n_weeks >= 12`
  - `c_bar >= P50`
  - `priority_q_1_scarcity_score >= 0.7`
- Current actionable split:
  - `tier1_actionable = 10`
  - `tier2_watchlist = 10`

## Stability Check (Q3 vs Q4)

Q3 artifacts generated:

- `alignment_jul_sep_2025.csv`
- `alignment_ridge_coefs_jul_sep_2025.json`
- `alignment_summary_jul_sep_2025.json`
- `alignment_priority_candidates_jul_sep_2025.csv`

Cross-window stability (tier1 actionable):

- Q4 actionable count: 10
- Q3 actionable count: 10
- Overlap count: 9
- Jaccard overlap: 0.8182
- Same `priority_q_1` within overlap: 9/9

Interpretation: candidate locations and recommended primary POI type are highly stable between Q3 and Q4 under the current workflow.

## Crowd Model Matrix Status

- Crowd model ranking and combined matrix generated:
  - `crowd_model_ranking.csv`
  - `performance_matrix.csv`
- Best weekly crowd model (among current runs): `cmp0428_sp8` by RMSE.

## Closeout Decision

Weekly pipeline is considered **ready to freeze** for reporting and mapping under the stated scope (Top-100 grids).

Recommended reporting outputs:

1. `performance_matrix.csv` (model performance summary)
2. `alignment_priority_candidates_oct_dec_2025.csv` (planning candidates)
3. `alignment_summary_oct_dec_2025.json` (thresholds and action counts)

## Next Step

Proceed to daily modeling in parallel, while keeping this weekly baseline unchanged for benchmark and fallback.
