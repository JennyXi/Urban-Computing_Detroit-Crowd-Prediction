## 4-week smoke test: training + evaluation summary (CPU, 3 epochs)

### Goal

Quickly validate the end-to-end pipeline (dataset loading → training → testing) and compare 3 models under the **same settings**:

- Autoformer
- Informer
- Transformer

This is a **small-sample sanity check**, not the final best-tuned result.

### Dataset

- File: `data/autoformer_hourly_4w_small.csv`
- Task: `features=MS`, `target=OT`, `freq=h`
- Sliding windows (reported by the loader):
  - train windows: `279`
  - val windows: `45`
  - test windows: `111`

### Common run settings (shared across all 3 models)

- `seq_len=168`, `label_len=48`, `pred_len=24`
- `enc_in=11`, `dec_in=11`, `c_out=1`
- `e_layers=2`, `d_layers=1`
- `d_model=128`, `n_heads=4`, `d_ff=256`
- `dropout=0.1`
- `batch_size=32`
- `train_epochs=3`, `itr=1`
- device: **CPU** (`--use_gpu 0`)

### Results (final test metrics)

Lower is better.

| Model        | MSE       | MAE       |
|-------------|-----------:|----------:|
| Autoformer   | 0.155056  | 0.296840  |
| Informer     | 0.955375  | 0.857600  |
| Transformer  | 0.994141  | 0.890341  |

### Interpretation

- Under this small 4-week setting, **Autoformer clearly outperformed** Informer and the vanilla Transformer baseline.
- The difference is large in both MSE and MAE, suggesting Autoformer fits the short periodic structure (weekly cycle) more effectively in this setup.

### Notes / limitations

- Because this is only 4 weeks of data and only 3 epochs, results can be noisy and should be treated as a pipeline check + rough comparison.
- For a stronger conclusion, increase the training span (e.g., 8–12 weeks or full year) and run longer training (with early stopping).

