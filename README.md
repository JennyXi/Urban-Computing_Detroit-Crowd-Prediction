# Urban-Computing_Detroit-Crowd-Prediction

CSCI-SHU 205 Topics in Computer Science · Urban Computing Final Project (Spring 2026)

---

## 中文说明（Chinese）

### 当前工作流、操作方法与目标效果

#### Weekly 封版状态（2026-04-29）

- 当前主线已完成 **Weekly crowd prediction + POI alignment** 的封版（Top-100 grids 范围）。
- 封版默认参数（`POI_Alignment_0429/compute_alignment.py`）：
  - 时间窗：`2025-10-01` ~ `2025-12-31`
  - 预测列：`y_pred_mean`
  - 特征：四类 POI 计数（默认 `log1p`）
  - 模型：Ridge `alpha=0.1` + `target_log1p=true`
- 约束筛选（action gates）已启用：
  - `r_alignment >= P80`
  - `n_weeks >= 12`
  - `c_bar >= P50`
  - `priority_q_1_scarcity_score >= 0.7`
- 当前主窗结果：`tier1_actionable=10`, `tier2_watchlist=10`。
- 跨季稳健性（Q3 vs Q4）：
  - actionable 重叠 9/11（Jaccard=0.8182）
  - 重叠格 `priority_q_1` 一致 9/9

#### 工作流（你现在在用的这条线）

1. **原始数据 → Detroit 范围**：用 `scripts/filter_detroit_duckdb.py` 得到 `data/detroit_filtered.parquet`（体积大，一般不提交 git）。  
2. **网格周汇总**：用 `scripts/aggregate_grid_weekly.py` 生成 **长表** Parquet（如 `data/grid100_weekly_2024_2025.parquet`），每行一周、一格。  
3. **Panel 表**：`panel_training_0426/build_panel_weekly_dataset.py` 选出 **Top-K 热门格**（默认按 **2024** 总 visits 排名，避免用 2025 选格的信息泄漏），补全每周、拼上 **仅用过去可得** 的城市尺度协变量（默认 `lag1`），可选 **`log1p(OT)`** 作为训练目标列 `OT`。  
4. **训练**：`panel_training_0426/train_panel_autoformer.py` 调用官方 Autoformer，在 **多网格长表** 上滑窗：**过去若干周 → 预测未来若干周**（默认 `seq_len=24`, `label_len=12`, `pred_len=4`）；**默认 Huber 损失**、**默认跑满 `--epochs`**（不加 `--early-stop` 则不会早停）；验证变好时写入 `checkpoint.pth`。  
5. **导出与对照实际**：`export_panel_predictions.py` 写出每个 `grid_id`、每个预测周的 **`y_true`（来自数据）与 `y_pred_*`**；再用 `use_official_autoformer_grid/validate_grid_predictions.py` 汇总 **整体 / 按格 / 按周** 的 MAE、RMSE、sMAPE、R² 等，并可筛 **2025** 日期只看外推段。
6. **POI 对齐（已接入）**：`POI_Alignment_0429/compute_alignment.py` 以多周平均预测 crowd（`c_bar`）对 POI-only 期望 crowd（`c_hat`）做残差 `r_alignment=c_bar-c_hat`；`summarize_alignment.py` 输出 Top 错配、优先补建类别（`priority_q_1/2`）及可行动分层（`tier1_actionable/tier2_watchlist`）。

（若做 **按格单独 Autoformer** 基线，则走 `use_official_autoformer_grid/`，与 Panel 并行对比。）

#### 操作方法（最小可复现顺序）

在项目根目录、已激活的虚拟环境下（示例路径请改成你自己的）：

1. 确认已有 **`data/grid100_weekly_2024_2025.parquet`**（或改 `build` 的 `--input`）。  
2. 构建 Panel：见 **`panel_training_0426/README.md`** 里的 `build_panel_weekly_dataset.py` 一行命令。  
3. 训练：`train_panel_autoformer.py --autoformer-root "<Autoformer 仓库路径>"`；若要 **MSE 基线** 加 `--loss mse`；若要早停加 `--early-stop`。  
4. 导出：`export_panel_predictions.py`；若 `checkpoints` 下有多套实验目录，务必加 **`--setting <文件夹名>`** 指到本次训练对应目录。  
5. 验证：`validate_grid_predictions.py --pred-by-date panel_training_0426/outputs/panel_pred_test_2025_by_date.csv`，需要时加 **`--date-start` / `--date-end`** 限定评估窗口。

详细命令可复制 **`panel_training_0426/README.md`**；根目录下文 **`scripts/`** 一节是 **城市级 / 聚合** 前序步骤。

#### 目标效果（你希望看到什么）

- **数值上**：在测试/外推段上，**整体** RMSE、MAE、R²、相关系数合理；**按格** 能看出多数 Top-K 格预测与真实周序形状一致；**最差若干格/若干周** 可定位、可解释（活动、体量阶跃等），而不是「全盘不可学」。  
- **方法上**：同一套 **滑窗 + 共享参数** 在 **多格** 上成立；可选对比 **Huber vs MSE**、**有无城市协变量**、不同 **`--city-cov`**，用同一验证脚本输出表格与 `best/worst_grids_topN`。  
- **报告/答辩上**：能说清 **数据怎么来、Top-K 怎么选、协变量为什么只用滞后、预测与真实怎么比、误差主要来自哪些格/周**。
- **规划落地上**：在 Top-100 热点格中，给出“**哪里错配高 + 更建议补哪类 POI**”的可行动清单，并与观察清单分层展示（tier1/tier2）。

---

### 项目概览

本仓库从 Dewey / SafeGraph 风格的 **Weekly Patterns** 数据中，整理 **Detroit 人流（visits）** 相关数据，并对接 **官方 Autoformer**（`thuml/Autoformer`）做时间序列预测实验。主要产出包括：

| 路线 | 说明 | 典型输出 / 目录 |
|------|------|------------------|
| **Panel 共享模型（推荐主线）** | Top-K 网格、**一个** Autoformer 学所有格；周频滑窗（如 24→4） | `panel_training_0426/` |
| **按格单独模型** | Top-100 网格、每格一个 Autoformer（简化 baseline） | `use_official_autoformer_grid/` |
| **城市级周序列** | 1 行 = 1 周，`h_0..h_167` 等通道 | `data/autoformer_weekly_preprocessed.csv` |
| **城市级小时序列** | 1 行 = 1 小时 | `data/autoformer_hourly_preprocessed.csv` |
| **空间聚合** | CBG / 100m 网格周汇总 | `scripts/aggregate_*` → `data/*.parquet` |

所有供 Autoformer 使用的 CSV 均遵循常见 **`Dataset_Custom`** 约定：第一列 `date`，最后一列 **`OT`**（预测目标），中间列为数值协变量。

---

### 推荐主线：Panel 多网格 + 共享 Autoformer（`panel_training_0426/`）

**思路**：每个训练样本来自 **某一个 grid_id** 上的一段连续周序列；模型输入为多通道（静态格属性、可选城市尺度滞后协变量等），目标仍为该周的 **`OT`（visits）**；`features=MS`，在官方 Autoformer 代码中训练 **单个** 共享权重。

**依赖**：本机需有官方 [Autoformer](https://github.com/thuml/Autoformer) 源码目录；Python 环境需包含 `torch`、`sklearn`、`pandas`、`numpy`、`pyarrow` 等（与 `duckdb` 等预处理依赖可并存）。

**数据前提**：长表网格周数据，例如 `data/grid100_weekly_2024_2025.parquet`（由 `scripts/aggregate_grid_weekly.py` 按需要时段聚合生成）。

**三步流程（命令细节与路径以子目录说明为准）**：

1. **构建 Panel CSV**：`panel_training_0426/build_panel_weekly_dataset.py`  
   - 默认 **Top-K 按 2024 总流量排名**（减少目标年信息泄漏）；可选 **城市周尺度协变量**（`--city-cov`：`none` / `lag1` / `lag4` / `roll4_mean_lag1`，均为**仅用过去可得**的量）；目标可做 **`log1p`** 写入 `OT`。  
2. **训练**：`panel_training_0426/train_panel_autoformer.py --autoformer-root "<你的Autoformer目录>"`  
   - 默认 **Huber** 损失（`--huber-delta 1.0`）；**`--loss mse`** 为 MSE 基线。  
   - 默认跑满 **`--epochs`**，**不使用 early stopping**；需要早停时加 **`--early-stop`**，并用 **`--patience`**。  
   - Checkpoint 目录名：Huber 为 `..._log1p_huber1` 等；纯 MSE 为 `..._log1p`。  
3. **导出与验证**：`export_panel_predictions.py`（建议显式 **`--setting`** 指向对应 checkpoint 文件夹）→ `use_official_autoformer_grid/validate_grid_predictions.py` 对比 `y_true` / `y_pred_*`。

**更完整的逐步命令与说明**：请阅读 **`panel_training_0426/README.md`**。

---

### 备选：按网格单独训练（`use_official_autoformer_grid/`）

每格独立训练与导出；适合与 Panel 共享模型做对比。说明与脚本见 **`use_official_autoformer_grid/README.md`**；预测质量汇总可用 **`validate_grid_predictions.py`**（需含 `grid_id, date, y_true, y_pred_last, y_pred_mean` 的 by-date 表）。

---

### 端到端数据准备（`scripts/`）

#### Step 0. 环境（建议）

虚拟环境中建议安装：`duckdb`, `pandas`, `numpy`, `pyarrow`；网格相关可选 `pyproj`；地图可选 `folium`。Panel 训练另需 `torch`, `scikit-learn`。

#### Step 1. 过滤 Detroit（从 Michigan 或更大范围）

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.csv.gz" --output data/detroit_filtered.parquet
```

Parquet 分片输入：

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.parquet" --format parquet --output data/detroit_filtered.parquet
```

#### Step 2. 快速检查（可选）

```bash
python scripts/probe_parquet_sample.py
python scripts/summarize_detroit_time_range.py
python scripts/peek_visits_by_hour.py
```

#### Step 3A. 城市级 weekly Autoformer CSV

```bash
python scripts/preprocess_weekly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

输出：`data/autoformer_weekly_preprocessed.csv`

#### Step 3B. 城市级 hourly Autoformer CSV

```bash
python scripts/preprocess_hourly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

输出：`data/autoformer_hourly_preprocessed.csv`

#### Step 4. 空间聚合（可选，Panel 前序）

```bash
python scripts/aggregate_cbg_weekly.py --date-start 2025-01-01 --date-end 2025-12-31 --output data/cbg_weekly_2025.parquet
python scripts/aggregate_grid_weekly.py --cell-meters 100 --date-start 2024-01-01 --date-end 2025-12-31 --output data/grid100_weekly_2024_2025.parquet
python scripts/visualize_grid_osm.py --input data/grid100_weekly_2024_2025.parquet --week 2025-01-06 --mode heatmap --output data/grid_map.html
```

（Windows 下路径分隔符可按习惯使用 `\`；大文件默认不提交 git。）

---

## English

### Current workflow, how-to, and intended outcomes

#### Weekly freeze status (2026-04-29)

- The weekly track is now frozen for **weekly crowd prediction + POI alignment** (Top-100 grids scope).
- Freeze defaults (`POI_Alignment_0429/compute_alignment.py`):
  - Window: `2025-10-01` ~ `2025-12-31`
  - Prediction column: `y_pred_mean`
  - Features: 4 POI counts (with default `log1p`)
  - Model: Ridge `alpha=0.1` + `target_log1p=true`
- Action gates are enabled:
  - `r_alignment >= P80`
  - `n_weeks >= 12`
  - `c_bar >= P50`
  - `priority_q_1_scarcity_score >= 0.7`
- Main-window split: `tier1_actionable=10`, `tier2_watchlist=10`.
- Cross-window stability (Q3 vs Q4):
  - actionable overlap Jaccard = `0.8182`
  - primary type consistency (`priority_q_1`) = `9/9` on overlap

#### Workflow (the track you are using)

1. **Raw → Detroit subset**: `scripts/filter_detroit_duckdb.py` → `data/detroit_filtered.parquet` (large; usually git-ignored).  
2. **Grid-weekly long table**: `scripts/aggregate_grid_weekly.py` → e.g. `data/grid100_weekly_2024_2025.parquet` (one row per grid × week).  
3. **Panel CSV**: `panel_training_0426/build_panel_weekly_dataset.py` selects **Top-K grids** (default: rank by **2024** total visits to avoid target-year selection leakage), fills missing weeks, adds **past-only** city-level covariates (default `lag1`), optional **`log1p`** on `OT` for training stability.  
4. **Train**: `train_panel_autoformer.py` calls the official Autoformer on the panel: **past L weeks → forecast H weeks** (defaults `seq_len=24`, `label_len=12`, `pred_len=4`). **Huber loss by default**, **full `--epochs` by default** (no early stopping unless `--early-stop`); `checkpoint.pth` updates when validation improves.  
5. **Export vs ground truth**: `export_panel_predictions.py` writes per-`grid_id`, per-date **`y_true` (from data) and `y_pred_*`**; `use_official_autoformer_grid/validate_grid_predictions.py` aggregates **global / per-grid / per-date** MAE, RMSE, sMAPE, R², etc., with optional **2025-only** filters.
6. **POI alignment (integrated)**: `POI_Alignment_0429/compute_alignment.py` fits POI-only expected crowd (`c_hat`) against multi-week mean predicted crowd (`c_bar`) and computes `r_alignment=c_bar-c_hat`; `summarize_alignment.py` exports top mismatch grids, recommended POI type (`priority_q_1/2`), and action tiers (`tier1_actionable/tier2_watchlist`).

(For a **one-model-per-grid** baseline, use `use_official_autoformer_grid/` in parallel.)

#### How to run (minimal reproducible order)

From repo root, in your activated venv (edit paths locally):

1. Ensure **`data/grid100_weekly_2024_2025.parquet`** exists (or pass `--input` to `build`).  
2. Build panel: see **`panel_training_0426/README.md`** (`build_panel_weekly_dataset.py`).  
3. Train: `train_panel_autoformer.py --autoformer-root "<path/to/Autoformer>"`; add **`--loss mse`** for an MSE baseline; add **`--early-stop`** if you want patience-based stopping.  
4. Export: `export_panel_predictions.py`; if multiple runs exist under `checkpoints/`, pass **`--setting <folder_name>`** explicitly.  
5. Validate: `validate_grid_predictions.py --pred-by-date panel_training_0426/outputs/panel_pred_test_2025_by_date.csv` (optional **`--date-start` / `--date-end`**).

Full copy-paste commands: **`panel_training_0426/README.md`**. The **`scripts/`** section below covers **city-level preprocessing and aggregation** upstream.

#### Intended outcomes (what “good” means)

- **Metrics**: reasonable global RMSE/MAE/R²/corr on the held-out / exported segment; per-grid diagnostics show most Top-K grids follow the true weekly pattern; worst grids/dates are **identifiable and interpretable** (events, scale shifts), not a fully broken model.  
- **Method**: one **shared** Autoformer with sliding windows works **across grids**; you can compare **Huber vs MSE**, **city cov on/off**, and **`--city-cov` variants** with the same validator (`best/worst_grids_topN`, etc.).  
- **Write-up / defense**: a clear story—**data provenance, Top-K rule, why covariates are lagged only, how preds are compared to truth, where errors concentrate**.
- **Planning outputs**: an actionable mismatch list in Top-100 hotspots with suggested POI category priorities and watchlist separation (tier1/tier2).

---

### What this repo does

This repository prepares **Detroit crowd-flow (visit)** time-series data from Dewey/SafeGraph-style **Weekly Patterns**, and wires experiments to the **official Autoformer** codebase (`thuml/Autoformer`). Main tracks:

| Track | Description | Where |
|--------|-------------|--------|
| **Panel shared model (recommended)** | Top-K grids, **one** Autoformer for all series; weekly sliding windows (e.g. 24→4) | `panel_training_0426/` |
| **Per-grid baselines** | Top-100 grids, one Autoformer per grid | `use_official_autoformer_grid/` |
| **City-level weekly series** | 1 row = 1 week, channels incl. `h_0..h_167` | `data/autoformer_weekly_preprocessed.csv` |
| **City-level hourly series** | 1 row = 1 hour | `data/autoformer_hourly_preprocessed.csv` |
| **Spatial aggregation** | CBG / 100m grid weekly totals | `scripts/aggregate_*` → `data/*.parquet` |

Autoformer-ready CSVs follow the usual **`Dataset_Custom`** layout: first column `date`, last column **`OT`**, numeric covariates in between.

---

### Recommended track: Panel + shared Autoformer (`panel_training_0426/`)

**Idea**: each training sample is a sliding window on **one `grid_id`**; inputs are multivariate (static grid fields, optional **past-only** city-level covariates, etc.); target is weekly **`OT` (visits)** with `features=MS`; **one** shared checkpoint is trained via the official `models.Autoformer`.

**Requirements**: a local checkout of [Autoformer](https://github.com/thuml/Autoformer); Python with `torch`, `sklearn`, `pandas`, `numpy`, `pyarrow` (plus preprocessing libs such as `duckdb` as needed).

**Data prerequisite**: long-format grid-weekly Parquet, e.g. `data/grid100_weekly_2024_2025.parquet`, produced by `scripts/aggregate_grid_weekly.py` for your date span.

**Three steps** (exact flags/paths: see **`panel_training_0426/README.md`**):

1. **Build panel CSV**: `build_panel_weekly_dataset.py` — default **Top-K ranked on 2024 totals**; **`--city-cov`** chooses past-only city features (`none` / `lag1` / `lag4` / `roll4_mean_lag1`); optional **`log1p`** on `OT` in the CSV.  
2. **Train**: `train_panel_autoformer.py --autoformer-root "<path/to/Autoformer>"` — default **Huber** loss (`--huber-delta 1.0`); use **`--loss mse`** for an MSE baseline.  
   - By default runs all **`--epochs`** with **no early stopping**; pass **`--early-stop`** (and **`--patience`**) to enable it.  
   - Checkpoint folder: Huber runs include a suffix like `..._log1p_huber1`; pure MSE uses `..._log1p`.  
3. **Export & validate**: `export_panel_predictions.py` (prefer explicit **`--setting`**) → `validate_grid_predictions.py` on the exported by-date CSV.

---

### Alternative: per-grid training (`use_official_autoformer_grid/`)

One model per grid for comparison with the panel model. See **`use_official_autoformer_grid/README.md`**. Use **`validate_grid_predictions.py`** on prediction tables that include `grid_id, date, y_true, y_pred_last, y_pred_mean`.

---

### End-to-end data prep (`scripts/`)

#### Step 0. Environment (recommended)

Use a venv with at least `duckdb`, `pandas`, `numpy`, `pyarrow`; optional `pyproj`, `folium`. Panel training additionally needs `torch`, `scikit-learn`.

#### Step 1. Filter Detroit-only rows

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.csv.gz" --output data/detroit_filtered.parquet
```

Parquet shards:

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.parquet" --format parquet --output data/detroit_filtered.parquet
```

#### Step 2. Quick sanity checks (optional)

```bash
python scripts/probe_parquet_sample.py
python scripts/summarize_detroit_time_range.py
python scripts/peek_visits_by_hour.py
```

#### Step 3A. City-level weekly Autoformer CSV

```bash
python scripts/preprocess_weekly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

Output: `data/autoformer_weekly_preprocessed.csv`

#### Step 3B. City-level hourly Autoformer CSV

```bash
python scripts/preprocess_hourly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

Output: `data/autoformer_hourly_preprocessed.csv`

#### Step 4. Spatial aggregation (optional, upstream of Panel)

```bash
python scripts/aggregate_cbg_weekly.py --date-start 2025-01-01 --date-end 2025-12-31 --output data/cbg_weekly_2025.parquet
python scripts/aggregate_grid_weekly.py --cell-meters 100 --date-start 2024-01-01 --date-end 2025-12-31 --output data/grid100_weekly_2024_2025.parquet
python scripts/visualize_grid_osm.py --input data/grid100_weekly_2024_2025.parquet --week 2025-01-06 --mode heatmap --output data/grid_map.html
```

Large generated files are git-ignored by design.

---

## Notes / 备注

- **English**: Weekly city-level CSV encodes within-week shape as `h_0..h_167`; hourly CSV uses real timestamps. Panel training consumes **grid-weekly** Parquet + panel CSV, not the city-level `autoformer_weekly_preprocessed.csv` unless you adapt the pipeline.  
- **中文**：城市级周数据用 `h_0..h_167` 编码周内形状；小时数据用真实时间戳。Panel 训练使用的是 **网格周 Parquet → panel CSV**，与城市级 `autoformer_weekly_preprocessed.csv` 是不同一条线，除非自行改接。
