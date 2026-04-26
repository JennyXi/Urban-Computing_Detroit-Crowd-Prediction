# Urban-Computing_Detroit-Crowd-Prediction

CSCI-SHU 205 Topics in Computer Science · Urban Computing Final Project (Spring 2026)

---

## 中文说明（Chinese）

### 这个仓库做什么？

本项目从 Dewey / SafeGraph 风格的 Weekly Patterns 数据中构建 **Autoformer / Transformer 可直接训练的时间序列数据集**，用于 **Detroit 人流（到访次数 visits）预测**。

当前主要包含两类建模数据：

- **Weekly（按周）预测数据集**：`data/autoformer_weekly_preprocessed.csv`
  - 1 行 = 1 周
  - 目标列：`OT`（weekly total visits）
  - 特征：周内 168 小时形状 `h_0..h_167` + 日历特征 + 波动特征
- **Hourly（按小时）预测数据集**：`data/autoformer_hourly_preprocessed.csv`
  - 1 行 = 1 小时
  - 目标列：`OT`（hourly total visits）
  - 特征：时/日/周/月周期特征 + rolling volatility 特征

### Autoformer 输入格式是否兼容？

是。生成的 CSV 已经符合 Autoformer 常用的 `Dataset_Custom` 约定：

- 第一列是 `date`
- 最后一列是 `OT`（预测目标）
- 中间列都是数值特征（covariates / channels）

也就是说：本仓库已经完成 “Autoformer 可用数据预处理”，后续只需要在官方 Autoformer 训练代码中配置 `freq / target / features` 等参数即可。

---

## 端到端流程（中文）

### Step 0. 环境准备（推荐）

建议创建虚拟环境并安装依赖。脚本最低依赖通常包括：

- `duckdb`, `pandas`, `numpy`, `pyarrow`
- （可选）`pyproj`（网格聚合）
- （可选）`folium`（地图可视化）

### Step 1. 过滤出 Detroit 数据（从 Michigan 或更大范围）

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.csv.gz" --output data\detroit_filtered.parquet
```

如果输入是 parquet shards：

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.parquet" --format parquet --output data\detroit_filtered.parquet
```

输出：`data/detroit_filtered.parquet`（很大，默认不提交到 git）

### Step 2. 快速检查（可选）

```bash
python scripts/probe_parquet_sample.py
python scripts/summarize_detroit_time_range.py
python scripts/peek_visits_by_hour.py
```

### Step 3A. 生成 weekly Autoformer 数据集（按周预测）

```bash
python scripts/preprocess_weekly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

输出：`data/autoformer_weekly_preprocessed.csv`

### Step 3B. 生成 hourly Autoformer 数据集（按小时预测）

```bash
python scripts/preprocess_hourly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

输出：`data/autoformer_hourly_preprocessed.csv`

### Step 4（可选）空间聚合数据集

CBG weekly totals（长表）：

```bash
python scripts/aggregate_cbg_weekly.py --date-start 2025-01-01 --date-end 2025-12-31 --output data\cbg_weekly_2025.parquet
```

100m × 100m 网格 weekly totals：

```bash
python scripts/aggregate_grid_weekly.py --cell-meters 100 --date-start 2025-01-01 --date-end 2025-12-31 --output data\grid100_weekly.parquet
```

网格热力图（HTML）：

```bash
python scripts/visualize_grid_osm.py --input data\grid100_weekly.parquet --week 2025-01-06 --mode heatmap --output data\grid_map.html
```

---

## 网格（grid）周预测工作流（含 log1p 与 test 验证）

目录：`use_official_autoformer_grid/`

- **说明文档**：`use_official_autoformer_grid/README.md`
- **验证脚本**：`use_official_autoformer_grid/validate_grid_predictions.py`

要点：

- 采用 “Top-100 热门网格 + 每格单独训练一个 Autoformer” 的简化策略
- 支持导出 `scope=test`（严格测试段）或 `scope=all`（滚动回测）
- 支持目标变换 `log1p`（训练时对 `OT` 做 \(ln(1+OT)\)，导出时自动还原到原始 visits 尺度）

---

## English

### What does this repo do?

This project builds **Autoformer/Transformer-ready time series datasets** for **Detroit crowd (visit) prediction** from Dewey/SafeGraph-style Weekly Patterns data.

It currently provides two main modeling datasets:

- **Weekly dataset**: `data/autoformer_weekly_preprocessed.csv`
  - 1 row = 1 week
  - Target: `OT` (weekly total visits)
  - Extra channels: within-week hourly shape `h_0..h_167` + calendar + volatility features
- **Hourly dataset**: `data/autoformer_hourly_preprocessed.csv`
  - 1 row = 1 hour
  - Target: `OT` (hourly total visits)
  - Extra channels: cyclical time features + rolling volatility features

### Autoformer input compatibility

Yes. The generated CSVs already follow the typical `Dataset_Custom` layout:

- first column: `date`
- last column: `OT` (target)
- middle columns: numeric covariates/channels

So the repo covers the preprocessing step; you only need to configure Autoformer training args (e.g., `freq`, `target`, `features`).

---

## End-to-end workflow (English)

### Step 0. Setup (recommended)

Create and activate a virtual environment, then install dependencies.

Minimum packages:

- `duckdb`, `pandas`, `numpy`, `pyarrow`
- (optional) `pyproj` for grid aggregation
- (optional) `folium` for map visualization

### Step 1. Filter Detroit-only data

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.csv.gz" --output data\detroit_filtered.parquet
```

If your source is Parquet shards:

```bash
python scripts/filter_detroit_duckdb.py --input "D:\path\to\michigan\*.parquet" --format parquet --output data\detroit_filtered.parquet
```

Output: `data/detroit_filtered.parquet` (large; intentionally ignored by git)

### Step 2. Quick sanity checks (optional)

```bash
python scripts/probe_parquet_sample.py
python scripts/summarize_detroit_time_range.py
python scripts/peek_visits_by_hour.py
```

### Step 3A. Build the weekly Autoformer dataset

```bash
python scripts/preprocess_weekly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

Output: `data/autoformer_weekly_preprocessed.csv`

### Step 3B. Build the hourly Autoformer dataset

```bash
python scripts/preprocess_hourly_for_autoformer.py --date-start 2025-01-01 --date-end 2025-12-31
```

Output: `data/autoformer_hourly_preprocessed.csv`

### Step 4 (optional). Spatial aggregation datasets

```bash
python scripts/aggregate_cbg_weekly.py --date-start 2025-01-01 --date-end 2025-12-31 --output data\cbg_weekly_2025.parquet
python scripts/aggregate_grid_weekly.py --cell-meters 100 --date-start 2025-01-01 --date-end 2025-12-31 --output data\grid100_weekly.parquet
python scripts/visualize_grid_osm.py --input data\grid100_weekly.parquet --week 2025-01-06 --mode heatmap --output data\grid_map.html
```

---

## Grid weekly workflow (log1p + test evaluation)

See `use_official_autoformer_grid/README.md`.

Highlights:

- Top-100 100m grids, one model per grid (simple panel strategy)
- Export `scope=test` (strict holdout) or `scope=all` (rolling backtest)
- Optional `log1p` target transform for stability; exports are inverted back to raw visits scale

---

## Notes

- Large files are intentionally ignored by git.
- Weekly dataset encodes within-week structure as `h_0..h_167`; hourly dataset uses real hourly timestamps.
