@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Compare panel training: base vs spatial neighbor covariates
REM - base: --spatial-cov none
REM - sp8 : --spatial-cov nbr8_meanstd_lag1
REM Outputs are written to separate folders to avoid overwriting.
REM ============================================================

REM --- Repo root (this script's directory) ---
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM --- Python (prefer .venv) ---
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

REM --- Autoformer official repo path (EDIT THIS IF NEEDED) ---
set "AUTOFORMER_ROOT=E:\Urban Computing Final Project\autoformer_spatial_0425\Autoformer"

REM --- Common data build args (edit if needed) ---
set "DATE_START=2024-01-01"
set "DATE_END=2025-12-31"
set "TOPK=100"
set "TOPK_YEAR=2024"
set "CITY_COV=lag1"
set "WEEKEND_COV=share_lag1"
set "TARGET_TRANSFORM=log1p"

REM --- Train/export split (strict 2024->2025) ---
set "SPLIT_MODE=year"
set "TRAIN_END=2024-12-31"
set "TEST_START=2025-01-01"
set "VAL_WEEKS=10"

REM --- Model hyperparams (match your defaults) ---
set "SEQ_LEN=24"
set "LABEL_LEN=12"
set "PRED_LEN=4"
set "BATCH=64"
set "EPOCHS=20"
set "LOSS=huber"
set "HUBER_DELTA=1.0"

REM Setting folder name (must match train_panel_autoformer.py logic)
set "SETTING=panel_Autoformer_ftMS_sl%SEQ_LEN%_ll%LABEL_LEN%_pl%PRED_LEN%_dm128_el2_dl1_%TARGET_TRANSFORM%_huber1"

echo.
echo === Using python: %PY%
echo === Repo root: %ROOT%
echo === Autoformer: %AUTOFORMER_ROOT%
echo.

REM ------------------------------------------------------------
REM 1) Build panel CSVs
REM ------------------------------------------------------------
echo [1/4] Build panel CSVs (base vs spatial)...

set "TAG_BASE=cmp0428_base"
set "TAG_SP=cmp0428_sp8"

"%PY%" panel_training_0426\build_panel_weekly_dataset.py ^
  --top-k %TOPK% --date-start %DATE_START% --date-end %DATE_END% ^
  --topk-year %TOPK_YEAR% --city-cov %CITY_COV% --weekend-cov %WEEKEND_COV% ^
  --spatial-cov none --target-transform %TARGET_TRANSFORM% ^
  --tag %TAG_BASE%
if errorlevel 1 goto :fail

"%PY%" panel_training_0426\build_panel_weekly_dataset.py ^
  --top-k %TOPK% --date-start %DATE_START% --date-end %DATE_END% ^
  --topk-year %TOPK_YEAR% --city-cov %CITY_COV% --weekend-cov %WEEKEND_COV% ^
  --spatial-cov nbr8_meanstd_lag1 --target-transform %TARGET_TRANSFORM% ^
  --tag %TAG_SP%
if errorlevel 1 goto :fail

set "PANEL_BASE=panel_training_0426/outputs/panel_weekly_top%TOPK%_2024_2025_%TAG_BASE%.csv"
set "PANEL_SP=panel_training_0426/outputs/panel_weekly_top%TOPK%_2024_2025_%TAG_SP%.csv"

echo    base panel: %PANEL_BASE%
echo    sp8  panel: %PANEL_SP%
echo.

REM ------------------------------------------------------------
REM 2) Train (separate checkpoints dirs to avoid overwriting)
REM ------------------------------------------------------------
echo [2/4] Train base model...
"%PY%" panel_training_0426\train_panel_autoformer.py ^
  --panel-csv "%PANEL_BASE%" ^
  --autoformer-root "%AUTOFORMER_ROOT%" ^
  --checkpoints-dir panel_training_0426/checkpoints_cmp/%TAG_BASE% ^
  --seq-len %SEQ_LEN% --label-len %LABEL_LEN% --pred-len %PRED_LEN% ^
  --batch-size %BATCH% --epochs %EPOCHS% ^
  --loss %LOSS% --huber-delta %HUBER_DELTA% ^
  --split-mode %SPLIT_MODE% --train-end %TRAIN_END% --test-start %TEST_START% --val-weeks %VAL_WEEKS%
if errorlevel 1 goto :fail

echo.
echo [2/4] Train spatial-lag model (8-neighborhood)...
"%PY%" panel_training_0426\train_panel_autoformer.py ^
  --panel-csv "%PANEL_SP%" ^
  --autoformer-root "%AUTOFORMER_ROOT%" ^
  --checkpoints-dir panel_training_0426/checkpoints_cmp/%TAG_SP% ^
  --seq-len %SEQ_LEN% --label-len %LABEL_LEN% --pred-len %PRED_LEN% ^
  --batch-size %BATCH% --epochs %EPOCHS% ^
  --loss %LOSS% --huber-delta %HUBER_DELTA% ^
  --split-mode %SPLIT_MODE% --train-end %TRAIN_END% --test-start %TEST_START% --val-weeks %VAL_WEEKS%
if errorlevel 1 goto :fail

echo.

REM ------------------------------------------------------------
REM 3) Export predictions (write to separate out dirs)
REM ------------------------------------------------------------
echo [3/4] Export + validate base...
"%PY%" panel_training_0426\export_panel_predictions.py ^
  --panel-csv "%PANEL_BASE%" ^
  --autoformer-root "%AUTOFORMER_ROOT%" ^
  --checkpoints-dir panel_training_0426/checkpoints_cmp/%TAG_BASE% ^
  --setting %SETTING% ^
  --out-dir panel_training_0426/outputs_cmp/%TAG_BASE% ^
  --scope test --target-year 2025 ^
  --seq-len %SEQ_LEN% --label-len %LABEL_LEN% --pred-len %PRED_LEN% ^
  --split-mode %SPLIT_MODE% --train-end %TRAIN_END% --test-start %TEST_START% --val-weeks %VAL_WEEKS%
if errorlevel 1 goto :fail

"%PY%" use_official_autoformer_grid\validate_grid_predictions.py ^
  --pred-by-date panel_training_0426/outputs_cmp/%TAG_BASE%/panel_pred_test_2025_by_date.csv ^
  --out-dir panel_training_0426/outputs_cmp/%TAG_BASE%/validation ^
  --top-n 20
if errorlevel 1 goto :fail

echo.
echo [3/4] Export + validate spatial...
"%PY%" panel_training_0426\export_panel_predictions.py ^
  --panel-csv "%PANEL_SP%" ^
  --autoformer-root "%AUTOFORMER_ROOT%" ^
  --checkpoints-dir panel_training_0426/checkpoints_cmp/%TAG_SP% ^
  --setting %SETTING% ^
  --out-dir panel_training_0426/outputs_cmp/%TAG_SP% ^
  --scope test --target-year 2025 ^
  --seq-len %SEQ_LEN% --label-len %LABEL_LEN% --pred-len %PRED_LEN% ^
  --split-mode %SPLIT_MODE% --train-end %TRAIN_END% --test-start %TEST_START% --val-weeks %VAL_WEEKS%
if errorlevel 1 goto :fail

"%PY%" use_official_autoformer_grid\validate_grid_predictions.py ^
  --pred-by-date panel_training_0426/outputs_cmp/%TAG_SP%/panel_pred_test_2025_by_date.csv ^
  --out-dir panel_training_0426/outputs_cmp/%TAG_SP%/validation ^
  --top-n 20
if errorlevel 1 goto :fail

echo.
echo [4/4] Done.
echo Base results:    panel_training_0426/outputs_cmp/%TAG_BASE%/validation
echo Spatial results: panel_training_0426/outputs_cmp/%TAG_SP%/validation
echo.
exit /b 0

:fail
echo.
echo ERROR: command failed with exit code %errorlevel%.
exit /b %errorlevel%

