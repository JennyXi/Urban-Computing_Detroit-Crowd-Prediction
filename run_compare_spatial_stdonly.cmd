@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Compare panel training:
REM - sp8_meanstd: --spatial-cov nbr8_meanstd_lag1
REM - sp8_stdonly: --spatial-cov nbr8_std_lag1
REM Everything else is held constant (year split, huber, etc.).
REM ============================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "AUTOFORMER_ROOT=E:\Urban Computing Final Project\autoformer_spatial_0425\Autoformer"

set "DATE_START=2024-01-01"
set "DATE_END=2025-12-31"
set "TOPK=100"
set "TOPK_YEAR=2024"
set "CITY_COV=lag1"
set "WEEKEND_COV=share_lag1"
set "TARGET_TRANSFORM=log1p"

set "SPLIT_MODE=year"
set "TRAIN_END=2024-12-31"
set "TEST_START=2025-01-01"
set "VAL_WEEKS=10"

set "SEQ_LEN=24"
set "LABEL_LEN=12"
set "PRED_LEN=4"
set "BATCH=64"
set "EPOCHS=20"
set "LOSS=huber"
set "HUBER_DELTA=1.0"

REM Must match train_panel_autoformer.py naming (huber1 suffix)
set "SETTING=panel_Autoformer_ftMS_sl%SEQ_LEN%_ll%LABEL_LEN%_pl%PRED_LEN%_dm128_el2_dl1_%TARGET_TRANSFORM%_huber1"

echo.
echo === Using python: %PY%
echo === Repo root: %ROOT%
echo === Autoformer: %AUTOFORMER_ROOT%
echo.

echo [1/4] Build panel CSVs (meanstd vs stdonly)...
set "TAG_MEANSTD=cmp0428_sp8_meanstd"
set "TAG_STDONLY=cmp0428_sp8_stdonly"

"%PY%" panel_training_0426\build_panel_weekly_dataset.py ^
  --top-k %TOPK% --date-start %DATE_START% --date-end %DATE_END% ^
  --topk-year %TOPK_YEAR% --city-cov %CITY_COV% --weekend-cov %WEEKEND_COV% ^
  --spatial-cov nbr8_meanstd_lag1 --target-transform %TARGET_TRANSFORM% ^
  --tag %TAG_MEANSTD%
if errorlevel 1 goto :fail

"%PY%" panel_training_0426\build_panel_weekly_dataset.py ^
  --top-k %TOPK% --date-start %DATE_START% --date-end %DATE_END% ^
  --topk-year %TOPK_YEAR% --city-cov %CITY_COV% --weekend-cov %WEEKEND_COV% ^
  --spatial-cov nbr8_std_lag1 --target-transform %TARGET_TRANSFORM% ^
  --tag %TAG_STDONLY%
if errorlevel 1 goto :fail

set "PANEL_MEANSTD=panel_training_0426/outputs/panel_weekly_top%TOPK%_2024_2025_%TAG_MEANSTD%.csv"
set "PANEL_STDONLY=panel_training_0426/outputs/panel_weekly_top%TOPK%_2024_2025_%TAG_STDONLY%.csv"

echo    meanstd panel: %PANEL_MEANSTD%
echo    stdonly panel: %PANEL_STDONLY%
echo.

echo [2/4] Train meanstd model...
"%PY%" panel_training_0426\train_panel_autoformer.py ^
  --panel-csv "%PANEL_MEANSTD%" ^
  --autoformer-root "%AUTOFORMER_ROOT%" ^
  --checkpoints-dir panel_training_0426/checkpoints_cmp/%TAG_MEANSTD% ^
  --seq-len %SEQ_LEN% --label-len %LABEL_LEN% --pred-len %PRED_LEN% ^
  --batch-size %BATCH% --epochs %EPOCHS% ^
  --loss %LOSS% --huber-delta %HUBER_DELTA% ^
  --split-mode %SPLIT_MODE% --train-end %TRAIN_END% --test-start %TEST_START% --val-weeks %VAL_WEEKS%
if errorlevel 1 goto :fail

echo.
echo [2/4] Train stdonly model...
"%PY%" panel_training_0426\train_panel_autoformer.py ^
  --panel-csv "%PANEL_STDONLY%" ^
  --autoformer-root "%AUTOFORMER_ROOT%" ^
  --checkpoints-dir panel_training_0426/checkpoints_cmp/%TAG_STDONLY% ^
  --seq-len %SEQ_LEN% --label-len %LABEL_LEN% --pred-len %PRED_LEN% ^
  --batch-size %BATCH% --epochs %EPOCHS% ^
  --loss %LOSS% --huber-delta %HUBER_DELTA% ^
  --split-mode %SPLIT_MODE% --train-end %TRAIN_END% --test-start %TEST_START% --val-weeks %VAL_WEEKS%
if errorlevel 1 goto :fail

echo.
echo [3/4] Export + validate meanstd...
"%PY%" panel_training_0426\export_panel_predictions.py ^
  --panel-csv "%PANEL_MEANSTD%" ^
  --autoformer-root "%AUTOFORMER_ROOT%" ^
  --checkpoints-dir panel_training_0426/checkpoints_cmp/%TAG_MEANSTD% ^
  --setting %SETTING% ^
  --out-dir panel_training_0426/outputs_cmp/%TAG_MEANSTD% ^
  --scope test --target-year 2025 ^
  --seq-len %SEQ_LEN% --label-len %LABEL_LEN% --pred-len %PRED_LEN% ^
  --split-mode %SPLIT_MODE% --train-end %TRAIN_END% --test-start %TEST_START% --val-weeks %VAL_WEEKS%
if errorlevel 1 goto :fail

"%PY%" use_official_autoformer_grid\validate_grid_predictions.py ^
  --pred-by-date panel_training_0426/outputs_cmp/%TAG_MEANSTD%/panel_pred_test_2025_by_date.csv ^
  --out-dir panel_training_0426/outputs_cmp/%TAG_MEANSTD%/validation ^
  --top-n 20
if errorlevel 1 goto :fail

echo.
echo [3/4] Export + validate stdonly...
"%PY%" panel_training_0426\export_panel_predictions.py ^
  --panel-csv "%PANEL_STDONLY%" ^
  --autoformer-root "%AUTOFORMER_ROOT%" ^
  --checkpoints-dir panel_training_0426/checkpoints_cmp/%TAG_STDONLY% ^
  --setting %SETTING% ^
  --out-dir panel_training_0426/outputs_cmp/%TAG_STDONLY% ^
  --scope test --target-year 2025 ^
  --seq-len %SEQ_LEN% --label-len %LABEL_LEN% --pred-len %PRED_LEN% ^
  --split-mode %SPLIT_MODE% --train-end %TRAIN_END% --test-start %TEST_START% --val-weeks %VAL_WEEKS%
if errorlevel 1 goto :fail

"%PY%" use_official_autoformer_grid\validate_grid_predictions.py ^
  --pred-by-date panel_training_0426/outputs_cmp/%TAG_STDONLY%/panel_pred_test_2025_by_date.csv ^
  --out-dir panel_training_0426/outputs_cmp/%TAG_STDONLY%/validation ^
  --top-n 20
if errorlevel 1 goto :fail

echo.
echo [4/4] Done.
echo Mean+Std results: panel_training_0426/outputs_cmp/%TAG_MEANSTD%/validation
echo Std-only results: panel_training_0426/outputs_cmp/%TAG_STDONLY%/validation
echo.
exit /b 0

:fail
echo.
echo ERROR: command failed with exit code %errorlevel%.
exit /b %errorlevel%

