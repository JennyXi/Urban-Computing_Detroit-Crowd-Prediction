$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"

# Prefer an external official Autoformer clone (avoids putting the whole repo under git)
$ExternalAutoformerRoot = "E:\Urban Computing Final Project\autoformer_spatial_0425\Autoformer"
$AutoformerRoot = if (Test-Path $ExternalAutoformerRoot) { $ExternalAutoformerRoot } else { Join-Path $RepoRoot "use_official_autoformer\third_party\Autoformer" }

$GridDataRoot = Join-Path $RepoRoot "use_official_autoformer_grid\data\grid_weekly_top100_visits"
$GridDataRootEnv = $env:GRID_DATA_ROOT
if ($GridDataRootEnv -and $GridDataRootEnv.Trim().Length -gt 0) {
  $GridDataRoot = $GridDataRootEnv
}
$Manifest = Join-Path $GridDataRoot "grid_topk_manifest.csv"
$Checkpoints = Join-Path $RepoRoot "use_official_autoformer_grid\checkpoints"

$SeqLen = if ($env:SEQ_LEN) { [int]$env:SEQ_LEN } else { 24 }
$LabelLen = if ($env:LABEL_LEN) { [int]$env:LABEL_LEN } else { 12 }
$PredLen = if ($env:PRED_LEN) { [int]$env:PRED_LEN } else { 4 }
$Des = if ($env:RUN_DES) { $env:RUN_DES } else { "grid_weekly_${SeqLen}to${PredLen}" }
$TrainEpochs = if ($env:TRAIN_EPOCHS) { [int]$env:TRAIN_EPOCHS } else { 50 }

$env:PYTHONUNBUFFERED = "1"

if (!(Test-Path $Manifest)) {
  throw "Missing manifest: $Manifest. Run prepare_top100_grid_weekly_csvs.py first."
}

$ids = & $Py -c "import pandas as pd; m=pd.read_csv(r'$Manifest'); print('\n'.join(m['grid_id'].astype(str).tolist()))"
$gridIds = $ids -split "`n" | Where-Object { $_.Trim().Length -gt 0 }

# Optional: limit training to first N grids for quick validation runs
$LimitEnv = $env:GRID_LIMIT
if ($LimitEnv -and $LimitEnv.Trim().Length -gt 0) {
  $n = [int]$LimitEnv
  if ($n -gt 0 -and $gridIds.Count -gt $n) {
    $gridIds = $gridIds[0..($n-1)]
  }
}

Write-Host "Training Autoformer per-grid (Top $($gridIds.Count))"
Write-Host "Data root: $GridDataRoot"
Write-Host "Checkpoints: $Checkpoints"
Write-Host "Autoformer root: $AutoformerRoot"

Set-Location $AutoformerRoot

foreach ($gridId in $gridIds) {
  $dataPath = "grid_$gridId.csv"
  if (!(Test-Path (Join-Path $GridDataRoot $dataPath))) {
    Write-Warning "Missing $dataPath, skipping."
    continue
  }

  $modelId = "detroit_grid100_$gridId"
  Write-Host ""
  Write-Host "=== $modelId ==="

  & $Py -u .\run.py `
    --is_training 1 `
    --model_id $modelId `
    --model Autoformer `
    --data custom `
    --root_path "$GridDataRoot" `
    --data_path "$dataPath" `
    --features S `
    --target OT `
    --freq w `
    --seq_len $SeqLen `
    --label_len $LabelLen `
    --pred_len $PredLen `
    --enc_in 1 `
    --dec_in 1 `
    --c_out 1 `
    --e_layers 2 `
    --d_layers 1 `
    --n_heads 8 `
    --d_model 128 `
    --d_ff 512 `
    --dropout 0.05 `
    --moving_avg 25 `
    --factor 1 `
    --embed timeF `
    --des $Des `
    --itr 1 `
    --train_epochs $TrainEpochs `
    --batch_size 16 `
    --learning_rate 0.0001 `
    --patience 20 `
    --num_workers 0 `
    --checkpoints "$Checkpoints"
}

