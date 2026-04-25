$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$AutoformerRoot = Join-Path $RepoRoot "use_official_autoformer\third_party\Autoformer"
$DataRoot = Join-Path $RepoRoot "data"
$DataPath = "autoformer_weekly_preprocessed.csv"
$Checkpoints = Join-Path $RepoRoot "use_official_autoformer\checkpoints"

#
# Make Python print logs immediately (Windows)
#
$env:PYTHONUNBUFFERED = "1"

if (!(Test-Path (Join-Path $DataRoot $DataPath))) {
  throw "Missing data file: $(Join-Path $DataRoot $DataPath)"
}

$Dims = & $Py -c "import pandas as pd; df=pd.read_csv(r'$($DataRoot)\$($DataPath)', nrows=1); print(len(df.columns)-1)"
$Dims = $Dims.Trim()
Write-Host "Detected weekly dims (enc_in=dec_in=c_out) = $Dims"

Set-Location $AutoformerRoot

& $Py -u .\run.py `
  --is_training 1 `
  --model_id detroit_2025_weekly `
  --model Autoformer `
  --data custom `
  --root_path "$DataRoot" `
  --data_path "$DataPath" `
  --features MS `
  --target OT `
  --freq w `
  --seq_len 12 `
  --label_len 6 `
  --pred_len 4 `
  --enc_in $Dims `
  --dec_in $Dims `
  --c_out $Dims `
  --e_layers 2 `
  --d_layers 1 `
  --n_heads 8 `
  --d_model 256 `
  --d_ff 1024 `
  --dropout 0.05 `
  --moving_avg 25 `
  --factor 1 `
  --embed timeF `
  --des weekly2025 `
  --itr 3 `
  --train_epochs 80 `
  --batch_size 16 `
  --learning_rate 0.0001 `
  --patience 999 `
  --num_workers 0 `
  --checkpoints "$Checkpoints"

