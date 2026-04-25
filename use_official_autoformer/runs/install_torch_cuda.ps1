$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "Current torch build:"
& $Py -c "import torch; print(torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda)"

Write-Host ""
Write-Host "Installing CUDA-enabled PyTorch (will replace +cpu build)..."
Write-Host "If your network blocks PyTorch index, open the URL and copy the right command for CUDA 12.x:"
Write-Host "  https://pytorch.org/get-started/locally/"
Write-Host ""

# For NVIDIA driver CUDA 12.9, cu12.8 wheels are typically the closest supported build.
# This uses PyTorch's official wheel index for CUDA 12.8.
& $Py -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

Write-Host ""
Write-Host "After install:"
& $Py -c "import torch; print(torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('device_count', torch.cuda.device_count())"

