#!/bin/bash
# setup_pod.sh
# RunPod Pod 초기화 스크립트 (RTX 5090 / PyTorch 2.8 + Ubuntu 24.04)
#
# 네트워크 볼륨(/workspace)에 레포가 있으면 git pull, 없으면 clone 후
# 필요한 시스템 패키지와 Python 의존성을 설치합니다.
# 이미 설치된 의존성은 스킵합니다.
#
# 이 스크립트는 Pod 내부에서 실행됩니다.
# run_train.sh가 SSH를 통해 자동으로 호출합니다.
#
# Known issues (RTX 5090 + PyTorch 2.8 + Ubuntu 24.04):
#   - pip install --break-system-packages 필수 (PEP 668)
#   - datasets 4.x는 torchcodec + FFmpeg ABI 불일치 → datasets<4 고정
#     (datasets 3.x는 librosa + soundfile 기반 오디오 디코딩)

set -euo pipefail

REMOTE_PROJECT="/workspace/BAFNet-plus"
REPO_URL="https://github.com/yskim3271/BAFNet-plus.git"

echo "=== BAFNet+ Pod Setup ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo ""

# ---- 1. System packages ----
if ! command -v ffmpeg &>/dev/null; then
    echo "[1/4] Installing system packages..."
    apt-get update -qq
    apt-get install -y -qq ffmpeg > /dev/null 2>&1
    echo "  ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "[1/4] System packages OK (ffmpeg already installed)"
fi

# ---- 2. Repository ----
echo "[2/4] Setting up repository..."
if [[ -d "$REMOTE_PROJECT/.git" ]]; then
    echo "  Repository exists, pulling latest..."
    cd "$REMOTE_PROJECT" && git pull --ff-only
else
    echo "  Cloning repository..."
    cd /workspace && git clone "$REPO_URL"
fi
cd "$REMOTE_PROJECT"

# ---- 3. Python dependencies ----
# Quick check: if all critical imports work, skip installation
DEPS_OK=$(python3 -c "
import hydra, pesq, datasets, transformers, tensorboard
import joblib, scipy, librosa, soundfile
print('OK')
" 2>/dev/null || echo "MISSING")

if [[ "$DEPS_OK" == "OK" ]]; then
    echo "[3/4] Python dependencies already installed, skipping"
else
    echo "[3/4] Installing Python dependencies..."
    pip install --break-system-packages -q -r requirements.txt
fi

# ---- 4. Verification ----
echo "[4/4] Verifying installation..."
python3 -c "
import numpy; print(f'  numpy:        {numpy.__version__}')
import torch; print(f'  torch:        {torch.__version__} (CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)})')
import torchaudio; print(f'  torchaudio:   {torchaudio.__version__}')
import hydra; print(f'  hydra:        {hydra.__version__}')
import librosa; print(f'  librosa:      {librosa.__version__}')
from pesq import pesq; print(f'  pesq:         OK')
import datasets; print(f'  datasets:     {datasets.__version__}')
import transformers; print(f'  transformers: {transformers.__version__}')
import scipy; print(f'  scipy:        {scipy.__version__}')
import joblib; print(f'  joblib:       {joblib.__version__}')
import tensorboard; print(f'  tensorboard:  {tensorboard.__version__}')
print()
print('All imports OK')
"

echo ""
echo "=== Setup Complete ==="
