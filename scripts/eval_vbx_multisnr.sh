#!/bin/bash
# Sequential multi-SNR batch eval of N models on a given GPU for Vibravox.
# Unified with TAPS: speech_clean (default config) test split mixed with DNS
# noise_fullband_16k at controlled SNR (clean reference -> intrusive PESQ/STOI
# valid). Per-utterance JSON written to results/eval/multi-snr/<model>.json.
#
# Usage: ./scripts/eval_vbx_multisnr.sh <GPU_ID> <model1> [model2 ...]
#   SNR_STEP env overrides the SNR set (default "-20 -10 0 10 15").
#   Caution: too many SNR steps held in memory at once can OOM — reduce SNR_STEP.
set -u
GPU="${1:?usage: $0 <GPU> <model1> [model2 ...]}"
shift
SNR_STEP="${SNR_STEP:--20 -10 0 10 15}"
NOISE_DIR="${NOISE_DIR:-/home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k}"
RIR_DIR="${RIR_DIR:-/home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k}"
DIR="results/eval/multi-snr"
mkdir -p "$DIR"
for M in "$@"; do
  if [ -f "$DIR/$M.json" ]; then
    echo "=== [GPU$GPU] $M already done at $DIR/$M.json — skip $(date) ==="
    continue
  fi
  echo "=== [GPU$GPU] START $M $(date) ==="
  CUDA_VISIBLE_DEVICES="$GPU" python -u -m src.evaluate \
    --model_config "results/experiments/$M/.hydra/config.yaml" \
    --chkpt_dir "results/experiments/$M" \
    --chkpt_file best.th \
    --dataset vibravox \
    --snr_step $SNR_STEP \
    --noise_dir "$NOISE_DIR" \
    --noise_test dataset/taps/noise_test.txt \
    --rir_dir "$RIR_DIR" \
    --rir_test dataset/taps/rir_test.txt \
    --num_workers 5 \
    --save_per_utterance_json \
    --output_json "$DIR/$M.json" \
    --log_file "$DIR/$M.log"
  echo "=== [GPU$GPU] END $M $(date) exit=$? ==="
done
echo "=== [GPU$GPU] BATCH FINISHED $(date) ==="
