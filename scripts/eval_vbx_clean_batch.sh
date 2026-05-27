#!/bin/bash
# Sequential batch eval of N models on a given GPU for Vibravox speech_clean
# (default config, 3,064 utt test split). Intrusive + Squim + DNSMOS metrics,
# per-utterance JSON.
#
# Usage: ./scripts/eval_vbx_clean_batch.sh <GPU_ID> <model1> [model2 ...]
set -u
GPU="${1:?usage: $0 <GPU> <model1> [model2 ...]}"
shift
DIR="results/eval/multi-snr-clean"
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
    --test_bypass_mixing \
    --test_hf_dataset yskim3271/vibravox_16k \
    --test_subset default \
    --num_workers 5 \
    --eval_squim --eval_dnsmos \
    --save_per_utterance_json \
    --output_json "$DIR/$M.json" \
    --log_file "$DIR/$M.log"
  echo "=== [GPU$GPU] END $M $(date) exit=$? ==="
done
echo "=== [GPU$GPU] BATCH FINISHED $(date) ==="
