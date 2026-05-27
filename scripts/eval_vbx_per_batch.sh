#!/bin/bash
# Sequential PER batch on a given GPU for Vibravox.
# Reads existing JSON file (from --eval_squim/dnsmos batch) if present and
# adds PER metrics in a separate run that writes to a sibling --per.json file
# to avoid clobbering the canonical multi-snr-clean/multi-snr-squim JSON.
#
# Usage: ./scripts/eval_vbx_per_batch.sh <GPU> <SUBSET> <model1> [model2 ...]
#   SUBSET: 'default' (speech_clean test split, 3,064 utt) or 'speech_noisy'
set -u
GPU="${1:?usage: $0 <GPU> <SUBSET> <model1> [...]}"
SUBSET="${2:?usage: $0 <GPU> <SUBSET> <model1> [...]}"
shift 2
case "$SUBSET" in
  default) DIR="results/eval/multi-snr-per-clean" ;;
  speech_noisy) DIR="results/eval/multi-snr-per-noisy" ;;
  *) echo "Unknown subset $SUBSET" >&2; exit 2 ;;
esac
mkdir -p "$DIR"
for M in "$@"; do
  if [ -f "$DIR/$M.json" ]; then
    echo "=== [GPU$GPU/$SUBSET] $M already done — skip $(date) ==="
    continue
  fi
  echo "=== [GPU$GPU/$SUBSET] START $M $(date) ==="
  CUDA_VISIBLE_DEVICES="$GPU" python -u -m src.evaluate \
    --model_config "results/experiments/$M/.hydra/config.yaml" \
    --chkpt_dir "results/experiments/$M" \
    --chkpt_file best.th \
    --dataset vibravox \
    --test_bypass_mixing \
    --test_hf_dataset yskim3271/vibravox_16k \
    --test_subset "$SUBSET" \
    --eval_per \
    --save_per_utterance_json \
    --output_json "$DIR/$M.json" \
    --log_file "$DIR/$M.log"
  echo "=== [GPU$GPU/$SUBSET] END $M $(date) exit=$? ==="
done
echo "=== [GPU$GPU/$SUBSET] BATCH FINISHED $(date) ==="
