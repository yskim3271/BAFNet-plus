#!/bin/bash
# Evaluate ablation models on TAPS at multiple SNR levels
# GPU0: abl_full, abl_no_calibration
# GPU1: abl_common_gain_only, abl_mask_only_alpha

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

EXP_DIR="results/experiments"
NOISE_DIR="/home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k"
RIR_DIR="/home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k"
NOISE_TEST="dataset/taps/noise_test.txt"
RIR_TEST="dataset/taps/rir_test.txt"
SNR_STEPS="-20 -10 0 10 15"

EVAL_DIR="results/eval_ablation_multi_snr"
mkdir -p "$EVAL_DIR"

# Pod paths that need to be replaced with local paths
POD_PREFIX="/workspace"
LOCAL_PREFIX="/home/yskim/workspace"

run_eval() {
    local gpu=$1
    local name=$2
    local chkpt_dir="${EXP_DIR}/${name}"
    local config_orig="${chkpt_dir}/.hydra/config.yaml"
    local config="${EVAL_DIR}/${name}_config.yaml"
    local output_json="${EVAL_DIR}/${name}.json"
    local log_file="${EVAL_DIR}/${name}.log"

    # Create temp config with local paths
    sed "s|${POD_PREFIX}|${LOCAL_PREFIX}|g" "$config_orig" > "$config"

    echo "[GPU${gpu}] Evaluating ${name} ..."
    CUDA_VISIBLE_DEVICES=$gpu python -m src.evaluate \
        --model_config "$config" \
        --chkpt_dir "$chkpt_dir" \
        --chkpt_file best.th \
        --dataset taps \
        --noise_dir "$NOISE_DIR" \
        --noise_test "$NOISE_TEST" \
        --rir_dir "$RIR_DIR" \
        --rir_test "$RIR_TEST" \
        --snr_step $SNR_STEPS \
        --eval_stt \
        --output_json "$output_json" \
        --log_file "$log_file"
    echo "[GPU${gpu}] Done: ${name} -> ${output_json}"
}

# GPU0: full, no_calibration (sequential within GPU)
gpu0_tasks() {
    run_eval 0 abl_full
    run_eval 0 abl_no_calibration
}

# GPU1: common_gain_only, mask_only_alpha (sequential within GPU)
gpu1_tasks() {
    run_eval 1 abl_common_gain_only
    run_eval 1 abl_mask_only_alpha
}

# Run GPU0 and GPU1 in parallel
gpu0_tasks &
GPU0_PID=$!

gpu1_tasks &
GPU1_PID=$!

echo "Started GPU0 (PID=$GPU0_PID) and GPU1 (PID=$GPU1_PID)"
echo "Logs: ${EVAL_DIR}/*.log"

wait $GPU0_PID
echo "GPU0 tasks completed."

wait $GPU1_PID
echo "GPU1 tasks completed."

echo "All evaluations done. Results in ${EVAL_DIR}/"
