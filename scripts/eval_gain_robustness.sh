#!/bin/bash
# Evaluate calibration robustness to BCS-ACS level mismatch
# Sweeps relative gain offset: BCS_gain = +offset/2, ACS_gain = -offset/2
# Compares: abl_full (with calibration) vs abl_no_calibration (without)
# Fixed SNR=0dB, no STT (audio metrics only)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

EXP_DIR="results/experiments"
NOISE_DIR="/home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k"
RIR_DIR="/home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k"
NOISE_TEST="dataset/taps/noise_test.txt"
RIR_TEST="dataset/taps/rir_test.txt"

EVAL_DIR="results/eval_gain_robustness"
mkdir -p "$EVAL_DIR"

# Pod-to-local path mapping
POD_PREFIX="/workspace"
LOCAL_PREFIX="/home/yskim/workspace"

# Relative gain offsets (dB): BCS - ACS
OFFSETS=(-20 -10 -5 0 5 10 20)

run_gain_eval() {
    local gpu=$1
    local name=$2
    local offset=$3

    local bcs_gain=$(echo "$offset / 2" | bc -l)
    local acs_gain=$(echo "-1 * $offset / 2" | bc -l)

    local chkpt_dir="${EXP_DIR}/${name}"
    local config_orig="${chkpt_dir}/.hydra/config.yaml"
    local config="${EVAL_DIR}/${name}_config.yaml"
    local output_json="${EVAL_DIR}/${name}_offset${offset}.json"
    local log_file="${EVAL_DIR}/${name}_offset${offset}.log"

    # Create temp config with local paths (once per model)
    if [ ! -f "$config" ]; then
        sed "s|${POD_PREFIX}|${LOCAL_PREFIX}|g" "$config_orig" > "$config"
    fi

    echo "[GPU${gpu}] ${name} offset=${offset}dB (BCS=${bcs_gain}dB, ACS=${acs_gain}dB)"
    CUDA_VISIBLE_DEVICES=$gpu python -m src.evaluate \
        --model_config "$config" \
        --chkpt_dir "$chkpt_dir" \
        --chkpt_file best.th \
        --dataset taps \
        --noise_dir "$NOISE_DIR" \
        --noise_test "$NOISE_TEST" \
        --rir_dir "$RIR_DIR" \
        --rir_test "$RIR_TEST" \
        --snr_step 0 \
        --bcs_gain_db "$bcs_gain" \
        --acs_gain_db "$acs_gain" \
        --output_json "$output_json" \
        --log_file "$log_file"
    echo "[GPU${gpu}] Done: ${name} offset=${offset}dB -> ${output_json}"
}

# GPU0: abl_full (all offsets sequentially)
gpu0_tasks() {
    for offset in "${OFFSETS[@]}"; do
        run_gain_eval 0 abl_full "$offset"
    done
}

# GPU1: abl_no_calibration (all offsets sequentially)
gpu1_tasks() {
    for offset in "${OFFSETS[@]}"; do
        run_gain_eval 1 abl_no_calibration "$offset"
    done
}

# Run GPU0 and GPU1 in parallel
gpu0_tasks &
GPU0_PID=$!

gpu1_tasks &
GPU1_PID=$!

echo "Started GPU0 (PID=$GPU0_PID) and GPU1 (PID=$GPU1_PID)"
echo "Offsets: ${OFFSETS[*]} dB"
echo "Logs: ${EVAL_DIR}/*.log"

wait $GPU0_PID
echo "GPU0 tasks completed."

wait $GPU1_PID
echo "GPU1 tasks completed."

echo "All evaluations done. Results in ${EVAL_DIR}/"
