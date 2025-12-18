#!/bin/bash
# Evaluation script for BAFNet experiments
# Evaluates multiple models across various SNR levels

set -e  # Exit on error

# Configuration
GPU_ID=0
SNR_STEPS="-15 -5 5 15"
DATASET="taps"
NUM_WORKERS=5
TEST_AUGMENT_NUMB=2

# Paths
PROJECT_DIR="/home/yskim/workspace/BAFNet-plus"
NOISE_DIR="/home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k"
NOISE_TEST="dataset/taps/noise_test.txt"
RIR_DIR="/home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k"
RIR_TEST="dataset/taps/rir_test.txt"

# Experiment directories
EXPERIMENTS=(
    "bafnet_full_1205"
    "bafnet_full_1208"
    "bafnetv2_1211"
)

# Output directory for evaluation logs
EVAL_OUTPUT_DIR="${PROJECT_DIR}/results/evaluation"
mkdir -p "${EVAL_OUTPUT_DIR}"

echo "========================================"
echo "BAFNet Evaluation Script"
echo "========================================"
echo "GPU: ${GPU_ID}"
echo "SNR Steps: ${SNR_STEPS}"
echo "Dataset: ${DATASET}"
echo "========================================"
echo ""

for EXP_NAME in "${EXPERIMENTS[@]}"; do
    EXP_DIR="${PROJECT_DIR}/results/experiments/${EXP_NAME}"
    CONFIG_FILE="${EXP_DIR}/.hydra/config.yaml"
    CHKPT_FILE="best.th"
    LOG_FILE="${EVAL_OUTPUT_DIR}/${EXP_NAME}_eval.log"

    echo "----------------------------------------"
    echo "Evaluating: ${EXP_NAME}"
    echo "Config: ${CONFIG_FILE}"
    echo "Checkpoint: ${EXP_DIR}/${CHKPT_FILE}"
    echo "Log: ${LOG_FILE}"
    echo "----------------------------------------"

    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        continue
    fi

    if [ ! -f "${EXP_DIR}/${CHKPT_FILE}" ]; then
        echo "ERROR: Checkpoint not found: ${EXP_DIR}/${CHKPT_FILE}"
        continue
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.evaluate \
        --model_config "${CONFIG_FILE}" \
        --chkpt_dir "${EXP_DIR}" \
        --chkpt_file "${CHKPT_FILE}" \
        --dataset "${DATASET}" \
        --noise_dir "${NOISE_DIR}" \
        --noise_test "${NOISE_TEST}" \
        --rir_dir "${RIR_DIR}" \
        --rir_test "${RIR_TEST}" \
        --test_augment_numb ${TEST_AUGMENT_NUMB} \
        --snr_step ${SNR_STEPS} \
        --num_workers ${NUM_WORKERS} \
        --log_file "${LOG_FILE}" \
        --eval_stt

    echo ""
    echo "Completed: ${EXP_NAME}"
    echo "Results saved to: ${LOG_FILE}"
    echo ""
done

echo "========================================"
echo "All evaluations completed!"
echo "Results are in: ${EVAL_OUTPUT_DIR}"
echo "========================================"
