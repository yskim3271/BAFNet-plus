#!/usr/bin/env bash
# D5d — Export 5 INT8 QDQ ONNX configs for ISOLATED HTP latency measurement.
#
# Each config:
#   - Uses its tier-specific trained model (bm_map_<T>ms / bm_mask_<T>ms)
#   - chunk_size_frames = T_chunk_frames(T)  (2/4/8/12/16 for 12/25/50/75/100 ms)
#   - encoder_lookahead = 0, decoder_lookahead = 0  (ρ = 0, causal extreme)
#     → export_time_frames = chunk_size_frames (no future buffering)
#
# This is the canonical Tab VI re-anchor: each row of Tab VI reports HTP
# per-chunk inference latency under controlled isolation, with ρ=0 (matching
# the latency formula T_alg = T_chunk).
#
# Outputs land at:
#   BAFNetPlus/results/onnx/bafnetplus_d5d_<T>ms/
#       bafnetplus.onnx
#       bafnetplus_qdq.onnx
#       bafnetplus_streaming_config.json
#
# Usage:
#   ./scripts/d5d_export_5configs.sh
#   CONFIGS="50" ./scripts/d5d_export_5configs.sh         # subset
#   FIXTURE_DIR=/path/to/fixture ./scripts/d5d_export_5configs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BAFNETPLUS="${REPO_ROOT}/BAFNetPlus"
DEFAULT_FIXTURE_DIR="${REPO_ROOT}/Android_projects/benchmark-app/src/androidTest/assets/bafnetplus_fixtures"

CONFIGS="${CONFIGS:-12 25 50 75 100}"
FIXTURE_DIR="${FIXTURE_DIR:-${DEFAULT_FIXTURE_DIR}}"

# T_alg ms → chunk_size_frames (T_alg / hop_ms, hop=6.25ms)
declare -A CHUNK_FRAMES=(
    [12]=2
    [25]=4
    [50]=8
    [75]=12
    [100]=16
)

if [ ! -f "${FIXTURE_DIR}/manifest.json" ]; then
    echo "ERROR: BAFNet+ calibration fixture not found at ${FIXTURE_DIR}" >&2
    exit 1
fi

echo "D5d isolated export sweep (Tab VI canonical, ρ=0)"
echo "  CONFIGS    : ${CONFIGS}"
echo "  FIXTURE_DIR: ${FIXTURE_DIR}"

cd "${BAFNETPLUS}"

for T in ${CONFIGS}; do
    CHUNK="${CHUNK_FRAMES[${T}]:-}"
    if [ -z "${CHUNK}" ]; then
        echo "ERROR: unknown T_alg=${T} (allowed: 12 25 50 75 100)" >&2
        exit 1
    fi
    echo "================================================================"
    echo "D5d export: T_alg=${T} ms, chunk_size_frames=${CHUNK}, lookahead=0"
    echo "================================================================"
    OUT_DIR="${BAFNETPLUS}/results/onnx/bafnetplus_d5d_${T}ms"
    mkdir -p "${OUT_DIR}"

    python -m src.models.streaming.onnx.export_bafnetplus_onnx \
        --chkpt_dir_mapping "${BAFNETPLUS}/results/experiments/bm_map_${T}ms" \
        --chkpt_dir_masking "${BAFNETPLUS}/results/experiments/bm_mask_${T}ms" \
        --chkpt_file best.th \
        --chunk_size "${CHUNK}" \
        --encoder_lookahead 0 \
        --decoder_lookahead 0 \
        --output_dir "${OUT_DIR}" \
        --output_name "bafnetplus.onnx" \
        --config_name "bafnetplus_streaming_config.json" \
        --simplify \
        --quantize_qdq \
        --calibration_fixture_dir "${FIXTURE_DIR}" \
        --qdq_activation_type QUInt8 \
        --qdq_weight_type QUInt8 \
        --skip_verify

    if [ ! -f "${OUT_DIR}/bafnetplus.onnx" ] || [ ! -f "${OUT_DIR}/bafnetplus_qdq.onnx" ]; then
        echo "ERROR: T=${T}ms export failed; expected outputs missing in ${OUT_DIR}" >&2
        exit 1
    fi

    echo "OK: ${OUT_DIR}/{bafnetplus,bafnetplus_qdq}.onnx"
done

echo
echo "D5d configs exported: ${CONFIGS}"
echo "Next: sync to benchmark-app/src/main/assets and run StreamingBenchmarkTest.benchmarkBafnetplusD5dIsolatedSweep()"
