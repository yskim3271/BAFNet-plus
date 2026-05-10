#!/usr/bin/env bash
# D5b — Export FP32 + INT8 QDQ ONNX for all 5 (or a subset of) latency configs.
#
# Each export reuses a BAFNet+ INT8 calibration fixture. The fixture is
# config-agnostic because the streaming I/O shape (bcs_mag/pha + acs_mag/pha at
# [1, 201, 11]) is identical across configs — encoder_padding_ratio is baked
# into each checkpoint's .hydra/config.yaml and only affects model internals,
# not I/O.
#
# Outputs land at:
#   BAFNetPlus/results/onnx/bafnetplus_<T>ms/
#       bafnetplus.onnx                   (FP32, simplified)
#       bafnetplus_qdq.onnx               (INT8 QDQ for HTP)
#       bafnetplus_streaming_config.json
#
# Environment variables:
#   CONFIGS         space-separated T_alg ms values to export (default: "12 25 50 75 100")
#   FIXTURE_DIR     calibration fixture path (default: existing bafnetplus_fixtures
#                   under Android_projects/benchmark-app/.../androidTest/assets/).
#                   To regenerate from real TAPS BCS+ACS pairs (silence-enriched),
#                   first run:
#                     python scripts/make_bafnetplus_calibration_fixture.py \
#                         --output_dir results/onnx/calibration_fixture_taps \
#                         --num_utts 20 --chunks_per_utt 20 --num_silence_chunks 50
#                   then export with:
#                     FIXTURE_DIR=$PWD/results/onnx/calibration_fixture_taps \
#                         scripts/d5b_export_all_configs.sh
#
# Calibration provenance note (M2 from D5b plan review):
#   The default fixture under Android_projects/benchmark-app/... was generated
#   from random-seeded synthetic audio (seed=42, input_scale=0.1) — it does
#   NOT match the paper prose claim of "silence-enriched in-distribution TAPS
#   subset" (latex/main.tex:1670-1671). Using the default reproduces the
#   calibration regime that produced the published 50 ms QDQ ONNX (so D5b
#   numbers reflect what users actually deploy), but the prose should be
#   reconciled — either (a) re-export with FIXTURE_DIR pointing at a fresh
#   TAPS-derived fixture, or (b) update the paper prose.
#
# Usage:
#   ./scripts/d5b_export_all_configs.sh
#   CONFIGS="50" ./scripts/d5b_export_all_configs.sh
#   FIXTURE_DIR=/path/to/fixture ./scripts/d5b_export_all_configs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BAFNETPLUS="${REPO_ROOT}/BAFNetPlus"
DEFAULT_FIXTURE_DIR="${REPO_ROOT}/Android_projects/benchmark-app/src/androidTest/assets/bafnetplus_fixtures"

CONFIGS="${CONFIGS:-12 25 50 75 100}"
FIXTURE_DIR="${FIXTURE_DIR:-${DEFAULT_FIXTURE_DIR}}"

if [ ! -f "${FIXTURE_DIR}/manifest.json" ]; then
    echo "ERROR: BAFNet+ calibration fixture not found at ${FIXTURE_DIR}" >&2
    echo "       Either fix FIXTURE_DIR, or generate one with:" >&2
    echo "         python scripts/make_bafnetplus_calibration_fixture.py \\" >&2
    echo "             --output_dir results/onnx/calibration_fixture_taps" >&2
    exit 1
fi

echo "D5b export sweep"
echo "  CONFIGS    : ${CONFIGS}"
echo "  FIXTURE_DIR: ${FIXTURE_DIR}"
FIXTURE_KIND="$(python -c "import json; m=json.load(open('${FIXTURE_DIR}/manifest.json')); print(m.get('calibration_distribution', m.get('generator', 'unknown')))")"
echo "  fixture provenance: ${FIXTURE_KIND}"

cd "${BAFNETPLUS}"

for T in ${CONFIGS}; do
    echo "================================================================"
    echo "D5b export: T_alg = ${T} ms"
    echo "================================================================"
    OUT_DIR="${BAFNETPLUS}/results/onnx/bafnetplus_${T}ms"
    mkdir -p "${OUT_DIR}"

    python -m src.models.streaming.onnx.export_bafnetplus_onnx \
        --chkpt_dir_mapping "${BAFNETPLUS}/results/experiments/bm_map_${T}ms" \
        --chkpt_dir_masking "${BAFNETPLUS}/results/experiments/bm_mask_${T}ms" \
        --chkpt_file best.th \
        --output_dir "${OUT_DIR}" \
        --output_name "bafnetplus.onnx" \
        --config_name "bafnetplus_streaming_config.json" \
        --simplify \
        --quantize_qdq \
        --calibration_fixture_dir "${FIXTURE_DIR}" \
        --qdq_activation_type QUInt8 \
        --qdq_weight_type QUInt8

    if [ ! -f "${OUT_DIR}/bafnetplus.onnx" ] || [ ! -f "${OUT_DIR}/bafnetplus_qdq.onnx" ]; then
        echo "ERROR: T=${T}ms export failed; expected outputs missing in ${OUT_DIR}" >&2
        exit 1
    fi

    echo "OK: ${OUT_DIR}/{bafnetplus,bafnetplus_qdq}.onnx"
done

echo
echo "Configs exported: ${CONFIGS}"
echo "Next: scripts/d5b_run_eval.py (use --configs to match the export subset)."
