#!/usr/bin/env bash
# S10 streaming-eval canonical command.
#
# Thin wrapper around `python -m src.analysis.eval_streaming` with the
# canonical defaults for the deployed unified bafnetplus_50ms ckpt.
# Useful for CI / smoke-checking after future model updates.
#
# Usage:
#   scripts/eval_streaming.sh                  # --preset full (default)
#   scripts/eval_streaming.sh smoke            # --preset smoke (LP target on idx 0)
#   scripts/eval_streaming.sh full             # --preset full (idx 0..4)
#   scripts/eval_streaming.sh full --verbose   # extra args after preset go through
#
# Outputs land under results/eval_streaming/ by default (gitignored under
# the top-level `results/` rule). Override --output-json / --output-markdown
# explicitly to redirect.
set -euo pipefail

# Resolve script dir -> repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRESET="${1:-full}"
case "${PRESET}" in
  smoke|full|custom)
    shift || true
    ;;
  *)
    # Forward as positional if it wasn't a preset keyword (custom flow).
    PRESET="full"
    ;;
esac

TIMESTAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
OUTPUT_DIR="results/eval_streaming"
mkdir -p "${OUTPUT_DIR}"

# Default artefacts (overridable via extra args).
JSON_DEFAULT="${OUTPUT_DIR}/run_${PRESET}_${TIMESTAMP}.json"
MD_DEFAULT="${OUTPUT_DIR}/run_${PRESET}_${TIMESTAMP}.md"

# Detect whether the caller already passed --output-json / --output-markdown.
PASSED_JSON=0
PASSED_MD=0
for arg in "$@"; do
  case "${arg}" in
    --output-json|--output-json=*) PASSED_JSON=1 ;;
    --output-markdown|--output-markdown=*) PASSED_MD=1 ;;
  esac
done

EXTRA_OUTPUTS=()
if [ "${PASSED_JSON}" -eq 0 ]; then
  EXTRA_OUTPUTS+=(--output-json "${JSON_DEFAULT}")
fi
if [ "${PASSED_MD}" -eq 0 ] && [ "${PRESET}" != "smoke" ]; then
  EXTRA_OUTPUTS+=(--output-markdown "${MD_DEFAULT}")
fi

python -m src.analysis.eval_streaming \
  --preset "${PRESET}" \
  --unified-ckpt-dir results/experiments/bafnetplus_50ms \
  --chkpt-file best.th \
  --onnx-artifact results/onnx/bafnetplus_50ms_fp32.onnx \
  "${EXTRA_OUTPUTS[@]}" \
  "$@"
