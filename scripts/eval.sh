#!/bin/bash
# eval.sh — Unified evaluation script for paper experiments
#
# Usage:
#   ./scripts/eval.sh <scenario> [models...] [options]
#
# Scenarios:
#   multi-snr         Evaluate at multiple SNR levels
#   gain-robustness   Sweep BCS-ACS gain offset at multiple SNRs
#
# Models: experiment names under results/experiments/, or group shortcuts:
#   --group backbone-map    bm_map_{12,25,50,75,100}ms
#   --group backbone-mask   bm_mask_{12,25,50,75,100}ms
#   --group bafnetplus      bafnetplus_{12,25,50,75,100}ms
#   --group ablation        abl_common_gain_only, abl_mask_only_alpha, abl_no_calibration
#   --group all             All 18 experiments
#
# Options:
#   --snrs "..."        Override SNR list (default: scenario-dependent)
#   --offsets "..."      Override gain offset list (gain-robustness only)
#   --eval-stt           Include STT evaluation (CER/WER)
#   --gpus 0,1           GPU IDs to use (default: 0)
#   --force              Re-run even if result JSON already exists
#   --dry-run            Print commands without executing
#
# Examples:
#   # Evaluate all BAFNet+ models at multiple SNRs
#   ./scripts/eval.sh multi-snr --group bafnetplus --gpus 0,1
#
#   # Gain robustness for two ablation models
#   ./scripts/eval.sh gain-robustness abl_full abl_no_calibration --gpus 0,1
#
#   # Single model, custom SNRs, with STT
#   ./scripts/eval.sh multi-snr bafnetplus_50ms --snrs "-10 0 10" --eval-stt
#
#   # Dry run to check what will be executed
#   ./scripts/eval.sh multi-snr --group all --gpus 0,1 --dry-run

set -euo pipefail

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="$PROJECT_ROOT/results/experiments"
EVAL_ROOT="$PROJECT_ROOT/results/eval"
NOISE_DIR="/home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k"
RIR_DIR="/home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k"
NOISE_TEST="dataset/taps/noise_test.txt"
RIR_TEST="dataset/taps/rir_test.txt"

# Pod-to-local path mapping
POD_PREFIX="/workspace"
LOCAL_PREFIX="/home/yskim/workspace"

# ── Group definitions ──────────────────────────────────────────────────────────
declare -A MODEL_GROUPS=(
    [backbone-map]="bm_map_12ms bm_map_25ms bm_map_50ms bm_map_75ms bm_map_100ms"
    [backbone-mask]="bm_mask_12ms bm_mask_25ms bm_mask_50ms bm_mask_75ms bm_mask_100ms"
    [bafnetplus]="bafnetplus_12ms bafnetplus_25ms bafnetplus_50ms bafnetplus_75ms bafnetplus_100ms"
    [ablation]="abl_common_gain_only abl_mask_only_alpha abl_no_calibration"
)
MODEL_GROUPS[all]="${MODEL_GROUPS[backbone-map]} ${MODEL_GROUPS[backbone-mask]} ${MODEL_GROUPS[bafnetplus]} ${MODEL_GROUPS[ablation]}"

# ── Defaults ───────────────────────────────────────────────────────────────────
SCENARIO=""
MODELS=()
SNRS=""
OFFSETS=""
EVAL_STT=false
GPUS=(0)
FORCE=false
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 1
}

[[ $# -lt 1 ]] && usage
SCENARIO="$1"; shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --group)
            key="$2"; shift 2
            if [[ -z "${MODEL_GROUPS[$key]+x}" ]]; then
                echo "ERROR: Unknown group '$key'. Available: ${!MODEL_GROUPS[*]}"
                exit 1
            fi
            read -ra _models <<< "${MODEL_GROUPS[$key]}"
            MODELS+=("${_models[@]}")
            ;;
        --snrs)      SNRS="$2"; shift 2 ;;
        --offsets)   OFFSETS="$2"; shift 2 ;;
        --eval-stt)  EVAL_STT=true; shift ;;
        --gpus)      IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --force)     FORCE=true; shift ;;
        --dry-run)   DRY_RUN=true; shift ;;
        --help|-h)   usage ;;
        -*)          echo "ERROR: Unknown option '$1'"; exit 1 ;;
        *)           MODELS+=("$1"); shift ;;
    esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: No models specified. Use model names or --group <name>."
    exit 1
fi

# ── Scenario defaults ─────────────────────────────────────────────────────────
case "$SCENARIO" in
    multi-snr)
        [[ -z "$SNRS" ]] && SNRS="-20 -10 0 10 15"
        EVAL_DIR="$EVAL_ROOT/multi-snr"
        ;;
    gain-robustness)
        [[ -z "$SNRS" ]] && SNRS="-10 0 10"
        [[ -z "$OFFSETS" ]] && OFFSETS="-20 -10 -5 0 5 10 20"
        EVAL_DIR="$EVAL_ROOT/gain-robustness"
        ;;
    *)
        echo "ERROR: Unknown scenario '$SCENARIO'. Use: multi-snr | gain-robustness"
        exit 1
        ;;
esac

mkdir -p "$EVAL_DIR"

# ── Validate models ───────────────────────────────────────────────────────────
for model in "${MODELS[@]}"; do
    if [[ ! -d "$EXP_DIR/$model" ]]; then
        echo "ERROR: Experiment directory not found: $EXP_DIR/$model"
        exit 1
    fi
    if [[ ! -f "$EXP_DIR/$model/.hydra/config.yaml" ]]; then
        echo "ERROR: Config not found: $EXP_DIR/$model/.hydra/config.yaml"
        exit 1
    fi
    if [[ ! -f "$EXP_DIR/$model/best.th" ]]; then
        echo "ERROR: Checkpoint not found: $EXP_DIR/$model/best.th"
        exit 1
    fi
done

# ── Config preparation (Pod→Local path rewrite) ──────────────────────────────
CONFIG_DIR="$EVAL_DIR/.configs"
mkdir -p "$CONFIG_DIR"

prepare_config() {
    local model=$1
    local config_src="$EXP_DIR/$model/.hydra/config.yaml"
    local config_dst="$CONFIG_DIR/${model}.yaml"
    sed "s|${POD_PREFIX}|${LOCAL_PREFIX}|g" "$config_src" > "$config_dst"
    echo "$config_dst"
}

# Check if model uses BCS-only input (SNR-independent)
is_bcs_only() {
    local model=$1
    local input_type
    input_type=$(grep -oP 'input_type:\s*\K\S+' "$EXP_DIR/$model/.hydra/config.yaml")
    [[ "$input_type" == "bcs" ]]
}

# ── Job generation ────────────────────────────────────────────────────────────
# Each job is a string: "model|snr|offset|output_json|log_file"
# For multi-snr, offset is empty.
JOBS=()

read -ra SNR_ARR <<< "$SNRS"

if [[ "$SCENARIO" == "multi-snr" ]]; then
    # One job per model (evaluate.py handles multiple SNRs natively)
    for model in "${MODELS[@]}"; do
        output_json="$EVAL_DIR/${model}.json"
        log_file="$EVAL_DIR/.logs/${model}.log"

        if [[ "$FORCE" == "false" && -f "$output_json" ]]; then
            echo "[SKIP] $output_json already exists"
            continue
        fi
        # BCS-only models are SNR-independent → evaluate at SNR 0 only
        local_snrs="$SNRS"
        if is_bcs_only "$model"; then
            local_snrs="0"
            echo "[INFO] $model is BCS-only, evaluating at SNR 0 only"
        fi
        JOBS+=("${model}|${local_snrs}||${output_json}|${log_file}")
    done

elif [[ "$SCENARIO" == "gain-robustness" ]]; then
    read -ra OFFSET_ARR <<< "$OFFSETS"
    TMP_DIR="$EVAL_DIR/.tmp"
    mkdir -p "$TMP_DIR"

    for model in "${MODELS[@]}"; do
        final_json="$EVAL_DIR/${model}.json"

        # If final merged JSON exists and not --force, skip entirely
        if [[ "$FORCE" == "false" && -f "$final_json" ]]; then
            echo "[SKIP] $final_json already exists"
            continue
        fi

        for snr in "${SNR_ARR[@]}"; do
            for offset in "${OFFSET_ARR[@]}"; do
                tmp_json="$TMP_DIR/${model}_snr${snr}_offset${offset}.json"
                log_file="$EVAL_DIR/.logs/${model}_snr${snr}_offset${offset}.log"

                if [[ "$FORCE" == "false" && -f "$tmp_json" ]]; then
                    continue
                fi

                bcs_gain=$(python3 -c "print($offset / 2)")
                acs_gain=$(python3 -c "print(-$offset / 2)")
                # Encode gain in offset field: "offset|bcs_gain|acs_gain"
                JOBS+=("${model}|${snr}|${offset}|${bcs_gain}|${acs_gain}|${tmp_json}|${log_file}")
            done
        done
    done
fi

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "All results already exist. Use --force to re-run."
    # Still run merge for gain-robustness in case tmp files exist but final doesn't
    if [[ "$SCENARIO" == "gain-robustness" ]]; then
        JOBS=("__merge_only__")
    else
        exit 0
    fi
fi

# ── Print summary ─────────────────────────────────────────────────────────────
NUM_GPUS=${#GPUS[@]}
echo ""
echo "=== Evaluation: $SCENARIO ==="
echo "  Models (${#MODELS[@]}): ${MODELS[*]}"
echo "  SNRs: $SNRS"
[[ "$SCENARIO" == "gain-robustness" ]] && echo "  Offsets: $OFFSETS"
echo "  GPUs: ${GPUS[*]} (${NUM_GPUS})"
echo "  STT: $EVAL_STT"
echo "  Jobs: ${#JOBS[@]}"
echo "  Output: $EVAL_DIR/"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "--- DRY RUN: commands that would be executed ---"
    echo ""
fi

# ── Execute: multi-snr ────────────────────────────────────────────────────────
run_multi_snr_job() {
    local gpu=$1
    local model=$2
    local snrs=$3
    local output_json=$4
    local log_file=$5

    local config
    config=$(prepare_config "$model")
    mkdir -p "$(dirname "$log_file")"

    local stt_flag=""
    [[ "$EVAL_STT" == "true" ]] && stt_flag="--eval_stt"

    local cmd="CUDA_VISIBLE_DEVICES=$gpu python -m src.evaluate \
        --model_config $config \
        --chkpt_dir $EXP_DIR/$model \
        --chkpt_file best.th \
        --dataset taps \
        --noise_dir $NOISE_DIR \
        --noise_test $NOISE_TEST \
        --rir_dir $RIR_DIR \
        --rir_test $RIR_TEST \
        --snr_step $snrs \
        $stt_flag \
        --output_json $output_json \
        --log_file $log_file"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "$cmd"
        echo ""
        return 0
    fi

    echo "[GPU$gpu] $model (SNR: $snrs)"
    eval "$cmd"
    echo "[GPU$gpu] Done: $model -> $output_json"
}

# ── Execute: gain-robustness ──────────────────────────────────────────────────
run_gain_job() {
    local gpu=$1
    local model=$2
    local snr=$3
    local offset=$4
    local bcs_gain=$5
    local acs_gain=$6
    local output_json=$7
    local log_file=$8

    local config
    config=$(prepare_config "$model")
    mkdir -p "$(dirname "$log_file")"

    local cmd="CUDA_VISIBLE_DEVICES=$gpu python -m src.evaluate \
        --model_config $config \
        --chkpt_dir $EXP_DIR/$model \
        --chkpt_file best.th \
        --dataset taps \
        --noise_dir $NOISE_DIR \
        --noise_test $NOISE_TEST \
        --rir_dir $RIR_DIR \
        --rir_test $RIR_TEST \
        --snr_step $snr \
        --bcs_gain_db $bcs_gain \
        --acs_gain_db $acs_gain \
        --output_json $output_json \
        --log_file $log_file"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "$cmd"
        echo ""
        return 0
    fi

    echo "[GPU$gpu] $model SNR=${snr}dB offset=${offset}dB"
    eval "$cmd"
    echo "[GPU$gpu] Done: $model SNR=${snr}dB offset=${offset}dB"
}

# ── Merge gain-robustness tmp JSONs into per-model JSON ───────────────────────
merge_gain_results() {
    local model=$1
    local final_json="$EVAL_DIR/${model}.json"
    local tmp_dir="$EVAL_DIR/.tmp"

    python3 -c "
import json, glob, re, sys

pattern = '${tmp_dir}/${model}_snr*_offset*.json'
files = sorted(glob.glob(pattern))
if not files:
    print(f'WARNING: No tmp files found for ${model}', file=sys.stderr)
    sys.exit(0)

merged = {}
for f in files:
    m = re.search(r'snr(-?\d+)_offset(-?\d+)', f)
    if not m:
        continue
    snr_key = f'snr={m.group(1)}dB'
    offset_key = f'offset={m.group(2)}dB'

    with open(f) as fp:
        data = json.load(fp)

    if snr_key not in merged:
        merged[snr_key] = {}
    # evaluate.py outputs {\"<snr>dB\": {metrics}}, extract the inner dict
    inner = list(data.values())[0] if data else {}
    merged[snr_key][offset_key] = inner

with open('${final_json}', 'w') as fp:
    json.dump(merged, fp, indent=2)
print(f'Merged {len(files)} files -> ${final_json}')
"
}

# ── GPU work distribution ─────────────────────────────────────────────────────
# Distribute jobs round-robin across GPUs, run each GPU's queue sequentially,
# all GPUs in parallel.

if [[ "${JOBS[0]}" == "__merge_only__" ]]; then
    JOBS=()
fi

if [[ ${#JOBS[@]} -gt 0 ]]; then
    # Create per-GPU job lists
    declare -a GPU_JOBS
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_JOBS[$i]=""
    done

    for ((j=0; j<${#JOBS[@]}; j++)); do
        gpu_idx=$((j % NUM_GPUS))
        GPU_JOBS[$gpu_idx]+="${JOBS[$j]}"$'\n'
    done

    # Run each GPU's work in a subshell
    cd "$PROJECT_ROOT"
    PIDS=()

    for ((i=0; i<NUM_GPUS; i++)); do
        gpu_id=${GPUS[$i]}
        (
            while IFS= read -r job; do
                [[ -z "$job" ]] && continue

                if [[ "$SCENARIO" == "multi-snr" ]]; then
                    IFS='|' read -r model snrs _ output_json log_file <<< "$job"
                    run_multi_snr_job "$gpu_id" "$model" "$snrs" "$output_json" "$log_file"
                elif [[ "$SCENARIO" == "gain-robustness" ]]; then
                    IFS='|' read -r model snr offset bcs_gain acs_gain output_json log_file <<< "$job"
                    run_gain_job "$gpu_id" "$model" "$snr" "$offset" "$bcs_gain" "$acs_gain" "$output_json" "$log_file"
                fi
            done <<< "${GPU_JOBS[$i]}"
        ) &
        PIDS+=($!)
        echo "Started GPU${gpu_id} worker (PID=${PIDS[-1]}, $(echo "${GPU_JOBS[$i]}" | grep -c . || true) jobs)"
    done

    # Wait for all GPUs
    FAIL=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=$((FAIL + 1))
    done

    if [[ $FAIL -gt 0 ]]; then
        echo ""
        echo "WARNING: $FAIL GPU worker(s) had failures. Check logs in $EVAL_DIR/.logs/"
    fi
fi

# ── Post-processing: merge gain-robustness results ────────────────────────────
if [[ "$SCENARIO" == "gain-robustness" && "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "Merging gain-robustness results..."
    for model in "${MODELS[@]}"; do
        merge_gain_results "$model"
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "=== Done ==="
    echo "Results:"
    for model in "${MODELS[@]}"; do
        json="$EVAL_DIR/${model}.json"
        if [[ -f "$json" ]]; then
            echo "  $json"
        fi
    done
fi
