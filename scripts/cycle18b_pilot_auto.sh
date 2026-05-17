#!/bin/bash
# Cycle 18b — self-managing pilot wrapper.
#
# Sequence:
#   1. 2-epoch QAT pilot training (Step 6)
#   2. Option β export — trained PT2E -> INT8 QDQ ONNX with init_overrides (Step 8)
#   3. Build combined split sidecar for Step 10 host parity
#   4. Self-terminate the RunPod pod via GraphQL podTerminate mutation
#
# All artifacts land on the postech network volume and survive pod removal:
#   - outputs/cycle18b_pilot/{best.th,checkpoint.th,trainer.log,tensorbd/}
#   - results/onnx/bafnetplus_50ms_int8_qat_pilot_trunk_t2.onnx (+ sidecar + stats)
#   - results/onnx/bafnetplus_50ms_split_int8_qat_pilot_t2.json
#   - /workspace/cycle18b_pilot_auto.log (this script's own log)
#
# Required env vars:
#   POD_ID         — current pod's ID for the GraphQL terminate call
#   RUNPOD_API_KEY — RunPod API key (Bearer auth)
#
# Detach with:
#   nohup setsid /workspace/cycle18b_pilot_auto.sh > /workspace/cycle18b_pilot_auto.out 2>&1 < /dev/null &

set -uo pipefail
LOG=/workspace/cycle18b_pilot_auto.log
exec > >(tee -a "$LOG") 2>&1

banner() {
    echo
    echo "==================================================================="
    echo "  $1"
    echo "  $(date -Iseconds)"
    echo "==================================================================="
}

banner "Cycle 18b auto-runner START"
echo "  POD_ID=${POD_ID:-<unset>}"
echo "  RUNPOD_API_KEY=${RUNPOD_API_KEY:+<set, ${#RUNPOD_API_KEY} chars>}"
echo "  cwd=$(pwd)"
echo "  pid=$$"

cd /workspace/BAFNet-plus

# ============================== STEP 6: training ============================
banner "STEP 6: 2-epoch QAT pilot training"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.train \
  +model=bafnetplus_qat +dset=taps \
  model.param.checkpoint_mapping=/workspace/BAFNet-plus/results/experiments/bm_map_50ms/best.th \
  model.param.checkpoint_masking=/workspace/BAFNet-plus/results/experiments/bm_mask_50ms/best.th \
  epochs=2 batch_size=2 num_workers=2 \
  valid_start_epoch=1 eval_every=2 warmup_epochs=0 \
  dset.noise_dir=/workspace/dataset/datasets_fullband/noise_fullband_16k \
  dset.rir_dir=/workspace/dataset/datasets_fullband/impulse_responses_16k \
  hydra.run.dir=outputs/cycle18b_pilot
TRAIN_EXIT=$?
echo "[wrapper] Training exited with code $TRAIN_EXIT"

# ============================== STEP 8: Option β =============================
banner "STEP 8: Option β export (init_overrides)"

PILOT_CKPT=outputs/cycle18b_pilot/best.th
if [ ! -f "$PILOT_CKPT" ]; then
    PILOT_CKPT=outputs/cycle18b_pilot/checkpoint.th
fi

OPTION_B_EXIT=99
if [ -f "$PILOT_CKPT" ]; then
    echo "[wrapper] Using pilot ckpt: $PILOT_CKPT"
    md5sum "$PILOT_CKPT" || true
    python -m scripts.qat_export_option_b \
      --pilot-ckpt "$PILOT_CKPT" \
      --fp32-trunk results/onnx/bafnetplus_50ms_fp32_trunk_t2.onnx \
      --reference-int8 results/onnx/bafnetplus_50ms_int8_qdq_trunk_t2.onnx \
      --output results/onnx/bafnetplus_50ms_int8_qat_pilot_trunk_t2.onnx \
      --calib-dir /tmp/bafnet_calib_taps_v3 \
      --bm-map results/experiments/bm_map_50ms/best.th \
      --bm-mask results/experiments/bm_mask_50ms/best.th \
      --batch-size 2 --time-frames 400 \
      --stats-json results/onnx/bafnetplus_50ms_int8_qat_pilot_trunk_t2.export_stats.json
    OPTION_B_EXIT=$?
    echo "[wrapper] Option β exited with code $OPTION_B_EXIT"
else
    echo "[wrapper] No pilot ckpt found at outputs/cycle18b_pilot/{best,checkpoint}.th — skipping Option β"
fi

# =========================== STEP 10 prep: split sidecar ====================
banner "STEP 10 prep: combined split sidecar"

QAT_ONNX=results/onnx/bafnetplus_50ms_int8_qat_pilot_trunk_t2.onnx
if [ -f "$QAT_ONNX" ]; then
    python -m scripts.build_qat_split_sidecar \
      --base results/onnx/bafnetplus_50ms_split_int8_t2.json \
      --qat-trunk-onnx "$QAT_ONNX" \
      --output results/onnx/bafnetplus_50ms_split_int8_qat_pilot_t2.json || true
fi

# =============================== self-terminate =============================
banner "Pod self-terminate"

if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${POD_ID:-}" ]; then
    echo "[wrapper] Terminating pod $POD_ID via GraphQL podTerminate..."
    RESPONSE=$(curl -s -X POST \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) }\"}" \
      https://api.runpod.io/graphql || echo '{"error":"curl failed"}')
    echo "[wrapper] Terminate response: $RESPONSE"
else
    echo "[wrapper] RUNPOD_API_KEY or POD_ID unset — pod NOT terminated; remove manually with:"
    echo "         runpodctl remove pod tg69liqxfia2mp"
fi

banner "Cycle 18b auto-runner DONE (train=$TRAIN_EXIT, option_b=$OPTION_B_EXIT)"
