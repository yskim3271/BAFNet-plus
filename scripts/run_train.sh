#!/bin/bash
# run_train.sh
# RunPod Pod에서 학습을 실행하고, 로그를 실시간 전송하며, 완료 후 Pod을 종료합니다.
#
# Usage:
#   ./scripts/run_train.sh [--keep-pod] <exp_name> [hydra overrides...]
#
# Examples:
#   # 1-epoch smoke test (eval 없이)
#   ./scripts/run_train.sh --keep-pod test_1ep epochs=1 eval_every=9999
#
#   # 200 epoch full training (완료 후 pod 자동 종료)
#   ./scripts/run_train.sh bm_200ep
#
#   # Pod 유지 (종료하지 않음)
#   ./scripts/run_train.sh --keep-pod bm_200ep
#
# Prerequisites:
#   - runpodctl 설치 및 인증 완료
#   - "bafnet-train" 이름의 RUNNING pod 존재
#   - SSH key가 runpod에 등록되어 있어야 함

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results/experiments"
REMOTE_PROJECT="/workspace/BAFNet-plus"
POD_NAME="bafnet-train"
KEEP_POD=false

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-pod) KEEP_POD=true; shift ;;
        --pod-name) POD_NAME="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) break ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--keep-pod] <exp_name> [hydra overrides...]"
    exit 1
fi

EXP_NAME="$1"
shift
HYDRA_OVERRIDES=("$@")
LOG_FILE="$PROJECT_DIR/${EXP_NAME}_train.log"

mkdir -p "$RESULTS_DIR"

# ---- Helper functions ----

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

find_pod_id() {
    runpodctl get pod 2>/dev/null | grep "$POD_NAME" | grep "RUNNING" | head -1 | awk '{print $1}'
}

get_ssh_cmd() {
    # "ssh root@HOST -p PORT" 형태의 문자열을 파싱
    local pod_id="$1"
    runpodctl ssh connect "$pod_id" 2>/dev/null | grep -oP 'ssh \S+ -p \d+'
}

remote_exec() {
    local ssh_host="$1"
    local ssh_port="$2"
    shift 2
    ssh $SSH_OPTS -p "$ssh_port" "$ssh_host" "$*"
}

remote_scp_from() {
    local ssh_host="$1"
    local ssh_port="$2"
    local remote_path="$3"
    local local_path="$4"
    scp $SSH_OPTS -P "$ssh_port" "${ssh_host}:${remote_path}" "$local_path"
}

# ---- Main ----

log "=== BAFNet+ Training: $EXP_NAME ==="
log "Hydra overrides: ${HYDRA_OVERRIDES[*]:-none}"
log "Keep pod after training: $KEEP_POD"

# 1. Find running pod and get SSH info
log "Finding RUNNING pod '$POD_NAME'..."
POD_ID=$(find_pod_id)
if [[ -z "$POD_ID" ]]; then
    log "ERROR: No RUNNING pod named '$POD_NAME' found."
    log "Run ./scripts/wait_and_create_pod.sh first."
    exit 1
fi
log "Found pod: $POD_ID"

SSH_CMD=$(get_ssh_cmd "$POD_ID")
if [[ -z "$SSH_CMD" ]]; then
    log "ERROR: Could not get SSH connection info for pod $POD_ID"
    exit 1
fi

# Parse "ssh root@HOST -p PORT"
SSH_HOST=$(echo "$SSH_CMD" | awk '{print $2}')
SSH_PORT=$(echo "$SSH_CMD" | awk '{print $4}')
log "SSH: $SSH_HOST port $SSH_PORT"

# 2. Pod setup: ensure repo exists, then run setup script
log "Setting up pod (repo + dependencies)..."
# Ensure repo exists on network volume (first-time clone or pull)
remote_exec "$SSH_HOST" "$SSH_PORT" \
    "if [ -d $REMOTE_PROJECT/.git ]; then cd $REMOTE_PROJECT && git pull --ff-only; else cd /workspace && git clone https://github.com/yskim3271/BAFNet-plus.git; fi" \
    2>&1 | tee -a "$LOG_FILE"
# Run full setup (system packages + python deps + verification)
remote_exec "$SSH_HOST" "$SSH_PORT" "bash $REMOTE_PROJECT/scripts/setup_pod.sh" 2>&1 | tee -a "$LOG_FILE"

# 4. Build training command
HYDRA_DIR="./results/experiments/${EXP_NAME}"
TRAIN_CMD="cd $REMOTE_PROJECT && python3 -m src.train +model=backbone_mapping +dset=taps hydra.run.dir=$HYDRA_DIR"
for override in "${HYDRA_OVERRIDES[@]:-}"; do
    if [[ -n "$override" ]]; then
        TRAIN_CMD="$TRAIN_CMD $override"
    fi
done

log "Training command: $TRAIN_CMD"

# 5. Run training (stream output to local log)
log "Starting training..."
remote_exec "$SSH_HOST" "$SSH_PORT" "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

if [[ $TRAIN_EXIT -ne 0 ]]; then
    log "ERROR: Training failed with exit code $TRAIN_EXIT"
    log "Pod will NOT be terminated due to training failure."
    exit $TRAIN_EXIT
fi

log "Training completed successfully!"

# 6. Transfer results
log "Transferring results from pod..."
LOCAL_EXP_DIR="$RESULTS_DIR/$EXP_NAME"
mkdir -p "$LOCAL_EXP_DIR"

# 원격에서 결과 압축
remote_exec "$SSH_HOST" "$SSH_PORT" \
    "cd $REMOTE_PROJECT && tar czf /tmp/${EXP_NAME}_results.tar.gz -C results/experiments $EXP_NAME" 2>&1 | tee -a "$LOG_FILE" || true

# SCP로 전송
remote_scp_from "$SSH_HOST" "$SSH_PORT" "/tmp/${EXP_NAME}_results.tar.gz" "$RESULTS_DIR/${EXP_NAME}_results.tar.gz" 2>&1 | tee -a "$LOG_FILE" || {
    log "WARNING: SCP transfer failed."
}

# 압축 해제
if [[ -f "$RESULTS_DIR/${EXP_NAME}_results.tar.gz" ]]; then
    tar xzf "$RESULTS_DIR/${EXP_NAME}_results.tar.gz" -C "$RESULTS_DIR/" 2>/dev/null || true
    rm -f "$RESULTS_DIR/${EXP_NAME}_results.tar.gz"
    log "Results transferred to: $LOCAL_EXP_DIR"
else
    log "WARNING: Could not transfer result files. Check pod manually."
fi

# 7. Terminate pod (unless --keep-pod)
if [[ "$KEEP_POD" == "false" ]]; then
    log "Terminating pod $POD_ID..."
    runpodctl remove pod "$POD_ID" 2>&1 | tee -a "$LOG_FILE"
    log "Pod terminated."
else
    log "Pod $POD_ID kept running (--keep-pod)."
fi

log "=== Done: $EXP_NAME ==="
log "Log file: $LOG_FILE"
