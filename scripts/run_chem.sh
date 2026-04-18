#!/bin/bash
# run_chem.sh
# RunPod Pod에서 CHEM finetuning 실험을 실행하고 결과를 회수합니다.
#
# Usage:
#   ./scripts/run_chem.sh --pod-name <name> --latency <12ms|50ms|100ms> [--keep-pod]
#
# Examples:
#   ./scripts/run_chem.sh --pod-name chem-12ms --latency 12ms
#   ./scripts/run_chem.sh --pod-name chem-50ms --latency 50ms --keep-pod
#
# What it does:
#   1. Find RUNNING pod by name
#   2. Run setup_pod.sh (git pull + deps)
#   3. rsync CHEM data + pretrained checkpoint to pod (if missing)
#   4. Run run_experiments.py --all --latency XXms --no_eval_stt
#   5. Transfer results back to local _backup/ablation/CHEM_dataset/results/
#   6. Terminate pod (unless --keep-pod)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHEM_LOCAL="$PROJECT_DIR/_backup/ablation/CHEM_dataset"
REMOTE_PROJECT="/workspace/BAFNet-plus"
REMOTE_CHEM="$REMOTE_PROJECT/_backup/ablation/CHEM_dataset"
REMOTE_BM_MAP_BASE="$REMOTE_PROJECT/results/experiments"

POD_NAME=""
LATENCY=""
KEEP_POD=false

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --pod-name) POD_NAME="$2"; shift 2 ;;
        --latency)  LATENCY="$2"; shift 2 ;;
        --keep-pod) KEEP_POD=true; shift ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$POD_NAME" ]] && { echo "ERROR: --pod-name required"; exit 1; }
[[ -z "$LATENCY" ]]  && { echo "ERROR: --latency required (12ms|50ms|100ms)"; exit 1; }

case "$LATENCY" in
    12ms|50ms|100ms) ;;
    *) echo "ERROR: invalid latency '$LATENCY' (must be 12ms, 50ms, or 100ms)"; exit 1 ;;
esac

LOG_FILE="$PROJECT_DIR/chem_${LATENCY}_train.log"

# ---- Lock file ----
LOCK_FILE="/tmp/run_chem_${LATENCY}.lock"
if [[ -f "$LOCK_FILE" ]]; then
    OLD_PID=$(cat "$LOCK_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: CHEM $LATENCY is already running (PID $OLD_PID)"
        exit 1
    fi
fi
echo $$ > "$LOCK_FILE"
trap "rm -f '$LOCK_FILE'" EXIT

# ---- Helpers ----
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

find_pod_id() {
    local matches count
    matches=$(runpodctl get pod 2>/dev/null | grep "$POD_NAME" | grep "RUNNING" || true)
    count=$(echo "$matches" | grep -c . 2>/dev/null || echo 0)
    if [[ "$count" -gt 1 ]]; then
        log "ERROR: Multiple RUNNING pods match '$POD_NAME'"
        echo "$matches"
        exit 1
    fi
    echo "$matches" | head -1 | awk '{print $1}'
}

get_ssh_cmd() {
    runpodctl ssh connect "$1" 2>/dev/null | grep -oP 'ssh \S+ -p \d+'
}

remote_exec() {
    ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_HOST" "$*"
}

# ---- Main ----
log "=== CHEM Finetuning: latency=$LATENCY ==="

# 1. Find pod
log "Finding RUNNING pod '$POD_NAME'..."
POD_ID=$(find_pod_id)
[[ -z "$POD_ID" ]] && { log "ERROR: No RUNNING pod named '$POD_NAME'"; exit 1; }
log "Pod: $POD_ID"

SSH_CMD=$(get_ssh_cmd "$POD_ID")
[[ -z "$SSH_CMD" ]] && { log "ERROR: Could not get SSH info for $POD_ID"; exit 1; }
SSH_HOST=$(echo "$SSH_CMD" | awk '{print $2}')
SSH_PORT=$(echo "$SSH_CMD" | awk '{print $4}')
log "SSH: $SSH_HOST:$SSH_PORT"

# 2. Setup pod (git pull + deps)
log "Running setup_pod.sh on pod..."
# Ensure repo exists and then run setup; use stream-tee to capture output
remote_exec "
    if [ ! -d $REMOTE_PROJECT/.git ]; then
        cd /workspace && git clone https://github.com/yskim3271/BAFNet-plus.git
    fi
    cd $REMOTE_PROJECT && git pull --ff-only || true
    bash scripts/setup_pod.sh
" 2>&1 | tee -a "$LOG_FILE"

# 3. Upload CHEM data (shared network volume — upload once via lock file)
# All pods share the same network volume, so use a remote lock to serialize uploads
log "Checking CHEM data on shared volume..."
remote_exec "mkdir -p $REMOTE_CHEM" 2>&1 | tee -a "$LOG_FILE"

UPLOAD_LOCK="/workspace/.chem_upload.lock"
UPLOAD_DONE="/workspace/.chem_upload.done"

# Wait for upload lock (max 10 min) or claim it
CLAIMED=$(remote_exec "
    # If already done, skip
    if [ -f $UPLOAD_DONE ]; then echo SKIP; exit 0; fi
    # Try to claim lock atomically
    if mkdir $UPLOAD_LOCK 2>/dev/null; then echo CLAIMED; else echo WAITING; fi
" | tr -d '\r')

if [[ "$CLAIMED" == "SKIP" ]]; then
    log "CHEM data already uploaded (marker file exists), skipping"
elif [[ "$CLAIMED" == "CLAIMED" ]]; then
    log "Uploading CHEM dataset (~210MB) via rsync..."
    # Trap to release lock on failure
    trap "remote_exec 'rmdir $UPLOAD_LOCK 2>/dev/null || true'; rm -f '$LOCK_FILE'" EXIT

    if rsync -rvz --no-owner --no-group --no-perms \
        --exclude='omniasr_venv' \
        --exclude='__pycache__' \
        --exclude='results' \
        --exclude='figures' \
        -e "ssh $SSH_OPTS -p $SSH_PORT" \
        "$CHEM_LOCAL/" \
        "$SSH_HOST:$REMOTE_CHEM/" 2>&1 | tail -20 | tee -a "$LOG_FILE"; then
        remote_exec "touch $UPLOAD_DONE && rmdir $UPLOAD_LOCK"
        log "CHEM data upload complete, marker set"
    else
        remote_exec "rmdir $UPLOAD_LOCK 2>/dev/null || true"
        log "ERROR: rsync failed"; exit 1
    fi
    # Restore original trap
    trap "rm -f '$LOCK_FILE'" EXIT
else
    log "Another pod is uploading CHEM data, waiting..."
    for i in $(seq 1 60); do
        STATE=$(remote_exec "[ -f $UPLOAD_DONE ] && echo DONE || echo WAIT" | tr -d '\r')
        if [[ "$STATE" == "DONE" ]]; then
            log "CHEM data upload completed by another pod"
            break
        fi
        sleep 10
    done
    if [[ "$STATE" != "DONE" ]]; then
        log "ERROR: Timed out waiting for CHEM data upload"; exit 1
    fi
fi

# Upload pretrained checkpoint for this latency
REMOTE_CKPT="$REMOTE_BM_MAP_BASE/bm_map_${LATENCY}/best.th"
CKPT_EXISTS=$(remote_exec "test -f $REMOTE_CKPT && echo YES || echo NO" | tr -d '\r')

if [[ "$CKPT_EXISTS" != "YES" ]]; then
    log "Uploading pretrained checkpoint (bm_map_${LATENCY}/best.th)..."
    remote_exec "mkdir -p $REMOTE_BM_MAP_BASE/bm_map_${LATENCY}"
    scp $SSH_OPTS -P "$SSH_PORT" \
        "$PROJECT_DIR/results/experiments/bm_map_${LATENCY}/best.th" \
        "$SSH_HOST:$REMOTE_CKPT" 2>&1 | tee -a "$LOG_FILE"
else
    log "Pretrained checkpoint already on pod, skipping upload"
fi

# 4. Run experiment
log "Starting CHEM experiments (all folds x 3 experiments for $LATENCY)..."
SAFE_NAME="chem_${LATENCY}"
REMOTE_LOG="/tmp/${SAFE_NAME}_stdout.log"
REMOTE_PID_FILE="/tmp/${SAFE_NAME}.pid"
REMOTE_EXIT_FILE="/tmp/${SAFE_NAME}.exit"

# Kill stale + clean temp
remote_exec "
    if [ -f $REMOTE_PID_FILE ]; then
        OLD=\$(cat $REMOTE_PID_FILE)
        kill \$OLD 2>/dev/null || true
    fi
    rm -f $REMOTE_EXIT_FILE $REMOTE_PID_FILE $REMOTE_LOG
"

TRAIN_CMD="cd $REMOTE_PROJECT && python3 _backup/ablation/CHEM_dataset/scripts/run_experiments.py --all --latency $LATENCY --no_eval_stt"
log "Command: $TRAIN_CMD"

REMOTE_PID=$(remote_exec "
    nohup bash -c '$TRAIN_CMD > $REMOTE_LOG 2>&1; echo \$? > $REMOTE_EXIT_FILE' </dev/null >/dev/null 2>&1 &
    echo \$! > $REMOTE_PID_FILE
    sleep 1
    cat $REMOTE_PID_FILE
")
log "Remote PID: $REMOTE_PID"

# Wait for log to appear
log "Waiting for training to initialize..."
for i in $(seq 1 60); do
    if remote_exec "test -f $REMOTE_LOG" 2>/dev/null; then break; fi
    sleep 5
done

# Stream logs
log "Streaming logs..."
SKIP=1
while true; do
    remote_exec "
        tail -n +${SKIP} -f $REMOTE_LOG | stdbuf -oL uniq & TAIL=\$!
        while ! test -f $REMOTE_EXIT_FILE; do sleep 10; done
        kill \$TAIL 2>/dev/null
        wait \$TAIL 2>/dev/null
    " 2>&1 | tee -a "$LOG_FILE" || true

    if remote_exec "test -f $REMOTE_EXIT_FILE" 2>/dev/null; then
        TRAIN_EXIT=$(remote_exec "cat $REMOTE_EXIT_FILE")
        break
    fi

    log "SSH dropped, reconnecting in 30s..."
    sleep 30
    SKIP=$(remote_exec "wc -l < $REMOTE_LOG 2>/dev/null || echo 0")
    SKIP=$((SKIP + 1))
done

if [[ "$TRAIN_EXIT" -ne 0 ]]; then
    log "ERROR: Training failed with exit code $TRAIN_EXIT"
    log "Pod kept for inspection."
    exit "$TRAIN_EXIT"
fi

log "Training completed!"

# 5. Transfer results back
log "Transferring results..."
LOCAL_RESULTS="$CHEM_LOCAL/results/bm_map_${LATENCY}"
mkdir -p "$LOCAL_RESULTS"

REMOTE_RESULTS_DIR="$REMOTE_CHEM/results/bm_map_${LATENCY}"
remote_exec "test -d $REMOTE_RESULTS_DIR && cd $REMOTE_CHEM/results && tar czf /tmp/chem_${LATENCY}_results.tar.gz bm_map_${LATENCY}" 2>&1 | tee -a "$LOG_FILE" || {
    log "WARNING: Results dir not found on pod"
}

scp $SSH_OPTS -P "$SSH_PORT" \
    "$SSH_HOST:/tmp/chem_${LATENCY}_results.tar.gz" \
    "$CHEM_LOCAL/results/chem_${LATENCY}_results.tar.gz" 2>&1 | tee -a "$LOG_FILE" || {
    log "WARNING: scp failed"
}

if [[ -f "$CHEM_LOCAL/results/chem_${LATENCY}_results.tar.gz" ]]; then
    tar xzf "$CHEM_LOCAL/results/chem_${LATENCY}_results.tar.gz" -C "$CHEM_LOCAL/results/" 2>/dev/null || true
    rm -f "$CHEM_LOCAL/results/chem_${LATENCY}_results.tar.gz"
    log "Results: $LOCAL_RESULTS"
fi

# 6. Terminate pod
if [[ "$KEEP_POD" == "false" ]]; then
    log "Terminating pod $POD_ID..."
    runpodctl remove pod "$POD_ID" 2>&1 | tee -a "$LOG_FILE"
else
    log "Pod $POD_ID kept (--keep-pod)"
fi

log "=== Done: CHEM $LATENCY ==="
