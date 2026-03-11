#!/bin/bash
# wait_and_create_pod.sh
# GPU On-Demand 가용성을 1분마다 폴링하고, 사용 가능해지면 Pod을 자동 생성합니다.
# Pod이 RUNNING + SSH 가능 상태가 되면 setup_pod.sh를 자동 실행합니다.
#
# Usage:
#   ./scripts/wait_and_create_pod.sh [options]
#
# Options:
#   --pod-name NAME       Pod 이름 (default: bafnet-train)
#   --gpu-type TYPE       GPU 종류 (default: NVIDIA GeForce RTX 5090)
#   --gpu-count N         GPU 개수 (default: 1)
#
# Examples:
#   ./scripts/wait_and_create_pod.sh
#   ./scripts/wait_and_create_pod.sh --gpu-type "NVIDIA GeForce RTX 4090" --gpu-count 2
#   ./scripts/wait_and_create_pod.sh --pod-name my-train --gpu-type "NVIDIA H100 80GB HBM3"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_PROJECT="/workspace/BAFNet-plus"

POD_NAME="bafnet-train"
GPU_TYPE="NVIDIA GeForce RTX 5090"
GPU_COUNT=1
DATA_CENTER_ID="EU-RO-1"
VOLUME_ID="8mdudh5imp"
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
POLL_INTERVAL=60  # seconds

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5"

while [[ $# -gt 0 ]]; do
    case $1 in
        --pod-name)  POD_NAME="$2"; shift 2 ;;
        --gpu-type)  GPU_TYPE="$2"; shift 2 ;;
        --gpu-count) GPU_COUNT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RUNPOD_API_KEY=$(grep -oP 'apikey = "\K[^"]+' ~/.runpod/config.toml)

check_gpu_availability() {
    local result
    result=$(curl -s -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\":\"{ gpuTypes { id displayName lowestPrice(input: {gpuCount: $GPU_COUNT, dataCenterId: \\\"$DATA_CENTER_ID\\\"}) { uninterruptablePrice stockStatus } } }\"}" \
        https://api.runpod.io/graphql 2>/dev/null)

    local price
    price=$(echo "$result" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for g in data['data']['gpuTypes']:
    if g['id'] == '$GPU_TYPE':
        p = g.get('lowestPrice')
        if p and p.get('uninterruptablePrice'):
            print(p['uninterruptablePrice'])
            break
" 2>/dev/null)

    if [[ -n "$price" ]]; then
        echo "$price"
        return 0
    fi
    return 1
}

get_ssh_cmd() {
    local pod_id="$1"
    runpodctl ssh connect "$pod_id" 2>/dev/null | grep -oP 'ssh \S+ -p \d+'
}

echo "=== GPU On-Demand 가용성 폴링 시작 ==="
echo "  GPU: $GPU_TYPE x$GPU_COUNT"
echo "  리전: $DATA_CENTER_ID"
echo "  폴링 간격: ${POLL_INTERVAL}초"
echo "  Pod 이름: $POD_NAME"
echo ""

attempt=0
while true; do
    attempt=$((attempt + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if price=$(check_gpu_availability); then
        echo "[$timestamp] $GPU_TYPE On-Demand 사용 가능! (\$${price}/hr) - Pod 생성 중..."

        # Pod 생성 시도
        output=$(runpodctl create pod \
            --name "$POD_NAME" \
            --gpuType "$GPU_TYPE" \
            --gpuCount "$GPU_COUNT" \
            --imageName "$IMAGE" \
            --networkVolumeId "$VOLUME_ID" \
            --volumePath "/workspace" \
            --containerDiskSize 20 \
            --ports "22/tcp" \
            --startSSH \
            --secureCloud 2>&1) || true

        if echo "$output" | grep -qi "no longer any instances\|error"; then
            echo "[$timestamp]   Pod 생성 실패 (인스턴스 소진). 재시도..."
            echo "  $output"
            sleep "$POLL_INTERVAL"
            continue
        fi

        echo ""
        echo "=== Pod 생성 완료 ==="
        echo "$output"
        echo ""

        # Pod이 RUNNING 상태가 될 때까지 대기
        echo "Pod이 RUNNING 상태가 될 때까지 대기 중..."
        POD_ID=""
        for i in $(seq 1 60); do
            pod_line=$(runpodctl get pod 2>/dev/null | grep "$POD_NAME" || true)
            if echo "$pod_line" | grep -q "RUNNING"; then
                POD_ID=$(echo "$pod_line" | awk '{print $1}')
                echo ""
                echo "=== Pod RUNNING ==="
                echo "$pod_line"
                break
            fi
            printf "."
            sleep 10
        done

        if [[ -z "$POD_ID" ]]; then
            echo ""
            echo "WARNING: Pod이 10분 내에 RUNNING 상태가 되지 않았습니다."
            echo "runpodctl get pod 으로 상태를 확인하세요."
            exit 1
        fi

        # 1분 대기 (Pod 내부 초기화 시간)
        echo ""
        echo "Pod 내부 초기화 대기 (60초)..."
        sleep 60

        # SSH 접속 가능할 때까지 폴링
        echo "SSH 접속 대기 중..."
        SSH_CMD=$(get_ssh_cmd "$POD_ID")
        if [[ -z "$SSH_CMD" ]]; then
            echo "WARNING: SSH 접속 정보를 가져올 수 없습니다."
            echo "다음 단계: ./scripts/run_train.sh <exp_name> [hydra overrides...]"
            exit 1
        fi

        SSH_HOST=$(echo "$SSH_CMD" | awk '{print $2}')
        SSH_PORT=$(echo "$SSH_CMD" | awk '{print $4}')

        for i in $(seq 1 30); do
            if ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_HOST" "echo ok" 2>/dev/null; then
                echo ""
                echo "=== SSH 접속 성공 ==="
                break
            fi
            printf "."
            sleep 10
        done

        # setup_pod.sh 자동 실행
        echo ""
        echo "=== setup_pod.sh 실행 ==="
        # 레포가 없으면 clone
        ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_HOST" \
            "test -d $REMOTE_PROJECT/.git || (cd /workspace && git clone https://github.com/yskim3271/BAFNet-plus.git)"
        ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_HOST" "bash $REMOTE_PROJECT/scripts/setup_pod.sh"

        echo ""
        echo "=== Setup 완료 ==="
        echo "다음 단계: ./scripts/run_train.sh <exp_name> [hydra overrides...]"
        exit 0
    else
        echo "[$timestamp] (${attempt}) $GPU_TYPE On-Demand 사용 불가. ${POLL_INTERVAL}초 후 재시도..."
        sleep "$POLL_INTERVAL"
    fi
done
