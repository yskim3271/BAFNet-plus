#!/bin/bash
# wait_and_create_pod.sh
# RTX 5090 On-Demand 가용성을 1분마다 폴링하고, 사용 가능해지면 Pod을 자동 생성합니다.
#
# Usage:
#   ./scripts/wait_and_create_pod.sh [--pod-name NAME]
#
# Example:
#   ./scripts/wait_and_create_pod.sh --pod-name bafnet-train

set -euo pipefail

POD_NAME="bafnet-train"
GPU_TYPE="NVIDIA GeForce RTX 5090"
DATA_CENTER_ID="EU-RO-1"
VOLUME_ID="8mdudh5imp"
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
POLL_INTERVAL=60  # seconds

while [[ $# -gt 0 ]]; do
    case $1 in
        --pod-name) POD_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RUNPOD_API_KEY=$(grep -oP 'apikey = "\K[^"]+' ~/.runpod/config.toml)

check_gpu_availability() {
    local result
    result=$(curl -s -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\":\"{ gpuTypes { id displayName lowestPrice(input: {gpuCount: 1, dataCenterId: \\\"$DATA_CENTER_ID\\\"}) { uninterruptablePrice stockStatus } } }\"}" \
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

echo "=== RTX 5090 On-Demand 가용성 폴링 시작 ==="
echo "  리전: $DATA_CENTER_ID"
echo "  폴링 간격: ${POLL_INTERVAL}초"
echo "  Pod 이름: $POD_NAME"
echo ""

attempt=0
while true; do
    attempt=$((attempt + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if price=$(check_gpu_availability); then
        echo "[$timestamp] ✓ RTX 5090 On-Demand 사용 가능! (\$${price}/hr) - Pod 생성 중..."

        # Pod 생성 시도
        output=$(runpodctl create pod \
            --name "$POD_NAME" \
            --gpuType "$GPU_TYPE" \
            --gpuCount 1 \
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
        for i in $(seq 1 60); do
            pod_status=$(runpodctl get pod 2>/dev/null | grep "$POD_NAME" || true)
            if echo "$pod_status" | grep -q "RUNNING"; then
                echo ""
                echo "=== Pod RUNNING ==="
                echo "$pod_status"
                echo ""
                echo "다음 단계: ./scripts/run_train.sh <exp_name> [hydra overrides...]"
                exit 0
            fi
            printf "."
            sleep 10
        done

        echo ""
        echo "WARNING: Pod이 10분 내에 RUNNING 상태가 되지 않았습니다."
        echo "runpodctl get pod 으로 상태를 확인하세요."
        exit 1
    else
        echo "[$timestamp] (${attempt}) RTX 5090 On-Demand 사용 불가. ${POLL_INTERVAL}초 후 재시도..."
        sleep "$POLL_INTERVAL"
    fi
done
