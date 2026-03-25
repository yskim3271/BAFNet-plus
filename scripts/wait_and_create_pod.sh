#!/bin/bash
# wait_and_create_pod.sh
# GPU On-Demand 가용성을 1분마다 폴링하고, 사용 가능해지면 Pod을 자동 생성합니다.
# Pod이 RUNNING + SSH 가능 상태가 되면 setup_pod.sh를 자동 실행합니다.
#
# Usage:
#   ./scripts/wait_and_create_pod.sh [options] [pod_name ...]
#
# Options:
#   --gpu-type TYPE       GPU 종류 (default: NVIDIA GeForce RTX 5090)
#
# Pod names are positional arguments after options. Default: bafnet-train
#
# Examples:
#   ./scripts/wait_and_create_pod.sh
#   ./scripts/wait_and_create_pod.sh train-a train-b train-c
#   ./scripts/wait_and_create_pod.sh --gpu-type "NVIDIA H100 80GB HBM3" exp-1 exp-2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_PROJECT="/workspace/BAFNet-plus"

GPU_TYPE="NVIDIA GeForce RTX 5090"
DATA_CENTER_ID="EU-RO-1"
VOLUME_ID="8mdudh5imp"
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
POLL_INTERVAL=60  # seconds

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5"

POD_NAMES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-type)  GPU_TYPE="$2"; shift 2 ;;
        --*)         echo "Unknown option: $1"; exit 1 ;;
        *)           POD_NAMES+=("$1"); shift ;;
    esac
done

# Default pod name if none provided
if [[ ${#POD_NAMES[@]} -eq 0 ]]; then
    POD_NAMES=("bafnet-train")
fi

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

get_ssh_cmd() {
    local pod_id="$1"
    runpodctl ssh connect "$pod_id" 2>/dev/null | grep -oP 'ssh \S+ -p \d+'
}

setup_pod() {
    local pod_name="$1"
    local pod_id="$2"

    # 1분 대기 (Pod 내부 초기화 시간)
    echo ""
    echo "[$pod_name] Pod 내부 초기화 대기 (60초)..."
    sleep 60

    # SSH 접속 정보 획득
    echo "[$pod_name] SSH 접속 대기 중..."
    local ssh_cmd ssh_host ssh_port
    ssh_cmd=$(get_ssh_cmd "$pod_id")
    if [[ -z "$ssh_cmd" ]]; then
        echo "[$pod_name] WARNING: SSH 접속 정보를 가져올 수 없습니다."
        return 1
    fi

    ssh_host=$(echo "$ssh_cmd" | awk '{print $2}')
    ssh_port=$(echo "$ssh_cmd" | awk '{print $4}')

    # SSH 접속 가능할 때까지 폴링
    local ssh_ok=false
    for i in $(seq 1 30); do
        if ssh $SSH_OPTS -p "$ssh_port" "$ssh_host" "echo ok" 2>/dev/null; then
            echo ""
            echo "[$pod_name] === SSH 접속 성공 ==="
            ssh_ok=true
            break
        fi
        printf "."
        sleep 10
    done

    if [[ "$ssh_ok" != "true" ]]; then
        echo ""
        echo "[$pod_name] WARNING: SSH 접속 시간 초과."
        return 1
    fi

    # setup_pod.sh 자동 실행
    echo "[$pod_name] === setup_pod.sh 실행 ==="
    ssh $SSH_OPTS -p "$ssh_port" "$ssh_host" \
        "test -d $REMOTE_PROJECT/.git || (cd /workspace && git clone https://github.com/yskim3271/BAFNet-plus.git)"
    ssh $SSH_OPTS -p "$ssh_port" "$ssh_host" "bash $REMOTE_PROJECT/scripts/setup_pod.sh"
    echo "[$pod_name] === Setup 완료 ==="
}

echo "=== GPU On-Demand 가용성 폴링 시작 ==="
echo "  GPU: $GPU_TYPE"
echo "  리전: $DATA_CENTER_ID"
echo "  폴링 간격: ${POLL_INTERVAL}초"
echo "  Pod 이름: ${POD_NAMES[*]} (${#POD_NAMES[@]}개)"
echo ""

# Track remaining pods to create and successfully created pod IDs
REMAINING_PODS=("${POD_NAMES[@]}")
declare -A POD_IDS

attempt=0
while [[ ${#REMAINING_PODS[@]} -gt 0 ]]; do
    attempt=$((attempt + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if price=$(check_gpu_availability); then
        echo "[$timestamp] $GPU_TYPE On-Demand 사용 가능! (\$${price}/hr) - 남은 Pod ${#REMAINING_PODS[@]}개 생성 중..."

        newly_created=()
        for pod_name in "${REMAINING_PODS[@]}"; do
            output=$(runpodctl create pod \
                --name "$pod_name" \
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
                echo "[$timestamp] [$pod_name] Pod 생성 실패 (인스턴스 소진). 남은 Pod은 다음 시도에서 재시도."
                break
            fi

            echo "[$pod_name] Pod 생성 요청 완료: $output"
            newly_created+=("$pod_name")
        done

        # Remove successfully created pods from REMAINING list
        if [[ ${#newly_created[@]} -gt 0 ]]; then
            updated=()
            for pod_name in "${REMAINING_PODS[@]}"; do
                skip=false
                for created in "${newly_created[@]}"; do
                    if [[ "$pod_name" == "$created" ]]; then
                        skip=true
                        break
                    fi
                done
                if [[ "$skip" == "false" ]]; then
                    updated+=("$pod_name")
                fi
            done
            REMAINING_PODS=("${updated[@]}")
            echo "생성 완료: ${newly_created[*]} | 남은 Pod: ${REMAINING_PODS[*]:-없음}"
        fi

        # Wait for newly created pods to become RUNNING, then setup
        if [[ ${#newly_created[@]} -gt 0 ]]; then
            echo "새로 생성된 Pod이 RUNNING 상태가 될 때까지 대기 중..."
            for i in $(seq 1 60); do
                pod_list=$(runpodctl get pod 2>/dev/null || true)
                all_running=true
                for pod_name in "${newly_created[@]}"; do
                    if [[ -n "${POD_IDS[$pod_name]+x}" ]]; then
                        continue
                    fi
                    pod_line=$(echo "$pod_list" | grep "$pod_name" || true)
                    if echo "$pod_line" | grep -q "RUNNING"; then
                        POD_IDS["$pod_name"]=$(echo "$pod_line" | awk '{print $1}')
                        echo ""
                        echo "[$pod_name] === RUNNING === ($pod_line)"
                    else
                        all_running=false
                    fi
                done
                if [[ "$all_running" == "true" ]]; then
                    break
                fi
                printf "."
                sleep 10
            done

            # Setup RUNNING pods
            for pod_name in "${newly_created[@]}"; do
                if [[ -n "${POD_IDS[$pod_name]+x}" ]]; then
                    setup_pod "$pod_name" "${POD_IDS[$pod_name]}"
                else
                    echo "[$pod_name] WARNING: RUNNING 상태가 되지 않았습니다."
                fi
            done
        fi

        # Check if all done
        if [[ ${#REMAINING_PODS[@]} -eq 0 ]]; then
            echo ""
            echo "=== 전체 완료 (${#POD_NAMES[@]}개 Pod) ==="
            echo "다음 단계: ./scripts/run_train.sh <exp_name> [hydra overrides...]"
            exit 0
        fi

        echo "남은 Pod ${#REMAINING_PODS[@]}개 대기 중... (${REMAINING_PODS[*]})"
        sleep "$POLL_INTERVAL"
    else
        echo "[$timestamp] (${attempt}) $GPU_TYPE On-Demand 사용 불가. ${POLL_INTERVAL}초 후 재시도..."
        sleep "$POLL_INTERVAL"
    fi
done
