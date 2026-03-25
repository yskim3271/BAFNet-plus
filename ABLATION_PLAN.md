# BAFNetPlus Ablation Study Plan

## 실험 목록

| ID | 이름 | ablation_mode | 검증 포인트 | 상태 |
|----|------|---------------|-------------|------|
| A3 | Full (proposed) | `full` | 제안 모델 기준선 | [ ] |
| A1 | w/o Calibration | `no_calibration` | Calibration 모듈의 기여도 | [ ] |
| B1 | Common gain only | `common_gain_only` | Common/relative gain 분리 설계 근거 | [ ] |
| A2 | Mask-only alpha | `mask_only_alpha` | Alpha 입력에 magnitude 추가의 기여도 (BAFNet-style 비교) | [ ] |

## 공통 학습 설정

```bash
# 기본 하이퍼파라미터 (bafnetplus_50ms 실험과 동일)
epochs=200
eval_every=50
batch_size=16
valid_start_epoch=100    # 초반 100 epoch validation 스킵 (시간 절감)
lr=0.0005
warmup_ratio=0.1

# Backbone checkpoints (Pod 내 경로)
checkpoint_mapping=/workspace/BAFNet-plus/results/experiments/taps/bm_50ms/best.th
checkpoint_masking=/workspace/BAFNet-plus/results/experiments/taps/bm_mask_50ms/best.th

# 데이터셋 경로 (Pod 내 경로)
noise_dir=/workspace/dataset/datasets_fullband/noise_fullband_16k
rir_dir=/workspace/dataset/datasets_fullband/impulse_responses_16k
```

## 실행 명령어

### Pod 생성
```bash
./scripts/wait_and_create_pod.sh --gpu-type "NVIDIA RTX PRO 6000 Blackwell Server Edition" <pod-name>
```

### 학습 실행
```bash
# A3 (full)
nohup ./scripts/run_train.sh --pod-name <pod-name> --model bafnetplus abl_full \
  epochs=200 eval_every=50 batch_size=16 valid_start_epoch=100 \
  model.param.checkpoint_mapping=/workspace/BAFNet-plus/results/experiments/taps/bm_50ms/best.th \
  model.param.checkpoint_masking=/workspace/BAFNet-plus/results/experiments/taps/bm_mask_50ms/best.th \
  dset.noise_dir=/workspace/dataset/datasets_fullband/noise_fullband_16k \
  dset.rir_dir=/workspace/dataset/datasets_fullband/impulse_responses_16k \
  > /dev/null 2>&1 &

# A1 (no_calibration)
nohup ./scripts/run_train.sh --pod-name <pod-name> --model bafnetplus abl_no_calibration \
  epochs=200 eval_every=50 batch_size=16 valid_start_epoch=100 \
  model.param.ablation_mode=no_calibration \
  model.param.checkpoint_mapping=/workspace/BAFNet-plus/results/experiments/taps/bm_50ms/best.th \
  model.param.checkpoint_masking=/workspace/BAFNet-plus/results/experiments/taps/bm_mask_50ms/best.th \
  dset.noise_dir=/workspace/dataset/datasets_fullband/noise_fullband_16k \
  dset.rir_dir=/workspace/dataset/datasets_fullband/impulse_responses_16k \
  > /dev/null 2>&1 &

# B1 (common_gain_only)
nohup ./scripts/run_train.sh --pod-name <pod-name> --model bafnetplus abl_common_gain_only \
  epochs=200 eval_every=50 batch_size=16 valid_start_epoch=100 \
  model.param.ablation_mode=common_gain_only \
  model.param.checkpoint_mapping=/workspace/BAFNet-plus/results/experiments/taps/bm_50ms/best.th \
  model.param.checkpoint_masking=/workspace/BAFNet-plus/results/experiments/taps/bm_mask_50ms/best.th \
  dset.noise_dir=/workspace/dataset/datasets_fullband/noise_fullband_16k \
  dset.rir_dir=/workspace/dataset/datasets_fullband/impulse_responses_16k \
  > /dev/null 2>&1 &

# A2 (mask_only_alpha)
nohup ./scripts/run_train.sh --pod-name <pod-name> --model bafnetplus abl_mask_only_alpha \
  epochs=200 eval_every=50 batch_size=16 valid_start_epoch=100 \
  model.param.ablation_mode=mask_only_alpha \
  model.param.checkpoint_mapping=/workspace/BAFNet-plus/results/experiments/taps/bm_50ms/best.th \
  model.param.checkpoint_masking=/workspace/BAFNet-plus/results/experiments/taps/bm_mask_50ms/best.th \
  dset.noise_dir=/workspace/dataset/datasets_fullband/noise_fullband_16k \
  dset.rir_dir=/workspace/dataset/datasets_fullband/impulse_responses_16k \
  > /dev/null 2>&1 &
```

## 비용 예측

| 항목 | 값 |
|------|-----|
| GPU | RTX PRO 6000 Server ($1.69/hr) |
| 실험당 시간 | ~163.9h (valid_start_epoch=100 적용) |
| 4회 총 시간 | ~655.6h |
| 4회 총 비용 | ~$1,108 |

## 인프라

- **리전**: EU-RO-1
- **볼륨**: postech (8mdudh5imp, 50GB)
- **이미지**: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
- **GPU 타입 ID**: `NVIDIA RTX PRO 6000 Blackwell Server Edition`

## 결과 저장 경로

학습 완료 시 `run_train.sh`가 자동으로 로컬에 전송:
```
results/experiments/abl_full/
results/experiments/abl_no_calibration/
results/experiments/abl_common_gain_only/
results/experiments/abl_mask_only_alpha/
```

## 로그 확인

```bash
# 실시간 로그
tail -f abl_full_train.log

# 학습 진행 확인
grep "Overall Summary" abl_full_train.log | tail -5

# Pod 상태
runpodctl get pod
```
