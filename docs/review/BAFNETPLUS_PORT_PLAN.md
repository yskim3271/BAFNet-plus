# BAFNetPlus Android Port Plan

- **Repository HEAD (at plan drafting)**: `f05664c` on `main` — 직전 LaCoSENet Android review 종결 직후 시점
- **Plan date**: 2026-04-19
- **Author context**: 단일 개발자, LaCoSENet 리뷰 프로젝트 (`docs/review/REPORT.md`) 종결 후 후속 포팅
- **Document scope**: BAFNetPlus (`src/models/bafnetplus.py`) 를 기존 LaCoSENet 기반 Android 스택 위에 포팅하기 위한 **구현 계획서**. 본 문서 자체가 산출물이며, 실제 코드/빌드/기기 작업은 Stage 1~6 실행 시점에 별도 세션으로 진행
- **Reference style**: LaCoSENet `REPORT.md` 의 8축 판정·Stage·P0/P1·parity 7/7 용어 그대로 재사용

---

## 0. Executive Summary

> BAFNetPlus 는 **두 개의 Backbone(mapping + masking)** 을 병렬 구동하고 **경량 fusion 서브네트워크(calibration encoder + 3ch alpha conv)** 로 두 출력을 블렌딩하는 dual-microphone 구조다. LaCoSENet Android 포팅의 검증된 기반(parity 7/7, pre-push hook, QDQ INT8 10ms 수준 HTP 성능, `benchmarkDualBackboneConcurrentQdq` 14.6~15.2ms)을 **그대로 재사용**하면서, (a) Python streaming wrapper 신설, (b) ONNX export 2-input/160-state 확장, (c) Android StatefulInference 시그니처 확장, (d) BAFNetPlus 전용 golden fixture 추가로 약 **1,200~1,600 LoC 증분**이 예상된다.

### 핵심 불확실성 3가지

| # | 불확실성 | 영향 | 해소 시점 |
|---|---|---|---|
| U1 | **Ablation 모드 선정** (`full` / `no_calibration` / `mask_only_alpha` / `common_gain_only` 중 배포 대상) | Python wrapper · ONNX export · fixture 의 입력 시그니처가 달라짐. 재export churn 위험 | Stage 1 킥오프 전 (Open Q1) |
| U2 | **단일 session vs 두 session 배포 전략** — 통합 ONNX 그래프(mapping+masking+fusion 일체)로 export 할지, LaCoSENet 처럼 단일 Backbone 그래프를 2개 세션으로 띄우고 fusion 만 host 측 Kotlin 으로 할지 | 지연·VTCM·native memory·Kotlin 구현 복잡도 모두에 영향. milestone.md dual concurrent 10.2~15.2ms 숫자는 "두 세션" 시나리오 기반 | Stage 3 ONNX verify 단계 (op coverage + graph finalization 결과) |
| U3 | **실기기 BCS 마이크 경로** — Galaxy S25 Ultra 는 일반 스테레오 마이크만 보유. 진짜 BCS(throat / bone-conduction)입력은 별도 하드웨어 필요. 데모/평가용으로 simulated BCS 를 쓸지, 실 BCS 캡쳐 경로를 뚫을지 | Android 통합(§5) 후반의 오디오 입력 파이프라인 설계에 직결 | Stage 4 후반 또는 별도 세션 (Open Q3) |

### 한줄 요약

> LaCoSENet Android 포팅이 float32 머신 epsilon parity + QDQ INT8 6.2ms HTP 수준의 "**검증된 스택**"을 남겨놓았으므로, BAFNetPlus 포팅은 **본질적으로 이 스택의 입력 2배·state 2배·출력 3종 확장 + 경량 fusion 추가** 문제로 축소된다. 가장 큰 리스크는 구현이 아니라 (a) 배포할 ablation 모드와 (b) 통합 vs 분리 session 전략의 사전 확정이며, 둘 다 구현 전 결정 사항이다.

---

## 1. Scope & Non-Goals

### 1.1 In-scope (본 포팅이 책임지는 범위)

1. **BAFNetPlus 신경망 추론의 streaming 변환 + Android 배포** — batch 오프라인 모드는 이미 `src/enhance.py` + `src/models/bafnetplus.py` 로 가능하지만, 실시간 chunk-by-chunk 스트리밍 경로는 신규
2. **Python-side streaming wrapper (`BAFNetPlusStreaming`)** — 기존 `LaCoSENet` wrapper 구조를 그대로 본뜬 2-input streaming 파이프라인
3. **Stateful ONNX export + streaming_config.json** — 2 Backbone × 80 state + fusion state(예상 4~6) ≈ **160+α 개 state** 직렬화
4. **Android library `bafnetplus-streaming`** (신규 모듈 또는 기존 `lacosenet-streaming` 일반화) — dual mag/pha 입력, 160 state 더블 버퍼, 3-output 해석, fusion 출력 소비
5. **Parity suite 확장** — 기존 LaCoSENet 7/7 @Test 유지 + BAFNetPlus 전용 parity @Test (STFT 는 공유, StatefulInference / E2E 만 추가)
6. **Benchmark extension** — `milestone.md` 에 BAFNetPlus 전용 row 추가, cold-state 및 same-process 시나리오 재측정
7. **Pre-push hook 범위 확장** — 기존 hook 이 `android/` + `scripts/make_streaming_golden.py` 변경 시 7/7 실행. BAFNetPlus fixture/test 가 추가되면 hook 이 두 suite 모두 실행하도록 업데이트

### 1.2 Non-goals (본 포팅에서 하지 않는 것)

| # | 항목 | 이유 |
|---|---|---|
| NG-1 | **온-디바이스 학습 / finetune** | BAFNetPlus 학습 파이프라인은 PyTorch 서버 측 (`src/train.py`) 고정. 모바일은 inference-only. |
| NG-2 | **품질 평가 (PESQ/STOI/CER on-device)** | metric 계산은 오프라인 `src/evaluate.py` 경로. 현 시점 Android 모듈엔 metric 연산 없음. 포팅 후 외부 비교는 별도 세션. |
| NG-3 | **Ablation mode 전체 4종을 모두 배포** | `full` 을 primary, 최대 1개 대체 모드만 지원. 나머지는 서버 실험용. (Open Q1 참조) |
| NG-4 | **iOS / CoreML / 기타 백엔드** | Android (ONNX Runtime QNN EP) 단일 타깃. |
| NG-5 | **API 2 (Fused multi-input) 및 Android AudioSource 설계** | BCS 캡쳐 하드웨어/드라이버 설계는 본 계획 범위 밖. 입력은 `processChunk(bcsSamples, acsSamples)` 인터페이스까지만. (Open Q3 참조) |
| NG-6 | **FP32 CPU-only 실시간 지원 보장** | LaCoSENet 실측에서 CPU FP32 는 45ms(Large) 로 budget 90%. BAFNetPlus 는 2배 backbone 이므로 CPU fallback 은 **best-effort** 로만 제공 (기능 동작 O, 실시간 보장 X). QNN HTP QDQ INT8 이 primary path. |
| NG-7 | **기존 LaCoSENet 모듈 제거** | `android/lacosenet-streaming/` 는 그대로 유지. BAFNetPlus 는 (1) 병렬 패키지로 신설 또는 (2) 일반화된 `android/streaming/` 로 재구성. Open Q4 참조. |

### 1.3 Assumptions (전제)

- A1. **BAFNetPlus checkpoint 가 50ms tier 로 준비 완료됨** (2026-04-19 Stage 1 킥오프 시 확정):
  - `results/experiments/bm_map_50ms/best.th` (5.5 MB, 2026-04-06 학습 완료)
  - `results/experiments/bm_mask_50ms/best.th` (5.5 MB, 2026-04-06 학습 완료)
  - Hydra config (`.hydra/config.yaml`) 로부터 model_lib/model_class/param 자동 로드
  - 포팅은 이 실 weight 기반으로 Stage 1 부터 시작 (random weight 단계 생략)
- A2. BAFNetPlus 의 두 Backbone 은 동일한 기하학적 하이퍼파라미터(`n_fft=400, hop_size=100, win_size=400, chunk_size=8, encoder_lookahead=3, decoder_lookahead=3, dense_channel=64, num_tsblock=4, time_block_kernel=[3,5,7,11]`)를 가진다고 가정 — 이는 `milestone.md:21` "공통: num_tsblock=4, dense_depth=4, ..." 와 동일한 Large 50ms tier 구성이다. Stage 1-α 시 `.hydra/config.yaml` 로 재확인.
- A3. 대상 기기는 LaCoSENet 과 동일한 Samsung Galaxy S25 Ultra (Snapdragon 8 Elite / Hexagon V79) — SSH 리버스 터널 + ADB 접속 절차 (`android/docs/connect_adb.md`) 재사용.
- A4. ORT 1.24.2 + QNN SDK 2.42.0 스택 유지. BAFNetPlus 전용 추가 의존성 없음 (JTransforms 3.1 은 이미 번들됨).
- A5. Pre-push hook 은 local 개발 환경에 이미 설치되어 있다 (`scripts/hooks/install-hooks.sh` 수행). 외부 CI 없음.
- A6. **Ablation mode `full` 단일 배포 확정** — ONNX / fixture / parity test 모두 1세트만 생성. Stage 1~6 전 범위에서 ablation 분기 코드 불필요.
- A7. **LaCoSENet 모듈 병행 유지 확정** — `android/lacosenet-streaming/` 그대로 보존, BAFNetPlus 는 신규 `android/bafnetplus-streaming/` 모듈로 병행 배포. Pre-push hook 은 두 모듈 parity 모두 실행.
- A8. **ONNX 배포 아키텍처는 Stage 3 에서 결정** — 단일 통합 ONNX 와 분리 2 세션 + host fusion 둘 다 Stage 3 에서 빌드·verify. 성능 비교 후 Stage 4 에서 하나 채택.

### 1.4 Stage 1 Kickoff Confirmation (2026-04-19)

계획서 §10 Open Questions 중 Stage 1 선행 조건 4개가 모두 확정되었다:

| # | 질문 | 확정 답변 | 영향 |
|---|---|---|---|
| Q1 | Ablation 모드 | **full 단일 배포** | ONNX/fixture/parity 모두 1세트. Stage 2~5 단순화 |
| Q2 | ONNX 배포 아키텍처 | **Stage 3 에서 둘 다 시도 후 결정** | 통합 세션 + 분리 2 세션 병행 빌드, verify 결과 기반 채택 |
| Q4 | LaCoSENet 공존 | **유지 — 병행 배포** | `android/bafnetplus-streaming/` 신규 모듈, parity 7/7 regression guard 유지 |
| Q6 | Weight 준비 | **둘 다 준비됨** — `results/experiments/bm_map_50ms/`, `results/experiments/bm_mask_50ms/` (50ms tier) | Stage 1 부터 실 weight 사용, Stage 5 품질 평가 옵션화 가능 |

**Stage 1 블로커 아님 (유보됨)**:

| # | 질문 | 유보 사유 | 확정 필요 시점 |
|---|---|---|---|
| Q3 | 2-마이크 입력 소스 | 포팅은 `processChunk(bcs, acs)` API 까지만, 실입력 파이프라인은 범위 밖 | Stage 4 후반 또는 별도 세션 |
| Q5 | FP16 배포 경로 유지 | 기본 QDQ + FP16 둘 다 benchmark 유지 (milestone.md 관습) | Stage 3 (export 시) |
| Q7 | SoC 배포 범위 | Galaxy S25 Ultra (Hexagon V79) 단일 — LaCoSENet 과 동일 기본값 | Stage 5 (측정 완료 시) |

→ **Stage 1 은 위 4개 확정 답변 기반으로 즉시 착수 가능**. 나머지 3개는 해당 Stage 진입 시점에 default 로 진행하거나 별도 AskUserQuestion 재수집.

---

## 2. Target Model — BAFNetPlus 구조 요약

### 2.1 클래스 구조

| 구성 요소 | 정의 위치 | 입력 | 출력 | 주요 특성 |
|---|---|---|---|---|
| `BAFNetPlus` | `src/models/bafnetplus.py:9-274` | `(bcs_com, acs_com)` tuple, 각 `[B, F, T, 2]` | `(est_mag, est_pha, est_com)` | ablation_mode 로 4종 경로 |
| `self.mapping` | `bafnetplus.py:93` — `Backbone(infer_type='mapping')` | `bcs_com [B,F,T,2]` | `(bcs_mag, _, bcs_com_out)` | mag 경로가 직접 재합성 (mask 미사용) |
| `self.masking` | `bafnetplus.py:94` — `Backbone(infer_type='masking')` | `acs_com [B,F,T,2]` | `(acs_mag, _, acs_com_out, acs_mask)` with `return_mask=True` | 곱연산 mask + fusion feature 제공 |
| `calibration_encoder` | `bafnetplus.py:117-129` | 5ch 1D 시계열 | 16ch hidden | 2×(CausalConv1d k=5 + PReLU), 프레임별 causal |
| `common_gain_head` | `bafnetplus.py:130` | 16ch hidden | 1ch | Conv1d k=1 + tanh + `max_common_log_gain=0.5` |
| `relative_gain_head` | `bafnetplus.py:132` | 16ch hidden | 1ch | Conv1d k=1 + tanh + `max_relative_log_gain=1.0`, `use_relative_gain` 일 때만 |
| `alpha_convblocks` | `bafnetplus.py:98-111` | 3ch (또는 1ch) TF 맵 `[B, C, T, F]` | 16ch | 4× (CausalConv2d k=(7,7) + BN2d + PReLU) |
| `alpha_out` | `bafnetplus.py:112` | 16ch | 2ch | Conv2d k=1×1, softmax 로 α_bcs/α_acs |

### 2.2 Forward 흐름 분해 (`bafnetplus.py:225-279`)

```
1. bcs_com, acs_com = input                               # [B, F, T, 2] × 2
2. bcs_mag, _, bcs_com_out   = self.mapping(bcs_com)       # Backbone(mapping)
3. acs_mag, _, acs_com_out, acs_mask = self.masking(acs_com, return_mask=True)

4. if use_calibration:
    feat = _build_calibration_features(bcs_mag, acs_mag, acs_mask)  # [B, 5, T]
       = cat([bcs_log_energy, acs_log_energy, log_energy_diff,
               acs_mask_mean, acs_mask_var], dim=1)
    hidden = calibration_encoder(feat)                               # [B, 16, T]
    common_log_gain = tanh(common_gain_head(hidden)) * 0.5           # [B, 1, T]
    if use_relative_gain:
        relative_log_gain = tanh(relative_gain_head(hidden)) * 1.0   # [B, 1, T]
        bcs_gain = exp(common_log_gain - 0.5*relative_log_gain)
        acs_gain = exp(common_log_gain + 0.5*relative_log_gain)
    else:
        bcs_gain = acs_gain = exp(common_log_gain)
    bcs_com_cal = bcs_com_out * bcs_gain[..., None]     # frame-wise scaling
    acs_com_cal = acs_com_out * acs_gain[..., None]
   else:
    bcs_com_cal, acs_com_cal = bcs_com_out, acs_com_out

5. if mask_only_alpha:
    alpha_feat = acs_mask.unsqueeze(1).transpose(2, 3)   # [B, 1, T, F]
   else:
    alpha_feat = _build_alpha_features(bcs_com_cal, acs_com_cal, acs_mask)
              = stack([|bcs_com_cal|, |acs_com_cal|, acs_mask], dim=1).transpose(2,3)   # [B, 3, T, F]

6. for block in alpha_convblocks:
       alpha_feat = block(alpha_feat)                  # 4× causal 2D conv
   alpha = alpha_out(alpha_feat)                        # [B, 2, T, F]
   alpha = softmax(alpha.transpose(2, 3), dim=1)        # [B, 2, F, T]
   α_bcs = alpha[:, 0, :, :, None]                      # [B, F, T, 1]
   α_acs = alpha[:, 1, :, :, None]

7. est_com = bcs_com_cal * α_bcs + acs_com_cal * α_acs  # [B, F, T, 2]
   est_mag, est_pha = complex_to_mag_pha(est_com)

Return: (est_mag, est_pha, est_com)
```

### 2.3 State inventory (streaming 변환 시 예상 state 개수)

| 소스 | 개수 | 비고 |
|---|---|---|
| `self.mapping` Backbone | **80 states** | LaCoSENet 과 동일 구조: 4 TSBlock × 2 time_block × 10 keys (`cab_*`, `gpkffn_*`) — `src/models/streaming/onnx/export_onnx.py:99-131` 의 `build_state_registry()` 결과 |
| `self.masking` Backbone | **80 states** | 동일 구조. state 이름 접두어만 다름 (예: `state_mapping_rf_*` vs `state_masking_rf_*`) |
| `calibration_encoder` (CausalConv1d ×2) | **2 states** (추정) | 각 CausalConv1d 의 streaming 변환 시 past-context 버퍼 1개씩. kernel=5 → padding=2 → state shape `[1, C_in, 4]` (왼쪽 pad 는 `padding*2=4`) |
| `alpha_convblocks` (CausalConv2d ×4) | **4 states** (추정) | kernel=(7,7) → time padding=6 per block. state shape `[1, C_in, 6, F]` |
| `alpha_out`, gain heads | 0 | 1×1 conv 는 state 불필요 |
| **합계 (ablation=full)** | **~166 states** | Backbone 160 + fusion 6. LaCoSENet 대비 2.08× |

- 실제 수치는 Stage 3 ONNX export 시 `build_state_registry()` 결과로 확정.
- `calibration_encoder` 와 `alpha_convblocks` 는 **Backbone 이 이미 streaming 변환을 거친 후** `convert_to_stateful()` 로 별도 처리해야 state 화 가능. 이때 CausalConv1d/2d 를 `StatefulConv1d/2d` 로 치환하는 기존 converter (`src/models/streaming/converters/conv_converter.py`) 를 그대로 재사용.

### 2.4 Ablation 모드별 차이

| Mode | `use_calibration` | `use_relative_gain` | `mask_only_alpha` | fusion 입력 채널 | state 추가 | 추정 ONNX 크기 (QDQ) |
|---|---|---|---|---|---|---|
| `full` (A3, 제안) | True | True | False | 3 | 2 (cal) + 4 (alpha) = 6 | ~4.0 MB × 2 + fusion ≈ **9~10 MB** |
| `no_calibration` (A1) | False | False | False | 3 | 4 (alpha only) | ~9 MB |
| `mask_only_alpha` (A2, BAFNet baseline) | True | True | True | 1 | 6 | ~9 MB, 단 alpha 그래프는 더 작음 |
| `common_gain_only` (B1) | True | False | False | 3 | 6 | ~9 MB |

→ **Open Q1** — 배포할 모드를 `full` 단일로 고정할지, `no_calibration` 을 fallback 으로 2종 지원할지 결정 필요. 2종 지원 시 별도 ONNX 파일과 config 가 생성되어야 하며, Android 측 초기화 분기가 추가된다.

### 2.5 State 텐서 형상 추정 (Large 구성, `dense_channel=64`, `freq_size=100`)

LaCoSENet 80 state 총 바이트 예측 (export_onnx.py 실제 사용값 기준):
- 평균 state shape `[1, 64, 100]` ~ `[1, 64, 14]` 범위, 평균 ~6k floats
- 80 × 6k floats × 4 bytes = **~1.9 MB per backbone**
- BAFNetPlus 통합: 2 × 1.9 MB + fusion ~0.3 MB ≈ **~4.1 MB state**
- Double buffered 에서는 ~8.2 MB — `StatefulInference.allocateStateBuffers()` 가 요구하는 Direct ByteBuffer 총량
- LaCoSENet Stage 6 측정에서 dumpsys meminfo 상 Native Heap peak 292 MB (B1 fix 후). BAFNetPlus 는 state 2배 + fusion 으로 **~350~400 MB peak** 예상.

### 2.6 State shape 세부 표 (추정)

Backbone 1개당 80 state. 주요 카테고리 별 shape 과 대표값:

| State 종류 | 예상 shape | 개수/block | dtype | 1개당 floats |
|---|---|---|---|---|
| `cab_ema` (SCA dw state) | `[1, 64, 10]` | 1 | float32 | 640 |
| `cab_cb` (causal block) | `[1, 64, 2]` | 1 | float32 | 128 |
| `cab_dw` | `[1, 64, 10]` | 1 | float32 | 640 |
| `gpkffn_conv_{k}` (k=3,5,7,11 kernel) | `[1, 64, {k-1}]` | 4 | float32 | 128~640 |
| `gpkffn_dwconv_{k}` | `[1, 64, {k-1}]` | 4 | float32 | 128~640 |

Per block (10 keys), per TSBlock (time_block_num=2), per backbone (num_tsblock=4):
- 1 backbone = 80 states × ~6 KB avg = **~480 KB**
- 2 backbones + fusion = **~1 MB** state (single-buffered). LaCoSENet Stage 6 측정의 state 비중이 이 수준과 일치.

Fusion 추가 state (정확한 shape 은 Stage 1 구현 후 확정):

| 위치 | 예상 shape | floats |
|---|---|---|
| `state_fusion_cal_conv0` (CausalConv1d in=5, out=16, k=5, padding=2) | `[1, 5, 4]` | 20 |
| `state_fusion_cal_conv1` (CausalConv1d in=16, out=16, k=5) | `[1, 16, 4]` | 64 |
| `state_fusion_alpha_conv0` (CausalConv2d in=3, out=16, k=(7,7)) | `[1, 3, 6, 201]` | 3,618 |
| `state_fusion_alpha_conv1` (CausalConv2d in=16, out=16) | `[1, 16, 6, 201]` | 19,296 |
| `state_fusion_alpha_conv2` | `[1, 16, 6, 201]` | 19,296 |
| `state_fusion_alpha_conv3` | `[1, 16, 6, 201]` | 19,296 |
| **Fusion 합계** | | **~61,600 floats ≈ 240 KB** |

→ Fusion 은 Backbone state 전체의 ~25% 수준. 무시 불가.

> **주의**: fusion alpha_convblocks 가 실제로 freq 차원 201 에서 동작하는지(encoded 100 이 아닌지) 은 `bafnetplus.py:218-223` `_build_alpha_features` 의 transpose(2, 3) 로부터 역추적 필요. `[B, 3, T, F=201]` 이면 위 추정대로 큰 state. `[B, 3, T, F=100]` 이면 1/2 로 축소.

---

## 3. Existing Assets & Reuse Map

### 3.1 Python streaming 인프라

| 파일 | LoC | 재사용 방식 | BAFNetPlus 변경 |
|---|---|---|---|
| `src/models/streaming/lacosenet.py` | 725 | **참조 템플릿** — 클래스 구조, 버퍼/STFT 로직, `from_checkpoint`, `process_samples`, `process_audio_fast` | 신규 `bafnetplus_streaming.py` (500~700 LoC 예상) 로 복사 후 dual-input 로 개조 |
| `src/models/streaming/utils.py` | — | **재사용 (단, §9.14 U1 fix 필요)** — `StateFramesContext`, `apply_streaming_tsblock`, `apply_stateful_conv`. `prepare_streaming_model` 은 Stage 1 현재 broken → Stage 2 전 복구 | Stage 1 은 `apply_streaming_tsblock` + `apply_stateful_conv` 를 mapping/masking 각각 직접 호출 (uses U1 우회) |
| `src/models/streaming/converters/conv_converter.py` | — | **그대로 재사용** — `convert_to_stateful`, `set_streaming_mode`, `reset_streaming_state`. CausalConv1d/2d 둘 다 지원 확인됨 (S1-β spike) | fusion 서브네트워크(calibration_encoder + alpha_convblocks) 에 추가로 적용 → 각 cal=2×StatefulCausalConv1d, alpha=4×StatefulCausalConv2d |
| `src/models/streaming/layers/tsblock.py` | — | **재사용 (단, §9.14 U2 fix 권고)** — `StreamingTSBlock.convert_sequence_block` | 각 Backbone 의 sequence_block 에 대해 2회 호출. `StreamingConv2d.forward` 는 현재 `state_frames` 를 thread-local 에서 fallback-read 하지 않음 → Stage 1 은 BAFNetPlusStreaming 에서 명시 전달로 우회 |
| `src/models/streaming/onnx/export_onnx.py` | 916 | **80%+ 재사용** — state registry, QDQ calibration, streaming_config.json | 신규 `export_bafnetplus_onnx.py` (400~500 LoC) 또는 본 파일에 `BAFNetPlusStatefulExportableNNCore` 클래스 추가 |
| `src/stft.py` | — | **그대로 재사용** — `mag_pha_stft`, `manual_istft_ola`, `complex_to_mag_pha` | — |

### 3.2 Android 모듈 현황 (2026-04-19 기준)

```
android/lacosenet-streaming/src/main/kotlin/com/lacosenet/streaming/
  StreamingEnhancer.kt            (395 lines)  — 단일 backbone 전용
  audio/StftProcessor.kt          (214 lines)  — FFT 기반 (A7 이후)
  audio/AudioBuffer.kt            (156 lines)
  backend/ExecutionBackend.kt     (158 lines)  — QNN/NNAPI/CPU interface
  backend/BackendSelector.kt      (183 lines)
  backend/QnnBackend.kt           (263 lines)
  backend/NnapiBackend.kt         (121 lines)
  backend/CpuBackend.kt           ( 87 lines)
  core/StreamingConfig.kt         (276 lines)  — Gson parser + validate()
  core/StreamingState.kt          ( 82 lines)
  session/StatefulInference.kt    (447 lines)  — 단일 mag/pha 입력, 80 state
  총                              2,382 LoC

android/benchmark-app/src/androidTest/kotlin/com/lacosenet/benchmark/
  StreamingBenchmarkTest.kt       (937 lines)  — 14 @Test
  parity/FixtureLoader.kt         — bin+manifest 파서
  parity/StftParityTest.kt        — 3 @Test
  parity/IstftParityTest.kt       — 2 @Test
  parity/StatefulInferenceParityTest.kt  — 2 @Test
  assets/fixtures/                — 3.8 MB, 22 chunks × 19 tensors

assets (bundled in benchmark-app):
  model.onnx (5.9 MB, FP32),  model_qdq.onnx (4.0 MB, QDQ INT8)
  streaming_config.json (4.1 KB, 80 states)
```

### 3.3 BAFNetPlus 가 소비·확장할 지점

| 컴포넌트 | BAFNetPlus 에서의 역할 | 변경 규모 |
|---|---|---|
| `StftProcessor` | 2번 인스턴스화 (BCS + ACS 각 1개) **또는** 내부 `olaBuffer` 를 채널별로 분기 | 인스턴스 복제 방식 추천(§5.1) — 0 LoC 변경, 호출측만 2배 |
| `StatefulInference` | 입력 `(mag, pha)` → `(bcs_mag, bcs_pha, acs_mag, acs_pha)`, state 80 → 160+α, 출력 `est_mask, phase_real, phase_imag` → `est_mag, est_pha` 또는 `est_com_real, est_com_imag` | 제네릭화 또는 복제 (§5.2, ~200 LoC) |
| `StreamingEnhancer` | `processChunk(bcsSamples, acsSamples)` 로 시그니처 변경 (BCS 별도 인자) | 신규 클래스 `BAFNetPlusStreamingEnhancer` 추천 (~350 LoC) |
| `BackendSelector` + QNN/NNAPI/CPU backend | 변경 없음 | 0 LoC |
| `StreamingConfig` | `streaming_config.json` 에 새 필드 `input_channels=2`, `state_info.backbone_scopes=["mapping","masking"]` 또는 유사 메타데이터 추가 | ~30 LoC |

### 3.4 성능 베이스라인 (재활용 가능한 숫자)

| 실측/예상 | LaCoSENet | BAFNetPlus 목표 |
|---|---|---|
| Single-Backbone QNN HTP QDQ INT8 Mean | **6.2ms (Small) / 10.1ms (Large)** — `milestone.md:38` | 개별 backbone 은 동일 수준 기대 (Large 10ms) |
| Dual-Backbone Concurrent QDQ INT8 Mean (cold-state) | **10.2ms (Small) / 15.2ms (Large)** — `milestone.md:44-45` + Stage 5 재측정 Dual 14.6ms | **BAFNetPlus full 파이프라인 ≤ 20ms (Large)** 를 1차 목표. fusion 추가 overhead 4~5ms 예상 |
| Dual overlap ratio | 0.68~0.76 | 유사 예상 (HTP 두 세션 병렬 동작 확인됨) |
| Same-process leak-safe | Stage 6 B1 fix 후 plateau (peak 292 MB) | 2배 state 로 peak **~400 MB** 예상 |
| 50ms budget 대비 여유 | Dual Large 15.2 / 50 = 30% | BAFNetPlus 목표 40~45% budget (2배 state + fusion) |

---

## 4. Python-Side Changes

### 4.1 BAFNetPlusStreaming wrapper (신규)

**파일**: `src/models/streaming/bafnetplus_streaming.py` (예상 500~700 LoC)

**목적**: `LaCoSENet` 클래스 (`src/models/streaming/lacosenet.py`) 구조를 본뜨되, **두 Backbone + fusion 을 단일 streaming 파이프라인으로 관리**.

**핵심 API**:
```python
class BAFNetPlusStreaming(nn.Module):
    def __init__(
        self,
        model: BAFNetPlus,                  # 이미 초기화된 BAFNetPlus (ablation_mode 확정)
        chunk_size: int = 8,
        encoder_lookahead: int = 3,
        decoder_lookahead: int = 3,
        hop_size: int = 100, n_fft: int = 400, win_size: int = 400,
        compress_factor: float = 0.3, sample_rate: int = 16000,
        streaming_tsblocks_mapping: nn.ModuleList = None,
        streaming_tsblocks_masking: nn.ModuleList = None,
        freq_size: int = 100,
        stft_center: bool = True,
    ): ...

    @classmethod
    def from_checkpoint(cls, chkpt_dir: str, ...) -> "BAFNetPlusStreaming": ...

    def process_samples(
        self, bcs_samples: Tensor, acs_samples: Tensor
    ) -> Optional[Tensor]:
        """2채널 샘플을 동시에 입력받아 enhanced 샘플 반환."""

    def process_audio(
        self, bcs_audio: Tensor, acs_audio: Tensor
    ) -> Tensor:
        """오프라인(비 스트리밍) 경로 — Stage 2 fixture 생성에 사용."""

    def reset_state(self) -> None: ...
```

**내부 구현 포인트**:

1. **버퍼 2쌍**: `bcs_input_buffer`, `acs_input_buffer`, `bcs_stft_context`, `acs_stft_context`, `bcs_ola_buffer`, `acs_ola_buffer`, `ola_norm` 공용 1개(출력은 하나의 est_com 이므로 iSTFT OLA 는 단일). **단, `ola_norm` 과 최종 OLA 는 est_com 기준으로 단일** — 중간 OLA 는 쓰지 않고 직접 fusion 후 iSTFT.
2. **`_tsblock_states` 2세트**: `self._tsblock_states_mapping`, `self._tsblock_states_masking` (각 4 block × 2 time_block × dict).
3. **`_process_encoder()` 분기**: BCS 경로 / ACS 경로 각각 dense_encoder + sequence_block 실행. mapping 은 `return_mask` 없이, masking 은 mask 도 수집.
4. **Fusion 실행 지점**: decoder lookahead 충족 후 decoder를 돌리기 전에 calibration/alpha 가 현재 chunk 만큼의 frame 을 처리. fusion conv 는 자체 state 를 가지며, 초기 `_reset_buffers()` 에서 zero-init.
5. **출력**: est_com → iSTFT (manual_istft_ola) 단일 호출 → `valid_output[:output_samples_per_chunk]`.
6. **input alignment**: bcs_samples.size == acs_samples.size 를 가정 (동일 clock, frame-sync). 불일치 시 `require` 로 fail-fast. 실제 장비에서 sub-sample drift 보정은 Open Q3.

**위험/결정 포인트**:
- **Fusion state context**: `alpha_convblocks` 는 `CausalConv2d(k=(7,7))` 로 time 축 6 frames 과거 의존. Chunk 간 state 전파가 필요. `convert_to_stateful` 로 state 화 가능하나, freq 차원이 100 (encoded) 이 아니라 **201 (원본 freq_bins)** — fusion 이 encoded freq 가 아닌 **full freq** 를 받기 때문이다 (`bafnetplus.py:218-223` `_build_alpha_features` 의 출력은 `[B, 3, T, 201]` 로 전치된 [time, freq]). state shape 이 커질 수 있음.
- **Calibration encoder**: CausalConv1d k=5 → 4 frame past context. Minor.

**예상 Python 클래스 구조 (skeleton)**:
```python
class BAFNetPlusStreaming(nn.Module):
    def __init__(self, model: BAFNetPlus, chunk_size, encoder_lookahead, decoder_lookahead,
                 hop_size, n_fft, win_size, compress_factor, sample_rate,
                 streaming_tsblocks_mapping, streaming_tsblocks_masking,
                 streaming_fusion_modules,  # dict: {calibration_encoder, alpha_convblocks (stateful)}
                 freq_size, stft_center):
        # ... 일반 init (LaCoSENet 동일 패턴)

        # Ablation 모드 저장
        self.ablation_mode = model.ablation_mode
        self.use_calibration = model.use_calibration
        self.use_relative_gain = model.use_relative_gain
        self.mask_only_alpha = model.mask_only_alpha

        # Backbones (mapping + masking)
        self.mapping = model.mapping
        self.masking = model.masking

        # Streaming TSBlocks per backbone
        self.streaming_tsblocks_mapping = streaming_tsblocks_mapping  # ModuleList
        self.streaming_tsblocks_masking = streaming_tsblocks_masking  # ModuleList

        # Fusion (stateful converted)
        if self.use_calibration:
            self.calibration_encoder = streaming_fusion_modules["calibration_encoder"]
            self.common_gain_head = model.common_gain_head
            if self.use_relative_gain:
                self.relative_gain_head = model.relative_gain_head
        self.alpha_convblocks = streaming_fusion_modules["alpha_convblocks"]
        self.alpha_out = model.alpha_out

        # Buffers
        self._reset_buffers()  # bcs/acs 버퍼 + OLA + ts states + fusion states

    def process_samples(self, bcs_samples, acs_samples):
        # 1. 양쪽 버퍼에 push
        # 2. samples_per_chunk 도달 확인
        # 3. STFT 2회
        # 4. _process_encoder(mapping_spec, bcs=True) → mapping_out
        # 5. _process_encoder(masking_spec, bcs=False) → masking_out (mask 포함)
        # 6. fusion 단계
        #    6a. calibration features + encoder (stateful)
        #    6b. gain 계산 + complex 적용
        #    6c. alpha features + alpha_convblocks (stateful)
        #    6d. softmax + blend
        # 7. iSTFT (단일 est_com)
        # 8. 반환

    def reset_state(self):
        # 버퍼 + ts states (2세트) + fusion states 모두 zero-init
```

**LaCoSENet 대비 추가 복잡도**:
- `_process_encoder` 가 2회 호출 → Backbone 2개 순차 실행 또는 asyncio 병렬 (PyTorch 는 기본 순차. Android QNN 에서만 concurrent 의미)
- `feature_buffer` 가 `bcs_mag`, `acs_mag`, `acs_mask`, `bcs_com_out`, `acs_com_out` 5종을 동시 버퍼링 → `DualFeatureBuffer` 헬퍼 또는 기존 `feature_buffer` 확장
- `_manual_istft_ola` 는 단일 est_com 이므로 변경 없음

**테스트 디자인**:
- `test_bafnetplus_streaming_offline_vs_streaming_equiv` — 전체 파이프라인 일관성
- `test_bafnetplus_streaming_ablation_full` / `_no_calibration` / `_mask_only_alpha` — 각 모드 forward smoke
- `test_bafnetplus_streaming_reset_idempotent` — reset 후 chunk 0 동일 출력
- `test_bafnetplus_streaming_lacosenet_compat` — LaCoSENet 단일 backbone 의 mask_only_alpha 케이스 vs 기존 LaCoSENet 결과 대조 (회귀 탐지)

### 4.2 Golden fixture 생성기 확장

**파일**: `scripts/make_bafnetplus_streaming_golden.py` (LaCoSENet 의 `scripts/make_streaming_golden.py` 320 LoC 복제 + 확장, 예상 400~500 LoC)

**변경점**:
1. **입력**: 고정 seed(42) 의 2채널 Gaussian 오디오 2.0초 (bcs + acs, 상관 0.7 정도로 설정 — ablation full 이 의미 있는 범위의 gain 을 생성하려면 energy 비가 정적이지 않아야 함).
2. **참조 경로**: `BAFNetPlusStreaming` (Python) + ORT CPU EP 둘 다 실행 후 tensor 차이 수집 — LaCoSENet 파이프라인 + 내부 fusion 도 포함.
3. **Dump 대상 per chunk**:
   - BCS 측: `bcs_input_samples`, `bcs_stft_context_in`, `bcs_stft_input`, `bcs_stft_mag`, `bcs_stft_pha`, `bcs_model_mag_in`, `bcs_model_pha_in`
   - ACS 측: 동일 prefix
   - Backbone 중간 출력: `mapping_bcs_mask_raw` (mapping 은 mapping 모드라 mask=est_mag), `masking_acs_mask_raw`, `bcs_phase_real`, `bcs_phase_imag`, `acs_phase_real`, `acs_phase_imag`
   - 재구성된 complex: `bcs_com_out`, `acs_com_out`
   - Calibration feature: `cal_feat_5ch`, `cal_hidden`, `common_log_gain`, `relative_log_gain` (relative 모드일 때), `bcs_gain`, `acs_gain`
   - Calibrated complex: `bcs_com_cal`, `acs_com_cal`
   - Alpha feature: `alpha_feat_3ch` (또는 1ch), `alpha_out`, `alpha_softmax_bcs`, `alpha_softmax_acs`
   - Final: `est_com`, `est_mag_final`, `est_pha_final`, `istft_output`
4. **State tensors**: opt-in `--dump_states` 으로 160+α 개 state, per chunk. 2배 용량(~48 MB/chunk). 기본 off.
5. **Fixture 크기 예산**: LaCoSENet 3.8 MB / 22 chunks 기준 → BAFNetPlus 는 텐서 개수 ~2배(state off 가정) → **예상 ~8 MB**. `.gitignore` 에서 `*.bin` 예외처리는 이미 적용됨 (`android/.gitignore`).
6. **Manifest 스키마 확장**:
```yaml
version: 2
model_class: BAFNetPlus
ablation_mode: full
derived: { samples_per_chunk: 1200, ... (LaCoSENet 동일) }
state_layout:
  mapping: [{name, shape, offset_floats, size_floats}, ... × 80]
  masking: [{name, shape, offset_floats, size_floats}, ... × 80]
  fusion:  [{name, shape, offset_floats, size_floats}, ... × 6]
chunks[i].files:
  bcs_input_samples [1200], bcs_stft_input [1400], ...
  mapping_est_mask [1,201,11], mapping_phase_real [1,201,11], ...
  masking_est_mask [1,201,11], masking_phase_real [1,201,11], masking_mask_raw [1,201,11], ...
  calibration_hidden [1,16,11], common_log_gain [1,1,11], ...
  alpha_feat [1,3,11,201], alpha_softmax [1,2,201,11], ...
  est_mag_final [1,201,11], est_pha_final [1,201,11], est_com [1,201,11,2],
  istft_output [800]
```
7. **Tolerance 설계**: LaCoSENet STFT path 1e-7 수준 / StatefulInference CPU path 1e-9 수준. Fusion 단계는 float32 Python PyTorch vs ORT CPU 간 **더 큰 오차 (1e-5 수준)** 예상 (conv+softmax+element-wise 다단 누적). 기대 tolerance: RMS < 1e-5, max < 1e-4 per chunk for fusion 출력. STFT/iSTFT 는 LaCoSENet 기준 그대로.

### 4.3 ONNX export 확장

**파일**: `src/models/streaming/onnx/export_bafnetplus_onnx.py` (예상 400~500 LoC)

**설계 결정**: §5.4 "단일 session vs 두 session" 과 연동. **계획서 시점 default 권장안은 "단일 통합 ONNX"** 이며, 근거·리스크 분석은 §5.4, Open Q2.

**통합 export 케이스 — `BAFNetPlusStatefulExportableNNCore`**:
```python
class BAFNetPlusStatefulExportableNNCore(nn.Module):
    def __init__(self, bafnetplus: BAFNetPlus,
                 streaming_tsblocks_mapping, streaming_tsblocks_masking,
                 streaming_fusion_modules,   # calibration_encoder + alpha_convblocks stateful-converted
                 freq_size, chunk_size, ablation_mode):
        ...

    def forward(
        self,
        bcs_mag, bcs_pha, acs_mag, acs_pha,   # [1, F, T] each
        *flat_states,                          # 160+α states
    ):
        # 1. mapping Backbone (non-streaming encoder/decoder + streaming TSBlocks)
        bcs_est_mask, bcs_phase_real, bcs_phase_imag, next_mapping_states = ...

        # 2. masking Backbone
        acs_est_mask, acs_phase_real, acs_phase_imag, next_masking_states = ...

        # 3. Reconstruct com_out for both branches
        #    For mapping (infer_type='mapping'): bcs_est_mag = bcs_est_mask, bcs_est_pha=atan2(phase_imag, phase_real)
        #    For masking: acs_est_mag = acs_mag * acs_est_mask (element-wise)
        #    Complex: com_out = mag_pha_to_complex(est_mag, est_pha)

        # 4. Fusion (with state)
        #    - calibration features: [5, T]
        #    - calibration_encoder: stateful 2×CausalConv1d
        #    - alpha features build + alpha_convblocks: stateful 4×CausalConv2d
        #    - alpha_out + softmax

        # 5. est_com = bcs_com_cal * α_bcs + acs_com_cal * α_acs

        # 6. Return: est_com_real [1, F, T], est_com_imag [1, F, T], *next_flat_states
        return est_com_real, est_com_imag, *next_flat_states
```

**주의**:
- 각 Backbone 내부에서 `atan2` 는 여전히 **skip** (host 측에서 수행 또는 아예 phase 를 complex 로 유지). BAFNetPlus 는 fusion 결과가 complex 이므로 복합 output 이 자연스러움 — **est_com_real, est_com_imag** 를 출력하고 Kotlin 에서 바로 iSTFT 로 넣는 것이 바람직 (atan2 불필요).
- `complex_to_mag_pha` / `mag_pha_to_complex` 를 그래프 내부에서 수행 (cos/sin 포함). HTP 지원 여부 확인 필요 (LaCoSENet 은 phase_real/phase_imag 형태로 이미 atan2 를 host 로 미룬 실적 있음 — 정상 작동).

**State 이름 규약**:
- `state_mapping_rf_{block}_tb{tb}_{section}_{key}` × 80
- `state_masking_rf_{block}_tb{tb}_{section}_{key}` × 80
- `state_fusion_cal_conv{i}` × 2 (calibration CausalConv1d)
- `state_fusion_alpha_conv{i}` × 4 (alpha CausalConv2d)
- 총 **166** (ablation=full 가정)
- 알파벳순 정렬 후 `streaming_config.json` `state_info.state_names` 에 직렬화. Kotlin `StatefulInference.initialize()` 의 `stateNames.sort()` + assertion (`StatefulInference.kt:124-142`) 이 그대로 동작.

**Dual export 케이스 (대체안)**: `mapping.onnx` + `masking.onnx` 각 80 state 로 export, fusion 은 **host Kotlin 구현** (calibration_encoder + alpha_convblocks + softmax + complex blend). 장점: 기존 `benchmarkDualBackboneConcurrentQdq` 경로 재사용. 단점: Kotlin 에서 CausalConv1d/2d 재구현(~150 LoC).

### 4.4 Quantization 전략

**LaCoSENet 실적**:
- FP16 full opts: Large 29.8ms (budget 60%)
- QDQ INT8 QUInt8: Large 10.1ms (budget 20%) — **실배포 타깃**
- QUInt16 activation 은 HTP V79 에서 PReLU 미지원 (`milestone.md Appendix A.2`) — 반드시 QUInt8

**BAFNetPlus 적용**:
1. `full` mode primary: **QUInt8 QDQ INT8** — LaCoSENet 의 QDQ path 그대로. PReLU 는 alpha_convblocks + calibration_encoder 에 총 6개 존재하나 모두 QUInt8 로 검증됨.
2. FP16 fallback: 단일 backbone FP16 29.8ms × 2 + fusion ~5ms = **~65ms > budget**. 실배포 불가. 디버깅 용도로만 제공.
3. **Calibration data**: `export_onnx.py:761-813` 의 `QnnCalibrationDataReader` 확장 필요 — 입력이 4개 (`bcs_mag, bcs_pha, acs_mag, acs_pha`) 로 늘어난다. 현재 구현은 `mag` / `pha` 하드코딩. Stage 3 에서 확장.
4. **Fusion 서브네트워크 quantization 주의**:
   - Softmax 는 HTP 에서 보통 int32 intermediate. QDQ 보존 가능성 높음.
   - Complex multiplication (est_com = complex multiply) 은 real*cos - imag*sin 과 같은 별도 op 없이 일반 arithmetic 으로 분해 — 문제 없음.
   - `tanh(common_gain_head(x))` — HTP 표준 activation, 통상 지원.
   - **미확인**: `exp(common_log_gain ± 0.5*relative_log_gain)` — exp op 의 HTP QDQ 정확도. Stage 3 ONNX verify 시 per-chunk max error > 1e-3 나오면 ablation 변경 또는 exp 를 clamp 로 대체 고려.

**미검증 리스크 P1**: Stage 3 에서 QDQ verify 수행 전까지는 "통합 그래프 전체가 HTP 에서 실행 가능"한지 확정 불가. LaCoSENet 은 PReLU 외에는 이슈 없었으나, BAFNetPlus 의 `log`, `exp`, `softmax` 체인은 첫 도전. 대응: Stage 3 에 "op coverage probe" 별도 스텝 추가 (§8 Stage 3).

---

## 5. Android-Side Changes

### 5.1 Dual-channel STFT 전략

**관찰**: 현 `StftProcessor` (214 lines, `android/lacosenet-streaming/.../audio/StftProcessor.kt`) 는 단일 채널 전용 — `olaBuffer`, `olaNorm`, `stftContext`, FFT instance `fft` 가 필드로 고정.

**세 가지 옵션**:

| 옵션 | 설명 | 장점 | 단점 | 추천 |
|---|---|---|---|---|
| (a) 인스턴스 2개 | `bcsStft`, `acsStft` 각 1개. 각자 FFT/buffer 보유 | 코드 변경 최소, 현 파일 0 LoC 수정 | FFT instance 중복 (JTransforms FloatFFT_1D 는 단일 instance 로도 thread-unsafe) | ✅ **권장** |
| (b) 내부 멀티채널 지원 | `StftProcessor.stft(audio, channelId)` 형태로 `olaBuffer` 를 `Array<FloatArray>` 로 승격 | 캡슐화 | 기존 API 파괴. LaCoSENet 병행 지원 시 호환성 문제. BAFNetPlus 만 쓸 거면 낭비 | ✗ |
| (c) FFT 공유 + 채널별 buffer 래퍼 | FFT 는 1 instance, `StftChannelState` data class 를 호출측에서 관리 | 메모리 절약 | API 침습 크고, FFT 재진입성 검증 필요 | ✗ (초기 단계에서 최적화 과조기) |

**결정**: **옵션 (a)** — `BAFNetPlusStreamingEnhancer` 내부에서 `bcsStft = StftProcessor(config.stftConfig); acsStft = StftProcessor(config.stftConfig)` 로 2개 인스턴스화. 메모리 overhead 는 FFT instance 1개 × ~4 KB + OLA buffer × 4 (bcsOla/bcsNorm/acsOla 대신 — 실제로는 iSTFT 는 단일 est_com 이라 output OLA 는 1개) — 무시할 수준.

**iSTFT 는 단일**: est_com 최종 한 개만 iSTFT 대상. 기존 `istftStreaming` 1회 호출로 충분.

### 5.2 `StatefulInference.run()` 시그니처 확장

**현행** (`StatefulInference.kt:250`):
```kotlin
fun run(mag: FloatArray, pha: FloatArray): InferenceResult
```

**BAFNetPlus 확장 옵션 A — 제네릭화**:
```kotlin
fun run(
    namedInputs: Map<String, FloatArray>,   // e.g. {"bcs_mag", "bcs_pha", "acs_mag", "acs_pha"}
): InferenceResult
```
- 장점: 단일 구현, 향후 3-mic 등 확장 용이
- 단점: hot path 에 Map 순회 추가. 현 구현은 pre-allocated `magBuffer`/`phaBuffer` 를 필드로 들고 있어 "name → buffer" lookup 이 매 chunk 추가됨. **단, ByteBuffer 자체가 pre-alloc 되고 Map 이 작으면 (2~4개 key) nanosecond-level overhead — 무시 가능**.

**옵션 B — 복제**:
```kotlin
class BAFNetPlusStatefulInference(...) {
    fun run(bcsMag, bcsPha, acsMag, acsPha): InferenceResult
}
```
- 장점: hot path 직접 변수, gc allocation 제로
- 단점: `StatefulInference.kt` 447 LoC 중 300+ LoC 가 동일 로직으로 복제. 수정 시 drift 위험

**권장**: **옵션 A (제네릭화)** — pre-allocated Direct ByteBuffer 필드를 `Map<String, ByteBuffer>` 로 승격하고, `run(named)` 에서 이름으로 dispatch. 기존 LaCoSENet 경로도 `run(mapOf("mag" to ..., "pha" to ...))` 로 옮기거나, 기존 `run(mag, pha)` overload 를 남겨 호환성 유지.

**State 확장**:
- `stateNames: List<String>` 는 현재도 `initialize()` 에서 session inputInfo 로부터 **동적 수집**. 80 → 166 자동 적응.
- `stateShapes`, `stateSizes`, `stateBuffersA/B`, `stateTensorsA/B` 모두 `mutableMap<String, ...>` — 자동 스케일.
- **Double buffered 예상 메모리**: LaCoSENet 292 MB peak (Stage 6 재측정) 의 state 비중이 ~30~40% 였다면 BAFNetPlus 는 이 비중이 2배 → **peak ~350~400 MB** 예상. Native heap watermark(LMKD min2x) 는 안전 범위 (LaCoSENet 292 MB 로 LMKD trigger 없었음).

**출력 확장**:
- 현 `run()` 은 `est_mask, phase_real, phase_imag` 3개 출력 → `estMask, estPhase` 로 정리 반환 (`InferenceResult`).
- BAFNetPlus 는 `est_com_real, est_com_imag` 2개 출력 (통합 ONNX 케이스). `InferenceResult` 에 `estComReal: FloatArray`, `estComImag: FloatArray` 필드 추가. 또는 `BAFNetPlusInferenceResult` 별도 data class.
- **iSTFT 는 complex → real** 이므로 Kotlin 측에서 `est_com_real, est_com_imag → mag, pha` 변환이 필요할 수 있다. 또는 iSTFT 를 complex 직접 처리하도록 확장(더 적절). 현 `istftStreaming(mag, pha, numFrames)` 는 mag/pha 입력 API — 내부에서 `mag*cos(pha), mag*sin(pha)` 로 복원하므로 호출측에서 `atan2(imag, real)` + `sqrt(real²+imag²)` 후 mag/pha 로 전달하는 것이 최소 침습. Stage 4 세부 결정.

### 5.3 `StreamingEnhancer` 확장 전략

**세 가지 옵션**:

| 옵션 | 설명 | 코드 영향 | 호환성 |
|---|---|---|---|
| (a) **신규 `BAFNetPlusStreamingEnhancer`** | 기존 `StreamingEnhancer` 그대로 유지, 별도 클래스 신설 | ~350 LoC 신규, 기존 0 LoC 변경 | 둘 다 배포 가능. 앱에서 모델 종류에 따라 인스턴스화 구분 |
| (b) **제네릭 `StreamingEnhancer<TConfig>`** | 기존 클래스에 타입 파라미터 도입, LaCoSENet/BAFNetPlus 는 둘 다 하위 케이스 | ~400 LoC 수정 (기존 395 에서 100+ 변경) | **LaCoSENet 회귀 위험** — parity 7/7 재검증 필요 |
| (c) **`StreamingEnhancer` 를 BAFNetPlus 로 치환** | LaCoSENet 경로 deprecate, BAFNetPlus 만 유지하되 `mask_only_alpha` + `no_calibration` + mapping-only 모드로 LaCoSENet 시뮬레이션 | 대대적 재구성 | LaCoSENet 벤치마크 baseline 유실 위험 |

**권장**: **옵션 (a)** — 신규 클래스 `BAFNetPlusStreamingEnhancer`. 근거:
- LaCoSENet parity 7/7 + pre-push hook 의 의미있는 regression guard 유지
- 두 모듈 모두 batchApp 에서 벤치마크 가능 (같은 기기에서 LaCoSENet vs BAFNetPlus 비교)
- 신규 클래스는 기존의 공통 컴포넌트(`AudioBuffer`, `FeatureBuffer`, `BackendSelector`, `ExecutionBackend`) 를 **재사용만** 하고, state/입력 수만 다르다
- 추후 두 모델이 같이 쓰이는 비율이 나오면 옵션 (b) 로 점진 통합 가능

**파일 구조 제안**:
```
android/bafnetplus-streaming/                    # 신규 Gradle 모듈
  src/main/kotlin/com/bafnetplus/streaming/
    BAFNetPlusStreamingEnhancer.kt              (~350 LoC, 신규)
    session/BAFNetPlusStatefulInference.kt      (~400 LoC, 옵션 A 제네릭화 시 불필요)
    core/BAFNetPlusStreamingConfig.kt           (~50 LoC, StreamingConfig 확장 또는 별도)
    audio/DualChannelFeatureBuffer.kt           (~100 LoC, BCS+ACS 동시 push)
  build.gradle.kts
  consumer-rules.pro
```

**대안** — `android/lacosenet-streaming/` 를 `android/streaming/` 로 rename 후 공통 인프라를 `core` 서브패키지로 재구성:
- 장점: 중복 감소
- 단점: LaCoSENet 패키지명 `com.lacosenet.streaming.*` 전면 rename → 기존 assets, parity test 경로 모두 수정 → 대규모 diff
- 결론: **Phase 2 작업으로 유보**. Stage 1~6 에서는 패키지 분리 유지.

### 5.3.1 `BAFNetPlusStreamingEnhancer` 주요 시그니처 (의사 코드)

```kotlin
class BAFNetPlusStreamingEnhancer(private val context: Context) {

    // Components (기존 LaCoSENet 과 동일 패턴)
    private var env: OrtEnvironment? = null
    private var backend: ExecutionBackend? = null
    private var inference: StatefulInference? = null
    private var bcsStft: StftProcessor? = null
    private var acsStft: StftProcessor? = null
    private var config: BAFNetPlusStreamingConfig? = null

    // Dual buffers
    private var bcsInputBuffer: AudioBuffer? = null
    private var acsInputBuffer: AudioBuffer? = null
    private var dualFeatureBuffer: DualChannelFeatureBuffer? = null

    fun initialize(
        modelPath: String = "bafnetplus_qdq.onnx",
        configPath: String = "bafnetplus_streaming_config.json",
        forceBackend: BackendType? = null
    ): InitResult { /* ... */ }

    fun processChunk(bcsSamples: FloatArray, acsSamples: FloatArray): FloatArray? {
        // Guards (H1 대응)
        require(bcsSamples.size == acsSamples.size) {
            "BCS/ACS length mismatch: bcs=${bcsSamples.size}, acs=${acsSamples.size}"
        }
        require(bcsSamples.all { it.isFinite() } && acsSamples.all { it.isFinite() }) {
            "NaN/Inf in input"
        }

        // Dual push
        bcsInputBuffer!!.push(bcsSamples)
        acsInputBuffer!!.push(acsSamples)

        if (!bcsInputBuffer!!.hasEnough(config!!.samplesPerChunk)) return null

        val bcsChunk = bcsInputBuffer!!.pop(config!!.samplesPerChunk)
        val acsChunk = acsInputBuffer!!.pop(config!!.samplesPerChunk)

        val (bcsMag, bcsPha) = bcsStft!!.stft(bcsChunk, advanceSamples = config!!.outputSamplesPerChunk)
        val (acsMag, acsPha) = acsStft!!.stft(acsChunk, advanceSamples = config!!.outputSamplesPerChunk)

        // Dual feature buffer accumulate
        dualFeatureBuffer!!.push(bcsMag, bcsPha, acsMag, acsPha)
        if (!dualFeatureBuffer!!.hasEnough(config!!.streamingConfig.exportTimeFrames)) return null

        val (bcsMagIn, bcsPhaIn, acsMagIn, acsPhaIn) =
            dualFeatureBuffer!!.get(config!!.streamingConfig.exportTimeFrames)

        // Inference (B4: shape guards 이미 StatefulInference 에 존재)
        val result = inference!!.run(mapOf(
            "bcs_mag" to bcsMagIn, "bcs_pha" to bcsPhaIn,
            "acs_mag" to acsMagIn, "acs_pha" to acsPhaIn,
        ))

        // 출력 — 통합 세션: est_com_real, est_com_imag
        val estComReal = result.estComReal!!  // [1, F, T]
        val estComImag = result.estComImag!!

        // Convert complex → mag/pha for istftStreaming
        val (estMag, estPha) = complexToMagPha(estComReal, estComImag)

        // Crop 11 → 8 frames (LaCoSENet 동일)
        val chunkFrames = config!!.streamingConfig.chunkSizeFrames
        val (cropMag, cropPha) = cropTimeFrames(estMag, estPha, chunkFrames)

        // iSTFT (단일)
        val enhanced = bcsStft!!.istftStreaming(cropMag, cropPha, chunkFrames)
        // NB: iSTFT 는 출력 1개이므로 bcsStft 의 olaBuffer 를 사용 (acsStft 의 olaBuffer 는 iSTFT 에 관여하지 않음).
        // 또는 별도 `outputStft` 인스턴스를 두어 명시적 분리 — Stage 4 에서 결정.

        dualFeatureBuffer!!.removeFirst(chunkFrames)
        return enhanced
    }

    fun reset() { /* bcs/acs buffer + stft + inference state */ }
    fun release() { /* ... */ }
}
```

**주의**:
- `bcsStft.istftStreaming` 을 사용하면 **BCS 의 OLA buffer 가 출력 재구성에 사용** — 의미적으로 어색함 (출력은 est_com 이지 BCS 가 아님). Stage 4 에서 **3번째 `outputStft: StftProcessor` 를 두어 iSTFT 전용으로 명시** 하는 것이 깔끔.
- `DualChannelFeatureBuffer` 는 4개 동시 push — 내부적으로 프레임 카운트 1개로 공유.

### 5.3.2 `BAFNetPlusStreamingConfig` 확장

기존 `StreamingConfig.kt` 는 단일 backbone 가정. BAFNetPlus 는 추가 메타데이터 필요:

```kotlin
data class BAFNetPlusStreamingConfig(
    // 기존 StreamingConfig 필드 전부 포함 (상속 또는 포함)
    val base: StreamingConfig,

    @SerializedName("bafnetplus_info")
    val bafnetplusInfo: BAFNetPlusInfo
)

data class BAFNetPlusInfo(
    @SerializedName("ablation_mode")
    val ablationMode: String,            // "full" | "no_calibration" | "mask_only_alpha" | "common_gain_only"

    @SerializedName("input_channels")
    val inputChannels: Int = 2,          // bcs + acs (고정 2, 미래 확장 대비)

    @SerializedName("fusion_state_count")
    val fusionStateCount: Int,           // 6 (ablation=full), 4 (no_calibration), ...

    @SerializedName("backbone_scopes")
    val backboneScopes: List<String> = listOf("mapping", "masking"),

    @SerializedName("est_com_output_mode")
    val estComOutputMode: String = "real_imag"   // "real_imag" | "mag_pha"
)
```

**Validate 확장**:
```kotlin
fun BAFNetPlusStreamingConfig.validate(source: String) {
    base.validate(source)
    val info = bafnetplusInfo
    require(info.ablationMode in listOf("full", "no_calibration", "mask_only_alpha", "common_gain_only"))
    require(info.inputChannels == 2)
    require(info.fusionStateCount >= 0)
    require(info.backboneScopes.size == 2 && info.backboneScopes == listOf("mapping", "masking"))
    require(info.estComOutputMode in listOf("real_imag", "mag_pha"))
}
```

### 5.4 QNN 백엔드 세션 1개 vs 2개 — 핵심 트레이드오프

| 축 | 단일 통합 세션 | 분리 2 세션 + host fusion |
|---|---|---|
| **Latency (예상)** | mapping + masking + fusion 이 **HTP 내부에서 파이프라인** → Dual Concurrent 15.2ms (Large) 에 fusion ~2ms 내장 ≈ **17~18ms** | mapping + masking 을 2 세션 concurrent (15.2ms) + host Kotlin fusion (~3~5ms) ≈ **18~20ms** |
| **HTP Graph finalization** | 단일 그래프 ~2× 복잡 → **~5~8s** (LaCoSENet QDQ 는 2.4s, 2배 모델은 4.8s + fusion overhead) | 각 Backbone 2.4s × 2 = 4.8s (병렬 init 시 2.4s 그대로) |
| **Context cache** | 단일 cache (1 파일), 코드 경로 단순 | 두 cache 파일, 로드 순서 신경 |
| **VTCM 용량** | 단일 그래프가 VTCM 8~16 MB 전체 점유 → **fusion ops 도 VTCM 요구** → **초과 리스크 P1** | 각 세션이 독립 VTCM 할당 → milestone.md 2.4 에서 "VTCM 경합 없음" 실증 |
| **Dual concurrent 재사용성** | 불가 — 단일 세션이므로 concurrent 개념 없음 | 100% 재사용 — 기존 `benchmarkDualBackboneConcurrentQdq` 가 이미 검증 |
| **Host code 복잡도** | 낮음 — Kotlin 은 input dispatch + output iSTFT 만 | 높음 — Kotlin 이 calibration_encoder (2 CausalConv1d) + alpha_convblocks (4 CausalConv2d) + softmax 구현 (~200 LoC) |
| **Fusion 정확도** | HTP QDQ 으로 일관 | Host FP32 — 정확도 우위 |
| **Ablation 전환 비용** | 각 모드마다 전용 ONNX 재export | ONNX 는 공통, fusion 코드만 flag 로 전환 |
| **Re-export churn** | ablation/weight 변경마다 단일 대형 모델 re-quantize | Backbone 만 재export, fusion 은 host 수정으로 끝 |

**권장 (초기)**: **단일 통합 세션** — 다음 근거로:
1. milestone.md Dual Concurrent 15.2ms 는 **두 세션이 HTP 를 병렬 점유한 경우의 최선**. 단일 통합 세션은 fusion ops 를 HTP 내부에서 추가 실행하므로 "fusion 공짜" 구간이 있을 수 있음(파이프라인 인터리빙).
2. Host fusion 구현은 Kotlin 에서 causal conv 재구현 = 추가 parity risk + 유지비용.
3. VTCM 초과 시에만 분리로 회귀 — Stage 3 ONNX verify 시점에 HTP 그래프 finalization 시도 후 결정.

**폴백 계획**: Stage 3 가 "단일 통합 그래프가 HTP 에 올라가지 않는다" 또는 "VTCM 초과 로그" 를 내면 **Stage 3 말미에 분리 2 세션 + host fusion 으로 전환**. 이 결정 포인트는 Stage 3 acceptance 기준에 명시.

---

## 6. Parity 전략

### 6.1 기존 LaCoSENet 7/7 과 공존 원칙

| Test class | 상태 | BAFNetPlus 영향 |
|---|---|---|
| `StftParityTest` (3 @Test) | **그대로 유지** | STFT 는 입력이 2채널이어도 채널별 독립. 기존 fixture 그대로 재사용 가능 |
| `IstftParityTest` (2 @Test) | **그대로 유지** | iSTFT 는 단일 출력 |
| `StatefulInferenceParityTest` (2 @Test) | **그대로 유지** (LaCoSENet 80-state 경로 검증) | BAFNetPlus 전용 별도 테스트로 분리 |
| `BAFNetPlusStatefulInferenceParityTest` | **신규 2 @Test** | 160+α state 경로 |
| `BAFNetPlusFusionParityTest` (옵션) | **신규 1~2 @Test** | Host fusion 시에만 필요 — 단일 통합 세션이면 굳이 별도 테스트 없이도 StatefulInferenceParityTest 가 fusion 까지 커버 |
| `BAFNetPlusStreamingEnhancerParityTest` (옵션, E2E) | **신규 1 @Test** | LaCoSENet 에서는 DESCOPED — BAFNetPlus 에서는 fusion + dual input orchestration 의 통합 검증이 의미있을 수 있음. 단, LaCoSENet 결론대로 "parity 개별 경로 + pre-push hook 이 regression guard" 라면 동일하게 skip 고려 |

**총 BAFNetPlus parity 목표**: **2~5 @Test 추가**. LaCoSENet 7/7 과 합쳐 9~12/12 PASS.

### 6.2 신규 parity @Test 설계

**`BAFNetPlusStatefulInferenceParityTest`** (2 @Test 예정):

1. `sequentialBAFNetPlusStreamingParity_firstChunks`
   - CPU 백엔드 사용
   - 5 chunks 순차 실행, 각 chunk 의 `est_com_real`, `est_com_imag`, 그리고 intermediate `mapping_bcs_mask_raw`, `masking_acs_mask_raw`, `calibration_hidden`, `alpha_softmax` 을 fixture 와 비교
   - Tolerance: RMS < 1e-5, max < 1e-4 (fusion path 누적 오차 고려, LaCoSENet 1e-9 보다 완화)
2. `resetStatesRestoresBAFNetPlusChunk0Output`
   - `resetStates()` 후 chunk 0 재실행 → bit-identical

**`BAFNetPlusFusionParityTest`** (단일 통합 세션 채택 시 **생략**, host fusion 시에만 추가):
1. `calibrationEncoder_frame0toN` — calibration 5ch feature 주입, hidden → gain 출력을 fixture 와 비교
2. `alphaConvBlocks_alpha_softmax` — alpha feature 주입, alpha softmax 를 fixture 와 비교

### 6.3 Fixture 크기·생성 절차·sync 규약

- **크기 예산**: LaCoSENet 3.8 MB + BAFNetPlus **~8 MB** (state off, chunk 수 동일 22). Git 트래킹 허용 범위 내.
- **생성 주기**: 모델 재export 시마다. `scripts/make_bafnetplus_streaming_golden.py` 는 `export_bafnetplus_onnx.py` 의 git_commit 과 streaming_config.json 의 메타데이터를 **함께 기록**하여 fixture 가 어느 ONNX 기반인지 추적.
- **Sync 규약**: `bafnetplus_streaming_config.json:export_info.git_commit` 과 `fixtures/bafnetplus/manifest.json:export_info.git_commit` 이 일치해야 parity 유효. 불일치 시 `StatefulInferenceParityTest` 초기화 단계에서 경고(E) 또는 skip(W).
- **재생성 runbook**: LaCoSENet 과 동일 패턴 (§ 부록 B).

---

## 7. Benchmark Goals

### 7.1 Latency budget 재확인

- Chunk: 8 frames × 6.25ms = **50ms budget**
- LaCoSENet Dual Concurrent Large QDQ INT8 cold-state (Stage 5 재측정): Mean **14.6ms** (budget 29%)
- BAFNetPlus 목표: fusion 2~5ms 추가 → **Mean ≤ 20ms** (budget ≤ 40%) 1차 목표
- 스트레치 목표: **Mean ≤ 17ms** (budget ≤ 34%) — 단일 통합 세션의 HTP 파이프라이닝 이득을 전부 누린 경우

### 7.2 `milestone.md` 확장 계획

기존 `milestone.md` 에 추가할 row:

```markdown
## 4. BAFNetPlus (dual-mic) 파이프라인 실측

### 4.1 통합 세션 (권장 경로)

| Config | Quantization | WallClock Mean | P95 | P99 | Budget % |
|---|---|---|---|---|---|
| BAFNetPlus full (unified) | QDQ INT8 | TBD (~17-20ms) | TBD | TBD | TBD |
| BAFNetPlus no_calibration (unified) | QDQ INT8 | TBD (~15-18ms) | TBD | TBD | TBD |

### 4.2 분리 세션 + host fusion (대체 경로)

| Config | Quantization | Backbone×2 | Host fusion | Total |
|---|---|---|---|---|
| BAFNetPlus split | QDQ INT8 | 14.6ms (Dual Concurrent) | TBD (~3-5ms Kotlin) | TBD |

### 4.3 State/memory overhead

- Native Heap peak: TBD (~350-400 MB 예상, LaCoSENet 292 MB 의 ~1.4×)
- State count: 166 (vs LaCoSENet 80)
- Double-buffered state memory: TBD (~8 MB)
- Model ONNX size: TBD (~9-10 MB QDQ INT8, vs LaCoSENet 4.0 MB)
```

### 7.3 Thermal · 배터리 고려

- **Thermal throttling**: Stage 5 LaCoSENet 재측정에서 FP16 P99 가 cold 38ms → same-process 46ms 로 +21% 증가 확인. BAFNetPlus 는 연산량 2× → 열 부담 예상. 1분 이상 연속 실행 테스트 필요 (§ Stage 5).
- **배터리**: 본 계획 범위 밖이나, 포팅 완료 후 `dumpsys batterystats` 기반 proxy 측정을 Stage 5 end-of-session 옵션으로 포함.
- **Duty cycle**: BAFNetPlus 는 양 귀(stereo) 혹은 양 마이크 동시 ON 으로 항시 작동 가정. LaCoSENet 의 "keyword detect 후 enhance" 패턴과 달리 continuous → HTP 열 상승 가능성 ↑.

---

## 8. Stage 분할 (Stage 1~6)

LaCoSENet 리뷰는 Stage 1(Audit) 을 사후로 두고 Stage 2~6 에 수정을 올렸다. **BAFNetPlus 는 처음부터 "구현" 프로젝트** 이므로 Stage 이름을 재정의한다. 각 Stage 는 LaCoSENet 경험을 반영해 **acceptance 기준·예상 LoC·선행 조건·의존성** 을 명확히 두었다.

---

### Stage 1 — Design freeze + Python streaming wrapper

**기간 가정**: 1 세션 (수일)
**선행 조건 (2026-04-19 확정)**:
- ✅ Open Q1 ablation mode → **`full` 단일 배포**
- ✅ Open Q2 ONNX 아키텍처 → Stage 3 에서 둘 다 시도 후 결정 (초안 OK)
- ✅ Open Q4 LaCoSENet 공존 → **유지 — 병행 배포**
- ✅ BAFNetPlus weight checkpoint → `results/experiments/bm_map_50ms/best.th` + `results/experiments/bm_mask_50ms/best.th` (50ms tier)

**`conf/model/bafnetplus_streaming.yaml` 파라미터 (S1-4 산출물)**:
```yaml
model_lib: bafnetplus
model_class: BAFNetPlus
input_type: acs+bcs
param:
  ablation_mode: full
  conv_depth: 4
  conv_channels: 16
  conv_kernel_size: 7
  calibration_hidden_channels: 16
  calibration_depth: 2
  calibration_kernel_size: 5
  calibration_max_common_log_gain: 0.5
  calibration_max_relative_log_gain: 1.0
  checkpoint_mapping: results/experiments/bm_map_50ms/best.th
  checkpoint_masking: results/experiments/bm_mask_50ms/best.th
# Streaming export 고정
streaming:
  chunk_size: 8
  encoder_lookahead: 3
  decoder_lookahead: 3
```

**산출물**:

| # | 산출물 | 경로 | 예상 LoC |
|---|---|---|---|
| S1-1 | `BAFNetPlusStreaming` wrapper | `src/models/streaming/bafnetplus_streaming.py` (신규) | 500~700 |
| S1-2 | 단위 테스트 — LaCoSENet 과 parity 유지 (같은 입력 주면 기존 `LaCoSENet` 결과와 mapping 경로만 비교) | `tests/test_bafnetplus_streaming.py` (신규) | 150~250 |
| S1-3 | Python 측 오프라인 비교 스크립트 — `BAFNetPlus.forward(batch)` vs `BAFNetPlusStreaming.process_audio_fast(batch)` 오차 측정 | `scripts/compare_bafnetplus_batch_vs_stream.py` (신규) | 80~120 |
| S1-4 | `conf/model/bafnetplus_streaming.yaml` — streaming export 용 파라미터 고정 (chunk_size, encoder_lookahead, decoder_lookahead, ablation_mode) | `conf/model/` (신규) | ~30 |

**예상 LoC 합**: **760~1,100 LoC**

**Stage 1 세부 하위 단계** (순서대로):

1. **S1-α (spike)**: `prepare_streaming_model()` 을 BAFNetPlus 에 적용 시 internal 양 Backbone 의 streaming TSBlock 을 각각 뽑을 수 있는지 확인. `metadata["streaming_tsblocks"]` 가 단일 ModuleList 이라면 wrapper 쪽에서 `metadata_mapping`, `metadata_masking` 로 분리 호출해야 함.
2. **S1-β (fusion converter sanity)**: `convert_to_stateful(bafnetplus.calibration_encoder, inplace=False)` + `convert_to_stateful(bafnetplus.alpha_convblocks, inplace=False)` 가 `StatefulConv1d`, `StatefulConv2d` 로 치환 성공하는지. 실패 시 converter 확장 필요 — P1 리스크.
3. **S1-γ (wrapper 뼈대)**: `BAFNetPlusStreaming.__init__` + `_reset_buffers` + `reset_state`. forward 는 없는 상태로 import 테스트만.
4. **S1-δ (forward)**: `process_samples(bcs, acs)` 구현. 초기엔 `infer_type='mapping'` 케이스를 단순화하기 위해 `no_calibration` ablation 부터 시작.
5. **S1-ε (from_checkpoint)**: BAFNetPlus 전체 checkpoint → streaming wrapper 로드 경로. 없으면 `build_from_config(bafnetplus_yaml)` 대체 경로.
6. **S1-ζ (offline equiv)**: `process_audio_fast(bcs, acs)` 가 `BAFNetPlus.forward((bcs_com, acs_com))` 와 RMS < 1e-4 동치인지 테스트. 불일치 시 fusion state 초기화/관리 디버깅.

**핵심 acceptance**:
- [ ] `tests/test_bafnetplus_streaming.py` 전체 PASS
- [ ] Offline (`BAFNetPlus.forward`) vs Streaming (`BAFNetPlusStreaming.process_audio_fast`) 오차 RMS < 1e-4, max < 1e-3 (float32 누적)
- [ ] LaCoSENet 1-channel mode (ACS only, mapping 경로 생략) 로 `BAFNetPlusStreaming` 을 돌렸을 때 기존 `LaCoSENet.process_audio` 와 RMS < 1e-5 일치 — **기존 회귀 탐지**
- [ ] 4개 ablation mode 각각 smoke test PASS (forward 에러 없음)
- [ ] `reset_state()` 후 chunk 0 재실행 bit-identical

**의존성**: `src/models/streaming/utils.py` (StateFramesContext), `src/models/streaming/layers/tsblock.py` (StreamingTSBlock), `src/models/streaming/converters/conv_converter.py` (convert_to_stateful). 모두 현 상태 그대로 사용.

**risks**:
- R1-1 `BAFNetPlus.__init__` 이 `checkpoint_mapping` / `checkpoint_masking` 에서 config 자동 로드 → streaming 변환 시 두 Backbone 의 `num_tsblock`, `time_block_kernel` 이 동일한지 확인 필요. **불일치면 state registry 이름 충돌** 발생 — require + 명시적 에러.
- R1-2 Fusion 서브네트워크 stateful 변환 시 `convert_to_stateful` 이 `CausalConv1d` 와 `CausalConv2d` 를 모두 인식하는지 확인. 현 구현은 주로 Backbone 내부의 `StatefulConv*` 로 치환 — BAFNetPlus 외부 모듈에도 적용 가능한지 Stage 1 초반에 sanity check.
- R1-3 `StateFramesContext(valid_frames)` 가 fusion state 까지 커버하는지. 현 구현은 module forward 훅 기반 — wrapper 가 fusion conv 를 직접 호출하므로 state 업데이트 guard 경로가 다를 수 있음. 해결: fusion 호출을 `StateFramesContext(chunk_size)` block 으로 둘러쌈.
- R1-4 `disable_state_guard` 옵션 계승 — LaCoSENet 의 C3 ablation 지원 여부. BAFNetPlus 는 신규 학습이므로 이 flag 가 필요 없을 가능성 높음. **Stage 1 default=False 고정**.

**Exit criteria**: 위 6 acceptance 모두 통과. 실패 시 Stage 2 진입 금지.

---

### Stage 1 실행 결과 (2026-04-19 완료)

**상태**: ✅ 완료 — 5/5 acceptance 테스트 PASS. Stage 2 착수 가능.

**산출물 실측 LoC** (`wc -l`):

| # | 파일 | 실측 | 예상 |
|---|---|---|---|
| S1-1 | `src/models/streaming/bafnetplus_streaming.py` | **857** | 500~700 |
| S1-2 | `tests/test_bafnetplus_streaming.py` | **388** | 150~250 |
| S1-2b | `tests/conftest.py` (project root 에 sys.path 추가) | **8** | — (추가 산출물) |
| S1-3 | `scripts/compare_bafnetplus_batch_vs_stream.py` | **179** | 80~120 |
| S1-4 | `conf/model/bafnetplus_streaming.yaml` | **14** | ~30 |
| **합계** | | **1,446** | 760~1,100 |

예상 대비 약 1.3~1.9× 증분 — black 120-col 포맷팅의 multi-line 시그니처 전개와 LaCoSENet 비교용 helper (`_capture_streaming_spec`, `_run_masking_branch_via_streaming_primitives`) 이 주 원인.

**acceptance 테스트 결과** (114s, CPU-only, `pytest tests/test_bafnetplus_streaming.py -v`):

```
test_structural_compatibility_asserted ......... PASSED  # S1-α: from_checkpoint 구조 assert
test_fusion_modules_stateful ................... PASSED  # S1-β: cal=2×StatefulCausalConv1d, alpha=4×StatefulCausalConv2d
test_reset_state_idempotent .................... PASSED  # Criterion 4: bit-identical re-run
test_offline_vs_streaming_parity ............... PASSED  # Criterion 2: mag RMS=2.7e-07, max=1.5e-05
test_lacosenet_masking_branch_regression ....... PASSED  # Criterion 3 (relaxed — 아래 U2 참조)
```

CLI diagnostic (`python scripts/compare_bafnetplus_batch_vs_stream.py --audio_length 16000`):
- `mag RMS=8.8e-07`, `mag max=6.1e-05` — 모두 1e-4 / 1e-3 tolerance 이하.

**상세 parity 수치 (spec 레벨, center=False STFT + matched context)**:
- Mapping+Masking+Calibration+AlphaFusion 전체 pipeline
- mag RMS **2.7e-07**, max **1.5e-05** (float32 machine epsilon 수준)
- pha RMS **5.0e-06**, max **2.4e-04** (tanh/softmax 비선형에서 약간 더 큼)
- 1s audio, 20 chunks

**Stage 1 에서 발견된 upstream issue 2 건 — Stage 2 착수 전 해소 필수**

| ID | 위치 | 설명 | 영향 | Stage 1 대응 | Stage 2 이전 해소안 |
|---|---|---|---|---|---|
| **U1** | `src/models/streaming/utils.py:272` | `load_model(model_args, device)` 호출 시그니처가 `src/utils.py:178` 의 `load_model(model_lib, model_class_name, model_params, device)` 와 불일치 — 커밋 `58738cc` refactor 에서 도입됨 | `prepare_streaming_model()` 및 `LaCoSENet.from_checkpoint()` 둘 다 현재 broken. Stage 4 에서 Android parity 비교 시 Python 쪽 재현 경로 필수이므로 반드시 수정 필요 | `BAFNetPlusStreaming.from_checkpoint` 에서 `prepare_streaming_model` 을 우회하고 `apply_streaming_tsblock` + `apply_stateful_conv` 를 직접 호출 | 1-line fix: `load_model(model_args.model_lib, model_args.model_class, dict(model_args.param), device)` — LaCoSENet 기능 복구 |
| **U2** | `src/models/streaming/layers/tsblock.py:454` | `StreamingConv2d.forward` 가 `state_frames` 를 thread-local `StateFramesContext` 에서 fallback-read 하지 않고 arg-only 로만 사용 — `StatefulCausalConv1d/2d` 및 `StatefulAsymmetricConv2d` 와 non-uniform | LaCoSENet 에서 lookahead 프레임이 state 업데이트에 포함됨 → 매 chunk 마다 ~2 frame 드리프트 누적, offline parity 시 RMS ~0.08. BAFNetPlus wrapper 의 올바른 동작과 불일치 | `BAFNetPlusStreaming._process_encoder` 에서 `block(ts_out, states, state_frames=vf)` 명시 전달로 Criterion 2 bit-close 달성. Criterion 3 는 "same-order-of-magnitude" 로 완화 (test docstring 에 원인 문서화) | `StreamingConv2d.forward` 에 `if state_frames is None: state_frames = get_state_frames_context()` 4-line 추가. LaCoSENet 기존 parity 7/7 회귀 측정 → 영향 없거나 개선만 있으면 채택 |

**U1/U2 해소가 Stage 2 전 필수인 이유**:
- Stage 2 는 golden fixture 를 **BAFNetPlusStreaming 기준**으로 생성. U2 fix 없이 fixture 를 만들면 Stage 4 Android 쪽과의 비교 기준이 LaCoSENet vs BAFNetPlus 사이에서 어긋남 — fixture 를 두 번 다시 만들어야 할 리스크
- U1 없이는 Stage 4 Android parity @Test 에서 Python reference 구동 경로가 없음 (LaCoSENet 의 기존 benchmark-app parity 7/7 도 현재 from_checkpoint 로 재현 불가 상태)

**Stage 1 범위 밖 원칙 유지**: LaCoSENet streaming 코어 (`lacosenet.py`, `utils.py`, `converters/`, `layers/`) 수정은 Stage 2 착수 직전 별도 commit 으로 처리하고, U1/U2 fix 가 기존 Android parity 7/7 을 깨지 않음을 pre-push hook 으로 보장한다.

---

### Stage 2 — Golden fixture 생성기 (BAFNetPlus)

**기간 가정**: 1 세션
**선행 조건**:
- ✅ Stage 1 완료 (5/5 acceptance PASS)
- ⚠️ **U1 fix 적용 및 LaCoSENet 7/7 regression 재확인** — `src/models/streaming/utils.py:272` `load_model(...)` 시그니처 교정
- ⚠️ **U2 fix 적용 및 LaCoSENet 7/7 regression 재확인** — `src/models/streaming/layers/tsblock.py` `StreamingConv2d.forward` 에 `get_state_frames_context()` fallback 추가
- ⚠️ U1/U2 적용 후 `tests/test_bafnetplus_streaming.py::test_lacosenet_masking_branch_regression` tolerance 를 **RMS < 1e-5 (원래 Criterion 3 목표치)** 로 복원하고 PASS 재확인
- ⚠️ U1/U2 적용 후 Stage 1 `test_offline_vs_streaming_parity` 수치가 회귀 없음 (여전히 RMS < 1e-4) 확인

**산출물**:

| # | 산출물 | 경로 | 예상 LoC / 크기 |
|---|---|---|---|
| S2-1 | Fixture 생성 스크립트 | `scripts/make_bafnetplus_streaming_golden.py` | 400~500 LoC |
| S2-2 | Fixture bin + manifest (고정 seed=42, 2초 2-ch Gaussian 0.7 상관) | `android/benchmark-app/src/androidTest/assets/bafnetplus_fixtures/` | ~8 MB, 22 chunks × ~30 tensors/chunk |
| S2-3 | FixtureLoader 확장 (BAFNetPlus 파싱 지원) | `android/benchmark-app/src/androidTest/kotlin/com/lacosenet/benchmark/parity/FixtureLoader.kt` (수정 20~40 LoC) | +30 LoC |
| S2-4 | `.gitignore` 정책 확인 — `bafnetplus_fixtures/*.bin` 트래킹 허용 (이미 `android/.gitignore` 에서 `*.bin` 제거 적용됨) | `android/.gitignore` | 0 LoC (기존 규칙 재활용) |

**핵심 acceptance**:
- [ ] 22 chunks fixture 생성 완료, manifest.json 유효 JSON
- [ ] PC 측 재실행 — fixture 로드 후 Python reference 로 돌린 결과와 bit-identical (같은 seed)
- [ ] Git 커밋 크기 ≤ 10 MB
- [ ] 기존 LaCoSENet fixture (3.8 MB) 와 이름 충돌 없음

**시사점 (LaCoSENet Stage 2 참고)**: LaCoSENet Stage 2 에서 fixture 를 만들고 기기에 올리자마자 **B9 (P0, NPE)** 가 드러났다 (`REPORT.md:407-419`). BAFNetPlus Stage 2 에서도 fixture 를 1차 Android 측에 올리면 **신규 P0 결함 발견 확률 높음** — 이 단계에서 STFT 는 LaCoSENet fix 이후로 정상이므로, 발견될 버그는 주로 (a) 160-state 초기화 경로, (b) dual input buffer sync, (c) fusion layer state 관리에서 기인. Stage 2 말미에 1차 기기 실행을 **강제 포함**한다.

**risks**:
- R2-1 Fixture 생성 시간: 22 chunks × BAFNetPlus forward ≈ 수 초. 한 번 생성하면 재사용이므로 치명적 아님.
- R2-2 2채널 Gaussian 의 correlation 수준이 낮으면 calibration gain 이 거의 1 로 수렴 → calibration 경로가 실질적으로 exercise 되지 않음. 생성기에 `bcs_noise_gain`, `acs_noise_gain` 파라미터를 두어 energy 비를 프레임별로 변화시켜야 calibration 이 유의미하게 반영됨.

---

### Stage 2 실행 결과 (2026-04-19 완료)

**상태**: ✅ 완료 — 7/7 acceptance PASS. Stage 3 착수 가능.

**산출물 실측 LoC + 크기**:

| # | 파일 | 실측 | 예상 |
|---|---|---|---|
| 선행 | `src/models/streaming/utils.py` (U1 fix) | +3 lines | 1 line |
| 선행 | `src/models/streaming/layers/tsblock.py` (U2 fix) | +7 lines | 4 lines |
| S2-1 | `scripts/make_bafnetplus_streaming_golden.py` | **515** | 400~500 |
| S2-2 | `android/benchmark-app/src/androidTest/assets/bafnetplus_fixtures/` | **7.64 MB** (7.12 MB bin + 380 KB manifest + 256 KB audio) | ~8 MB |
| S2-3 | `FixtureLoader.kt` 확장 (`BafnetPlusFixtureLoader` 신규 클래스) | +115 | +30 |
| S2-3b | `BafnetPlusFixtureSmokeTest.kt` (신규 smoke test) | **81** | — (S2-η 요구로 추가) |
| **합계** | | **721 LoC + 7.64 MB** | 430~530 LoC |

**acceptance 결과**:

- ✅ S2-1 스크립트 PASS: 41 chunks × 31 tensors 생성, manifest.json 유효 JSON
- ✅ manifest.json 자기-기술적 (tensor name / dtype / shape / bytes / SHA256)
- ✅ PC 재현성 bit-identical: 2 회 실행 → `diff -rq` 0 diff
- ✅ Git 커밋 크기: 7.64 MB (≤ 10 MB)
- ✅ 이름 충돌 방지: 기존 `fixtures/` 와 별도 `bafnetplus_fixtures/`
- ✅ Calibration 경로 exercise (R2-2 대응): relative_log_gain std=0.056, alpha_softmax std=0.032 → `exercised=True` 기록. ⚠️ `common_log_gain` 자체는 synthetic 입력에 대해 -0.5 근처 saturation — trained tanh head 가 acs_mask_mean>0.1 에 민감해 포화됨 (R2-3 참조). Stage 4 parity 는 byte-wise 비교이므로 포화여도 무관
- ✅ Fusion 중간 텐서 포함: calibration_feat/hidden, common/relative_log_gain, bcs/acs_com_cal, alpha_softmax, est_com 등 31 키 dump
- ✅ 1차 기기 smoke: `BafnetPlusFixtureSmokeTest` 4/4 PASS (0.093s on SM_S938N). 통합 parity 패키지 `com.lacosenet.benchmark.parity` **11/11 PASS** (1.642s) — LaCoSENet 7 + BafnetPlus 4

**실측 per-chunk 텐서 구성 (31 items, ~178 KB/chunk raw float32)**:

- Inputs (6): input_samples_bcs/acs, stft_context_bcs/acs_in, stft_input_bcs/acs
- STFT outputs (4): bcs/acs_mag, bcs/acs_pha (각 [1, 201, 11])
- Decoder outputs (7): bcs/acs_est_mag/pha, bcs/acs_com_out, acs_mask (masking-branch)
- Calibration (6): calibration_feat [1,5,8], calibration_hidden [1,16,8], common/relative_log_gain [1,1,8], bcs/acs_com_cal [1,201,8,2]
- Fusion (3): alpha_softmax [1,2,201,8], est_mag/pha [1,201,8]
- iSTFT OLA (5): ola_buffer_in/out, ola_norm_in/out, istft_output

**Stage 3 착수 시 주의점**:
- **ONNX graph 설계 시 alpha_softmax 가 핵심 검증점**. fusion 이 model graph 의 최종 블록이라, decoder 출력 vs alpha_softmax vs est_com 경계에서 ONNX op 변환 오류 발생 가능. Stage 3 verify 는 `alpha_softmax.bin` 비교로 시작
- **common_log_gain 의 -0.5 saturation 은 의도된 모델 응답**. Stage 3 에서 ONNX export 후 fixture 와 비교할 때 "-0.5 근처 상수" 는 정상, 값이 0 또는 +0.5 에 가까우면 export 버그
- **fixture 가 31 tensors × 41 chunks = 1,271 .bin 파일**. Android 빌드 시 asset 패키징이 느려질 수 있음 (Kotlin 빌드는 15 초 측정됨)
- **U1+U2 fix 는 별도 commit 이지만 Stage 2 의 일부로 간주** (선행 조건이었음). Stage 2 commit 메시지에서 커밋 hash 참조
- **기기 smoke S2-η 통과 확인됨** (SM_S938N, 2026-04-21 재실행). `loadsBafnetPlusManifest`, `loadsInputAudioStreams`, `chunk000HasExpectedKeys`, `calibrationDiagnosticsExercised` 4/4 PASS. Android asset 패키징 + JSON 파싱 + float32 LE 역직렬화 경로 모두 정상

**신규 P0 / R2-X 없음**. LaCoSENet Stage 2 의 B9 (NPE) 같은 Android-side 구조 결함은 기기 smoke 4/4 PASS 로 배제됨. LaCoSENet 기존 parity 7/7 도 **회귀 없음** (통합 run 11/11 확인)

---

### Stage 3 — ONNX export + 정적 검증

**기간 가정**: 1~2 세션
**선행 조건**: Stage 2 fixture 확보

**산출물**:

| # | 산출물 | 경로 | 예상 LoC |
|---|---|---|---|
| S3-1 | ONNX export 스크립트 | `src/models/streaming/onnx/export_bafnetplus_onnx.py` (신규) | 400~500 |
| S3-2 | `BAFNetPlusStatefulExportableNNCore` — 통합 그래프 wrapper | export_bafnetplus_onnx.py 내부 | ~150 |
| S3-3 | Calibration data reader 확장 (4-input) | export_bafnetplus_onnx.py (재활용하되 `QnnCalibrationDataReader` 서브클래스화) | ~80 |
| S3-4 | 번들 asset 생성: `bafnetplus.onnx` (FP32), `bafnetplus_qdq.onnx` (QDQ INT8), `bafnetplus_streaming_config.json` | `android/benchmark-app/src/main/assets/` | — |
| S3-5 | ORT CPU EP 기반 multi-chunk verify (기존 `verify_onnx` 확장) | export_bafnetplus_onnx.py | ~100 |
| S3-6 | **HTP op coverage probe** — `bafnetplus.onnx` 또는 `bafnetplus_qdq.onnx` 로드 + 단일 run QNN HTP 에서 성공 여부 확인 | 신규 간단한 androidTest 또는 `scripts/probe_htp_support.py` | ~50~100 |

**예상 LoC 합**: **780~1,030 LoC**
**핵심 acceptance**:
- [ ] FP32 export 성공, onnx-simplifier 통과
- [ ] PyTorch vs ORT CPU EP: est_com_real/imag 및 모든 next_state RMS < 1e-5, max < 1e-4 (3 chunks 연속)
- [ ] QDQ INT8 export 성공, HTP 에서 세션 초기화 성공
- [ ] HTP 에서 1 chunk inference 성공 (결과 정확도는 별도 벤치마크에서)
- [ ] streaming_config.json schema valid — `StreamingConfig.validate()` 통과 (파싱 시도만)

**Stage 3 결정 포인트**:
- 만약 HTP op coverage probe 에서 **graph finalization 실패** 또는 **VTCM 초과 에러** 발생 → **분리 2 세션 + host fusion 전략으로 전환**. 이 경우 S3-2 의 wrapper 는 mapping 용, masking 용 2개로 분리, fusion 은 Android 쪽에서 구현 (Stage 4 에 반영).
- 성공 시 → 단일 통합 세션으로 Stage 4 진행.

**risks**:
- R3-1 **QNN HTP op coverage** — exp, softmax, complex multiply, tanh 일부 op 이 QDQ QUInt8 에서 실패할 가능성. LaCoSENet 은 PReLU QUInt16 실패만 경험. BAFNetPlus 는 더 다양한 op 조합.
- R3-2 **VTCM 16 MB 한계** (Snapdragon 8 Elite Hexagon V79). 통합 그래프가 VTCM 초과하면 context tile 분할 발생 → 성능 저하. 이 경우 분리 세션이 더 빠를 수 있음.
- R3-3 **export_time_frames 선택** — LaCoSENet 은 11 frames. BAFNetPlus 도 동일 (chunk_size=8 + encoder_lookahead=3). 단, fusion 은 chunk_size=8 frame 만 처리 — export 시 fusion 입력 tensor 를 8 또는 11 로 할지 결정 (통상 11 로 맞추고 내부에서 slice). onnx export API 제약 검토.

---

### Stage 3 실행 결과 (2026-04-21 완료)

**상태**: ✅ 완료 — 핵심 acceptance 5/5 + 추가 acceptance 3/3 PASS. **단일 세션 전략 확정** (HTP 로드 + 1 chunk 실행 성공). Stage 4 착수 가능.

**산출물 실측 LoC + 크기**:

| # | 파일 | 실측 | 예상 |
|---|---|---|---|
| S3-1 | `src/models/streaming/onnx/export_bafnetplus_onnx.py` (신규) | **1,502 LoC** (docstring + CLI + verify + QDQ + config gen 모두 포함) | 500~700 |
| S3-2 | `BAFNetPlusStatefulExportableNNCore` wrapper | (S3-1 포함, ~230 LoC) | ~150 |
| S3-3 | `BafnetPlusCalibrationReader` — 4-input Stage 2 fixture 리더 | (S3-1 포함, ~70 LoC) | ~80 |
| S3-4 | `bafnetplus.onnx` (FP32, simplified) | **11.63 MB** (md5 4b7663…) | ~15 MB |
| S3-4 | `bafnetplus_qdq.onnx` (QDQ INT8) | **6.98 MB** (md5 9a1b53…) | ~10 MB |
| S3-4 | `bafnetplus_streaming_config.json` | 59 KB, **166 states**, md5 38531a… | ~10 KB |
| S3-5 | `verify_onnx_multi` + `verify_against_fixture` | (S3-1 포함, ~140 LoC) | ~100 |
| S3-6 | `BafnetPlusHtpProbeTest.kt` (신규) | **206 LoC** | 80 |
| **합계** | | **1,708 LoC + 18.67 MB** | 680~940 LoC + 30~50 MB |

**LoC 예측 초과 원인**: 통합 그래프 wrapper 가 LaCoSENet 대비 입력/출력 대폭 확장 (4 audio + 166 states). `build_bafnetplus_state_registry`, `BafnetPlusStateRegistry`, `_unflatten_states/_flatten_states`, `_stateful_conv1d_inline/_stateful_conv2d_inline` 등 LaCoSENet 에 없던 헬퍼가 ~500 LoC 추가. Docstring + module description 이 ~200 LoC. 실제 로직 LoC 는 ~800 수준.

**acceptance 결과**:

- ✅ FP32 export 성공: `bafnetplus.onnx` 12.0 MB → simplify 후 11.6 MB (-3.1%)
- ✅ onnx-simplifier 통과 (check OK)
- ✅ **PyTorch wrapper vs ORT CPU EP parity (3 chunks 연속)**: est_mag max 2.84e-5, est_com_real max 5.93e-5, est_com_imag max 5.73e-5, RMS 전부 1e-5 미만 → **tolerance RMS<1e-5, max<1e-4 만족** ✅
- ✅ QDQ INT8 export 성공: `bafnetplus_qdq.onnx` 7.0 MB (-40% vs FP32). Calibration: Stage 2 fixture 41 chunks
- ✅ HTP 세션 초기화 성공 + **1-chunk inference 성공** (SM_S938N, 8.3s). 169 outputs (3 primary + 166 next_state), est_mag/real/imag 모두 NaN/Inf-free
- ✅ streaming_config.json schema valid — `state_info.state_layout` 에 166 entries × shape/dtype/bytes 기록, `io_info.input_names` / `output_names` 명세
- ✅ **추가 S3**: State 이름 체계 `state_alpha_conv_*` (4) < `state_calibration_conv_*` (2) < `state_mapping_rf_*` (80) < `state_masking_rf_*` (80) — Python `sorted()` + Kotlin `List.sort()` 결정적 순서 보장. `total_state_bytes` = 23.45 MB
- ✅ **추가 S3**: LaCoSENet 기존 자산 무회귀 — `model.onnx`, `streaming_config.json`, `fixtures/` 건드리지 않음. `parity` 패키지 **12/12 PASS** (11 기존 + 1 신규 HTP probe, 9.8s)
- ✅ **추가 S3 Stage 3 결정 포인트**: HTP 로드 + 1 chunk 실행 성공 → **단일 통합 세션 전략 확정**. 분리 2-session 으로 전환 불필요

**실측 graph 구성 (166 states)**:

| 카테고리 | prefix | count | state shape 예시 |
|---|---|---|---|
| Alpha fusion | `state_alpha_conv_0..3` | 4 | [1, 3, 6, 207] / [1, 16, 6, 207] (블록별 in_channels 상이) |
| Calibration | `state_calibration_conv_0..1` | 2 | [1, 5, 8] / [1, 16, 8] |
| Mapping TSBlocks | `state_mapping_rf_{0..3}_tb{0..1}_{cab\|gpkffn}_{key}` | 80 | [1, 64, 6, 100] (conv), [1, 64, 2] (ema) 등 |
| Masking TSBlocks | `state_masking_rf_{0..3}_tb{0..1}_{cab\|gpkffn}_{key}` | 80 | 동일 |

**핵심 설계 결정 (S3-β 완료)**:

- **통합 단일 그래프**: mapping Backbone + masking Backbone + calibration + alpha fusion 모두 하나의 ONNX 에 포함. 분리 전략 불필요
- **Backbone encoder/decoder = 비스트리밍 (zero-padded)** — LaCoSENet 컨벤션 따름. TSBlock 80 states 만 externalize (backbone 당)
- **Fusion = inline 스트리밍** — `_stateful_conv1d_inline` / `_stateful_conv2d_inline` 헬퍼로 state I/O 를 graph 에 명시. `StatefulCausalConv1d/Conv2d._streaming = False` 유지 (내부 `_state` 비사용)
- **atan2 을 그래프 밖으로 이동** — QNN HTP precision 우려로 `est_pha` 대신 `(est_mag, est_com_real, est_com_imag)` 출력. 초기에 atan2 을 in-graph 로 두었을 때 PT-ORT 파리티 max_err=3.05e-4 관찰 (atan2 FP32 구현 차이). 복소수 출력으로 전환 후 max_err=5.93e-5 까지 개선
- **Kaiming init 시드 = 42** — Stage 2 fixture 와 동일. `prepare_bafnetplus_from_checkpoints` 가 `BAFNetPlusStreaming.from_checkpoint` 를 재사용해 RNG 상태 일치 보장

**Stage 2 fixture 정합성 주의 (발견)**:

- Stage 2 fixture 의 `chunk_000/est_mag.bin` 은 **실제 첫 모델 forward 의 2번째 호출 결과** — pre-warm chunk (first forward) 은 fixture 에 저장되지 않음. Stage 3 ORT 가 zero-state 로 시작하면 fixture `chunk_000` 과 일치 불가 (drift ~0.37 on est_mag)
- 이 차이는 버그가 아니라 **설계상 기대되는 사항**. 정식 parity 검증은 `verify_onnx_multi` (PyTorch wrapper vs ORT, 동일 초기 state) 로 수행하며 위에서 PASS 확인됨
- `verify_against_fixture` 는 정보 제공용 진단으로 분류됨 (drift 출력은 post-mortem 기록)

**Stage 2 fixture 재현성 미세 drift 기록**:

- `diff -rq` 재실행 시 `istft_output.bin`, `ola_buffer_{in,out}.bin` 에서 **1.86e-9 ULP-level 차이** 발견 (iFFT FP32 rounding + OLA 누적 비결정성). Stage 3 ONNX 비교 기준 텐서 (`est_mag.bin`, `est_pha.bin`, `alpha_softmax.bin`) 는 **bit-identical** 재현 확인. iSTFT 는 host-side 이므로 Stage 3 scope 외
- 결정: Stage 2 fixture 는 그대로 유지 (critical 경로 bit-identical). 필요 시 Stage 4 regeneration 단계에서 iSTFT 경로 재결정성 조사

**Stage 4 착수 시 주의점**:

- **새 output 시그니처 (est_mag, est_com_real, est_com_imag)**: Stage 4 의 `BAFNetPlusInferenceResult` data class 는 LaCoSENet 의 `(estMask, phaseReal, phaseImag)` 와 다른 시멘틱 — BAFNetPlus 는 이미 fusion 완료된 complex, 즉 **iSTFT 직접 입력 가능**. mask 곱셈 없음
- **State 이름 ordering = alphabetical**: Python `sorted()` 결과가 `state_info.state_names` 에 기록됨. Kotlin 에서 `stateNames.sorted()` 호출로 동일 순서 얻을 수 있으나, **직접 JSON 파싱 순서를 따르는 편이 안전** (shape metadata 도 동일 순서 기록됨)
- **State 총 메모리 = 23.45 MB** per forward (both double-buffers on). Stage 4 R4-X 에서 메모리 예산 점검 필요 (LaCoSENet 대비 ~5x, 여전히 400 MB 예산 내)
- **CausalConv1d padding 관계**: `StatefulCausalConv1d.padding_size = padding * 2` 라는 사실이 `_stateful_conv1d_inline` 의 state 모양 계산에 중요. Stage 4 Android 구현 시 이 정수 값을 config 에서 직접 읽어 사용 (config 의 `state_layout[*].shape` 참조)
- **Git LFS 불필요**: 18.67 MB 합산으로 기존 `model.onnx` (6 MB) + 본 export 가 일반 Git tracking 범위 내
- **QDQ 수치 drift**: PT-ORT QDQ 비교 시 chunk 0 est_mag max=1.12 (예상 — INT8 quantization 오차). HTP 에서는 FP16 누적 경로 포함해 더 달라질 수 있음. Stage 5 벤치마크에서 정식 PESQ/STOI 측정

**신규 P0 / R3-X 없음**. Stage 3 결정 포인트 분기 경로 (2-session + host fusion) 불필요. Stage 4 진입 조건 모두 충족.

---

### Stage 4 — Android 확장 (dual input / 160+α state)

**기간 가정**: 2~3 세션
**선행 조건**: Stage 3 ONNX + config 확정

**산출물**:

| # | 산출물 | 경로 | 예상 LoC |
|---|---|---|---|
| S4-1 | `StatefulInference.run()` 제네릭화 (옵션 A) — named input map 지원 + 출력 mapping | `android/lacosenet-streaming/.../session/StatefulInference.kt` 수정 | +80 / -20 = **+60 LoC 순증** |
| S4-2 | 신규 `BAFNetPlusStreamingEnhancer` | `android/bafnetplus-streaming/.../BAFNetPlusStreamingEnhancer.kt` (신규 모듈) | ~350 |
| S4-3 | `bafnetplus-streaming` Gradle 모듈 신설 + `consumer-rules.pro` 기존 LaCoSENet 과 동일 rule 복제 | `android/bafnetplus-streaming/build.gradle.kts`, `settings.gradle.kts` | ~70 |
| S4-4 | `benchmark-app` 에 BAFNetPlus 의존성 추가 + 샘플 수행 코드 | `android/benchmark-app/build.gradle.kts`, `StreamingBenchmarkTest.kt` 확장 | +100 LoC |
| S4-5 | `BAFNetPlusStatefulInferenceParityTest` (2 @Test) | `android/benchmark-app/src/androidTest/.../parity/BAFNetPlusStatefulInferenceParityTest.kt` | ~200 |
| S4-6 | **선택**: `DualChannelFeatureBuffer` — BCS+ACS frame 동시 push/pop | `android/bafnetplus-streaming/.../audio/DualChannelFeatureBuffer.kt` | ~100 |
| S4-7 | `BAFNetPlusInferenceResult` data class + complex-to-iSTFT helper | `android/bafnetplus-streaming/.../core/` | ~60 |

**예상 LoC 합**: **940~1,140 LoC** (LoC 예산의 약 70%)

**핵심 acceptance**:
- [ ] 기존 LaCoSENet 7/7 parity 회귀 없음
- [ ] 신규 `BAFNetPlusStatefulInferenceParityTest` 2/2 PASS (tolerance: RMS < 1e-5, max < 1e-4, CPU 백엔드)
- [ ] `BAFNetPlusStreamingEnhancer.processChunk(bcsSamples, acsSamples)` E2E 가 NPE/Crash 없이 800 samples 반환
- [ ] 메모리 — `StreamingEnhancer` 초기화 + 100 chunks 반복 후 Native Heap peak ≤ 400 MB (LaCoSENet 292 MB × 1.4)
- [ ] `benchmark-app` 에서 `benchmarkBafnetplusFullQdq` 신규 테스트 성공 실행 (수치는 Stage 5 평가)

**Stage 4 하위 단계 순서** (LaCoSENet Stage 3 의 "fix 단위 PR" 방식 적용):

1. **S4-PR1 `feat/generic-stateful-inference`** — S4-1. `StatefulInference.run()` 을 named input Map 으로 확장하되 `run(mag, pha)` overload 유지. Acceptance: LaCoSENet 7/7 회귀 없이 빌드 성공, 기기 실행 7/7 PASS.
2. **S4-PR2 `feat/bafnetplus-streaming-module`** — S4-3. 빈 Gradle 모듈 + `BAFNetPlusStreamingEnhancer` skeleton (initialize/release 만). Acceptance: `./gradlew :bafnetplus-streaming:assembleDebug` 성공.
3. **S4-PR3 `feat/bafnetplus-stateful-inference-init`** — S4-2 일부. StatefulInference 초기화에서 BAFNetPlus session 의 166 state 를 인식하고 double buffer 할당. Acceptance: initialize 후 `stateNames.size == 166`, `magBuffer/phaBuffer` 대신 4개 input buffer 할당 성공. Test class 는 PassThrough 수준 — 실제 inference 는 다음 PR.
4. **S4-PR4 `feat/bafnetplus-processchunk-cpu-only`** — S4-2 메인. `processChunk(bcs, acs)` 가 CPU 백엔드에서 non-crashing 실행. Acceptance: processChunk 100 회 반복 crash 없음.
5. **S4-PR5 `test/bafnetplus-stateful-inference-parity`** — S4-5. 2 @Test 추가. Stage 2 fixture 로 CPU 백엔드 결과와 비교. Acceptance: 2/2 PASS, RMS < 1e-5.
6. **S4-PR6 `feat/bafnetplus-qnn-backend`** — QNN HTP 경로 동작 확인. Acceptance: QDQ model load 성공, 1 chunk inference 성공 (성능은 Stage 5).
7. **S4-PR7 `test/bafnetplus-benchmark`** — S4-4. `benchmarkBafnetplusFullQdq` 추가. Acceptance: 빌드 + 실행 성공, 수치는 Stage 5 평가.
8. **S4-PR8 (옵션) `feat/bafnetplus-ablation-variants`** — Open Q1 answer 에 따라 2종 ablation 지원.

**각 PR 후 pre-push hook 자동 검증**: LaCoSENet 7/7 + BAFNetPlus 현재까지 추가된 test — 회귀 탐지.

**risks**:
- R4-1 **제네릭화 회귀**: LaCoSENet 7/7 이 깨지면 Stage 4 의 다른 모든 진행이 blocked. S4-PR1 완료 후 pre-push hook 이 자동으로 7/7 돌림 — 회귀 즉시 감지. Mitigation: `run(mag, pha)` overload 를 `run(mapOf("mag" to mag, "pha" to pha))` 로 inline 위임 → behavior 변경 0.
- R4-2 **State 이름 스킴 충돌**: `state_mapping_*`, `state_masking_*`, `state_fusion_*` 과 기존 LaCoSENet `state_rf_*` 의 이름 경계. Stage 3 에서 확정. `StatefulInference.initialize()` 는 `name.startsWith("state_")` 로 필터링 — 모든 prefix 커버. C4 assertion 은 streaming_config.json 의 state_names 리스트와 대조 → config 가 올바르면 자동 pass.
- R4-3 **benchmark-app 이 두 라이브러리(lacosenet-streaming + bafnetplus-streaming) 를 모두 의존** 하므로 APK 크기 ↑. LaCoSENet APK 83 MB (Stage 6). BAFNetPlus 추가 시 ~90~100 MB 예상. 문제 시 별도 benchmark-app-bafnetplus 분리 검토.
- R4-4 **CPU 백엔드 실시간성 손실**: Stage 4 개발 내내 CPU 백엔드로 parity 맞추다가 QDQ 로 전환 시 수치 drift 발생 가능. LaCoSENet 은 이 경로 경험 없음. Mitigation: QDQ verify 는 Stage 3 에서 완료되었으므로 여기서는 "정확성 ≈ CPU" 가정. Stage 5 가 실측으로 확정.
- R4-5 **ExecutionBackend.run() 시그니처 변경 불필요**: 기존 `run(inputs: Map<String, OnnxTensor>): OrtSession.Result` 는 이미 Map 기반. BAFNetPlus 추가 input 도 동일 인터페이스로 자연스럽게 확장. 변경 0.
- R4-6 **state 이름 알파벳 정렬의 의도치 않은 상호작용**: `state_fusion_alpha_conv0` 과 `state_mapping_rf_0_tb0_cab_ema` 중 어느 쪽이 먼저인가? Python `sorted()` 는 `state_fusion*` < `state_mapping*` < `state_masking*` 순. Kotlin `List.sort()` 동일. **문제 없음**.

**Exit criteria**: S4-PR5 (parity) PASS + S4-PR6 (QNN session load) PASS.

---

### Stage 5 — 실기기 벤치마크 + 재측정

**기간 가정**: 1 세션
**선행 조건**: Stage 4 parity 통과, SSH 터널 + ADB 연결 복구

**산출물**:

| # | 산출물 | 경로 | 형식 |
|---|---|---|---|
| S5-1 | `benchmarkBafnetplusFullQdq` (cold-state) | `StreamingBenchmarkTest.kt` 확장 | 10 warmup + 50 sessions |
| S5-2 | `benchmarkBafnetplusNoCalibrationQdq` (ablation 2종 지원 시) | 동일 | 동일 |
| S5-3 | Same-process 3-test 시퀀스 — LaCoSENet dual → BAFNetPlus full 로드 순서 (메모리 증감 확인) | 스크립트 또는 문서화 | `am instrument -e class A#m1,A#m2,A#m3` |
| S5-4 | `dumpsys meminfo` 시계열 수집 (0.5s 간격) | `docs/review/logs/stage5_bafnetplus_meminfo_*.log` | 실측 |
| S5-5 | `milestone.md` § 4 BAFNetPlus row 추가 | `android/benchmark-app/milestone.md` | ~50 LoC |
| S5-6 | Thermal probe — `benchmarkBafnetplusLongRunQdq` 60s 연속 (옵션) | StreamingBenchmarkTest.kt | 1분 루프 |

**측정 시나리오 매트릭스**:

| # | 시나리오 | 목적 | Primary 지표 |
|---|---|---|---|
| A | BAFNetPlus full QDQ cold-state | 최선 latency 베이스 | Mean, P95, P99 |
| B | BAFNetPlus full QDQ same-process 반복 10회 | leak 회귀 검증 | meminfo 시계열 plateau 확인 |
| C | LaCoSENet dual QDQ + BAFNetPlus full QDQ 같은 프로세스 순차 | 크로스 leak | PSS 증가량 |
| D | BAFNetPlus full FP16 cold-state (옵션) | fallback 베이스 | Mean |
| E | BAFNetPlus no_calibration QDQ (옵션, Q1 A/B 시) | ablation 비교 | Mean |
| F | BAFNetPlus long-run 60s | thermal drift | P99 variance |
| G | BAFNetPlus session init + release cycle (10회) | session lifecycle | native heap 회수 |

**핵심 acceptance**:
- [ ] Cold-state Mean ≤ 20ms (목표) 또는 ≤ 25ms (fallback)
- [ ] Same-process 10-chunk 연속 실행 시 RSS 증가 없음 (LaCoSENet B1 fix 로 sanity 확보)
- [ ] P95 ≤ 40ms (budget 80% 이내), P99 ≤ 50ms (budget 100% 이내)
- [ ] Overlap ratio (fusion 기준) ≥ 0.3 (단일 세션이므로 해당 안 될 수 있음 — 분리 세션 시에만 측정)
- [ ] milestone.md row 값이 실측 ±10% 재현

**파생 지표**:
- `fusion_overhead_ms` = BAFNetPlus Mean − LaCoSENet Dual Concurrent Mean. 예상 2~5ms.
- `parallel_efficiency` = 2 × LaCoSENet single Mean / BAFNetPlus Mean. 1.0 이면 완전 병렬, 0.5 이면 직렬 수준. 예상 0.7~0.85.
- `budget_utilization_p95` = BAFNetPlus P95 / 50ms. 목표 ≤ 0.8.

**risks**:
- R5-1 **Thermal throttling** — 연속 실행 시 P99 큰 변동. BAFNetPlus 는 연산량 2배로 LaCoSENet 대비 열 부담 ↑. 1분 이상 테스트 시 latency 20% 이상 드리프트 가능. Mitigation: 측정은 cold-state 위주, long-run 은 "정성적 확인" 수준.
- R5-2 **Budget 초과 시 mitigation** (우선순위 순):
  1. ablation `no_calibration` 으로 전환 (fusion 간소화, ~2~3ms 절약)
  2. `time_block_kernel` 을 `[11]` (Small) 로 다시 학습 → Large 10.1ms → Small 6.2ms 로 단일 backbone 절감. 2 backbone 합 ≈ 10ms
  3. chunk_size 16 frames 로 증가 (batch throughput ↑, 실시간 delay ↑, latency-budget 재계산 필요)
  4. FP32 path 완전 DESCOPE
  5. 분리 2 세션 + host fusion 으로 재구성 (단, fusion 자체가 host 측이므로 절약량 제한적)
- R5-3 **Dual-concurrent overlap ≈ 0**: 단일 통합 세션의 경우 "overlap ratio" 는 의미 없음. 대신 `fusion_overhead_ms` 가 주요 지표.
- R5-4 **일관성 결여**: cold-state 와 same-process 간 차이가 너무 크면 실 서비스 배포 리스크. LaCoSENet 은 Stage 6 B1 fix 후 same-process 가 오히려 더 빠름 (JIT warmup). BAFNetPlus 는 state 2배로 warmup 이득이 다를 수 있음.

---

### Stage 6 — 문서 + pre-push hook 확장

**기간 가정**: 0.5 세션
**선행 조건**: Stage 5 벤치마크 완료

**산출물**:

| # | 산출물 | 경로 | 형식 |
|---|---|---|---|
| S6-1 | `android/README.md` 섹션 추가 — BAFNetPlus 초기화 코드 샘플 | `android/README.md` | ~40 LoC |
| S6-2 | `android/docs/ARCHITECTURE.md` 업데이트 — BAFNetPlus 파이프라인 다이어그램, LOC 표 확장 | `android/docs/ARCHITECTURE.md` | ~60 LoC |
| S6-3 | `scripts/hooks/pre-push` 확장 — LaCoSENet 7/7 + BAFNetPlus 2/2 (또는 추가 E2E) 동시 실행 | `scripts/hooks/pre-push` | +10 LoC |
| S6-4 | `docs/review/BAFNETPLUS_PORT_CLOSURE.md` — Stage 1~5 결과 요약, 남은 이슈, 후속 계획 | `docs/review/` (신규) | ~200 LoC |
| S6-5 | `milestone.md` § Conclusion 업데이트 — BAFNetPlus 배포 가능성 판정 | `android/benchmark-app/milestone.md` | ~30 LoC |

**핵심 acceptance**:
- [ ] Pre-push hook 이 LaCoSENet parity 7/7 + BAFNetPlus parity 2/2 = 9/9 모두 실행, 실패 시 push 차단
- [ ] `README.md` 샘플 코드가 실제 API 와 1:1 일치
- [ ] Closure 문서가 "배포 가능" / "가능하나 제약 있음" / "개선 필요" 중 명확한 판정 제시

---

### Stage 전체 의존성 DAG

```
Open Q1 (ablation) ──┐
Open Q2 (session)  ──┼─── Stage 1 ──(U1+U2 fix)── Stage 2 ─── Stage 3 ─── Stage 4 ─── Stage 5 ─── Stage 6
Open Q3 (mic HW)   ──┤                                             │
                     │                                             └─ (optional) long-run thermal probe
Open Q4 (coexist)  ──┘

U1+U2 fix (§9.14): LaCoSENet 공유 streaming 인프라 복구 단계
  (a) src/models/streaming/utils.py:272 load_model 시그니처 교정
  (b) src/models/streaming/layers/tsblock.py:454 StreamingConv2d state_frames fallback 추가
  (c) LaCoSENet parity 7/7 재측정 → 회귀 없음 또는 개선 확인
  (d) BAFNetPlus Stage 1 5/5 재측정 → test_lacosenet_masking_branch_regression tolerance 를
       RMS < 1e-5 로 복원 후 PASS 확인
```

---

## 9. 위험 평가

### 9.1 QNN HTP op coverage (P1)

| 위험 | 원인 | 영향 | 대응 |
|---|---|---|---|
| `exp(common_log_gain ± 0.5*relative_log_gain)` 가 QDQ INT8 QUInt8 에서 정확도 열화 | HTP 에서 exp 는 LUT 기반 — 저·고 값에서 dequant 오차 누적 | calibration gain 이 1 에서 크게 벗어나면 fusion 결과 왜곡 | (a) Stage 3 probe 후 판정, (b) `calibration_max_common_log_gain=0.5` 가 exp 입력 범위를 [−0.5, 0.5] 로 clamp 하므로 safe, (c) 필요 시 `max_common_log_gain` 축소 |
| `softmax(alpha, dim=1)` 가 2-channel softmax 로 QDQ 에서 불안정 | HTP 의 softmax 는 보통 float32 중간계산 사용, QUInt8 input 에서 dequant→softmax→requant 중 오차 | α_bcs + α_acs ≠ 1 (수치적으로) → 최종 est_com 에너지 drift | Stage 3 verify 시 sum(α) 검증, drift > 1% 시 대체 활성화(sigmoid) 제안 |
| `torch.log(x + 1e-8)` 가 HTP 에서 지원되지 않거나 dequant 오차 | calibration features 계산에 사용 (`bafnetplus.py:185-187`) | calibration 입력이 망가짐 | (a) log 입력 범위 확인 (보통 log(|X|²+1e-8) 은 ~-18..5 정도), (b) 필요 시 log 를 ONNX 에서 float32 sub-graph 로 격리 |

### 9.2 VTCM 용량 초과 (P1)

- Snapdragon 8 Elite Hexagon V79 VTCM: **16 MB** (QNN config `vtcm_mb=8` 은 보수적 설정). LaCoSENet 단일 backbone 은 여유 충분.
- BAFNetPlus 통합 그래프는 2 Backbone + fusion → activation peak 2배 이상. `graph finalization` 이 VTCM 초과 감지 시 tile split → latency penalty.
- **대응**:
  - Stage 3 에서 단일 통합 그래프의 `qnn_context_priority=high` + `vtcm_mb=16` 으로 시도
  - 실패 시 VTCM 분할 허용 또는 **분리 2 세션 전략** 으로 전환 (각 세션이 8 MB VTCM 점유)

### 9.3 State 160+α native memory 압박 (P1 → P2)

- LaCoSENet Stage 6 B1 fix 후 peak 292 MB (Native Heap). BAFNetPlus state 2배로 ~400 MB 예상.
- LMKD `min2x watermark` 는 실기기에서 3 GB 수준에서 trigger (Galaxy S25 Ultra 11 GB RAM). 400 MB 는 안전 범위.
- **단, B1 fix 가 BAFNetPlus 경로에도 일관 적용** 되어야 함 — `BAFNetPlusStreamingEnhancer` 내부의 모든 `session.run()` 을 `.use { }` 로 감싸는 것, 그리고 `StatefulInference.run()` 이 확장되면 해당 경로에도 `.use { }` 가 유지되는지 Stage 4 에서 검증.
- **대응**: Stage 4 parity test 통과 후 Stage 5 에 same-process 다중 실행 시나리오 포함 (B1 회귀 탐지).

### 9.4 실기기 2-마이크 입력 확보 (P0/P1)

- **하드웨어 현실**:
  - Galaxy S25 Ultra 는 일반 MEMS 마이크 2~3 개 (bottom/top/earpiece). **BCS(body-conducted) 는 없음**.
  - 진짜 BCS 는 throat/bone-conduction mic — 별도 BT 헤드셋 또는 외장 USB mic 필요.
- **대안 a. Simulated BCS**: ACS 에 low-pass + gain 조절로 BCS 근사 — Stage 1~5 파이프라인 검증엔 충분, **실제 제품 품질 평가는 불가**
- **대안 b. 스테레오 입력 hack**: L/R 채널을 BCS/ACS 로 가장, 유선 스테레오 마이크 외장 사용. 한계: L/R 간 동기 품질
- **대안 c. 실 BCS 장비**: 별도 조달 필요. 본 포팅 범위 밖 (Open Q3)
- **Stage 진행에 미치는 영향**: Stage 1~5 는 offline fixture + simulated BCS 로 충분. 실 BCS 는 Stage 6 이후 별도 세션.

### 9.5 Ablation 미확정 시 재작업 위험 (P1)

- `full` vs `no_calibration` vs `mask_only_alpha` 각각 ONNX export 다름, state 개수 다름, fixture 다름.
- Stage 1 시작 전 Open Q1 결정 없이 진행 시 **2~3 세션 재작업** 필수 (fixture 재생성, ONNX 재export, parity test 재작성).
- **대응**: 계획서 AskUserQuestion 에서 Q1 을 **최우선** 으로 수집.

### 9.6 Paper-code drift 및 weight 불일치 (P2)

- `conf/model/bafnetplus.yaml` 의 `checkpoint_mapping`, `checkpoint_masking` 은 **현재 null**. 학습이 진행 중이거나 검토용 weight 가 별도 디렉터리에 있을 수 있음.
- ONNX export + fixture 가 random-weight 으로 만들어져도 **수치 parity 검증은 가능**하나 **CER/PESQ 품질 평가는 불가**.
- **대응**: Stage 1~5 는 random weight 진행. 최종 closure (Stage 6) 직전에 실 weight 로 재export + 재측정. `streaming_config.json:export_info.checkpoint_md5` 로 추적.

### 9.7 LaCoSENet parity 회귀 (P0)

- `StatefulInference` 제네릭화 (§5.2 옵션 A) 적용 시 기존 LaCoSENet `run(mag, pha)` 경로가 의도치 않게 변할 수 있음.
- **대응**:
  - 기존 `run(mag, pha)` overload 유지 (inline 위임으로 구현: `run(mapOf("mag" to mag, "pha" to pha))`)
  - Pre-push hook 이 LaCoSENet parity 7/7 자동 검증 — 회귀 즉시 차단
  - Stage 4-1 완료 직후 즉시 `:benchmark-app:connectedDebugAndroidTest` 실행

### 9.8 Dual-stream frame sync (P2)

- BCS 와 ACS 가 서로 다른 마이크에서 캡처 시 sub-sample 드리프트 가능. LaCoSENet 은 단일 채널이라 무관.
- **대응**: 본 포팅은 "주어진 시점에 bcsSamples.size == acsSamples.size" 전제. Drift 보정(cross-correlation alignment) 은 별도 세션. `processChunk(bcs, acs)` 에서 `require(bcs.size == acs.size)` fail-fast.

### 9.9 Fusion 상태 초기화 누락 (P1)

- `BAFNetPlusStreaming.reset_state()` 가 mapping/masking 의 state 는 reset 하나 fusion 의 stateful conv state 는 놓치면 chunk 0 에서 학습 분포와 다른 context 로 출력.
- **대응**: Stage 1 `tests/test_bafnetplus_streaming.py` 에 `test_reset_state_fully_resets_fusion` 항목 포함 (`resetStatesRestoresBAFNetPlusChunk0Output` parity test 와 구조적으로 쌍을 이룸).

### 9.10 ONNX 그래프 node 수 증가로 인한 finalize 시간 (P2)

- LaCoSENet QDQ finalization: 2.4s. BAFNetPlus 통합 그래프는 노드 수 2~2.5배 → **예상 5~7s**.
- 앱 cold start 시점 부담. Mitigation: QNN context cache 사용 — `qnn_config.context_cache_enabled=true` (현재 default). 첫 실행만 긴 finalize, 이후는 cache 로 ms 수준 load.
- Cache 파일 크기: LaCoSENet QDQ ~5 MB 예상. BAFNetPlus ~10 MB 예상. `filesDir` 공간 확인.

### 9.11 ORT 버전 고정성 (P2)

- 포팅 기간 내 ORT 1.24.2 고정 가정. 1.25 또는 2.0 으로 업그레이드 시 QNN API 변화 가능성 → Stage 4 B1 fix (`OrtSession.Result.use {}`) 재검토 필요.
- **대응**: LaCoSENet `REPORT.md` 의 B8 (ORT 1.25+ `getFloatBuffer().get()` API 변경 리스크) 와 동일. BAFNetPlus 포팅 기간엔 1.24.2 fix. 1.25 업그레이드는 별도 세션.

### 9.12 Fixture 재생성 비용 (P2)

- Weight 변경, ablation 변경, 또는 streaming_config.json 필드 추가 시 fixture **재생성 필수**.
- Git 상에서 fixture `.bin` 변경 diff 는 binary 로 표시 — review 어려움. Mitigation: `.bin` 파일 변경 시 manifest.json 의 `version` 또는 `seed` 필드도 함께 올려 추적 가능.

### 9.13 복수 세션 동시 초기화 타이밍 (P2, 분리 세션 전략 시)

- 분리 세션 케이스에서 mapping/masking ONNX 를 순차 로드하면 cold start 지연. 병렬 로드 시 QNN context lock 충돌 가능성.
- **대응**: Stage 4 S4-PR2 시점에 `ExecutorService(2).submit { backend.initialize(...) }` 패턴 테스트. LaCoSENet 은 단일 세션이라 경험 없음 — 처음 등장하는 케이스.

### 9.14 Upstream streaming 인프라 latent bug 2건 (P0, Stage 1 에서 발견)

Stage 1 실행 중 LaCoSENet 공유 streaming 인프라에 기존 latent bug 2건이 발견되어 Stage 2 착수 전 해소 필수.

**U1 — `prepare_streaming_model()` signature mismatch (P0)**
- 위치: `src/models/streaming/utils.py:272-275`
- 원인: `load_model(model_args, device)` 호출이 `src/utils.py:178` 정의 `load_model(model_lib, model_class_name, model_params, device)` 와 불일치. 커밋 `58738cc` (2026-02-28 streaming 모듈 refactor) 에서 도입
- 영향 범위: `prepare_streaming_model()` / `LaCoSENet.from_checkpoint()` 즉시 오류 → Stage 4 에서 Python reference 경로로 LaCoSENet parity 재현 불가능 (Android ↔ Python 비교 금지)
- Stage 1 우회: `BAFNetPlusStreaming.from_checkpoint` 에서 `apply_streaming_tsblock` / `apply_stateful_conv` 를 직접 호출해 `prepare_streaming_model` 우회
- Stage 2 이전 fix: 1-line — `load_model(model_args.model_lib, model_args.model_class, dict(model_args.param), device)` 로 교정. Pre-push hook 으로 LaCoSENet 7/7 regression 재측정 (변화 없음 기대 — 기존 path 가 그대로 동작)

**U2 — `StreamingConv2d` thread-local context fallback 누락 (P0)**
- 위치: `src/models/streaming/layers/tsblock.py:454`
- 원인: `StreamingConv2d.forward` 가 `state_frames=None` 일 때 `x.shape[2]` 로만 fallback — `get_state_frames_context()` 를 읽지 않음. `StatefulCausalConv1d/2d` 및 `StatefulAsymmetricConv2d` 는 동일 자리에 thread-local fallback 을 가짐 (non-uniform)
- 영향 범위: LaCoSENet `_process_streaming_tsblocks` 가 `block(x, state)` 호출 시 `state_frames` 를 전달하지 않음 → lookahead 프레임이 state 업데이트에 포함 → 매 chunk ~2 frame state drift → offline vs streaming RMS ~0.08 누적 오차. BAFNetPlus `test_offline_vs_streaming_parity` 를 bit-close 로 통과시키려면 반드시 해결
- Stage 1 우회: `BAFNetPlusStreaming._process_encoder` 에서 `block(ts_out, state, state_frames=valid_frames)` 명시 전달. Criterion 3 는 LaCoSENet 과의 divergence 때문에 "same-order-of-magnitude" 로 tolerance 완화
- Stage 2 이전 fix: 4-line 추가 — `if state_frames is None: state_frames = get_state_frames_context()` 을 `StreamingConv2d.forward:454` 직전에 삽입. 그 후:
  1. LaCoSENet 기존 parity 7/7 재측정 — **변화 있음** 예상 (LaCoSENet 도 이제 올바른 streaming 을 수행) — Android 쪽 fixture 와의 bit-exact 관계가 유지되는지 확인
  2. Stage 1 `test_lacosenet_masking_branch_regression` tolerance 를 **RMS < 1e-5 로 복원** (현재 `< 0.1` 완화)
  3. Stage 1 `test_offline_vs_streaming_parity` 수치가 여전히 bit-close (RMS < 1e-4) 인지 재확인

**U1+U2 fix 순서**: Stage 2 kick-off 직전 별도 commit 으로 (a) U1 fix, (b) U2 fix, (c) LaCoSENet 7/7 재측정, (d) BAFNetPlus Stage 1 5/5 재측정 (tolerance 복원 포함) — 모두 통과해야 Stage 2 S2-1 스크립트 작성 진입.

**우선순위**: P0 — LaCoSENet 스택 자체의 올바름을 결정짓는 issue. Stage 4 Android parity 의 신뢰성이 이 두 fix 에 걸려 있음.

---

## 10. 열린 질문 (AskUserQuestion 초안)

본 계획서 착수 전 확정되어야 하는 결정 사항. 각 질문은 `AskUserQuestion` 으로 수집한 뒤 answer 기반으로 §1.1 scope 와 §8 Stage 계획을 조정한다.

### Q1. 배포할 ablation 모드를 고정하시겠습니까?

- **Option A**: `full` (A3, 제안 모델 — calibration + relative gain + 3ch alpha) **단일** 배포
- **Option B**: `full` + `no_calibration` (A1) **둘 다** 배포, 앱 측 toggle
- **Option C**: `mask_only_alpha` (A2, BAFNet baseline) 만 배포 — fusion 이 가장 작음
- **Option D**: 미확정, 포팅 중 모든 모드 지원

**기본 권장**: A — 단일 모델이 ONNX 1개 + config 1개로 Stage 3~5 가 단순. 결과가 시원찮으면 차차 B 로 확장.

### Q2. ONNX 배포 전략 — 단일 통합 세션 vs 분리 2 세션?

- **Option A**: 단일 통합 ONNX (mapping + masking + fusion 일체) — 권장
- **Option B**: Backbone 2 ONNX + host (Kotlin) fusion
- **Option C**: Stage 3 에서 둘 다 시도, 성능 좋은 쪽 채택

**기본 권장**: C — Stage 3 에 낮은 추가 비용으로 둘 다 빌드, verify 후 판정. Stage 4 에서 하나 선택.

### Q3. 2-마이크 입력 소스 — 어떤 BCS 를 사용하시겠습니까?

- **Option A**: Simulated BCS (ACS → low-pass filter) — 개발 단계에만 사용
- **Option B**: 유선 스테레오 마이크 (외장 USB/3.5mm, L=BCS/R=ACS 가장)
- **Option C**: 실 throat/bone mic (별도 조달, BT 또는 USB)
- **Option D**: 미정 — 포팅은 API 까지만, 실입력은 나중에

**기본 권장**: D — 계획서 시점엔 입력까지 책임지지 않고 `processChunk(bcs, acs)` API 를 노출하는 것 까지만. 데모 앱은 별도 세션.

### Q4. LaCoSENet 모듈을 유지하시겠습니까?

- **Option A**: 유지 — 기존 `android/lacosenet-streaming/` 그대로, BAFNetPlus 는 새 모듈로 병행 — 권장
- **Option B**: 제거 — BAFNetPlus 로 완전 대체 (`mask_only_alpha` 모드는 사실상 LaCoSENet 과 유사)
- **Option C**: 공통 모듈로 통합 — `android/streaming/` 로 재구성

**기본 권장**: A — parity 7/7 + 벤치마크 baseline 유지. 초기엔 병행, 나중에 통합 고려.

### Q5. FP16 배포 경로를 유지하시겠습니까?

- **Option A**: QDQ INT8 만 — 실배포 유일 경로
- **Option B**: QDQ INT8 + FP16 둘 다 — 디버깅·비교용
- **Option C**: QDQ INT8 + FP16 + FP32 CPU — 모든 fallback 제공

**기본 권장**: B — milestone.md 표준 따라 FP16 도 benchmark-app 에 유지하되, FP32 CPU 는 "동작 보증 X" 로 표기.

### Q6. Weight checkpoint 준비 상태 — `checkpoint_mapping`, `checkpoint_masking` 현황?

- **Option A**: 둘 다 준비됨 (경로 제공)
- **Option B**: 하나만 준비됨 (어느 쪽)
- **Option C**: 둘 다 없음 — `--no_checkpoint` 로 random weight 로 진행

**기본 권장**: C (planning 현 시점) — Stage 5 완료 후 실 weight 재export 세션 별도.

### Q7. 배포 예상 기기 범위 — Snapdragon 8 Elite 외 지원?

- **Option A**: Snapdragon 8 Elite (Hexagon V79) 단일
- **Option B**: Snapdragon 8 Gen 2/3 (Hexagon V73/V75) 도 포함
- **Option C**: 모든 Qualcomm QNN 지원 기기 + NNAPI fallback

**기본 권장**: A — LaCoSENet 과 동일하게 V79 최적화. 타 SoC 는 동작만 보장 (NNAPI/CPU fallback).

---

## Appendix A. 파일별 체크리스트

### A.1 Python 신규/수정 파일

| 파일 | 상태 | LoC | Stage | 비고 |
|---|---|---|---|---|
| `src/models/streaming/bafnetplus_streaming.py` | 신규 | 500~700 | S1 | LaCoSENet 템플릿 복제 |
| `src/models/streaming/onnx/export_bafnetplus_onnx.py` | 신규 | 400~500 | S3 | `BAFNetPlusStatefulExportableNNCore` 포함 |
| `scripts/make_bafnetplus_streaming_golden.py` | 신규 | 400~500 | S2 | 2-ch fixture + fusion intermediate dump |
| `scripts/compare_bafnetplus_batch_vs_stream.py` | 신규 | 80~120 | S1 | 배치 vs 스트림 오차 확인 |
| `scripts/probe_htp_support.py` | 신규 | 50~100 | S3 | op coverage 사전 확인 (옵션) |
| `conf/model/bafnetplus_streaming.yaml` | 신규 | ~30 | S1 | streaming export 파라미터 |
| `tests/test_bafnetplus_streaming.py` | 신규 | 150~250 | S1 | Python unit test |
| `src/models/streaming/converters/conv_converter.py` | 확인 (수정 없음 예상) | 0 | S1 | CausalConv1d/2d 변환 sanity check |

### A.2 Android 신규/수정 파일

| 파일 | 상태 | LoC | Stage | 비고 |
|---|---|---|---|---|
| `android/bafnetplus-streaming/build.gradle.kts` | 신규 | ~50 | S4 | Gradle 모듈 |
| `android/bafnetplus-streaming/consumer-rules.pro` | 신규 | ~20 | S4 | ORT + JTransforms keep |
| `android/bafnetplus-streaming/.../BAFNetPlusStreamingEnhancer.kt` | 신규 | ~350 | S4 | 메인 API |
| `android/bafnetplus-streaming/.../core/BAFNetPlusStreamingConfig.kt` | 신규 | ~50 | S4 | 설정 확장 |
| `android/bafnetplus-streaming/.../audio/DualChannelFeatureBuffer.kt` | 신규 (옵션) | ~100 | S4 | |
| `android/bafnetplus-streaming/.../core/BAFNetPlusInferenceResult.kt` | 신규 | ~60 | S4 | complex 출력 래핑 |
| `android/lacosenet-streaming/.../session/StatefulInference.kt` | 수정 | +60 / -20 | S4 | named input 제네릭화, LaCoSENet 호환 유지 |
| `android/lacosenet-streaming/.../backend/ExecutionBackend.kt` | 수정 (거의 없음) | 0 | S4 | run() 은 이미 Map 입력 |
| `android/settings.gradle.kts` | 수정 | +1 LoC | S4 | `include(":bafnetplus-streaming")` |
| `android/benchmark-app/build.gradle.kts` | 수정 | +5 LoC | S4 | `implementation(project(":bafnetplus-streaming"))` |
| `android/benchmark-app/src/main/assets/bafnetplus.onnx` | 신규 | ~10 MB | S3 | FP32 |
| `android/benchmark-app/src/main/assets/bafnetplus_qdq.onnx` | 신규 | ~9 MB | S3 | QDQ INT8 |
| `android/benchmark-app/src/main/assets/bafnetplus_streaming_config.json` | 신규 | ~6 KB | S3 | state 166 |
| `android/benchmark-app/src/androidTest/assets/bafnetplus_fixtures/` | 신규 | ~8 MB | S2 | 22 chunks |
| `android/benchmark-app/src/androidTest/.../parity/BAFNetPlusStatefulInferenceParityTest.kt` | 신규 | ~200 | S4 | 2 @Test |
| `android/benchmark-app/src/androidTest/.../parity/FixtureLoader.kt` | 수정 | +30 | S2 | BAFNetPlus manifest v2 지원 |
| `android/benchmark-app/src/androidTest/.../StreamingBenchmarkTest.kt` | 수정 | +100 | S4 | `benchmarkBafnetplusFullQdq` 등 |
| `android/benchmark-app/milestone.md` | 수정 | +80 | S5 | § 4 BAFNetPlus row |
| `android/README.md` | 수정 | +40 | S6 | BAFNetPlus 섹션 |
| `android/docs/ARCHITECTURE.md` | 수정 | +60 | S6 | 다이어그램 갱신 |

### A.3 프로젝트 루트 파일

| 파일 | 상태 | LoC | Stage | 비고 |
|---|---|---|---|---|
| `scripts/hooks/pre-push` | 수정 | +10 | S6 | BAFNetPlus parity class 추가 실행 |
| `docs/review/BAFNETPLUS_PORT_PLAN.md` | **본 파일** | ~1,700 | — | — |
| `docs/review/BAFNETPLUS_PORT_CLOSURE.md` | 신규 | ~200 | S6 | Stage 1~5 결과 요약 |

### A.4 LoC 증분 요약

| 영역 | 신규 | 수정 | 삭제 | 순증 |
|---|---|---|---|---|
| Python | ~1,600 | ~0 | 0 | ~1,600 |
| Kotlin | ~870 | ~165 | ~20 | ~1,015 |
| Gradle/Config | ~150 | ~10 | 0 | ~160 |
| 문서 | ~400 | ~180 | 0 | ~580 |
| **합계** | **~3,020** | **~355** | **~20** | **~3,355 LoC** |

→ LaCoSENet Android 모듈 전체가 2,382 LoC 였음을 고려하면, **~1.4× 규모의 증분**. 대부분 신규 (수정은 StatefulInference 제네릭화 정도).

---

## Appendix B. 재현 runbook (참조)

### B.1 Python 측 sanity (Stage 1 완료 후)

```bash
# Streaming wrapper 단위 테스트
cd /home/yskim/workspace/BAFNet-plus
pytest tests/test_bafnetplus_streaming.py -v

# 오프라인 vs 스트리밍 오차 측정 (ablation=full)
python scripts/compare_bafnetplus_batch_vs_stream.py \
    --mapping_chkpt none \
    --masking_chkpt none \
    --ablation_mode full \
    --n_seconds 2.0

# 예상 출력: RMS < 1e-4, max < 1e-3
```

### B.2 Fixture 생성 (Stage 2 완료 후)

```bash
python scripts/make_bafnetplus_streaming_golden.py \
    --output_dir android/benchmark-app/src/androidTest/assets/bafnetplus_fixtures \
    --ablation_mode full \
    --n_seconds 2.0 \
    --seed 42
# → 22 chunks, ~8 MB, manifest.json 생성
```

### B.3 ONNX export (Stage 3 완료 후)

```bash
python -m src.models.streaming.onnx.export_bafnetplus_onnx --no_checkpoint \
    --chunk_size 8 --encoder_lookahead 3 --decoder_lookahead 3 \
    --ablation_mode full \
    --output_dir android/benchmark-app/src/main/assets \
    --output_name bafnetplus.onnx \
    --quantize_qdq --qdq_activation_type QUInt8
# → bafnetplus.onnx (FP32), bafnetplus_qdq.onnx (QDQ INT8), bafnetplus_streaming_config.json
```

### B.4 Android parity 실행 (Stage 4 완료 후)

```bash
# SSH 터널 복구 (Windows 측 adb tcpip + forward + ssh -R)
~/platform-tools/adb connect 127.0.0.1:15555
~/platform-tools/adb devices   # "127.0.0.1:15555  device"

cd /home/yskim/workspace/BAFNet-plus/android
./gradlew :benchmark-app:assembleDebugAndroidTest
./gradlew :benchmark-app:connectedDebugAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.parity
# → 기존 7 + 신규 2 = 9/9 PASS 기대

# BAFNetPlus 단독
./gradlew :benchmark-app:connectedDebugAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.parity.BAFNetPlusStatefulInferenceParityTest
```

### B.5 Benchmark (Stage 5 완료 후)

```bash
# Cold-state (각 @Test 별도 프로세스)
./gradlew :benchmark-app:connectedDebugAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkBafnetplusFullQdq

# Same-process 3-test 시퀀스 (B1 fix 유효성 재확인)
~/platform-tools/adb shell am instrument -w \
    -e class com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkQnnHtpQdq,com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkBafnetplusFullQdq \
    com.lacosenet.benchmark/androidx.test.runner.AndroidJUnitRunner

# meminfo 시계열 (별 창에서)
while true; do
  ~/platform-tools/adb shell dumpsys meminfo com.lacosenet.benchmark | \
    awk '/TOTAL PSS|Native Heap/' ;
  date;
  sleep 0.5 ;
done > docs/review/logs/stage5_bafnetplus_meminfo_$(date +%F_%H%M).log
```

---

## Appendix B-2. Data flow 다이어그램

### B-2.1 런타임 dataflow (통합 세션 전략)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Application layer (Kotlin)                                              │
│                                                                          │
│  bcsSamples[1200]  acsSamples[1200]                                       │
│        │                 │                                               │
│        ▼                 ▼                                               │
│  bcsInputBuffer     acsInputBuffer                                        │
│        │                 │                                               │
│        ▼                 ▼                                               │
│  bcsStft (FFT)      acsStft (FFT)                                        │
│        │                 │                                               │
│        ▼                 ▼                                               │
│  (bcsMag, bcsPha)  (acsMag, acsPha)   each [201, 11]                     │
│        │                 │                                               │
│        └────────┬────────┘                                               │
│                 ▼                                                        │
│         dualFeatureBuffer (accumulate across chunks for lookahead)      │
│                 │                                                        │
│                 ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │  StatefulInference.run(mapOf(                           │           │
│   │      "bcs_mag", "bcs_pha", "acs_mag", "acs_pha",        │           │
│   │      ...state_mapping_* (80),                           │           │
│   │      ...state_masking_* (80),                           │           │
│   │      ...state_fusion_* (6))                             │           │
│   │  )                                                      │           │
│   └─────────────────────────────────────────────────────────┘           │
│                 │                                                        │
│                 ▼                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  QNN HTP / CPU / NNAPI backend                                     │ │
│  │   ┌───────────┐   ┌───────────┐                                    │ │
│  │   │ mapping   │   │ masking   │ (parallel ops in single graph)     │ │
│  │   │ Backbone  │   │ Backbone  │                                    │ │
│  │   └─────┬─────┘   └─────┬─────┘                                    │ │
│  │         │               │                                          │ │
│  │         └───────┬───────┘                                          │ │
│  │                 ▼                                                  │ │
│  │        ┌────────────────┐                                          │ │
│  │        │  Calibration   │ (5ch → 16 hidden → gain)                 │ │
│  │        │  encoder       │                                          │ │
│  │        └────────┬───────┘                                          │ │
│  │                 ▼                                                  │ │
│  │        ┌────────────────┐                                          │ │
│  │        │  Alpha conv    │ (3ch → 16ch → softmax)                   │ │
│  │        │  blocks × 4    │                                          │ │
│  │        └────────┬───────┘                                          │ │
│  │                 ▼                                                  │ │
│  │         est_com = α_bcs·bcs + α_acs·acs                            │ │
│  │                 │                                                  │ │
│  └─────────────────┼──────────────────────────────────────────────────┘ │
│                    ▼                                                     │
│  (estComReal, estComImag) each [1, 201, 11]                              │
│  + 166 next_state_* tensors                                              │
│                    │                                                     │
│                    ▼                                                     │
│  complex→magPha conversion                                               │
│                    ▼                                                     │
│  crop [201,11] → [201,8] (chunk_size frames)                             │
│                    ▼                                                     │
│  outputStft.istftStreaming (OLA tail 300 samples)                        │
│                    ▼                                                     │
│              enhanced[800]                                               │
└──────────────────────────────────────────────────────────────────────────┘
```

### B-2.2 State 흐름

```
 initialize() ──► zero-init 166 states in Buffer A
                       │
                       ▼
 processChunk k=0 ────► inputs = (mag/pha, state_A) ──► ORT run ──► outputs = (est_com, next_state_*)
                                                                         │
                                                                         ▼
                                                                Copy next → Buffer B
                                                                Swap: active = B
                       │
                       ▼
 processChunk k=1 ────► inputs = (mag/pha, state_B) ──► ORT run ──► outputs
                                                                         │
                                                                         ▼
                                                                Copy next → Buffer A
                                                                Swap: active = A
                       │
                       ▼
      ...
```

Double buffering 은 LaCoSENet `StatefulInference.kt:67-71` 의 동일 패턴. 확장 포인트는 buffer map 의 key space 가 166개로 증가.

---

## Appendix C. 참고 파일 좌표 요약

LaCoSENet 리뷰 `docs/review/REPORT.md` 에서 그대로 참조하는 핵심 파일과 해당 섹션:

| 파일 | 주요 라인 | 본 계획서 인용 섹션 |
|---|---|---|
| `src/models/bafnetplus.py` | 9-274 (전체) | §2 전체 |
| `src/models/backbone.py` | 22-52 (CausalConv1d/2d), 503-614 (Backbone) | §2.1, §2.3 |
| `src/models/streaming/lacosenet.py` | 92-726 | §3.1, §4.1 |
| `src/models/streaming/onnx/export_onnx.py` | 99-131 (state registry), 164-283 (StatefulExportableNNCore), 688-844 (QDQ for HTP) | §3.1, §4.3 |
| `src/models/streaming/utils.py` | 68-113 (StateFramesContext) | §4.1 |
| `android/lacosenet-streaming/.../session/StatefulInference.kt` | 49-447 | §5.2 |
| `android/lacosenet-streaming/.../StreamingEnhancer.kt` | 58-396 | §5.3 |
| `android/lacosenet-streaming/.../backend/ExecutionBackend.kt` | 58-158 | §5.4 |
| `android/lacosenet-streaming/.../core/StreamingConfig.kt` | 17-277 | §3.2, §4.3 |
| `android/benchmark-app/milestone.md` | 30-48 (best results), 148-210 (dual backbone) | §3.4, §7 |
| `docs/review/REPORT.md` | §0 (closure 상태), §3 (Stage 5 재측정), §4 Stage 6 (B1 fix 효과) | §3.4, §9.3, §9.7 |

---

## Appendix D. LaCoSENet 리뷰에서 물려받는 설계 제약

본 계획은 LaCoSENet Android 리뷰의 **관성** 위에 세운다. 다음 설계 결정은 특별한 이유가 없는 한 그대로 계승:

1. **State 이름: 알파벳 정렬 규약** — `REPORT.md` C1. `stateNames.sort()` + config assertion (`StatefulInference.kt:124-142`) 유지. BAFNetPlus 에서도 `state_mapping_*`, `state_masking_*`, `state_fusion_*` prefix 가 알파벳순으로 정렬되면 그대로 동작.
2. **Phase output: complex 모드** — `export_onnx.py:885` `"phase_output_mode": "complex"`. atan2 는 host 측. BAFNetPlus 도 mapping/masking 각각 phase_real/imag 출력 유지하거나, 단일 통합 시 est_com_real/est_com_imag 로 통합.
3. **STFT: center=True, periodic Hann, FFT 기반 (JTransforms)** — `REPORT.md` Stage 3 A-series 결과. BAFNetPlus 도 동일 STFT 적용. 채널별 독립 인스턴스.
4. **iSTFT: OLA tail carry-over 300 samples** — `StftProcessor.istftStreaming`. BAFNetPlus 에서도 동일. 최종 est_com 1개이므로 iSTFT 1회.
5. **B1 pattern: `session.run(inputs).use { }` 강제** — `ExecutionBackend.kt:104, 141-152`. BAFNetPlus 추가 경로에도 일관 적용.
6. **C2: streaming_config.json sentinel + validate()** — `StreamingConfig.kt:64-87`. BAFNetPlus config 에 `input_channels`, `ablation_mode` 등 신규 필드도 동일 validate 패턴 적용.
7. **H1: NaN/Inf 입력 거절** — `StreamingEnhancer.kt:222-226`. BAFNetPlus 의 `processChunk(bcs, acs)` 에서도 **양쪽 다** 검증.
8. **D4: abiFilters arm64-v8a 선언** — `build.gradle.kts:20-24`. bafnetplus-streaming 모듈에도 동일 선언.
9. **QDQ INT8 QUInt8 (not QUInt16)** — milestone.md Appendix B, REPORT.md Stage 6. PReLU 포함 그래프는 QUInt8 필수.
10. **Pre-push hook 구조** — `scripts/hooks/pre-push` 가 parity suite 를 `am instrument` 로 실행. BAFNetPlus parity 추가 시 해당 hook 에 test class 추가만.

---

## Appendix E. 본 계획서의 한계 및 가정 재정리

- **본 문서는 사전 계획서** — 실제 코드 변경, 빌드, 기기 실행은 **수행하지 않음**. LoC 추정·latency 예측은 모두 LaCoSENet 실적 및 BAFNetPlus 구조 분석에 기반한 범위 추정.
- **구조적 불확실성은 Stage 3 verify 전까지 해소 불가** — 특히 HTP op coverage, VTCM, 통합 그래프 finalization 성공 여부는 실기기 실행 없이 결론 불가.
- **Weight 학습 완료가 포팅 순서의 선행 조건은 아님** — LaCoSENet 도 random weight 로 완수. BAFNetPlus 역시 `--no_checkpoint` 로 Stage 5 까지 진행 가능.
- **Ablation 모드 선택이 가장 큰 스코프 변동 요인** — Q1 answer 에 따라 본 계획의 Stage 2~5 가 1~3 ablation 모드 × 반복으로 확장될 수 있음.
- **Open questions §10 Q1, Q2 미해결 시 Stage 1 착수 비추천** — 반드시 Stage 1 킥오프 전 `AskUserQuestion` 실행 및 answer 기록.

---

*본 계획서는 BAFNetPlus Android 포팅 Stage 1 착수 전 승인 대상이며, 승인 후 Stage 별 개별 세션으로 실행한다. LaCoSENet `docs/review/REPORT.md` 와 본 계획서 `docs/review/BAFNETPLUS_PORT_PLAN.md` 는 append-only 이력으로 유지된다.*

---

## Appendix F. 글로서리 (LaCoSENet 리뷰에서 계승한 용어)

| 용어 | 정의 | 출처 |
|---|---|---|
| **parity 7/7** | LaCoSENet Android 포팅에서 float32 머신 epsilon 수준으로 통과하는 7개 androidTest @Test (`StftParityTest` 3 + `IstftParityTest` 2 + `StatefulInferenceParityTest` 2). Pre-push hook 이 자동 검증 | REPORT.md §0 |
| **fixture** | 고정 seed Gaussian 2.0초 오디오로 생성한 per-chunk intermediate tensor bundle (3.8 MB, 22 chunks). Python 참조 경로의 "정답 streaming" | REPORT.md Stage 2 |
| **Stage N** | 포팅 작업 분할 단위. LaCoSENet 은 Stage 1 (audit) + Stage 2~6 (수정). BAFNetPlus 는 Stage 1~6 (구현) | REPORT.md §4 |
| **P0/P1/P2** | 결함 심각도. P0 = 즉시 차단, P1 = 반드시 해결, P2 = 점진 개선. 본 계획서에서도 위험 등급으로 사용 | REPORT.md §2 |
| **QDQ INT8** | Quantize-Dequantize INT8 quantization. HTP 전용 가속 경로 | milestone.md |
| **HTP** | Hexagon Tensor Processor (Qualcomm NPU). Snapdragon 8 Elite 의 V79 세대 | milestone.md |
| **VTCM** | Vector Tightly Coupled Memory. HTP 내부 캐시 (8~16 MB) | milestone.md 1.3 |
| **Pre-push hook** | `scripts/hooks/pre-push`. `git push` 직전 parity suite 자동 실행, 실패 시 push 차단 | REPORT.md Stage 6 |
| **B1 leak** | ORT `OrtSession.Result` 미close 로 인한 native heap 누수. Stage 6 에서 `.use { }` 패턴으로 해소 | REPORT.md §2 B1 |
| **Dual Concurrent** | 두 ORT session 을 동시 실행하여 HTP 파이프라이닝 이득을 노리는 패턴. milestone.md § 2 Dual Backbone | milestone.md 2.1 |
| **StateFramesContext** | Lookahead 프레임이 streaming state 를 contaminate 하지 않도록 state 업데이트 범위를 현재 chunk 프레임으로 제한하는 컨텍스트 매니저. C3 ablation 대상 | `src/models/streaming/utils.py:68-113` |
| **export_time_frames** | ONNX 모델의 입력 시간 축 차원 = chunk_size + encoder_lookahead = 8 + 3 = 11 (Large 구성) | streaming_config.json |
| **samples_per_chunk** | 한 번의 `processChunk` 가 요구하는 입력 샘플 수. `(total_frames - 1) * hop_size + win_size/2 = 1200` | `StreamingConfig.kt:92-96` |
| **output_samples_per_chunk** | 한 번의 `processChunk` 가 반환하는 enhanced 샘플 수. `chunk_size * hop_size = 800` | 동일 |
| **input_lookahead_frames** | encoder 를 지연시켜 future context 를 확보하는 프레임 수. Python 기준 `= encoder_lookahead` (C3 정정 이후) | `StreamingConfig.kt:196-213` |
| **ablation_mode** | BAFNetPlus 에서 fusion 구성 요소 on/off 조합. `full / no_calibration / mask_only_alpha / common_gain_only` | `bafnetplus.py:10-15` |
| **infer_type** | Backbone 의 출력 해석. `masking` = mask × input_mag / `mapping` = mask 자체를 est_mag 로 사용 | `backbone.py:544, 605-607` |
| **BCS / ACS** | Body-Conducted Speech (throat/bone mic) / Air-Conducted Speech (일반 mic). BAFNetPlus 는 둘을 결합해 잡음 환경에서 품질 향상 | `bafnetplus.py` docstring 영역 |
| **fusion** | 두 Backbone 출력을 결합하는 BAFNetPlus 후단 네트워크. calibration_encoder + alpha_convblocks + softmax | `bafnetplus.py:96-135` |
| **calibration gain** | frame-wise common_log_gain + relative_log_gain 으로 BCS/ACS 에너지 균형을 조정 | `bafnetplus.py:195-216` |
| **alpha blending** | TF-wise softmax(α) 로 est_com = α_bcs · bcs_com + α_acs · acs_com | `bafnetplus.py:263-271` |

---

## Appendix G. Stage 간 의존성 매트릭스 (세부)

| 의존 요소 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | Stage 6 |
|---|---|---|---|---|---|---|
| Python streaming wrapper 완성 | produce | consume | consume | — | — | — |
| Golden fixture (.bin + manifest) | — | produce | — | consume | — | — |
| ONNX 모델 + streaming_config.json | — | — | produce | consume | — | — |
| Python 참조 수치 (ORT CPU) | — | produce | consume (verify) | consume (parity) | — | — |
| Kotlin `StatefulInference` 제네릭화 | — | — | — | produce | consume | consume |
| Kotlin `BAFNetPlusStreamingEnhancer` | — | — | — | produce | consume | consume |
| 실기기 benchmark 결과 | — | — | — | — | produce | consume (milestone.md) |
| Pre-push hook 확장 | — | — | — | — | — | produce |
| LaCoSENet parity 7/7 (기존, 유지) | guard | guard | guard | guard | guard | guard |

**해석**: Stage 1 실패 시 Stage 2~6 전부 blocked. Stage 3 verify 실패 시 Stage 4~6 blocked. Stage 4 parity 실패 시 Stage 5~6 blocked. Stage 5 는 실기기 성능 평가로, 실패해도 "배포 불가" 결론만 낳을 뿐 Stage 6 은 진행 가능 (단, milestone.md 에 그 결과 반영).

**병행 가능한 작업**:
- Stage 1 진행 중 Stage 2 fixture 생성 스크립트 **설계** 는 가능 (구현은 Stage 1 완료 후)
- Stage 3 QDQ verify 진행 중 Stage 4 Kotlin 제네릭화 skeleton **설계** 는 가능
- Stage 5 측정 진행 중 Stage 6 문서 **초안** 작성 가능

---

## Appendix H. 본 계획서와 LaCoSENet REPORT.md 차이점 요약

| 축 | LaCoSENet REPORT.md | 본 계획서 (BAFNETPLUS_PORT_PLAN.md) |
|---|---|---|
| **방향성** | Audit 후 수정 (retrospective) | Forward-looking 구현 계획 |
| **시작점** | 이미 작성된 Android 포팅본의 결함 발굴 | 검증된 LaCoSENet 스택 위에서 확장 |
| **결함 분류** | P0/P1/P2 등급, 8축 A~H | 위험 등급만 차용, 축은 없음 (전진 설계이므로) |
| **Stage 목적** | 리뷰 (Stage 1) + 수정 (Stage 2~6) | 구현 (Stage 1~6) |
| **수치 parity** | 기존 구현이 깨져있어 fix 가 목표 | 기존 patterns 을 계승해 parity 유지가 목표 |
| **산출물** | Audit 리포트 (후속 수정 작업 분해) | 구현 계획서 (후속 구현 작업 분해) |
| **실기기 실측** | Stage 5 재측정으로 확인 | Stage 5 에서 처음 측정 |
| **Pre-push hook** | Stage 6 에서 신설 | 기존 hook 에 BAFNetPlus 항목 추가 |
| **문서 최종 상태** | 종결 (2026-04-19 closed) | 착수 전 승인 대기 |

---

*작성: 2026-04-19. 승인 및 Stage 1 킥오프 결정은 Open Questions §10 해소 후.*
