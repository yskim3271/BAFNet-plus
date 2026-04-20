# Android Codebase Architecture

**lacosenet-streaming** - 실시간 음성 향상을 위한 Android 라이브러리

---

## 1. 프로젝트 구조

```
android/
├── lacosenet-streaming/          # 라이브러리 모듈 (AAR)
│   └── src/main/kotlin/com/lacosenet/streaming/
│       ├── StreamingEnhancer.kt        # Public API (진입점)
│       ├── audio/                       # 오디오 처리
│       │   ├── AudioBuffer.kt          # 오디오 링 버퍼
│       │   └── StftProcessor.kt        # STFT/iSTFT 변환
│       ├── backend/                     # 실행 백엔드
│       │   ├── ExecutionBackend.kt     # 백엔드 인터페이스
│       │   ├── BackendSelector.kt      # 자동 백엔드 선택
│       │   ├── QnnBackend.kt           # Qualcomm NPU (QNN HTP)
│       │   ├── NnapiBackend.kt         # Android NNAPI
│       │   └── CpuBackend.kt           # CPU 폴백
│       ├── core/                        # 핵심 유틸리티
│       │   ├── StreamingConfig.kt      # JSON 설정 파싱
│       │   └── StreamingState.kt       # 상태 관리 + 데이터 클래스
│       └── session/                     # 추론 세션
│           └── StatefulInference.kt    # 통합 모델 추론
│
├── benchmark-app/                # 벤치마크 애플리케이션
│   └── src/
│       ├── main/
│       │   ├── assets/                 # ONNX 모델 + 설정 파일
│       │   └── kotlin/.../BenchmarkActivity.kt
│       └── androidTest/
│           └── kotlin/.../StreamingBenchmarkTest.kt
│
└── docs/                         # 문서
    └── ARCHITECTURE.md           # 이 문서
```

---

## 2. 핵심 컴포넌트

### 2.1 StreamingEnhancer (Public API)

```kotlin
// 사용 예시
val enhancer = StreamingEnhancer(context)
val result = enhancer.initialize("model_int8.onnx", "streaming_config.json")

if (result.success) {
    // 스트리밍 처리 (16kHz, Float PCM)
    val enhanced = enhancer.processChunk(audioChunk)  // 200ms 단위
}

enhancer.release()
```

**주요 메서드:**
| 메서드 | 설명 |
|--------|------|
| `initialize()` | 모델 로드 및 백엔드 초기화 |
| `processChunk()` | 오디오 청크 처리 (STFT → 추론 → iSTFT) |
| `reset()` | 스트리밍 상태 초기화 |
| `release()` | 리소스 해제 |

### 2.2 Backend 계층

```
┌─────────────────────────────────────────────────────┐
│                  BackendSelector                     │
│         (디바이스 능력 기반 자동 선택)                │
└─────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ QnnBackend│   │NnapiBackend│  │CpuBackend│
   │ (NPU)    │   │ (NNAPI)  │   │ (CPU)    │
   └──────────┘   └──────────┘   └──────────┘
```

**선택 우선순위:** QNN HTP > NNAPI > CPU

| 백엔드 | 조건 | 특징 |
|--------|------|------|
| QNN_HTP | Qualcomm SoC + INT8 모델 | 최고 성능, NPU 가속 |
| NNAPI | Android 8.1+ (API 27) | 범용 하드웨어 가속 |
| CPU | 항상 가용 | 폴백, 디버깅용 |

### 2.3 Session 계층

```
┌─────────────────────────────────────────────────────┐
│                 StreamingEnhancer                    │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ StatefulInference│
              │ (통합 모델)       │
              └──────────────────┘
```

---

## 3. 데이터 흐름

### 3.1 데이터 흐름

```
Audio Input (16kHz PCM)
    │
    ▼
┌─────────────┐
│ AudioBuffer │  ← 링 버퍼 (청크 누적)
└─────────────┘
    │
    ▼
┌─────────────┐
│StftProcessor│  ← STFT (시간 → 주파수)
└─────────────┘
    │
    ▼
┌─────────────┐
│FeatureBuffer│  ← Lookahead 버퍼링
└─────────────┘
    │
    ▼
┌─────────────────┐
│StatefulInference│  ← ONNX 추론 (mag, pha, states)
└─────────────────┘
    │
    ▼
┌─────────────┐
│StftProcessor│  ← iSTFT (주파수 → 시간)
└─────────────┘
    │
    ▼
Enhanced Audio Output
```

---

## 4. 설정 파일 (streaming_config.json)

```json
{
  "model_info": {
    "name": "prk_1117_1",
    "model_path": "model_int8.onnx",
    "quantization": "int8_qdq"
  },
  "stft_config": {
    "n_fft": 400,
    "hop_size": 100,
    "sample_rate": 16000
  },
  "streaming_config": {
    "chunk_size_frames": 32,    // 32 frames × 6.25ms = 200ms
    "decoder_lookahead": 7      // 알고리즘 지연
  },
  "qnn_config": {
    "htp_performance_mode": "burst",
    "context_cache_enabled": true
  }
}
```

---

## 5. 주요 최적화

### 5.1 Tensor Pooling

```kotlin
// StatefulInference.kt - Zero-copy JNI 전송
private var magBuffer: ByteBuffer? = null  // Direct ByteBuffer

magBuffer = ByteBuffer.allocateDirect(size * 4)
    .order(ByteOrder.nativeOrder())

magTensor = OnnxTensor.createTensor(env, magBuffer.asFloatBuffer(), shape)
```

### 5.2 Double Buffering

```kotlin
// 상태 텐서 스왑 (복사 없음)
private var stateBuffersA = mutableMapOf<String, ByteBuffer>()
private var stateBuffersB = mutableMapOf<String, ByteBuffer>()
private var useBufferA = true

fun updateStates() {
    // 비활성 버퍼에 쓰기
    val inactive = if (useBufferA) stateBuffersB else stateBuffersA
    // ... 데이터 복사 ...
    useBufferA = !useBufferA  // 스왑
}
```

### 5.3 QNN Context Caching

```kotlin
// QnnBackend.kt - 첫 로드 후 컴파일된 컨텍스트 캐싱
if (File(cachePath).exists()) {
    session = env.createSession(cachePath, options)  // 빠른 로드
} else {
    session = env.createSession(modelPath, options)  // 컴파일 + 캐시
}
```

---

## 6. 파일별 책임

LOC 수치는 2026-04-19 Stage 3 완료 시점 기준 (`wc -l`).

| 파일 | LOC | 책임 |
|------|-----|------|
| `StatefulInference.kt` | 406 | 통합 모델 추론, 상태 관리 |
| `StreamingEnhancer.kt` | 383 | Public API, 파이프라인 오케스트레이션 |
| `StftProcessor.kt` | 202 | STFT/iSTFT 변환 (JTransforms FFT) |
| `QnnBackend.kt` | 257 | Qualcomm NPU 백엔드 |
| `StreamingConfig.kt` | 241 | JSON 설정 파싱 |
| `BackendSelector.kt` | 180 | 백엔드 자동 선택 |
| `ExecutionBackend.kt` | 163 | 백엔드 인터페이스 |
| `AudioBuffer.kt` | 156 | 오디오/피처 버퍼 |
| `NnapiBackend.kt` | 121 | NNAPI 백엔드 |
| `CpuBackend.kt` | 87 | CPU 백엔드 |
| `StreamingState.kt` | 82 | 상태 관리 + InferenceResult |

**총 라인 수:** 2,278 LOC

---

## 7. 의존성

```kotlin
// build.gradle.kts
dependencies {
    // ONNX Runtime with QNN EP
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.2")

    // JSON parsing
    implementation("com.google.code.gson:gson:2.10.1")

    // Coroutines (optional)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

---

## 8. 성능 지표

상세 수치는 단일 진실 소스 `android/benchmark-app/milestone.md` 참조. 아래는 2026-04-18 Galaxy S25 Ultra(Snapdragon 8 Elite / Hexagon V79) 재현 측정본.

| 항목 | 값 | 출처 |
|------|-----|------|
| 청크 크기 | 50ms (8 frames × hop 6.25ms) | `streaming_config.json: chunk_size_frames=8` |
| 실시간 버짓 | 50ms (청크와 동일) | 스트리밍 realtime 제약 |
| 알고리즘 지연 | ~31ms | `encoder_lookahead * hop + win_size/2/sr` = 18.75ms + 12.5ms |
| 모델 | Large (`time_block_kernel=[3,5,7,11]`) | 번들된 `model.onnx` (FP32, 5.9MB) / `model_qdq.onnx` (INT8 QDQ, 4.0MB) |
| CPU (4-thread FP32) | **46.3ms** Mean / 49.1ms P95 | milestone.md Large 34.7ms 대비 +33% |
| QNN HTP FP16 | **27.3ms** Mean / 32.5ms P95 | milestone.md Large 29.8ms ±9% ✓ |
| QNN HTP QDQ INT8 | **10.4ms** Mean / 11.2ms P95 | milestone.md Large 10.1ms ±3% ✓ |
| Dual concurrent QDQ (mapping+masking) | 29.4ms WallClock / 64.6ms P95 | **milestone.md 15.2ms 대비 2배 회귀 — F-bench-new-2** |

상세 로그 및 신규 P1 이슈는 `docs/review/REPORT.md` §2 F축 / §3 참조.

---

## 9. 확장 포인트

### 새 백엔드 추가

```kotlin
class NewBackend : BaseExecutionBackend() {
    override val type = BackendType.NEW_TYPE
    override val isAvailable: Boolean = checkAvailability()

    override fun initialize(...): BackendInitResult { ... }
    override fun createSessionOptions(config): OrtSession.SessionOptions { ... }
}
```

### 설정 확장

```kotlin
// StreamingConfig.kt에 새 필드 추가
data class ModelInfo(
    // ... 기존 필드 ...
    @SerializedName("new_feature")
    val newFeature: Boolean = false
)
```

---

## 10. BAFNetPlus (Unified dual-channel graph)

LaCoSENet 과 공존하는 독립 모듈 `bafnetplus-streaming/`. BCS (골전도) + ACS (기전도) 2채널 입력을 받아 통합 ONNX 그래프 하나에서 mapping + masking + calibration + fusion (alpha/gamma) 을 수행. ORT session / state double-buffer / backend 는 `lacosenet-streaming` 의 인프라 (`StatefulInference`, `AudioBuffer`, `StftProcessor`, `BackendSelector`) 를 재사용하여 중복 구현을 피한다.

### 10.1 모듈 구조

```
android/bafnetplus-streaming/
└── src/main/kotlin/com/bafnetplus/streaming/
    ├── BAFNetPlusStreamingEnhancer.kt   # Public API (dual-channel 진입점)
    ├── audio/
    │   └── DualChannelFeatureBuffer.kt  # BCS/ACS 프레임 버퍼 (mag+pha 각 2채널)
    └── core/
        └── BAFNetPlusInferenceResult.kt # 출력 데이터 클래스 (est_mag + com_real + com_imag)
```

Cross-module 의존: `com.lacosenet.streaming.{audio.AudioBuffer, audio.StftProcessor, backend.*, core.*, session.StatefulInference}`.

### 10.2 파이프라인 다이어그램

```
BCS PCM (16 kHz) ──┐
ACS PCM (16 kHz) ──┤
                   ▼
         ┌─────────────────────┐
         │ AudioBuffer × 2      │  링 버퍼 (BCS / ACS 독립)
         └─────────────────────┘
                   ▼
         ┌─────────────────────┐
         │ StftProcessor × 2   │  BCS / ACS 독립 STFT
         │  (n_fft=400,        │   streaming context 각각 유지
         │   hop=100,          │
         │   center=True)      │
         └─────────────────────┘
                   ▼
         ┌─────────────────────────────────┐
         │ DualChannelFeatureBuffer        │  (bcs_mag, bcs_pha,
         │  exportTimeFrames = 8+3+3       │   acs_mag, acs_pha) frame-aligned
         └─────────────────────────────────┘
                   ▼
         ┌──────────────────────────────────────────┐
         │ StatefulInference (unified BAFNetPlus)   │
         │  inputs:  bcs_mag, bcs_pha, acs_mag,     │
         │           acs_pha                        │
         │  states:  166 (alpha 4 + cal 2 +         │
         │           mapping 80 + masking 80)       │
         │  outputs: est_mag, est_com_real,         │
         │           est_com_imag                   │
         └──────────────────────────────────────────┘
                   ▼
         atan2(est_com_imag, est_com_real) → est_pha   (host-side)
                   ▼
         ┌─────────────────────┐
         │ StftProcessor.iSTFT │  BCS stream 이 OLA 버퍼 담당
         └─────────────────────┘
                   ▼
          Enhanced PCM (800 samples / chunk, 50 ms)
```

### 10.3 State ordering invariant

BAFNetPlus 의 166 state tensor 는 ONNX 그래프 내부에서 lexicographically sorted. `StatefulInference` 가 Kotlin `sort()` 로 정렬하며 Python `sorted()` 와 일치 — Stage 1 parity 에서 bit-identical 검증 완료 (RMS = 0.0).

### 10.4 LaCoSENet 과의 차이

| 항목 | LaCoSENet | BAFNetPlus |
|---|---|---|
| 입력 채널 | 1 (single PCM) | 2 (BCS + ACS) |
| ONNX session | 1 (통합) 또는 2 (mapping/masking 분리) | 1 (unified graph) |
| States | mapping 80 + masking 80 ≈ 80~160 | 166 (alpha 4 + cal 2 + mapping 80 + masking 80) |
| Fusion | host-side (별도 fusion net) | graph 내부 (alpha/gamma + calibration) |
| Phase 복원 | atan2 (est_com_*) 또는 pha copy | atan2 (est_com_real, est_com_imag) |
| 실측 Mean (SM-S938N QDQ HTP) | 6.2 ms (single) / 10.2 ms (dual concurrent) | **13.4 ms** |
| Budget utilization (P95) | 0.23 (single) / 0.34 (dual) | 0.28 |

### 10.5 LOC 표 (Stage 1–6 누적, 2026-04-21 기준)

`wc -l` 기준 BAFNetPlus 포팅으로 추가된 코드 / 문서.

| Stage | 카테고리 | 파일 | LOC |
|---|---|---|---|
| Stage 1 | parity (코드 리뷰) | Python 측 `tests/test_bafnetplus*.py` (플랜 §Stage 1) | (기존 소스 측 테스트) |
| Stage 2 | fixture + smoke | `parity/FixtureLoader.kt` | 306 |
| Stage 2 | fixture + smoke | `parity/BafnetPlusFixtureSmokeTest.kt` | 87 |
| Stage 3 | ONNX export | `src/models/streaming/onnx/export_bafnetplus_onnx.py` | (Python, 플랜 §Stage 3) |
| Stage 3 | HTP probe | `parity/BafnetPlusHtpProbeTest.kt` | 206 |
| Stage 4 | streaming module | `bafnetplus-streaming/…/BAFNetPlusStreamingEnhancer.kt` | 383 |
| Stage 4 | streaming module | `bafnetplus-streaming/…/audio/DualChannelFeatureBuffer.kt` | 120 |
| Stage 4 | streaming module | `bafnetplus-streaming/…/core/BAFNetPlusInferenceResult.kt` | 67 |
| Stage 4 | parity (Kotlin) | `parity/BAFNetPlusEnhancerTest.kt` | 178 |
| Stage 4 | parity (Kotlin) | `parity/BAFNetPlusStatefulInferenceParityTest.kt` | 314 |
| Stage 5 | benchmark | `StreamingBenchmarkTest.kt` Repeat10x + SessionLifecycle | +218 |
| Stage 5 | milestone doc | `benchmark-app/milestone.md § 4` | +120 |
| Stage 6 | docs + hook | README + ARCHITECTURE + CLOSURE + Conclusion + pre-push | ~340 |
| **Kotlin 소계** | (runtime + tests + bench) | | **1,673 LOC** |
| **Doc 소계** | (milestone + closure + README/ARCH) | | **~460 LOC** |

Python 측 증분 (export + parity tests) 은 플랜 §Stage 3 / §Stage 1 참조.

### 10.6 Parity 커버리지

`com.lacosenet.benchmark.parity` 패키지의 총 **16** 테스트 — LaCoSENet 7 (STFT/iSTFT/StatefulInference) + BAFNetPlus 9 (FixtureSmoke 2 + HtpProbe 2 + Enhancer 2 + StatefulInferenceParity 3). Stage 5 종료 시점 기준 16/16 PASS. `scripts/hooks/pre-push` 가 `git push` 전 전체 실행을 강제한다.

### 10.7 성능 (Stage 5 실측)

| Scenario | Mean | P95 | P99 | Peak PSS | Budget % |
|---|---|---|---|---|---|
| A — cold-state QDQ | 13.4 ms | 14.0 ms | 14.9 ms | 440 MB | 27 % |
| B — same-process 10× | 13.4 ms | 14.3 ms | 17.3 ms | 440 MB | 27 % |
| C — cross-load (LaCoSENet dual → BAFNet+) | 13.7 ms | 14.7 ms | 16.0 ms | 452 MB (+12) | 27 % |
| G — session lifecycle ×10 | init 8.3–9.0 s / run 27 ms / close 200–230 ms | — | — | 443 → 260 MB | — |

상세 + 재현 명령은 `android/benchmark-app/milestone.md § 4`, closure 문서는 `docs/review/BAFNETPLUS_PORT_CLOSURE.md`.
