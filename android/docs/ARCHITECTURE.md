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
