# LaCoSENet Streaming Android

LaCoSENet 기반 실시간 음성 향상(Speech Enhancement) Android 라이브러리 및 벤치마크 앱.

ONNX Runtime을 사용하여 Qualcomm QNN(NPU), NNAPI, CPU 백엔드를 자동 선택하며, 스트리밍 추론을 통해 실시간 잡음 제거를 수행한다.

## 프로젝트 구조

```
android/
├── build.gradle.kts                     # 루트 빌드 설정
├── settings.gradle.kts                  # Gradle 모듈 설정
├── gradle.properties
│
├── lacosenet-streaming/                 # 핵심 라이브러리 모듈
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       └── kotlin/com/lacosenet/streaming/
│           ├── StreamingEnhancer.kt     # 메인 퍼블릭 API
│           ├── audio/                   # 오디오 처리 (STFT, 버퍼링)
│           ├── backend/                 # 추론 백엔드 (QNN, NNAPI, CPU)
│           ├── core/                    # 설정, 상태
│           └── session/                 # 추론 세션
│
├── benchmark-app/                       # 벤치마크 인스트루먼트 테스트
│   ├── build.gradle.kts
│   └── src/
│       └── androidTest/.../StreamingBenchmarkTest.kt
│
└── docs/                                # 설계 문서
    ├── ARCHITECTURE.md
    ├── MOBILE_REFERENCE.md
    └── QNN_TROUBLESHOOTING.md
```

## 요구 사항

| 항목 | 버전 |
|------|------|
| Android SDK (min) | 26 (Android 8.0) |
| Android SDK (target/compile) | 35 (Android 15) |
| Kotlin | 1.9.20 |
| ONNX Runtime | 1.24.2 (`onnxruntime-android-qnn`) |
| JVM Target | 17 |
| Gradle Plugin | 8.2.0 |

## 타겟 하드웨어

- **Primary**: Samsung Galaxy S23+ (Snapdragon 8 Gen 2 / SM8550)
- **Compatible**: Qualcomm Snapdragon SoC 탑재 기기 전반

## 설치

프로젝트에 라이브러리 모듈을 추가한다:

```kotlin
// settings.gradle.kts
include(":lacosenet-streaming")
project(":lacosenet-streaming").projectDir = file("path/to/lacosenet-streaming")

// app/build.gradle.kts
dependencies {
    implementation(project(":lacosenet-streaming"))
}
```

## 사용법

### 기본 사용

```kotlin
import com.lacosenet.streaming.StreamingEnhancer

class AudioProcessor(context: Context) {
    private val enhancer = StreamingEnhancer(context)

    fun initialize(): Boolean {
        val result = enhancer.initialize(
            modelPath = "model.onnx",
            configPath = "streaming_config.json"
        )
        if (result.success) {
            Log.i(TAG, "Backend: ${result.backend}, Latency: ${result.latencyMs}ms")
        }
        return result.success
    }

    fun processAudio(samples: FloatArray): FloatArray? {
        // PCM float 16kHz 샘플을 입력하면 향상된 오디오를 반환.
        // 버퍼링 중이면 null 반환.
        return enhancer.processChunk(samples)
    }

    fun onNewUtterance() {
        enhancer.reset()  // 새로운 발화 시 상태 초기화
    }

    fun release() {
        enhancer.release()
    }
}
```

### 백엔드 강제 지정

```kotlin
import com.lacosenet.streaming.backend.BackendType

val result = enhancer.initialize(
    modelPath = "model.onnx",
    configPath = "streaming_config.json",
    forceBackend = BackendType.QNN_HTP
)
```

### 디바이스 정보 조회

```kotlin
val info = enhancer.getDeviceInfo()
// Keys: manufacturer, model, hardware, board, sdk_int,
//       is_qualcomm, qnn_available, nnapi_available, nnapi_deprecated
```

## 아키텍처

### 전체 데이터 흐름

```
PCM Input (float, 16kHz)
    │
    ▼
┌─────────────┐
│ AudioBuffer  │  링 버퍼: 샘플 축적
└──────┬──────┘
       ▼
┌──────────────┐
│ StftProcessor │  Host FP32 STFT (Hann window, power-law compression)
└──────┬───────┘
       ▼
┌───────────────┐
│ FeatureBuffer  │  스펙트럼 프레임 버퍼링 (decoder lookahead)
└──────┬────────┘
       ▼
┌──────────────────────────────────────────────┐
│              Session Layer                    │
│              StatefulInference                │
│              (통합 ONNX 모델)                 │
└─────────────────────┬────────────────────────┘
                     ▼
┌──────────────────────────────────────────────┐
│            ExecutionBackend                   │
├──────────────┬──────────────┬────────────────┤
│  QnnBackend   │ NnapiBackend │   CpuBackend   │
│  (Hexagon NPU)│   (NNAPI)    │   (CPU)        │
└──────────────┴──────────────┴────────────────┘
                     │
                     ▼
          Mask Application: est_mag = mag × mask
                     │
                     ▼
         StftProcessor.istftStreaming()
                     │
                     ▼
           Enhanced PCM Output
```

### 백엔드 선택 우선순위

`BackendSelector`가 디바이스 능력에 따라 자동으로 최적 백엔드를 선택한다:

1. **QNN HTP** -- Qualcomm Hexagon NPU. INT8 양자화 모델 필요. Context binary 캐싱으로 재시작 시 빠른 로드.
2. **NNAPI** -- Android Neural Networks API. Android 15(API 35)부터 deprecated.
3. **CPU** -- ONNX Runtime 최적화 커널. 항상 사용 가능한 최종 폴백.

초기화 시 상위 백엔드가 실패하면 자동으로 다음 백엔드로 폴백한다.

### 패키지별 구성

#### `audio/` -- 오디오 신호 처리

| 클래스 | 역할 |
|--------|------|
| `AudioBuffer` | Thread-safe 링 버퍼. `ReentrantLock` 기반 동시성 제어. PCM 샘플 축적 및 청크 단위 추출. |
| `FeatureBuffer` | 주파수 도메인 프레임 버퍼. Magnitude/Phase 쌍 저장. Decoder lookahead 프레임 관리. (`AudioBuffer.kt` 내 정의) |
| `StftProcessor` | 순수 Kotlin STFT/iSTFT. Hann window, power-law compression (`c=0.3`), overlap-add 복원. 스트리밍 모드에서 context 보존. |

#### `backend/` -- 실행 백엔드

| 클래스 | 역할 |
|--------|------|
| `ExecutionBackend` | 백엔드 공통 인터페이스. `initialize()`, `run()`, `release()`, `createSessionOptions()`. |
| `BaseExecutionBackend` | 공통 구현. 세션 관리, 추론 시간 측정. |
| `BackendSelector` | 디바이스 감지 및 백엔드 자동 선택. Qualcomm SoC 판별, QNN 라이브러리 로드 확인, 폴백 체인 생성. |
| `QnnBackend` | Qualcomm QNN EP. HTP performance mode(burst), context binary 캐싱(MD5 기반), VTCM 설정, graph finalization 최적화. |
| `NnapiBackend` | Android NNAPI EP. CPU fallback 비활성화하여 성능 확보. |
| `CpuBackend` | ONNX Runtime CPU EP. 전체 코어 활용, graph optimization `ALL_OPT`. |

#### `core/` -- 설정 및 상태

| 클래스 | 역할 |
|--------|------|
| `StreamingConfig` | `streaming_config.json` 파싱. `ModelInfo`, `StftConfig`, `StreamingParams`, `QnnConfig`, `StateInfo` 포함. `samplesPerChunk`, `latencyMs` 등 파생값 계산. |
| `StreamingState` | ONNX 상태 텐서 관리. `InferenceResult`(mask + phase), `StreamingMetrics`(성능 추적) 정의. |

#### `session/` -- 추론 세션

| 클래스 | 역할 |
|--------|------|
| `StatefulInference` | 단일 ONNX 세션으로 통합 추론. 명시적 state I/O (`state_*` / `next_state_*`). 텐서 풀링 + 더블 버퍼링 + zero-copy JNI. Phase 복원: `atan2(imag, real)` on host. |

### 핵심 최적화 기법

| 기법 | 설명 |
|------|------|
| **Tensor Pooling** | 사전 할당된 `Direct ByteBuffer`를 재사용. 추론마다 텐서를 새로 할당하지 않아 GC 압력 최소화. |
| **Double Buffering** | 상태 텐서에 A/B 두 버퍼셋을 두고 swap. 복사 없이 상태 업데이트. |
| **Zero-copy JNI** | `Direct ByteBuffer` → ONNX Runtime JNI 전달 시 메모리 복사 없음. |
| **Host-side STFT** | STFT/iSTFT를 호스트 FP32로 수행. 양자화 아티팩트 방지, 위상 정밀도 보장. |
| **QNN Context Caching** | 컴파일된 QNN context binary를 MD5 해시 기반으로 캐싱. 후속 로드 시 컴파일 과정 생략. |

## 모델

변환된 `model.onnx`와 `streaming_config.json`을 `benchmark-app/src/main/assets/`에 배치한다. 단일 통합 ONNX 모델을 사용하며, `StreamingEnhancer`는 초기화 시 이 파일을 로드한다.

### `streaming_config.json` 구조

```json
{
  "model_info": {
    "name": "lacosenet",
    "version": "1.0.0",
    "export_format": "stateful_nncore",
    "quantization": "fp32",
    "phase_output_mode": "atan2",
    "infer_type": "masking",
    "qnn_compatible": false,
    "supported_backends": ["nnapi", "cpu"]
  },
  "stft_config": {
    "n_fft": 400,
    "hop_size": 100,
    "win_length": 400,
    "sample_rate": 16000,
    "center": true,
    "compress_factor": 0.3
  },
  "streaming_config": {
    "chunk_size_frames": 32,
    "encoder_lookahead": 7,
    "decoder_lookahead": 7,
    "export_time_frames": 40,
    "freq_bins": 201,
    "freq_bins_encoded": 100
  },
  "qnn_config": {
    "target_soc": "SM8550",
    "htp_performance_mode": "burst",
    "context_cache_enabled": true,
    "vtcm_mb": 8,
    "enable_htp_fp16_precision": false
  },
  "state_info": {
    "num_states": 0,
    "state_names": []
  },
  "export_info": {
    "timestamp": "2026-02-10T18:27:25.745467+00:00",
    "checkpoint_md5": "083e52e9...",
    "git_commit": "dccb9816..."
  }
}
```

- `export_info`: export 시점의 provenance 정보.
- `state_info.num_states`: ONNX 파일에서 실제 추출한 값 (hardcoded 대신).

## 성능

벤치마크 결과는 Android 디바이스에서 `StreamingBenchmarkTest`를 실행하여 측정한다. 벤치마크 출력에 모델 식별 정보(이름, 버전, git commit, checkpoint MD5)가 자동으로 포함되어 어떤 모델을 테스트했는지 추적할 수 있다.

**Real-time budget**: 250ms (40 frames x 6.25ms hop)

## 빌드

```bash
# 라이브러리 빌드
./gradlew :lacosenet-streaming:assembleRelease

# 벤치마크 앱 빌드
./gradlew :benchmark-app:assembleDebug

# 벤치마크 테스트 실행 (디바이스 연결 필요)
./gradlew :benchmark-app:connectedAndroidTest

# 특정 테스트만 실행
./gradlew :benchmark-app:connectedAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest
```

벤치마크 테스트는 `@Test` 메서드로 구성된다:

#### `benchmarkUnifiedModel` — 단일 모델 E2E 벤치마크

- 통합 `model.onnx`를 사용한 스트리밍 E2E 벤치마크
- 10 warmup sessions + 50 benchmark sessions = 200 per-chunk latency 샘플
- Per-chunk 통계: mean, P95, P99
- Per-session 통계: 4-chunk session total mean
- Chunk position 분석: chunk[0]~chunk[3] 위치별 평균 latency
- Real-time budget 판정

## 문서

| 문서 | 내용 |
|------|------|
| [Architecture Guide](docs/ARCHITECTURE.md) | 코드베이스 구조 및 데이터 흐름 상세 |
| [Mobile Reference](docs/MOBILE_REFERENCE.md) | 모바일 추론 참고 자료 |
| [QNN Troubleshooting](docs/QNN_TROUBLESHOOTING.md) | QNN EP 문제 해결 가이드 |

## 참고 자료

- [ONNX Runtime QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [NNAPI Migration Guide](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
