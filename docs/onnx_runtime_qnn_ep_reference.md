# ONNX Runtime + QNN EP Reference for StreamingPrimeKnet

> Galaxy S23+ (Snapdragon 8 Gen 2, SM8550) 타겟 | 조사일: 2026-02-09

## 1. 최신 버전 현황

| 항목 | 버전 | 비고 |
|------|------|------|
| ONNX Runtime | 1.24.1 (2025-02) | 최신 릴리스 |
| QNN SDK (QAIRT) | 2.42.0 | ORT 1.24.1에 내장 |
| Android AAR | com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.1 | Maven Central |
| QNN Runtime 의존성 | com.qualcomm.qti:qnn-runtime:2.42.0 | AAR에 자동 포함 |

> 기존 프로젝트 spec (ORT 1.22.0 + QAIRT 2.37.0) → 1.24.1 + 2.42.0으로 업그레이드 가능. 1.18.0부터 QNN SDK를 별도 설치할 필요 없음 (AAR에 내장).

Maven dependency:
```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime-android-qnn</artifactId>
    <version>1.24.1</version>
</dependency>
```

Gradle (Kotlin DSL):
```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.1")
```

## 2. ORT 1.22~1.24 QNN EP 릴리스 변경사항

| 버전 | QNN SDK | 핵심 변경 |
|------|---------|----------|
| 1.21.0 | 2.31 | QNN shared memory 도입, QNN EP shared lib 기본 빌드, Python 3.13 |
| 1.22.0 | 2.33→2.36.1 | GPU 백엔드 추가(QnnGpu), Upsample/Einsum/LSTM/CumSum op 지원, Softmax 퓨전, DSP queue polling (burst 모드), ARM64x NuGet |
| 1.23.0 | 2.37 | 동적 HTP performance mode 설정 지원 |
| 1.24.1 | 2.42.0 | STFT op 추가, RMSNorm, GatherND, ScatterElements, RandomUniformLike, Gelu 패턴 퓨전, LPBQ 양자화, Clip QDQ 버그 수정 (#26601), ARM64 wheel, v81 디바이스 |

### StreamingPrimeKnet에 직접적 영향

- **STFT op이 QNN에서 네이티브 지원 (1.24+)**: 현재 Host FP32에서 처리하는 STFT를 NPU로 이전할 가능성. 단, INT8 양자화 호환성 및 위상(phase) 정밀도 검증 필요
- **Gelu 패턴 퓨전**: 모델에 Gelu가 있다면 자동 최적화
- **LSTM op 지원 (1.22+)**: 향후 GRU/LSTM 기반 모델 변형 시 NPU 실행 가능
- **DSP queue polling (1.22+)**: burst 모드에서 HTP 응답 지연 감소

## 3. QNN EP Provider Options (전체)

### 3.1 백엔드 선택

| 옵션 | 값 | 설명 |
|------|---|------|
| `backend_type` | `"htp"` / `"cpu"` / `"gpu"` / `"saver"` | HTP 권장 (INT8 모델) |
| `backend_path` | 플랫폼별 경로 | e.g., `libQnnHtp.so` (Android) |

### 3.2 HTP 성능 모드

| `htp_performance_mode` 값 | 특성 | StreamingPrimeKnet 사용 시나리오 |
|---------------------------|------|--------------------------------|
| `"burst"` | 최대 성능, 최대 전력 | **실시간 음성향상 (권장)** |
| `"sustained_high_performance"` | 높은 지속 성능 | 장시간 통화 |
| `"high_performance"` | 높은 성능 | - |
| `"balanced"` | 성능/전력 균형 | 배터리 중시 |
| `"default"` | 시스템 기본 | - |
| `"low_balanced"` | 낮은 전력 | - |
| `"power_saver"` | 절전 | - |
| `"low_power_saver"` | 최소 전력 | 백그라운드 |
| `"high_power_saver"` | 절전 (고효율) | - |

### 3.3 그래프 최적화

| 옵션 | 값 | 설명 |
|------|---|------|
| `htp_graph_finalization_optimization_mode` | `"0"` (기본) ~ `"3"` (최대) | `"3"` 권장: 컴파일 시간 길지만 최적 그래프. Context cache와 결합하면 첫 실행만 느림 |

### 3.4 하드웨어 타겟

| 옵션 | 값 | 설명 |
|------|---|------|
| `soc_model` | `"SM8550"` | Galaxy S23+ SoC 명시 (미지정 시 "0"/자동감지) |
| `htp_arch` | (자동감지) | Hexagon TP 버전 (e.g., "73") |
| `device_id` | `"0"` (기본) | 디바이스 ID |

### 3.5 메모리 및 리소스

| 옵션 | 값 | 설명 |
|------|---|------|
| `vtcm_mb` | `"8"` | VTCM 메모리 할당 (MB). 미설정 시 시스템 결정 |
| `enable_htp_shared_memory_allocator` | `"0"` (기본) / `"1"` | **CPU↔NPU 복사 오버헤드 제거 — 실시간 추론에 권장** |
| `rpc_control_latency` | 마이크로초 | RPC 레이턴시 제어 |

### 3.6 정밀도

| 옵션 | 값 | 설명 |
|------|---|------|
| `enable_htp_fp16_precision` | `"0"` / `"1"` (기본) | INT8 QDQ 모델 사용 시 `"0"`으로 비활성화 |
| `offload_graph_io_quantization` | `"0"` / `"1"` (기본) | I/O 양자화를 HTP에 오프로드 |

### 3.7 프로파일링

| 옵션 | 값 | 설명 |
|------|---|------|
| `profiling_level` | `"off"` (기본), `"basic"`, `"detailed"`, `"optrace"` | optrace는 QAIRT 2.39+ 필요 |
| `profiling_file_path` | CSV 파일 경로 | 프로파일링 이벤트 출력 |

### 3.8 기타

| 옵션 | 값 | 설명 |
|------|---|------|
| `qnn_context_priority` | `"low"`, `"normal"` (기본), `"normal_high"`, `"high"` | 실시간 처리 시 `"high"` |
| `qnn_saver_path` | 파일 경로 | QnnSaver 백엔드 라이브러리 경로 (디버깅용) |

## 4. Session Options

| 옵션 | 값 | 설명 |
|------|---|------|
| `ep.context_enable` | `"1"` | Context binary 캐싱 활성화 |
| `ep.context_embed_mode` | `"1"` (기본) | Binary를 ONNX 파일에 임베딩. `"0"`이면 별도 .bin 파일 |
| `ep.context_file_path` | 커스텀 경로 | 미지정 시 `[model_name]_ctx.onnx` |
| `session.disable_cpu_ep_fallback` | `"1"` | QNN 실행 불가 시 에러 발생 (디버깅/검증용) |
| `ep.share_ep_contexts` | `"1"` | 다중 모델 weight sharing 활성화 |
| `ep.stop_share_ep_contexts` | `"1"` | 세션 그룹의 마지막 세션에 설정 |

## 5. QNN HTP 지원 연산자 (StreamingPrimeKnet 관련)

### 5.1 모델 사용 연산자별 지원 현황

| 연산자 | QNN HTP 지원 | StreamingPrimeKnet 사용처 | 비고 |
|--------|-------------|-------------------------|------|
| Conv2d | O | DenseEncoder, TS_BLOCK, Decoders | 3D도 1.18+ |
| ConvTranspose2d | O | MaskDecoder, PhaseDecoder | 3D도 1.18+ |
| PReLU | O | DenseEncoder 활성화 | FP16/INT32 1.18+ |
| Sigmoid | O | MaskDecoder 출력, SCA attention | |
| Tanh | O | PhaseDecoder 출력 | |
| Add/Mul/Sub/Div | O | FrozenIN (scale*x+shift), 잔차 연결 | |
| Concat | O | DenseBlock 특성 연결 | |
| Reshape/Transpose | O | TS_BLOCK 차원 변환 (B,C,T,F ↔ BF,C,T) | |
| Squeeze/Unsqueeze | O | 디코더 출력 형상 조정 | |
| ReduceMean | O | StreamingSCA (AdaptiveAvgPool 대체) | |
| MatMul | O | SCA의 1×1 Conv 대체 가능 | uint8/uint16 조합 |
| Pad | O | StatefulConv의 state concatenation | |
| Slice | O | StateFramesContext에서 프레임 슬라이싱 | |
| Cos/Sin | O | Phase → complex 변환 (est_mag*cos/sin) | |
| STFT | O (1.24+) | 현재 Host에서 처리 — NPU 이전 가능성 | QNN SDK 2.42.0 신규 |
| InstanceNormalization | O | 원본 모델 (FrozenIN으로 대체하므로 미사용) | |
| BatchNormalization | O | 미사용 (참고용) | FP16 1.18+ |

### 5.2 미지원 연산자 (주의)

| 연산자 | 상태 | 영향 |
|--------|------|------|
| Loop/If | **미지원** | 동적 제어 흐름 불가 → 현재 설계에 해당 없음 |
| Dynamic shapes | **미지원** | 정적 shape export 필수 (현재 설계와 일치) |

### 5.3 FrozenInstanceNorm2d의 QNN 호환성

FrozenIN은 사전계산된 상수로 변환되므로 ONNX 그래프에서 단순 Mul+Add로 표현:
```
y = x * scale + shift
→ ONNX: Mul(x, scale_constant) → Add(_, shift_constant)
```
QNN HTP에서 완전 지원. InstanceNorm op 자체보다 더 효율적.

## 6. INT8 QDQ 양자화 파이프라인

### 6.1 전체 워크플로우

```python
from pathlib import Path
from onnxruntime.quantization import quantize, CalibrationMethod, QuantType
from onnxruntime.quantization.execution_providers.qnn import (
    get_qnn_qdq_config,
    qnn_preprocess_model,
)

# Step 1: QNN 전처리 (Conv-ReLU 퓨전 등)
model_changed = qnn_preprocess_model(
    model_input=Path("model_fp32_static.onnx"),
    model_output=Path("model_preproc.onnx"),
    fuse_layernorm=False,
)
model_to_quantize = "model_preproc.onnx" if model_changed else "model_fp32_static.onnx"

# Step 2: 양자화 설정 생성
qnn_config = get_qnn_qdq_config(
    model_input=Path(model_to_quantize),
    calibration_data_reader=data_reader,
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    per_channel=False,  # QNN EP는 per-channel 미지원
)

# Step 3: 양자화 실행
quantize(model_to_quantize, "model_int8_qdq.onnx", qnn_config)
```

### 6.2 `get_qnn_qdq_config` 파라미터

| 파라미터 | 지원 값 | 기본값 | 설명 |
|---------|--------|-------|------|
| `activation_type` | QUInt8, QInt16, QUInt16 | QUInt8 | 활성화 양자화 타입 |
| `weight_type` | QUInt8, QInt8, QInt16, QUInt16 | QUInt8 | 가중치 양자화 타입 |
| `calibrate_method` | MinMax, Entropy, Percentile | MinMax | 캘리브레이션 방법 |
| `per_channel` | **False만 허용** | False | QNN EP 제약 |

내부 자동 설정:
- `MinimumRealRange`: 0.0001
- `DedicatedQDQPair`: False
- 16-bit 타입 사용 시 `UseQDQContribOps`: True
- Sigmoid/Tanh에 hardcoded scale/zero-point (16-bit)
- Cast 연산자 양자화 제외

### 6.3 `qnn_preprocess_model` 함수

```python
qnn_preprocess_model(
    model_input: Path,
    model_output: Path,
    fuse_layernorm: bool = False,
) -> bool  # True if model was modified
```

수행하는 최적화:
- Conv + ReLU/PReLU 퓨전
- LayerNorm 퓨전 (옵션)
- QNN 비호환 패턴 변환

**반드시 양자화 전에 실행해야 함** — Conv-ReLU 미퓨전 시 QDQ 노드 과다로 OOM 발생 가능.

### 6.4 주의사항

- 양자화 도구는 **x86_64 Python에서만 실행** (ARM64 불가)
- 양자화된 모델 추론은 ARM64(Android)에서 실행
- CalibrationDataReader는 실제 입력 데이터 분포를 대표해야 함

## 7. Context Binary 캐싱 (EP Context)

### 7.1 동작 방식

```
첫 실행 (on-device 또는 offline):
  model_int8_qdq.onnx → [QNN 컴파일] → HTP 그래프 (2-6초)
                                       ↓
                               model_int8_qdq_ctx.onnx (캐시)

후속 실행:
  model_int8_qdq_ctx.onnx → [로드] → HTP 그래프 (0.3-0.5초)
```

### 7.2 Context 생성 코드

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.add_session_config_entry("ep.context_enable", "1")
sess_options.add_session_config_entry("ep.context_embed_mode", "1")
sess_options.add_session_config_entry("ep.context_file_path", "model_ctx.onnx")

# 세션 생성만으로 context binary 생성 (추론 불필요)
session = ort.InferenceSession(
    "model_int8_qdq.onnx",
    sess_options,
    providers=["QNNExecutionProvider"],
    provider_options=[{
        "backend_type": "htp",
        "soc_model": "SM8550",
        "htp_graph_finalization_optimization_mode": "3",
    }]
)
```

### 7.3 배포 전략 비교

| 전략 | 장점 | 단점 | 권장 |
|------|------|------|------|
| embed_mode=1 (임베딩) | 단일 파일 배포, Android assets 관리 간편 | 파일 크기 증가 | **Android 권장** |
| embed_mode=0 (분리) | 원본 모델과 context 분리 | .onnx + .bin 두 파일 관리 | 서버/데스크톱 |
| 오프라인 사전 컴파일 | 첫 실행 지연 없음 | SoC/HTP arch별 별도 빌드 | **프로덕션 권장** |

### 7.4 캐시 무효화 조건

- ONNX 모델 파일 변경 (해시 불일치)
- QNN SDK 버전 변경
- SoC/HTP architecture 변경 (다른 기기)

### 7.5 Context Binary 출력 파일

- `[model_name]_ctx.onnx`: EPContext 노드가 포함된 모델
- `[model_name]_QNN_[hash_id].bin`: Context binary (embed_mode=0일 때)
- `[model_name]_schematic.bin`: Optrace 프로파일링용

### 7.6 Weight Sharing (다중 모델)

Encoder/Decoder를 분리 ONNX 모델로 export하는 경우:

```python
# 세션 그룹으로 weight 공유
sess_options = ort.SessionOptions()
sess_options.add_session_config_entry("ep.context_enable", "1")
sess_options.add_session_config_entry("ep.share_ep_contexts", "1")
session1 = ort.InferenceSession("encoder.onnx", sess_options, ...)

sess_options.add_session_config_entry("ep.stop_share_ep_contexts", "1")
session2 = ort.InferenceSession("decoder.onnx", sess_options, ...)
# → encoder_ctx.onnx, decoder_ctx.onnx + 공유 .bin 1개 생성
```

추론 시:
```python
sess_options.add_session_config_entry("ep.share_ep_contexts", "1")
session1 = ort.InferenceSession("encoder_ctx.onnx", sess_options, ...)
session2 = ort.InferenceSession("decoder_ctx.onnx", sess_options, ...)
```

## 8. Galaxy S23+ 특이사항 및 해결책

### 8.1 Graph Finalization OOM (해결됨)

- **문제**: QDQ 모델의 FinalizeGraphs에서 4GB+ 메모리 사용, Android OOM killer 발동
- **원인**: Conv-ReLU 미퓨전으로 QDQ 노드 과다 (345개+)
- **해결**: ORT 1.22+ main branch에서 수정 완료 (2024-04 close, GitHub #18353)
- **추가 대응**: `qnn_preprocess_model()`로 Conv-ReLU 퓨전 적용 + Context binary 사전 캐싱

### 8.2 HTP Shared Memory Allocator

- `enable_htp_shared_memory_allocator = "1"` 설정
- CPU↔NPU 간 데이터 복사 오버헤드 제거
- 실시간 음성향상에서 chunk당 수백μs 절약 가능

### 8.3 DSP Queue Polling (1.22+)

- `htp_performance_mode = "burst"` 설정 시 DSP queue polling 자동 활성화
- HTP 추론 요청의 응답 지연 감소

### 8.4 SM8550 (8 Gen 2) HTP 아키텍처

| 항목 | 값 |
|------|---|
| HTP Architecture | v73 |
| Context binary 호환 SoC | SM8650 (8 Gen 3), SM7550 (7+ Gen 3) |
| 비호환 (arch 다름) | SM8450 (8 Gen 1, v69), SM8750 (8 Elite, v75) |

## 9. GPU 백엔드 (대안 경로)

ORT 1.22+에서 QNN GPU 백엔드(Adreno 740) 추가:

| 항목 | HTP (NPU) | GPU (Adreno) |
|------|-----------|--------------|
| 모델 포맷 | INT8 QDQ 필수 | FP32/FP16 (양자화 모델 비호환) |
| 성능 | 최고 | 중간 |
| 전력 효율 | 최고 | 중간 |
| 설정 | `backend_type: "htp"` | `backend_type: "gpu"` |
| FP16 자동 변환 | `enable_htp_fp16_precision` | FP16 모델 직접 사용 |

StreamingPrimeKnet 관점: INT8 QDQ 모델이 이미 준비되어 있으므로 **HTP가 최적**. GPU 백엔드는 FP32/FP16 모델용 폴백으로만 고려.

## 10. Stateful ONNX 추론 패턴

### 10.1 StreamingPrimeKnet 채택 패턴

```python
# 매 chunk마다:
inputs = {
    "mag": mag_tensor,           # [1, F, T]
    "pha": pha_tensor,           # [1, F, T]
    "state_0": prev_state_0,     # StatefulConv buffer
    "state_1": prev_state_1,     # EMA state
    ...
    "state_N": prev_state_N,     # 총 24+ 상태 텐서
}
outputs = session.run(None, inputs)
est_mask, est_pha, *next_states = outputs
# next_states → 다음 chunk의 inputs로 전달
```

### 10.2 참고 프로젝트: sherpa-onnx (k2-fsa)

- 오픈소스 음성 처리 프레임워크 (ASR, TTS, 음성향상)
- GTCRN 모델로 16kHz 스트리밍 음성향상 구현
- ONNX 모델에 state를 명시적 input/output으로 노출
- Android NDK + Kotlin 바인딩 지원
- QNN 백엔드 지원 (Qualcomm NPU)
- GitHub: https://github.com/k2-fsa/sherpa-onnx

### 10.3 Kotlin (Android) 추론 예시

```kotlin
// ORT Session 생성
val sessionOptions = OrtSession.SessionOptions()
sessionOptions.addConfigEntry("ep.context_enable", "1")

val providerOptions = mapOf(
    "backend_type" to "htp",
    "htp_performance_mode" to "burst",
    "soc_model" to "SM8550",
    "vtcm_mb" to "8",
    "enable_htp_shared_memory_allocator" to "1",
    "htp_graph_finalization_optimization_mode" to "3",
    "enable_htp_fp16_precision" to "0",
    "qnn_context_priority" to "high",
)

val session = env.createSession(modelPath, sessionOptions,
    listOf("QNNExecutionProvider", "CPUExecutionProvider"),
    listOf(providerOptions, emptyMap()))

// Stateful 추론
fun processChunk(mag: FloatArray, pha: FloatArray, states: List<OnnxTensor>): InferenceResult {
    val inputs = mutableMapOf<String, OnnxTensor>()
    inputs["mag"] = OnnxTensor.createTensor(env, mag, longArrayOf(1, F, T))
    inputs["pha"] = OnnxTensor.createTensor(env, pha, longArrayOf(1, F, T))
    states.forEachIndexed { i, state ->
        inputs["state_$i"] = state
    }

    val results = session.run(inputs)
    val estMask = results["est_mask"].get() as OnnxTensor
    val estPha = results["est_pha"].get() as OnnxTensor
    val nextStates = (0 until numStates).map {
        results["next_state_$it"].get() as OnnxTensor
    }
    return InferenceResult(estMask, estPha, nextStates)
}
```

## 11. 프로파일링 가이드

### 11.1 프로파일링 레벨

| 레벨 | 내용 | 사용 시점 |
|------|------|----------|
| `"off"` | 비활성화 | 프로덕션 |
| `"basic"` | 초기화/실행/해제 통계 | 초기 성능 확인 |
| `"detailed"` | 레이어별 성능 분석 | 병목 식별 |
| `"optrace"` | HTP Op 상세 타이밍 | 심층 최적화 (QAIRT 2.39+) |

### 11.2 Optrace 워크플로우

1. Context binary를 optrace 모드로 생성
2. 생성된 `_ctx.onnx`로 새 세션 생성
3. 추론 실행 → `_qnn.log`, `_qnn.bin` 생성
4. `qnn-profile-viewer`로 분석 → HTML 시각화

## 12. 기존 구현 대비 업데이트 권장사항

| 항목 | 기존 (spec 기준) | 권장 업데이트 | 영향도 |
|------|-----------------|-------------|--------|
| ORT 버전 | 1.22.0 | **1.24.1** | QNN SDK 2.42.0, STFT op, 버그 수정 |
| Maven 의존성 | `onnxruntime-android-qnn:1.22.0` | **`:1.24.1`** | Gradle 한 줄 변경 |
| 양자화 전처리 | 미사용 | **`qnn_preprocess_model()` 추가** | Conv-ReLU 퓨전으로 OOM 방지 |
| Shared Memory | 미설정 | **`enable_htp_shared_memory_allocator: "1"`** | 데이터 복사 오버헤드 제거 |
| Context 최적화 | mode 미지정 | **`htp_graph_finalization_optimization_mode: "3"`** | 최적 HTP 그래프 |
| Context embed | 분리 가능 | **`ep.context_embed_mode: "1"`** | Android 배포 단일 파일 |
| STFT 위치 | Host FP32 고정 | Host FP32 유지 (1.24+ STFT op 검증 후 NPU 이전 검토) | 당장 변경 불요 |
| Context priority | 미설정 | **`qnn_context_priority: "high"`** | 실시간 처리 우선 |

## 13. 참고 자료

### 공식 문서
- [QNN EP 공식 문서](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)
- [ORT Quantization 문서](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ORT 릴리스 노트](https://github.com/microsoft/onnxruntime/releases)
- [ORT 1.24.1 릴리스](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1)
- [ORT Android 빌드 가이드](https://onnxruntime.ai/docs/build/android.html)

### Maven / 패키지
- [onnxruntime-android-qnn Maven Central](https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-android-qnn)
- [onnxruntime-android-qnn 1.22.0 MVN](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android-qnn/1.22.0)

### GitHub 이슈 / PR
- [Galaxy S23 OOM 이슈 #18353](https://github.com/microsoft/onnxruntime/issues/18353)
- [QNN Shared Memory PR #23136](https://github.com/microsoft/onnxruntime/pull/23136)
- [QNN GPU crashes #24004](https://github.com/microsoft/onnxruntime/issues/24004)
- [QNN HTP Setup on Android #21214](https://github.com/microsoft/onnxruntime/issues/21214)

### 소스 코드 참조
- [QNN quant_config.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/execution_providers/qnn/quant_config.py)
- [QNN preprocess.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/execution_providers/qnn/preprocess.py)
- [QNN EP inference examples (DeepWiki)](https://deepwiki.com/microsoft/onnxruntime-inference-examples/3.1-qnn-ep-examples)
- [onnxruntime-android-qnn-builds (커뮤니티)](https://github.com/Lucchetto/onnxruntime-android-qnn-builds)

### 블로그 / 기술 문서
- [QNN GPU 백엔드 블로그 (Qualcomm 2025)](https://www.qualcomm.com/developer/blog/2025/05/unlocking-power-of-qualcomm-qnn-execution-provider-gpu-backend-onnx-runtime)
- [Qualcomm AI Hub Release Notes](https://app.aihub.qualcomm.com/docs/hub/release_notes.html)

### 참고 프로젝트
- [sherpa-onnx (음성향상/ASR/TTS)](https://github.com/k2-fsa/sherpa-onnx)
- [Qualcomm AI Hub Models](https://github.com/quic/ai-hub-models)
