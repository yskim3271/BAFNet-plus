# Galaxy S23+ On-Device AI Serving Frameworks

> Snapdragon 8 Gen 2 (Hexagon NPU / Adreno GPU / Kryo CPU) 기준
> 조사일: 2026-02-09

## 프레임워크 비교표

| 프레임워크 | 개발사 | 모델 포맷 | HW 가속 (S23+) | LLM 지원 | 성숙도 |
|---|---|---|---|---|---|
| **LiteRT** (구 TFLite) | Google | `.tflite` | NPU, GPU(OpenCL), CPU | LiteRT-LM으로 지원 | ★★★★★ |
| **Qualcomm AI Engine Direct** | Qualcomm | `.dlc` | NPU, GPU, DSP (직접 제어) | 제한적 | ★★★★☆ |
| **Qualcomm AI Hub** | Qualcomm | TFLite/ONNX/QNN | NPU, GPU, CPU (자동 배분) | 지원 | ★★★★☆ |
| **ExecuTorch** | Meta (PyTorch) | `.pte` | Qualcomm/Vulkan/XNNPACK | 지원 | ★★★☆☆ |
| **ONNX Runtime Mobile** | Microsoft | `.onnx` | NNAPI(NPU/GPU/DSP) | 제한적 | ★★★★☆ |
| **Samsung ONE** | Samsung | TFLite/ONNX/Circle | NPU, GPU, DSP, CPU | 미지원 | ★★★☆☆ |
| **MediaPipe** | Google | `.tflite` (task 번들) | GPU, CPU | LLM Inference API | ★★★★☆ |
| **llama.cpp** | 커뮤니티 | `.gguf` | CPU (ARM NEON) | 전용 | ★★★☆☆ |
| **MLC LLM** | CMU/OctoAI | TVM 컴파일 | GPU (OpenCL/Vulkan) | 전용 | ★★☆☆☆ |

## HW 가속 방식 비교

| 프레임워크 | NPU (Hexagon) | GPU (Adreno) | CPU (Kryo) | 비고 |
|---|---|---|---|---|
| LiteRT | NNAPI 경유 | OpenCL 직접 | XNNPACK | GPU 성능 TFLite 대비 1.4x |
| QNN (AI Engine Direct) | 직접 접근 | 직접 접근 | 직접 접근 | 최고 수준 HW 제어 |
| ExecuTorch | Qualcomm 백엔드 | Vulkan 백엔드 | XNNPACK | 12+ 백엔드 |
| ONNX Runtime | NNAPI 경유 | NNAPI 경유 | 내장 | NNAPI 의존적 |
| llama.cpp | X | X | ARM NEON | CPU 전용 |
| MLC LLM | X | OpenCL/Vulkan | 폴백 | GPU 중심 |

## 모델 포맷 변환 경로

```
PyTorch (.pt)
  ├─ torch.export ──────────► ExecuTorch (.pte)
  ├─ torch.onnx.export ────► ONNX (.onnx) ──► ONNX Runtime Mobile
  │                                    └────► QNN (.dlc) via qnn-onnx-converter
  └─ ai_edge_torch ────────► LiteRT (.tflite)

TensorFlow (.pb/.h5)
  └─ TFLiteConverter ──────► LiteRT (.tflite)
```

## 양자화 지원

| 프레임워크 | INT8 | INT4 | FP16 | Dynamic Quant | PTQ | QAT |
|---|---|---|---|---|---|---|
| LiteRT | O | X | O | O | O | O |
| QNN | O | O | O | X | O | X |
| Qualcomm AI Hub | O | O | O | O | O | X |
| ExecuTorch | O | O | O | O | O | O |
| ONNX Runtime | O | X | O | O | O | X |
| llama.cpp | O | O (Q2~Q8) | O | X | O | X |

## 용도별 추천

### 일반 DNN (음성향상, 분류, 탐지 등)

| 순위 | 프레임워크 | 이유 |
|---|---|---|
| 1 | **LiteRT** | 가장 성숙, 문서 풍부, NPU 가속, Android 표준 |
| 2 | **Qualcomm AI Hub** | Snapdragon 최적화, 300+ 사전검증 모델 |
| 3 | **ONNX Runtime** | 크로스플랫폼 호환, PyTorch→ONNX 변환 용이 |
| 4 | **ExecuTorch** | PyTorch 네이티브 export, 빠르게 성장 중 |

### LLM 서빙

| 순위 | 프레임워크 | 이유 |
|---|---|---|
| 1 | **llama.cpp** | 가장 넓은 모델 지원, 활발한 커뮤니티, 양자화 유연 |
| 2 | **MediaPipe LLM** | Google 공식, Gemma/Gemini 최적화 |
| 3 | **MLC LLM** | GPU 가속 활용, 컴파일러 최적화 |

### Snapdragon NPU 최대 활용

| 순위 | 프레임워크 | 이유 |
|---|---|---|
| 1 | **QNN (AI Engine Direct)** | NPU 직접 제어, 최고 성능/효율 |
| 2 | **Qualcomm AI Hub** | QNN 기반 + 자동 최적화 파이프라인 |
| 3 | **LiteRT** | NNAPI 경유로 NPU 활용, 범용성 우수 |

## 주요 링크

| 프레임워크 | 공식 문서 | GitHub |
|---|---|---|
| LiteRT | [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert/overview) | [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT) |
| QNN SDK | [qualcomm.com/developer](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | - |
| Qualcomm AI Hub | [app.aihub.qualcomm.com](https://app.aihub.qualcomm.com/docs/) | [quic/ai-hub-models](https://github.com/quic/ai-hub-models) |
| ExecuTorch | [executorch.ai](https://executorch.ai/) | [pytorch/executorch](https://github.com/pytorch/executorch) |
| ONNX Runtime | [onnxruntime.ai](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html) | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| Samsung ONE | - | [Samsung/ONE](https://github.com/Samsung/ONE) |
| MediaPipe | [ai.google.dev/edge/mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide) | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) |
| llama.cpp | - | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| MLC LLM | [mlc.ai/mlc-llm](https://mlc.ai/mlc-llm/) | [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) |
