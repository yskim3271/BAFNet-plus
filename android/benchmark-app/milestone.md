# Android ONNX Inference Benchmark

## Overview

Backbone 모델의 Qualcomm HTP (Hexagon Tensor Processor) 추론 최적화 실험 기록.
BAFNet은 두 개의 독립 Backbone(mapping + masking)을 사용하며, 각 backbone의 단일 추론 성능과 두 backbone 병렬 실행 시의 성능을 측정한다.

### Test Environment

| Item | Value |
|---|---|
| Device | SM-S938N (Galaxy S25 Ultra) |
| SoC | Snapdragon 8 Elite (SM8750), Hexagon V79 |
| ORT | 1.24.2 (QNN SDK 2.42.0) |
| Budget | 50.0ms per chunk (8 frames × 6.25ms) |
| Benchmark | 10 warmup + 50 sessions × 4 chunks = 200 chunks |
| Weights | Random (`--no_checkpoint`, architecture-only) |

### Model Configurations

두 가지 아키텍처를 실험. 공통: num_tsblock=4, dense_depth=4, dense_channel=64, freq_block_kernel=[3,11,23,31], causal=True, chunk_size=8, lookahead=3.

| Name | time_block_kernel | float32 | QDQ INT8 |
|---|---|---|---|
| **Small** | [11] | 4.5 MB | 3.1 MB |
| **Large** | [3, 5, 7, 11] | 5.9 MB | 4.0 MB |

---

## Best Results

### Single Backbone

| Backend | Quantization | Small | Large | Budget % (Small) |
|---|---|---|---|---|
| CPU (4 threads) | float32 | 34.7ms | — | 69% |
| HTP full opts | FP16 | 23.0ms | 29.8ms | 46% |
| **HTP full opts** | **QDQ INT8** | **6.2ms** | **10.1ms** | **12%** |

### Dual Backbone Concurrent (BAFNet 실제 배포 시나리오)

| Model | Quantization | WallClock | Overlap Ratio | Budget % |
|---|---|---|---|---|
| Small [11] | QDQ INT8 | **10.2ms** | 0.681 | **20%** |
| Large [3,5,7,11] | QDQ INT8 | **15.2ms** | 0.761 | **30%** |

→ Budget 대비 70~80% 여유. Fusion 네트워크(경량 4-conv) 포함해도 충분.

---

## 1. Single Backbone 최적화

### 1.1 ORT 업그레이드: 가장 큰 임팩트

ORT 1.22.0 → 1.24.2 업그레이드만으로 CPU 추론 46% 개선. 아키텍처 축소, 양자화 등 모든 다른 최적화보다 효과가 컸음.

| ORT | Backend | freq_block_kernel | Mean | Verdict |
|---|---|---|---|---|
| 1.22.0 | CPU | [3, 11, 23, 31] | 65.7ms | NOT REALTIME |
| 1.22.0 | CPU | [3, 11] | 50.3ms | NOT REALTIME |
| **1.24.2** | **CPU** | **[3, 11, 23, 31]** | **35.5ms** | **REALTIME** |
| 1.24.2 | CPU | [3, 11] | 27.6ms | REALTIME |

→ ORT 1.24.2에서는 아키텍처 축소 효과(35.5→27.6ms, 22%)보다 ORT 버전 효과(65.7→35.5ms, 46%)가 2배 큼.

### 1.2 CPU Runtime 최적화: 효과 없음

ORT 1.24.2의 CPU 커널이 이미 충분히 최적화되어 추가 런타임 튜닝 여지 없음.

| Technique | Expected | Actual | Note |
|---|---|---|---|
| Thread sweep (1,2,4,6,8) | 10-25%↓ | 없음 | 4 threads 최적, 6+ 역효과 |
| INT8 dynamic quantization | 10-20%↓ | **9배 악화** | ORT 1.24.2 ARM에서 역효과 |
| onnx-simplifier | 3-8%↓ | 없음 | 크기 3%↓, 속도 무변화 |
| Memory pattern opt | 1-3%↓ | 없음 | 고정 shape에도 효과 없음 |

<details>
<summary>상세 데이터</summary>

**Thread sweep** (SM-S938N, 8코어 전부 performance급):

| Threads | Mean | P95 | P99 |
|---|---|---|---|
| 1 | 93.3ms | 94.0ms | 101.1ms |
| 2 | 60.0ms | 63.1ms | 65.5ms |
| **4** | **38.4ms** | **45.1ms** | **50.2ms** |
| 6 | 66.6ms | 167.1ms | 237.8ms |
| 8 | 89.5ms | 157.9ms | 317.4ms |

**INT8 dynamic quantization**:

| Config | Model Size | Mean | P95 | P99 |
|---|---|---|---|---|
| float32 | 4.5 MB | 35.5ms | 36.7ms | 41.6ms |
| int8_dynamic | 2.4 MB | 321.5ms | 441.7ms | 444.8ms |

**onnx-simplifier**:

| Config | Model Size | Mean | P95 | P99 |
|---|---|---|---|---|
| original | 4.5 MB | 35.5ms | 36.7ms | 41.6ms |
| simplified | 4.4 MB | 37.9ms | 55.5ms | 59.3ms |

</details>

### 1.3 HTP 최적화: 핵심 돌파구

초기 HTP 벤치마크(108ms)는 `backend_path`만 설정한 minimal 구성이었음. Provider options 전체 적용으로 79% 개선, QDQ INT8으로 추가 71% 개선.

**HTP Provider Options (full):**
```
htp_performance_mode                    = burst
htp_graph_finalization_optimization_mode = 3 (최대)
enable_htp_fp16_precision               = 1 (FP16) or 0 (QDQ)
enable_htp_shared_memory_allocator      = 1
vtcm_mb                                 = 8
qnn_context_priority                    = high
```

**Small model [11] 기준 — 단계별 개선:**

| Step | Config | Mean | P95 | P99 | vs CPU |
|---|---|---|---|---|---|
| baseline | HTP minimal opts | 108.3ms | 114.4ms | 124.5ms | 0.3x |
| +full opts | HTP FP16 full opts | 23.0ms | 25.0ms | 32.2ms | 1.5x |
| +QDQ INT8 | HTP QDQ INT8 full opts | 6.6ms | 7.8ms | 10.9ms | 5.3x |
| **+ctx cache** | **HTP QDQ INT8 + cache** | **6.2ms** | **6.6ms** | **10.1ms** | **5.6x** |

**Key insight**:
- Provider options만으로 108→23ms (79%↓). burst mode + finalization opt 3 + shared memory + VTCM 결합 효과.
- QDQ INT8로 추가 23→6.6ms (71%↓). HTP는 INT8 전용 가속기.
- Context cache는 추론 속도 향상 미미하나, 그래프 컴파일 시간(float32: 5.7s, QDQ: 2.4s) 절약이 실서비스 cold start에서 유의미.

### 1.4 모델 크기 확장: time_block_kernel 비교

time_block_kernel=[11] → [3,5,7,11]로 GPKFFN 병렬 branch 증가 시 성능 변화.

| Model | HTP FP16 | QDQ INT8 |
|---|---|---|
| Small [11] (4.5/3.1 MB) | 23.0ms | 6.2ms |
| Large [3,5,7,11] (5.9/4.0 MB) | 29.8ms (+30%) | 10.1ms (+63%) |

→ 모델 31% 커지면 FP16 30%↑, QDQ INT8 63%↑. INT8은 커널 수에 더 민감.
→ 그래도 Large QDQ INT8 10.1ms = budget의 20%, 충분한 여유.

---

## 2. Dual Backbone 병렬 추론

### 2.1 실험 설계

BAFNet의 mapping(BCS) + masking(ACS) 두 backbone은 추론 경로가 완전히 독립(공유 연산 없음).
추론 후 경량 fusion 네트워크(4 conv blocks + sigmoid)가 출력을 블렌딩.

**핵심 질문**: Qualcomm HTP는 두 개의 ORT session을 동시에 실행할 수 있는가?

**방법**: `ExecutorService(2 threads)`로 두 session.run()을 동시 제출.

**핵심 지표 — overlap ratio**:
```
overlap_ratio = (mappingMs + maskingMs − wallClockMs) / wallClockMs
```
- `> 0.3` → HTP가 유의미한 병렬 처리
- `≈ 0.0` → 직렬화 (concurrent 무의미)
- `< 0.0` → contention overhead (concurrent 해로움)

### 2.2 Small Model (time_block_kernel=[11])

| Config | Mapping | Masking | WallClock | Overlap | Budget % |
|---|---|---|---|---|---|
| Sequential FP16 | 23.4ms | 23.6ms | 47.0ms | 0.000 | 94% |
| Concurrent FP16 | 35.4ms | 31.4ms | 43.2ms | 0.546 | 86% |
| **Concurrent QDQ INT8** | **7.8ms** | **9.4ms** | **10.2ms** | **0.681** | **20%** |

P95/P99:

| Config | WallClock P95 | WallClock P99 | Overlap P95 |
|---|---|---|---|
| Sequential FP16 | 49.7ms | 56.2ms | 0.000 |
| Concurrent FP16 | 48.8ms | 57.2ms | 0.600 |
| Concurrent QDQ INT8 | 11.6ms | 13.8ms | 0.732 |

### 2.3 Large Model (time_block_kernel=[3,5,7,11])

| Config | Mapping | Masking | WallClock | Overlap | Budget % |
|---|---|---|---|---|---|
| Concurrent FP16 | 34.9ms | 40.3ms | 47.8ms | 0.574 | 96% |
| **Concurrent QDQ INT8** | **13.4ms** | **13.1ms** | **15.2ms** | **0.761** | **30%** |

P95/P99:

| Config | WallClock P95 | WallClock P99 | Overlap P95 |
|---|---|---|---|
| Concurrent FP16 | 57.7ms | 101.1ms | 0.646 |
| Concurrent QDQ INT8 | 17.1ms | 27.7ms | 0.825 |

### 2.4 Dual Backbone Insights

1. **HTP는 두 session을 병렬 실행한다**: 모든 구성에서 overlap > 0.3. FP16 0.55~0.57, QDQ INT8 0.68~0.76.

2. **QDQ INT8이 더 높은 병렬성**: INT8 연산이 HTP ALU를 적게 점유하여 두 그래프가 더 잘 인터리빙.

3. **모델 커질수록 overlap 개선**: Small INT8 0.681 → Large INT8 0.761. HTP 파이프라이닝 기회 증가.

4. **Concurrent에서 개별 latency 증가**: 리소스 공유로 개별 실행 시간 상승 (FP16 단일 23ms → concurrent 33ms). 그러나 wall-clock은 합산보다 감소.

5. **VTCM 경합 없음**: 두 session 모두 vtcm=8로 생성·실행 성공. Snapdragon 8 Elite의 VTCM이 충분히 크거나 QNN 런타임이 자동 분할.

6. **FP16 dual은 비실용**: FP16 concurrent wall-clock 43~48ms로 budget 경계. 실배포에서는 QDQ INT8 필수.

---

## 3. Conclusion

```
Budget: 50.0ms
────────────────────────────────────────────────── 50ms

[1.1] ORT 1.22.0 CPU:
  █████████████████████████████████████████████████████████████████ 65.7ms  ✗

[1.1] ORT 1.24.2 CPU:
  ██████████████████████████████████ 34.7ms  ✓

[1.3] HTP FP16 full opts:
  ██████████████████████ 23.0ms  ✓✓

[1.3] HTP QDQ INT8:
  ██████ 6.2ms  ✓✓✓  Single best

[2.2] Dual QDQ INT8 [11]:
  ██████████ 10.2ms  ✓✓✓  Dual small

[2.3] Dual QDQ INT8 [3,5,7,11]:
  ███████████████ 15.2ms  ✓✓✓  Dual large

[4.1] BAFNetPlus full QDQ INT8 (unified):
  █████████████ 13.4ms  ✓✓✓  Unified BAFNet+
```

> **QDQ INT8 + HTP full opts가 최적 경로.**
> 단일 backbone 6.2~10.1ms (budget 12~20%), dual backbone 10.2~15.2ms (budget 20~30%).
> 모델 표현력(time_block_kernel 확대)을 키워도 budget 내 여유 충분.
> BAFNet 전체 파이프라인(dual backbone + fusion)은 50ms budget 내에서 편안하게 동작.
> BAFNetPlus unified graph 도 13.4ms (budget 27%) 로 실시간 여유.

---

## 4. BAFNetPlus (Unified Graph, Stage 5)

BAFNetPlus 는 BAFNet 을 확장한 통합 아키텍처 — mapping + masking + calibration + fusion (alpha/gamma) 을 **단일 ONNX 그래프** 로 export. dual backbone 병렬 전략 대신 HTP 가 통합 그래프 내부의 parallelism 을 직접 최적화.

### 4.1 Test Environment (Stage 5)

| Item | Value |
|---|---|
| Device | SM-S938N (Galaxy S25+) — Snapdragon 8 Elite, Hexagon V79 |
| VTCM | 16 MB available (provider option vtcm_mb=8) |
| ORT | 1.24.2 (QNN SDK 2.42.0) |
| Model | `bafnetplus_qdq.onnx` 6.98 MB (md5=9a1b533de08e03ba9bf08fc69cb03d3a) |
| States | 166 (alpha 4 + calibration 2 + mapping 80 + masking 80) |
| Budget | 50.0ms per chunk (8 frames × 6.25ms) |
| Benchmark | 10 warmup + 50 sessions × 4 chunks = 200 chunks (per iter) |

### 4.2 Scenario Results

| Scenario | Backend | Mean | P50 | P95 | P99 | Peak PSS | Budget % (Mean) | Notes |
|---|---|---|---|---|---|---|---|---|
| A — cold-state | QNN HTP QDQ | **13.4ms** | 13.2ms | 14.0ms | 14.9ms | 440 MB | 27% | S5-1, REALTIME |
| B — same-process 10x | QNN HTP QDQ | 13.4ms | 13.2ms | 14.3ms | 17.3ms | 440 MB | 27% | S5-2, drift Mean +1.3% / P99 −2.1% |
| C1 — after LaCoSENet dual | QNN HTP QDQ | 13.3ms | — | 14.3ms | 16.0ms | 452 MB | 27% | S5-3, same-process sequence |
| C2 — after LaCoSENet + BAFNet+ | QNN HTP QDQ | 13.7ms | 13.4ms | 14.7ms | 16.0ms | 452 MB | 27% | S5-3, 3rd position, drift Mean +5.7% / P99 +4.3% |
| G — session lifecycle (×10) | QNN HTP QDQ | 27.9ms (1st chunk) | — | — | — | 443 MB / 260 MB | — | S5-7, init 8.3–9.0s, close 200–230ms, PSS recovers every cycle |

### 4.3 Derived metrics

Baselines measured same session (cold-state, SM-S938N):
- LaCoSENet single QDQ Small [11] Mean: **6.2ms**
- LaCoSENet dual concurrent QDQ Small [11] WallClock Mean: **10.2ms** (Overlap=0.637)

Derived (BAFNetPlus vs LaCoSENet baseline):

| Metric | Formula | Value | Target | Verdict |
|---|---|---|---|---|
| `fusion_overhead_ms` | BAFNet+ Mean − LaCoSENet Dual Mean | **+3.2ms** | 2–5 ms | ✅ within |
| `parallel_efficiency` | 2 × LaCoSENet Single Mean / BAFNet+ Mean | **0.925** | 0.7–0.85 | ✅ exceeds (unified > dual-session) |
| `budget_utilization_p95` | BAFNet+ P95 / 50ms | **0.28** | ≤ 0.8 | ✅ 72% margin |

> **핵심 관찰**: BAFNetPlus unified 그래프(13.4ms)가 LaCoSENet dual concurrent(10.2ms) 대비 3.2 ms overhead만 발생. fusion + calibration + alpha/gamma + 추가 TS block 5개를 포함해도 dual 세션 대비 overhead 3.2ms. QNN HTP 가 통합 그래프에서 추가 최적화 기회를 활용.

### 4.4 Acceptance checklist (plan §Stage 5)

| Item | Target | Actual | Status |
|---|---|---|---|
| Cold-state Mean | ≤ 20ms (goal) / ≤ 25ms (fallback) | 13.4ms | ✅ goal |
| Cold-state P95 | ≤ 40ms (budget 80%) | 14.0ms | ✅ 28% |
| Cold-state P99 | ≤ 50ms (budget 100%) | 14.9ms | ✅ 30% |
| Same-process RSS plateau | no growth | plateau 428–440 MB, no drift | ✅ |
| Cross-load PSS delta | ≤ 100 MB | +12 MB (452 vs 440) | ✅ |
| Milestone ±10% reproducibility | drift < 10% | +1.3% / +5.7% | ✅ |
| Session lifecycle recovery | heap returns to baseline | PSS 443 → 260 MB/cycle | ✅ |

### 4.5 Deployment verdict

**✅ 배포 가능** — 모든 acceptance 통과, budget 마진 72%, 12개월 이상 thermal stability 을 장기 후속 과제로 유지.

후속 최적화 여지 (모두 선택적, 배포 블로커 아님):
- **Session init 8.3s**: 실 서비스 cold start 시 context cache 로 < 500 ms 단축 가능 (Stage 6 R5 참고)
- **PSS 428 MB**: Scenario B 기준 일정 수준. 200 MB 대가 필요하면 VTCM 동적 해제 또는 배치/세션 lifecycle 조정 필요
- **P99 스파이크 17.3ms (Scenario B)**: single-test 14.9ms 대비 tail 약간 증가. 실 사용 시나리오에서 1 % 이하 chunk 가 budget 의 ~35 % 차지하나 여전히 realtime

### 4.6 Reproduction

```bash
# Cold-state
adb shell am instrument -w -r \
  -e class com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkBafnetplusFullQdq \
  com.lacosenet.benchmark.test/androidx.test.runner.AndroidJUnitRunner

# Same-process 10x
adb shell am instrument -w -r \
  -e class com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkBafnetplusFullQdqRepeat10x \
  com.lacosenet.benchmark.test/androidx.test.runner.AndroidJUnitRunner

# Cross-load sequence (3 tests in one process)
adb shell am instrument -w -r \
  -e class com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkDualBackboneConcurrentQdq,com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkBafnetplusFullQdq,com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkBafnetplusFullQdqRepeat10x \
  com.lacosenet.benchmark.test/androidx.test.runner.AndroidJUnitRunner

# Session lifecycle (10 cycles)
adb shell am instrument -w -r \
  -e class com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkBafnetplusSessionLifecycleQdq \
  com.lacosenet.benchmark.test/androidx.test.runner.AndroidJUnitRunner

# Meminfo sampling (0.5s interval, background):
while :; do echo "$(date +%s.%3N) $(adb shell 'dumpsys meminfo com.lacosenet.benchmark | grep TOTAL\\ PSS')"; sleep 0.5; done > meminfo.log &
```

Logs: `docs/review/logs/stage5_bafnetplus_meminfo_*.log`, `stage5_crossload_meminfo_*.log`, `stage5_lifecycle_meminfo_*.log`.

---

## Appendix

### A. QNN HTP Debugging Log

ORT 1.22.0 → 1.24.2 업그레이드 과정에서 해결한 이슈:

| # | 문제 | 해결 |
|---|---|---|
| 1 | `dlopen failed: libQnnHtp.so not found` | `backend_path`를 라이브러리 이름으로 변경 |
| 2 | `QNN_DEVICE_ERROR_INVALID_CONFIG` | ORT 1.22.0 QNN SDK가 Hexagon V79 미지원 → 1.24.2로 업그레이드 |
| 3 | `dlopen failed: libcdsprpc.so not found` | AndroidManifest에 `<uses-native-library>` 추가 |
| 4 | HTP 동작하지만 CPU보다 느림 (108ms) | minimal opts → full opts + QDQ INT8 |

### B. QDQ INT8 Export

```bash
python -m src.models.streaming.onnx.export_onnx --no_checkpoint \
    --chunk_size 8 --encoder_lookahead 3 --decoder_lookahead 3 \
    --output_dir android/benchmark-app/src/main/assets \
    --output_name model.onnx --quantize_qdq --qdq_activation_type QUInt8
```

- QUInt16 activation은 HTP V79에서 PReLU 미지원 (`could not create op: q::prelu.opt`)
- QUInt8로 변경하면 모든 op이 HTP에서 실행

### C. Reproduction Commands

```bash
# Single backbone benchmarks
cd android && ./gradlew :benchmark-app:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkCpu
cd android && ./gradlew :benchmark-app:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkQnnHtp
cd android && ./gradlew :benchmark-app:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkQnnHtpQdq
cd android && ./gradlew :benchmark-app:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkQnnHtpCached
cd android && ./gradlew :benchmark-app:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkQnnHtpQdqCached

# Dual backbone benchmarks
cd android && ./gradlew :benchmark-app:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkDualBackboneSequential
cd android && ./gradlew :benchmark-app:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkDualBackboneConcurrent
cd android && ./gradlew :benchmark-app:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkDualBackboneConcurrentQdq

# HTP profiling
cd android && ./gradlew :benchmark-app:connectedAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=com.lacosenet.benchmark.StreamingBenchmarkTest#benchmarkQnnHtpWithProfiling
adb pull /data/user/0/com.lacosenet.benchmark/cache/qnn_profile.csv ./qnn_profile.csv

# Check results
adb logcat -s StreamingBenchmark -d | tail -80
```

### D. Code Changes

| File | Change |
|---|---|
| `StreamingBenchmarkTest.kt` | CPU/HTP/NNAPI/cached/QDQ/dual benchmark 전체 구현 |
| `AndroidManifest.xml` | `<uses-native-library android:name="libcdsprpc.so">` |
| `build.gradle.kts` (benchmark, streaming) | ORT 1.22.0 → 1.24.2 |
| `export_onnx.py` | `--quantize_qdq`, `--quantize`, `--simplify` |

### E. Known Issues

1. **Tail latency 변동**: P95/P99가 실험 간 편차 있음 (thermal throttling 영향)
2. **QUInt16 미지원**: HTP V79에서 PReLU의 QUInt16 미지원, QUInt8만 사용 가능
3. **FP16 dual 비실용**: FP16 concurrent wall-clock 43~48ms로 budget 경계, QDQ INT8 필수
