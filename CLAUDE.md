# CLAUDE.md - Development Guide

## Quick Start Commands

### Build/Run
```bash
# Install dependencies
pip install -r requirements.txt
# Or with dev dependencies
pip install -e ".[dev]"

# Train model
CUDA_VISIBLE_DEVICES=0 python -m src.train +model=backbone_masking +dset=taps

# Resume training
python -m src.train +model=backbone_masking +dset=taps \
  continue_from=outputs/checkpoint_dir

# Run inference
python -m src.enhance --chkpt_dir outputs/model_dir --chkpt_file best.th \
  --noise_dir /path/to/noise --rir_dir /path/to/rir \
  --snr 0 --output_dir samples

# Run evaluation
python -m src.evaluate --model_config <cfg> --chkpt_dir <dir> --chkpt_file best.th \
  --dataset taps --snr_step 0 \
  --noise_dir <...> --noise_test dataset/taps/noise_test.txt \
  --rir_dir <...>   --rir_test   dataset/taps/rir_test.txt

# Track experiments (parse results to CSV/Markdown)
python results/track_experiment.py --update

# Compute receptive field and latency
python src/compute_rf.py --experiment prk_1117_1 --csv results/experiments.csv

# Streaming inference — PT wrapper / ORT host wrapper (post-2026-05-13 rebuild)
# - PT streaming: BAFNetPlusStreaming.from_checkpoint('results/experiments/bafnetplus_50ms')
#   (src/models/streaming/bafnetplus_streaming.py — S6)
# - FP32 ONNX export: export_bafnetplus_to_onnx_from_checkpoint(...)
#   (src/models/streaming/onnx/export.py — S8; sidecar JSON schema
#   's8-bafnetplus-functional-fp32', 190 states, T_export=14)
# - ORT host wrapper: BAFNetPlusOrtStreaming.from_onnx('results/onnx/bafnetplus_50ms_fp32.onnx')
#   (src/models/streaming/onnx/ort_wrapper.py — S9; same public API as PT wrapper)

# Full-utterance 3-path eval harness (S10):
python -m src.analysis.eval_streaming --preset full \
  --unified-ckpt-dir results/experiments/bafnetplus_50ms \
  --onnx-artifact results/onnx/bafnetplus_50ms_fp32.onnx \
  --output-json results/eval_streaming/run.json \
  --output-markdown results/eval_streaming/run.md
# Or via the canonical shell wrapper:
./scripts/eval_streaming.sh full
./scripts/eval_streaming.sh smoke   # TAPS idx [0] only, LP target 1e-4
```

### Linting and Testing
```bash
# Run all tests
make test
# Or: pytest tests/ -v

# Run single test file
pytest tests/test_models.py -v

# Run single test
pytest tests/test_models.py::TestBackbone::test_backbone_forward -v

# Format code
make format
# Or: black . && isort .

# Lint code
make lint
# Or: flake8 . && mypy .
```

## Code Style Rules

### Python Style
- **Line length**: 120 characters (Black + isort configured)
- **Formatting**: Use Black for auto-formatting
- **Import sorting**: Use isort with Black profile
- **Naming conventions**:
  - Classes: `PascalCase` (e.g., `Backbone`, `DenseEncoder`)
  - Functions/methods: `snake_case` (e.g., `mag_pha_stft`, `tailor_dB_FS`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SR`)
  - Private methods: `_leading_underscore` (e.g., `_run_one_epoch`)

### Documentation
- Use docstrings for all public functions, classes, and modules
- Format: Google-style docstrings
- Example:
```python
def mag_pha_to_complex(mag: Tensor, pha: Tensor) -> Tensor:
    """Convert magnitude and phase to complex spectrogram.

    Args:
        mag: Magnitude spectrogram [B, F, T]
        pha: Phase spectrogram [B, F, T]

    Returns:
        Complex spectrogram [B, F, T, 2] with real and imaginary parts
    """
```

### Type Hints
- Use type hints for function signatures
- Import types from `typing` module
- Example: `from typing import List, Tuple, Optional`

## Important Files

### Core Files
- `src/train.py`: Main training entry point using Hydra
- `src/enhance.py`: Inference script
- `src/evaluate.py`: Evaluation script with metrics
- `src/compute_metrics.py`: Metric computation utilities
- `src/analysis/`: Public offline analysis utilities
- `src/solver.py`: Training loop implementation (Solver class)
- `src/data.py`: Dataset and data augmentation logic
- `src/stft.py`: STFT/iSTFT utilities
- `src/runtime_common.py`: Shared runtime helpers for evaluation/enhancement CLIs
- `src/utils.py`: Utility functions
- `src/models/backbone.py`: Backbone model architecture (BatchNorm2d, CausalConv1d SCA)
- `src/models/streaming/`: Streaming inference modules (post-2026-05-13 rebuild)
  - `layers/stateful_conv.py`: Stateful `CausalConv1d` / `AsymmetricConv2d` / `CausalConv2d` (S2)
  - `lookahead.py`: `compute_lookahead(backbone) -> (L_enc, L_dec)` (S2)
  - `context.py`: `StateFramesContext` thread-local (S2)
  - `converters.py`: `convert_to_stateful` / `prepare_streaming_model` (S2)
  - `backbone_streaming.py`: PT streaming single-Backbone wrapper (S3)
  - `bafnetplus_streaming.py`: PT streaming `BAFNetPlusStreaming` wrapper (S6)
  - `onnx/functional_stateful.py`: Explicit-state functional conv variants (S2)
  - `onnx/state_registry.py`: Functional-state collection (S2)
  - `onnx/backbone_core.py`: `ExportableBackboneCore` single-Backbone export form (S4)
  - `onnx/bafnetplus_core.py`: `BAFNetPlusCore` (non-streaming, S5) +
    `ExportableBAFNetPlusCore` (functional-stateful, S8)
  - `onnx/export.py`: FP32 ONNX export drivers + multi-step verify (S4 + S8)
  - `onnx/ort_wrapper.py`: `BAFNetPlusOrtStreaming` (S9; FP32 ONNX host wrapper,
    same public API as the PT wrapper)
- `results/track_experiment.py`: Experiment tracking and result parser

### Configuration
- `conf/config.yaml`: Base configuration
- `conf/model/*.yaml`: Model-specific configs
- `conf/dset/*.yaml`: Dataset-specific configs
- Hydra manages outputs in `outputs/<timestamp>/`

### Testing
- `tests/test_*.py`: Unit tests for each module
- `pytest.ini` (in pyproject.toml): Test configuration
- Use fixtures for reusable test data

## Repository Etiquette

### Before Committing
1. **Format your code**: `make format`
2. **Run linters**: `make lint`
3. **Run tests**: `make test`
4. **Check git status**: Ensure no unintended files are staged

### Commit Messages
- Use conventional commits format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `test:` for tests
  - `refactor:` for refactoring
  - `style:` for formatting changes
- Example: `feat: add Backbone-Mamba variant`

### Pull Requests
- Create feature branches: `git checkout -b feature/your-feature`
- Keep PRs focused and small
- Write descriptive PR descriptions
- Ensure CI passes before requesting review

### Branches
- `main`: Production-ready code
- `dev`: Development branch
- `feature/*`: Feature branches
- `fix/*`: Bug fix branches

## Single-Test Recipe

To run a single test quickly during development:

```bash
# Test specific function
pytest tests/test_stft.py::TestSTFT::test_mag_pha_to_complex_shape -v

# Test entire class
pytest tests/test_models.py::TestBackbone -v

# Test with coverage
pytest tests/test_models.py -v --cov=src.models --cov-report=term-missing

# Test with verbose output and stop on first failure
pytest tests/ -vsx
```

## Project Structure

```
BAFNet-plus/
├── README.md               # Project overview
├── CLAUDE.md              # Development guide (this file)
├── requirements.txt       # Python dependencies
├── src/                   # Source code (library + scripts)
│   ├── data.py           # Dataset and data augmentation
│   ├── solver.py         # Training loop (Solver class)
│   ├── stft.py           # STFT/iSTFT utilities
│   ├── runtime_common.py # Shared runtime helpers
│   ├── utils.py          # Utility functions
│   ├── train.py          # Training entry point
│   ├── enhance.py        # Inference script
│   ├── evaluate.py       # Evaluation script
│   ├── analysis/         # Offline analysis utilities
│   ├── compute_metrics.py # Metric computation
│   └── models/           # Model architectures
│       ├── backbone.py   # Backbone (BatchNorm2d, CausalConv1d SCA)
│       ├── bafnet.py
│       ├── discriminator.py  # MetricGAN discriminator
│       └── streaming/    # Streaming inference (rebuilt 2026-05-13)
│           ├── context.py            # StateFramesContext (S2)
│           ├── converters.py         # convert_to_stateful, prepare_streaming_model (S2)
│           ├── lookahead.py          # compute_lookahead (S2)
│           ├── backbone_streaming.py # PT streaming single-Backbone wrapper (S3)
│           ├── bafnetplus_streaming.py # PT streaming BAFNet+ wrapper (S6)
│           ├── layers/stateful_conv.py # StatefulCausalConv1d / AsymmetricConv2d / CausalConv2d (S2)
│           └── onnx/             # FP32 ONNX export + ORT host wrapper
│               ├── functional_stateful.py # Explicit-state conv variants (S2)
│               ├── state_registry.py      # State collection (S2)
│               ├── backbone_core.py       # ExportableBackboneCore (S4)
│               ├── bafnetplus_core.py     # BAFNetPlusCore (S5) + ExportableBAFNetPlusCore (S8)
│               ├── export.py              # ONNX export driver + verify (S4 + S8)
│               └── ort_wrapper.py         # BAFNetPlusOrtStreaming (S9)
├── conf/                  # Hydra configurations
├── dataset/               # Dataset file lists
├── scripts/               # Experiment scripts
├── tests/                 # Unit tests
├── outputs/               # Training outputs (auto-generated)
└── results/               # Experiment tracking
    ├── track_experiment.py # Experiment tracking tool
    ├── experiments/       # Saved experiment results
    ├── experiments.csv   # Parsed experiment data
    └── EXPERIMENTS.md    # Human-readable summary
```

## Common Tasks

### Adding a New Model
1. Create `src/models/your_model.py` with model class
2. Add configuration in `conf/model/your_model.yaml`
3. Add unit tests in `tests/test_models.py`
4. Test: `pytest tests/test_models.py::TestYourModel -v`

### Modifying Training Loop
1. Edit `src/solver.py` (Solver class)
2. Update relevant tests in `tests/`
3. Run: `make test`

### Adding New Loss Function
1. Add implementation to `src/utils.py` or `src/solver.py`
2. Update loss configuration in `conf/config.yaml`
3. Add unit test for the loss function

### Computing Receptive Field and Latency

The `src/compute_rf.py` tool calculates the receptive field size and algorithmic latency of Backbone models based on their hyperparameters.

**Usage:**
```bash
# From experiment in CSV
python src/compute_rf.py --experiment prk_1117_1 --csv results/experiments.csv

# With direct parameters
python src/compute_rf.py --dense_depth 4 --num_tsblock 4 \
  --time_block_kernel 3 5 7 11 --causal True \
  --encoder_padding_ratio 0.5 0.5
```

**Output includes:**
- Time-axis and frequency-axis receptive field sizes
- Algorithmic latency in frames, samples, and milliseconds
- Layer-wise breakdown (DenseEncoder, TS_BLOCK, MaskDecoder)
- Configuration summary

**Example output:**
```
Configuration:
  Model: Backbone
  Causal: True
  Encoder Padding: (1.0, 0.0)  # Fully causal

Receptive Field:
  Time-axis RF: 105 frames (656.2ms @ 16kHz)
  Frequency-axis RF: 260 bins

Algorithmic Latency:
  STFT hop size: 100 samples (6.25ms)
  Total latency: 650.00ms

Layer-wise Breakdown:
  DenseEncoder:    RF_time=9, RF_freq=11
  TS_BLOCK x4:     RF_time=+92, RF_freq=+244
  MaskDecoder:     RF_time=9, RF_freq=10
```

**Key insights:**
- **Causal models** with `encoder_padding_ratio=(1.0, 0.0)` have ~650ms latency
- **Symmetric models** with `encoder_padding_ratio=(0.5, 0.5)` have ~325ms latency but use future context
- Latency is primarily determined by encoder padding ratio and receptive field size

### Tracking Experiments

The `results/track_experiment.py` tool automatically parses experiment results from `results/experiments/` subdirectories and generates CSV and Markdown documentation.

**Usage:**
```bash
# Full reparse of all experiments (first time)
python results/track_experiment.py --full

# Incremental update (add only new experiments)
python results/track_experiment.py --update

# Parse specific experiment
python results/track_experiment.py --experiment prk_1104_2

# Check schema and statistics
python results/track_experiment.py --check
```

**What it parses:**
- Performance metrics: PESQ, STOI, CER, WER, CSIG, CBAK, COVL (from `trainer.log`)
- Model hyperparameters: All params from `model.param` (from `.hydra/config.yaml`)
- Model version: Modification date + file hash of model code

**Output files:**
- `results/experiments.csv`: Complete data in CSV format (pandas-friendly)
- `results/EXPERIMENTS.md`: Human-readable tables grouped by model type

**Schema management:**
- Uses Union schema: all parameter columns across different model versions
- Missing parameters are filled with "N/A"
- Automatically adds new columns when new parameters are detected

### Dual-Recipe 관리 (baselines/, r7.7 2026-04-20)

External baseline (`baselines/<model>/`)는 **dual-recipe** 구조를 따릅니다:

- `baselines/<model>/config.yaml`: **unified** recipe (BAFNet+ 통일 recipe, Main Table I/II 등재용)
- `baselines/<model>/config_paper.yaml`: **paper** recipe (원 논문 recipe 근사, Supp Table S1 등재용)
- Outputs: `baselines/<model>/outputs/<recipe>/<timestamp>/` (recipe 별 분리)
- Paper recipe가 불가능한 모델은 `config_paper.yaml` 부재 (현재 VibVoice만 해당 — 3-ch IMU @ 1.6 kHz modality non-reproducible)

**학습 실행** (Hydra `--config-name` override):

```bash
# Unified (default)
python -m baselines.dccrn.train

# Paper recipe
python -m baselines.dccrn.train --config-name config_paper
```

**Smoke test** (`baselines/common/test_smoke.py` extended with `--config-name`):

```bash
python -m baselines.common.test_smoke baselines/dccrn --config-name config
python -m baselines.common.test_smoke baselines/dccrn --config-name config_paper
```

**평가** (`baselines/common/eval.sh --recipe {unified|paper}`):

```bash
./baselines/common/eval.sh multi-snr --model dccrn --recipe unified --gpus 0
./baselines/common/eval.sh multi-snr --model dccrn --recipe paper   --gpus 0
# 출력: results/eval/baselines/multi-snr/dccrn_<recipe>.json
```

**제약 사항 (S0-C4 policy)**:
- Cross-recipe checkpoint reference 금지 — 예: BAFNet conf `checkpoint_dccrn`은 unified BAFNet conf 학습 시 unified DCCRN best.th만 참조, paper BAFNet conf 학습 시 paper DCCRN best.th만 참조
- Unified recipe의 config.yaml hyperparameter 변경 금지 (S0-C2 r3 final)
- Paper recipe에서 solver 한계로 재현 불가한 항목 (L_time, sub-band loss, ExpLR scheduler 등)은 Supp §S1 footnote로 disclosure

상세: `baselines/README.md` §Training recipes (dual-recipe), `paper_work/review_pipeline/cycles/c02_external_baselines/changelog.md` r7.7 entry.

### Debugging Training
1. Check TensorBoard logs: `tensorboard --logdir outputs/your_run/tensorbd`
2. Inspect Hydra config: `cat outputs/your_run/.hydra/config.yaml`
3. Check training logs: `less outputs/your_run/trainer.log`

## Tips

- Use `hydra.utils.to_absolute_path()` for file paths in configs
- Training outputs are isolated in timestamped directories
- Checkpoint saves: `checkpoint.th` (latest), `best.th` (best validation)
- Use `continue_from` to resume training from a checkpoint
- Model code is auto-saved in output dir when `save_code=true`

## Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0,1

# Disable CUDA (CPU only)
export CUDA_VISIBLE_DEVICES=""

# Adjust number of workers for data loading
# (Set in conf/config.yaml: num_workers)
```

## Troubleshooting

### OOM (Out of Memory)
- Reduce `batch_size` in config
- Reduce `num_tsblock` or `dense_channel` in model config
- Check if gradients are being freed properly

### Hydra Path Issues
- Use `hydra.utils.to_absolute_path()` for file paths
- Check `outputs/<run>/.hydra/config.yaml` for resolved paths

### Dataset Loading Errors
- Verify dataset paths in `conf/dset/*.yaml`
- Check noise/RIR file lists exist in `dataset/`
- Ensure HuggingFace datasets are accessible

### Test Failures
- Run single test: `pytest tests/test_file.py::test_function -v`
- Add `-s` flag to see print statements: `pytest -s`
- Use `--pdb` to drop into debugger on failure

### Remote Training Log Duplication
- `run_train.sh`가 원격 stdout을 `tail -f`로 스트리밍할 때, 일부 메시지가 2번 나타날 수 있음
- 원인: 모델 코드의 `print()` (→ stdout)와 `logger.info()` (→ trainer.log → stdout)가 별도 경로로 출력
- 예: `[BAFNetPlus]` 메시지, HuggingFace Warning 등
- **학습 자체에는 영향 없음** — `trainer.log`에는 중복 없이 정상 기록됨
- 해결: 모델 코드에서 `print()` 대신 `logger`를 사용하면 근본적으로 해결 가능

## Resources

- **Hydra Docs**: https://hydra.cc/docs/intro/
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **Black Formatting**: https://black.readthedocs.io/
- **Pytest Guide**: https://docs.pytest.org/

## 코딩 가이드라인 (Karpathy-derived)

### 1. Think Before Coding
- 가정을 명시적으로 밝힌다. 불확실하면 먼저 질문한다.
- 여러 해석이 가능하면 선택지를 제시한다 — 임의로 하나를 고르지 않는다.
- 더 단순한 접근이 존재하면 말한다. 반박이 필요하면 반박한다.
- 혼란스러우면 멈추고, 무엇이 불명확한지 짚고 질문한다.

### 2. Simplicity First
- 요청된 것만 구현한다. 투기적 기능, 단일 용도 추상화, 요청되지 않은 "유연성"이나 "설정 가능성"을 추가하지 않는다.
- 200줄로 작성한 코드가 50줄로 가능하면 다시 쓴다.

### 3. Surgical Changes
- 요청과 직접 관련된 코드만 수정한다. 인접 코드, 주석, 포맷팅을 "개선"하지 않는다.
- 깨지지 않은 것을 리팩토링하지 않는다. 기존 스타일을 따른다.
- 내 변경으로 인해 미사용된 import/변수/함수는 제거한다. 기존 dead code는 요청 없이 건드리지 않는다.

### 4. Goal-Driven Execution
- 성공 기준을 먼저 정의하고, 검증될 때까지 루프한다.
- 다단계 작업은 검증 체크포인트가 포함된 간단한 계획을 먼저 제시한다.

## 사용자 워크플로 선호

### Stage 프롬프트 전달 방식
- 다음 Stage 착수용 프롬프트는 **파일로 저장하지 말고 세션 응답으로 직접 출력**한다.
- `docs/review/BAFNETPLUS_STAGE<N>_PROMPT.md` 같은 파일 생성 금지 (사용자가 프롬프트를 복사해 다음 세션에 붙여넣는 용도이므로 파일 불필요).
- Stage 실행 결과 / 포스트모템은 `docs/review/BAFNETPLUS_PORT_PLAN.md` 같은 상위 계획 문서에 기록하는 것만 허용.

### Host parity 1-utt cycle gating convention (2026-05-16)

Cycle 의사결정용 host parity 는 **1 utt (TAPS idx=0)** 로 진행한다. 기존 5-utt
"S1 validation list" 는 wiki promotion / deployable verification / 논문 측정
단계에서만 사용한다 (3-tier 운영).

| Tier | 용도 | utt 수 | `--taps-indices` | 소요 시간 (INT8 QDQ 기준) |
|---|---|:---:|---|---:|
| 1 | Cycle gating (의사결정) | **1** | `0` | 약 5 분 |
| 2 | Wiki promotion (결과 기록) | 5 | `0 1 2 3 4` | 약 25 분 |
| 3 | Paper / deployable verification | 30+ | full TAPS test split | 수 시간 |

**근거**:
- 이전 S17~S24 cycle 들의 5-utt `|dRO|` 분포 std 가 좁음 (≤ 0.04). Cycle-decisive
  차이 (Δ ≥ 0.1) 의 판정에는 1-utt 측정의 noise 가 안전한 마진 안에 있음.
- idx=0 은 5-utt mean 의 합리적 추정치 — 예: S21 baseline 의 idx=0 `|dRO|` 0.355
  vs 5-utt mean 0.331 (Δ 0.024, std 0.04 안).
- 기존 `eval_streaming.sh smoke` preset 의 convention 과 일치 (S10 부터).

**Tier 2 / Tier 3 로 promotion 해야 하는 경우**:
- Δ `|dRO|` < 0.05 같은 작은 차이의 통계적 유의성 필요 시 → Tier 2 (5 utt)
- Deployable bound (`|d_PESQ| ≤ 5e-2`) 직전 측정 → Tier 2 또는 Tier 3
- 논문 측정 (Main Table, Supp) → Tier 3 (full split + multi-seed)
- Wiki 결과 테이블 기록 직전 → Tier 2 로 재측정 후 기록

**코드/스크립트 정책**:
- `Android_projects/scripts/parity/check_onnx_parity.py` 의 `--taps-indices` default
  는 `[0,1,2,3,4]` 그대로 유지 (S11+ canonical contract, downstream tools 의존성
  존재). 변경하지 않는다.
- Cycle 작업 시에는 호출 측에서 `--taps-indices 0` 을 **명시적으로 전달**.
- Wiki promotion / paper 측정 시에는 명시 없이 default 사용 또는 `0 1 2 3 4` 명시.

Cross-ref:
[wiki § Host parity (current)](../docs/wiki/concepts/android-streaming-deployment.md#host-parity-current)
의 cycle gating note.
