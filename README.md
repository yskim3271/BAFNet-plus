# BAFNet-plus: Semi-Real-Time Speech Enhancement

A PyTorch-based speech enhancement framework designed for semi-real-time processing with high performance. The core architecture is **Backbone** — a convolution-based model with asymmetric padding for latency control — supporting both masking and mapping inference. **BAFNet** extends this with dual-input (throat mic + acoustic mic) fusion.

## Features

- **Backbone Architecture**: Convolution-based model with asymmetric padding for latency control (6.25ms–2s)
- **BAFNet Dual-Input Fusion**: Combines throat mic and acoustic mic signals via learned fusion weights
- **Comprehensive Training**: Multi-loss framework with MetricGAN discriminator
- **Flexible Data Pipeline**: Noise augmentation with SNR control and reverberation
- **Hydra Configuration**: Modular, reproducible experiment management
- **Mapping & Masking**: Two inference approaches for speech enhancement

## Pre-Trained Checkpoints & Audio Samples

Pre-trained PyTorch checkpoints, sanitized self-contained Hydra configurations, and the deployable 50 ms FP32 streaming ONNX artifact are hosted on HuggingFace Hub, accompanied by an interactive audio sample and spectrogram comparison page:

- **HuggingFace Hub Checkpoints & ONNX**: [yskim3271/BAFNet-plus](https://huggingface.co/yskim3271/BAFNet-plus)
- **Audio Samples & Spectrogram Demo**: [https://yskim3271.github.io/BAFNet-plus/](https://yskim3271.github.io/BAFNet-plus/)

### Released Model Artifacts

| Artifact Path | Corpus | $T_{\text{alg}}$ | Chunk Size | Notes |
|---|---|---:|---:|---|
| `checkpoints/bafnetplus_12ms/` (`best.th`, `config.yaml`) | TAPS | 12.5 ms | 2 frames | GPU real-time tier ($\rho=0.975$) |
| `checkpoints/bafnetplus_25ms/` (`best.th`, `config.yaml`) | TAPS | 25 ms | 4 frames | Low-latency GPU tier ($\rho=0.95$) |
| `checkpoints/bafnetplus_50ms/` (`best.th`, `config.yaml`) | TAPS | 50 ms | 8 frames | **Default operating point** ($\rho=0.90$, mobile DSP RTF 0.80) |
| `checkpoints/bafnetplus_75ms/` (`best.th`, `config.yaml`) | TAPS | 75 ms | 12 frames | Real-time across GPU, mobile DSP, and server CPU ($\rho=0.84$) |
| `checkpoints/bafnetplus_100ms/` (`best.th`, `config.yaml`) | TAPS | 100 ms | 16 frames | Highest low-SNR PESQ tier ($\rho=0.78$) |
| `checkpoints/vbx_bafnetplus_50ms/` (`best.th`, `config.yaml`) | Vibravox | 50 ms | 8 frames | Cross-corpus French validation ($\rho=0.90$) |
| `onnx/bafnetplus_50ms_fp32.onnx` (+ `.onnx.json`) | TAPS | 50 ms | 8 frames | 190-state FP32 streaming ONNX graph + sidecar JSON |

Every released `config.yaml` embeds `model.param.args_mapping` and `model.param.args_masking` directly, so checkpoints load standalone without separate branch pre-training files.

### Download & Inference Quickstart

```bash
pip install huggingface_hub omegaconf
```

```python
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from src.checkpoint import ConfigDict, load_checkpoint, load_model
from src.stft import mag_pha_istft, mag_pha_stft
from src.utils import get_stft_args_from_config

repo_dir = Path(snapshot_download("yskim3271/BAFNet-plus"))
ckpt_dir = repo_dir / "checkpoints" / "bafnetplus_50ms"

cfg = OmegaConf.load(ckpt_dir / "config.yaml")
params = OmegaConf.to_container(cfg.model.param, resolve=True)
params["args_mapping"] = ConfigDict(params["args_mapping"])
params["args_masking"] = ConfigDict(params["args_masking"])

model = load_model(cfg.model.model_lib, cfg.model.model_class, params, device="cpu")
model = load_checkpoint(model, str(ckpt_dir), "best.th", device="cpu")
model.eval()

stft_args = get_stft_args_from_config(cfg.model)
stft_args["compress_factor"] = float(cfg.compress_factor)

# Paired 16 kHz waveforms [1, T]
bcs = torch.randn(1, 16000)
noisy_acs = torch.randn(1, 16000)
with torch.inference_mode():
    _, _, bcs_com = mag_pha_stft(bcs, **stft_args)
    _, _, acs_com = mag_pha_stft(noisy_acs, **stft_args)
    est_mag, est_pha, _ = model((bcs_com, acs_com))
    enhanced = mag_pha_istft(est_mag, est_pha, **stft_args)
```

For chunked ONNX Runtime streaming inference (`800` samples / `50 ms` per step at `16 kHz`):

```python
from src.models.streaming.onnx.ort_wrapper import BAFNetPlusOrtStreaming

streamer = BAFNetPlusOrtStreaming.from_onnx(str(repo_dir / "onnx" / "bafnetplus_50ms_fp32.onnx"))
enhanced_stream = streamer.process_audio(bcs.squeeze(0), noisy_acs.squeeze(0))
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yskim3271/BAFNet-plus.git
cd BAFNet-plus

# Install PyTorch (choose your CUDA version from pytorch.org)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install HuggingFace datasets
pip install datasets

# Install other dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 1.13
- torchaudio
- CUDA (recommended for training)
- For full dependencies, see `requirements.txt`

### Training

```bash
# Train Backbone (masking) on TAPS dataset
CUDA_VISIBLE_DEVICES=0 python -m src.train +model=backbone_masking +dset=taps

# Resume from checkpoint
python -m src.train +model=backbone_masking +dset=taps \
  continue_from=results/experiments/2025-10-10_03-52-05
```

Training outputs (checkpoints, logs, samples) are saved to `results/experiments/<timestamp>/` via Hydra.

### Inference

```bash
python -m src.enhance.py \
  --chkpt_dir results/experiments/your_model_dir \
  --chkpt_file best.th \
  --noise_dir /path/to/noise \
  --noise_test dataset/noise_test.txt \
  --rir_dir /path/to/rir \
  --rir_test dataset/rir_test.txt \
  --snr 0 \
  --output_dir samples
```

### Evaluation

The unified `scripts/eval.sh` driver supports two scenarios:

```bash
# Scenario 1: multi-SNR evaluation (varying SNRs, fixed clean BCS/ACS gains)
./scripts/eval.sh multi-snr --group bafnetplus --gpus 0,1

# Scenario 2: gain-robustness (sweep BCS-ACS gain offset at multiple SNRs)
./scripts/eval.sh gain-robustness bafnetplus_50ms abl_no_calibration --gpus 0,1
```

Or run `src.evaluate` directly:

```bash
python -m src.evaluate \
  --model_config results/experiments/model_dir/.hydra/config.yaml \
  --chkpt_dir results/experiments/model_dir \
  --chkpt_file best.th \
  --noise_dir /path/to/noise \
  --noise_test dataset/noise_test.txt \
  --rir_dir /path/to/rir \
  --rir_test dataset/rir_test.txt \
  --snr_step 0 \
  --eval_stt
```

When `--output_json` is provided, `src.evaluate` writes a JSON object with
`metadata` and `metrics` top-level fields. The metadata records evaluation inputs
such as dataset/checkpoint paths, SNR steps, BCS/ACS gain offsets, and metric
flags; derived relative gain is intentionally not stored.

### Analysis Utilities

Reusable analysis utilities are available under `src.analysis`:

```bash
# Model Params/MACs/RTF summary
python -m src.analysis.complexity --experiment bafnetplus_50ms

# Per-utterance metrics for statistical analysis
# (nested under each SNR block as a 'per_utterance' key in the output JSON)
python -m src.evaluate \
  --model_config results/experiments/model_dir/.hydra/config.yaml \
  --chkpt_dir results/experiments/model_dir \
  --noise_dir /path/to/noise \
  --noise_test dataset/taps/noise_test.txt \
  --rir_dir /path/to/rir \
  --rir_test dataset/taps/rir_test.txt \
  --snr_step -20 -10 0 10 15 \
  --save_per_utterance_json \
  --output_json results/analysis/per_utterance.json

# BAFNetPlus calibration/fusion traces
python -m src.analysis.forward_traces \
  --model_config results/experiments/bafnetplus_50ms/.hydra/config.yaml \
  --chkpt_dir results/experiments/bafnetplus_50ms \
  --snr_step -10 0 10 \
  --output_npz results/analysis/bafnetplus_50ms_traces.npz
```

Paper-specific report generators and generated figures/tables are intentionally kept in ignored local workspaces.

## Model Architectures

### Backbone (Convolution-Based)

```
Input (Noisy Complex Spectrogram [B, F, T, 2])
    ↓
[Extract Mag & Phase → Concatenate [B, 2, T, F]]
    ↓
[Dense Encoder: DS_DDB blocks]
    ↓
[TS_BLOCKs x N: Two-Stage Time-Frequency Processing]
    ↓
[Dual Decoders: Mask Decoder + Phase Decoder]
    ↓
Output (Enhanced Mag, Phase, Complex Spectrogram)
```

**Key Components:**
- **DS_DDB**: Dilated-separable dense blocks with exponential dilation (1, 2, 4, 8)
- **TS_BLOCK**: Two-Stage blocks with sequential time-axis and frequency-axis convolutions
- **Asymmetric Padding**: Configurable `encoder_padding_ratio` for latency control
  - `(1.0, 0.0)`: Fully causal (~12.5ms latency)
  - `(0.5, 0.5)`: Symmetric (half-latency, uses future context)
- **Dual Decoders**: Separate magnitude masking/mapping and phase refinement paths
- **Inference Types**:
  - `masking`: Predicts mask to multiply with noisy magnitude
  - `mapping`: Directly predicts clean magnitude and phase

### BAFNet (Dual-Input Fusion)

```
Input (Throat Mic Signal + Acoustic Mic Signal)
    ↓
[Pre-trained Mapping Model] ← Throat Signal → Enhanced TM
[Pre-trained Masking Model] ← Acoustic Signal → Mask + Enhanced AM
    ↓
[STFT on Both Enhanced Signals]
    ↓
[Convolutional Blocks: Process Mask → Fusion Weight α]
    ↓
[Complex Spectrogram Fusion: α·TM + (1-α)·AM]
    ↓
[iSTFT]
    ↓
Output (Enhanced Acoustic Signal)
```

**Key Components:**
- **Dual-Input Processing**: Utilizes throat microphone (bcs) and acoustic microphone (acs) signals
- **Pre-trained Models**: Requires separate mapping and masking Backbone models
- **Adaptive Fusion**: Learns frequency-dependent fusion weights via convolutional blocks
- **Learnable Sigmoid**: Time-frequency adaptive weighting for optimal signal combination
- **Complex-Domain Fusion**: Operates on complex spectrograms for better phase preservation

### Model Variants

| Model | Architecture | Inference Type | Config Files |
|-------|-------------|----------------|--------------|
| **Backbone** | Convolution-based (LKFCA) | Masking / Mapping | `backbone_masking.yaml`, `backbone_mapping.yaml` |
| **BAFNet** | Dual-input (acs+bcs) fusion | Masking | `bafnet.yaml` |

**Inference Types:**
- **Masking**: Predicts magnitude/phase masks to apply to noisy spectrogram
- **Mapping**: Directly predicts clean magnitude/phase spectrograms

**Receptive Field (RF)** depends on kernel configurations. With default settings (num_tsblock=4, time_block_num=2, time_block_kernel=[3,5,7,11]), use `python src/receptive_field.py` to compute RF and algorithmic latency. See `src/models/backbone.py` for architecture details.

## Training Details

### Loss Function

The model is trained with a weighted combination of losses:

```python
loss = 0.9 * L_magnitude + 0.3 * L_phase + 0.1 * L_complex +
       0.05 * L_consistency + 0.05 * L_metric
```

- **L_magnitude**: MSE on magnitude spectrograms
- **L_phase**: Phase-aware loss (cosine distance)
- **L_complex**: MSE on complex spectrograms
- **L_consistency**: STFT-iSTFT consistency loss
- **L_metric**: MetricGAN adversarial loss (PESQ-driven)

### Data Augmentation

- **SNR Range**: -15 to 20 dB (uniform sampling)
- **Noise Types**: Environmental noise (DNS-Challenge, etc.)
- **Reverberation**: Optional RIR convolution (0-50% probability)
- **Dynamic Level**: Random dB FS adjustment (±10 dB)

### Hyperparameters

```yaml
sampling_rate: 16000
n_fft: 400
hop_size: 100
batch_size: 4
learning_rate: 5e-4
warmup_ratio: 0.1  # cosine annealing with linear warmup
optimizer: AdamW
epochs: 200
```

See `conf/config.yaml` for full configuration.

## Datasets

This project uses:
- **TAPS**: Throat and Acoustic Pairing Speech Dataset ([HuggingFace](https://huggingface.co/datasets/yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset))
- **Vibravox**: Contact microphone speech dataset ([HuggingFace](https://huggingface.co/datasets/yskim3271/vibravox_16k))

**Custom Datasets**: Implement a compatible HuggingFace dataset or modify `data.py` to support your audio format.

## Configuration

Hydra-based configuration allows flexible experiment management:

```bash
# Override model parameters
python -m src.train +model=backbone_masking model.param.dense_channel=128

# Override training settings
python -m src.train +dset=taps batch_size=8 lr=1e-3

# Train with different configs
python -m src.train +model=backbone_mapping +dset=vibravox epochs=100
python -m src.train +model=bafnet +dset=taps
```

Configuration files are in `conf/`:
- `config.yaml`: Base configuration
- `model/*.yaml`: Model architecture configs

## Project Structure

```
BAFNet-plus/
├── conf/                      # Hydra configurations
│   ├── config.yaml            # Main config
│   └── model/                 # Model configs
│       ├── backbone_masking.yaml
│       ├── backbone_mapping.yaml
│       └── bafnet.yaml
├── src/                       # Source code
│   ├── models/                # Model implementations
│   │   ├── backbone.py        # Backbone (conv-based, asymmetric padding)
│   │   ├── bafnet.py          # BAFNet (dual-input acs+bcs fusion)
│   │   ├── discriminator.py   # MetricGAN discriminator
│   │   └── streaming/         # Streaming inference modules
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Evaluation script
│   ├── enhance.py             # Inference script
│   ├── analysis/              # Public offline analysis utilities
│   ├── data.py                # Dataset implementation
│   ├── solver.py              # Training loop
│   ├── stft.py                # STFT utilities
│   ├── runtime_common.py      # Shared runtime helpers
│   ├── utils.py               # Helper functions
│   └── compute_metrics.py     # Metric computation
├── dataset/                   # Dataset file lists
├── scripts/                   # Experiment and evaluation scripts
│   ├── eval.sh                # Unified driver (multi-snr / gain-robustness)
│   └── run_train.sh           # Remote training orchestrator
├── results/                   # (gitignored) experiment outputs and private reports
│   ├── experiments/           # Per-model checkpoints and configs
│   ├── eval/                  # Per-scenario evaluation JSONs
│   └── analysis/              # Generated local analysis artifacts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

**Note**: Training outputs and analysis artifacts are saved to `results/` (not tracked in git).

## Evaluation Metrics

The framework computes the following objective metrics:

- **PESQ**: Perceptual Evaluation of Speech Quality (wideband, 16kHz)
- **STOI**: Short-Time Objective Intelligibility
- **CSIG**: Composite measure of signal distortion
- **CBAK**: Composite measure of background noise distortion
- **COVL**: Composite measure of overall quality
- **SegSNR**: Segmental Signal-to-Noise Ratio

Optional ASR-based metrics (with `--eval_stt` flag):
- **CER**: Character Error Rate using Whisper ASR model
- **WER**: Word Error Rate using Whisper ASR model

## Citation

Citation information will be provided upon publication.

## License

This project is licensed under the MIT License.

## Acknowledgments

- MetricGAN training framework
- Hydra configuration management
- TAPS and Vibravox dataset providers

## Contact

For questions or issues, please contact: ys.kim@postech.ac.kr
