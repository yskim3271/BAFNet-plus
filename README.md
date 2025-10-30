# BAFNet-plus: Semi-Real-Time Speech Enhancement

A PyTorch-based speech enhancement framework designed for semi-real-time processing with high performance. This repository includes multiple architectural variants (PrimeKnet, MPNet, BAFNet) with both masking and mapping approaches to achieve optimal trade-offs between quality and latency.

## Features

- **Multiple Model Architectures**: PrimeKnet (convolution-based), MPNet (GRU + Multi-head Attention), BAFNet (dual-input model)
- **Real-Time Capable**: Causal convolutions with ~2s receptive field (configurable)
- **Comprehensive Training**: Multi-loss framework with MetricGAN discriminator
- **Flexible Data Pipeline**: Noise augmentation with SNR control and reverberation
- **Hydra Configuration**: Modular, reproducible experiment management
- **Mapping & Masking**: Two inference approaches for speech enhancement

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/BAFNet-plus.git
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
# Train PrimeKnet on TAPS dataset
CUDA_VISIBLE_DEVICES=0 python -m src.train +model=primeknet_masking +dset=taps

# Resume from checkpoint
python -m src.train +model=primeknet_masking +dset=taps \
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

```bash
# Evaluate model on test set
bash eval.sh
# Or customize:
python -m src.evaluate.py \
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

## Model Architectures

### PrimeKnet (Convolution-Based)

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
- **Causal Support**: Asymmetric padding for latency control (6.25ms - 100ms)
- **Dual Decoders**: Separate magnitude masking/mapping and phase refinement paths
- **Inference Types**:
  - `masking`: Predicts mask to multiply with noisy magnitude
  - `mapping`: Directly predicts clean magnitude and phase

### MPNet (Attention-Based)

```
Input (Noisy Time-Domain Signal)
    ↓
[STFT → Extract Mag & Phase [B, 2, T, F]]
    ↓
[Dense Encoder]
    ↓
[TSTransformerBlocks x N: Multi-head Attention + GRU FFN]
    ↓
[Dual Decoders: Magnitude + Phase Estimation]
    ↓
[iSTFT]
    ↓
Output (Enhanced Time-Domain Signal)
```

**Key Components:**
- **TSTransformerBlock**: Combines Multi-head Attention with GRU-based feed-forward network
- **Dense Encoder**: Initial feature extraction with convolutional layers
- **Bidirectional GRU**: Captures temporal dependencies in feed-forward layers
- **End-to-End**: STFT/iSTFT operations integrated within the model
- **Inference Types**:
  - `masking`: Magnitude masking approach
  - `mapping`: Direct magnitude/phase prediction

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
- **Pre-trained Models**: Requires separate mapping and masking models (e.g., PrimeKnet or MPNet)
- **Adaptive Fusion**: Learns frequency-dependent fusion weights via convolutional blocks
- **Learnable Sigmoid**: Time-frequency adaptive weighting for optimal signal combination
- **Complex-Domain Fusion**: Operates on complex spectrograms for better phase preservation

### Model Variants

| Model | Architecture | Inference Type | Config Files |
|-------|-------------|----------------|--------------|
| **PrimeKnet** | LKFCA (conv-based) | Masking / Mapping | `primeknet_masking.yaml`, `primeknet_mapping.yaml` |
| **MPNet** | GRU + Multi-head Attention | Masking / Mapping | `mpnet_masking.yaml`, `mpnet_mapping.yaml` |
| **BAFNet** | Dual-input (acs+bcs) | Masking | `bafnet.yaml` |

**Inference Types:**
- **Masking**: Predicts magnitude/phase masks to apply to noisy spectrogram
- **Mapping**: Directly predicts clean magnitude/phase spectrograms

**Receptive Field (RF)** for PrimeKnet depends on kernel configurations. With default settings (num_tsblock=4, time_block_num=2, time_block_kernel=[3,11,23,31]), RF ≈ 317 frames (32,000 samples or ~2.0s @ 16kHz). See `src/models/primeknet.py:8-86` for detailed RF calculations.

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
lr_decay: 0.99  # exponential decay
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
python -m src.train +model=primeknet_masking model.param.dense_channel=128

# Override training settings
python -m src.train +dset=taps batch_size=8 lr=1e-3

# Train with different models
python -m src.train +model=mpnet_masking +dset=vibravox epochs=100
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
│   └── model/                 # Model configs (primeknet, mpnet, bafnet)
│       ├── primeknet_masking.yaml
│       ├── primeknet_mapping.yaml
│       ├── mpnet_masking.yaml
│       ├── mpnet_mapping.yaml
│       └── bafnet.yaml
├── src/                       # Source code
│   ├── models/                # Model implementations
│   │   ├── primeknet.py       # PrimeKnet (LKFCA conv-based)
│   │   ├── mpnet.py           # MPNet (GRU + Multi-head Attention)
│   │   ├── bafnet.py          # BAFNet (dual-input acs+bcs)
│   │   └── discriminator.py  # MetricGAN discriminator
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Evaluation script
│   ├── enhance.py             # Inference script
│   ├── data.py                # Dataset implementation
│   ├── solver.py              # Training loop
│   ├── stft.py                # STFT utilities
│   ├── utils.py               # Helper functions
│   └── compute_metrics.py     # Metric computation
├── dataset/                   # Dataset file lists
├── enhance.sh                 # Inference script
├── eval.sh                    # Evaluation script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

**Note**: Training outputs are saved to `results/experiments/<timestamp>/` (not tracked in git).

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

- PrimeKnet architecture inspiration
- MetricGAN training framework
- Hydra configuration management
- TAPS and Vibravox dataset providers

## Contact

For questions or issues, please contact: ys.kim@postech.ac.kr
