# BAFNet-plus: Semi-Real-Time Speech Enhancement

A PyTorch-based speech enhancement framework built on PrimeKnet architecture, designed for semi-real-time processing with high performance. This repository explores various architectural variants (GRU, LSTM, Mamba) to achieve optimal trade-offs between quality and latency.

## Features

- **Multiple Model Architectures**: PrimeKnet with GRU/LSTM/Mamba variants
- **Real-Time Capable**: Causal convolutions with ~2s receptive field (configurable)
- **Comprehensive Training**: Multi-loss framework with MetricGAN discriminator
- **Flexible Data Pipeline**: Noise augmentation with SNR control and reverberation
- **Hydra Configuration**: Modular, reproducible experiment management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/BAFNet-plus.git
cd BAFNet-plus

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 1.13
- CUDA (recommended for training)

### Training

```bash
# Train PrimeKnet-GRU on TAPS dataset
CUDA_VISIBLE_DEVICES=0 python train.py +model=primeknet_gru_masking +dset=taps

# Resume from checkpoint
python train.py +model=primeknet_gru_masking +dset=taps \
  continue_from=outputs/2025-10-10_03-52-05
```

Training outputs (checkpoints, logs, samples) are saved to `outputs/<timestamp>/` via Hydra.

### Inference

```bash
python enhance.py \
  --chkpt_dir outputs/your_model_dir \
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
python evaluate.py \
  --model_config outputs/model_dir/.hydra/config.yaml \
  --chkpt_dir outputs/model_dir \
  --chkpt_file best.th \
  --noise_dir /path/to/noise \
  --noise_test dataset/noise_test.txt \
  --rir_dir /path/to/rir \
  --rir_test dataset/rir_test.txt \
  --snr_step 0 \
  --eval_stt
```

## Model Architecture

### PrimeKnet Overview

```
Input (Noisy Complex Spectrogram)
    ↓
[Dense Encoder: DS_DDB + Frequency Downsample]
    ↓
[LKFCAnet/Two-Stage Blocks: Time + Frequency Processing]
    ↓
[Dual Decoders: Magnitude Mask + Phase Refinement]
    ↓
Output (Enhanced Complex Spectrogram)
```

**Key Components:**
- **DS_DDB**: Dilated-separable dense blocks with exponential dilation (1, 2, 4, 8)
- **Two-Stage Block**: Sequential time-axis (GRU/LSTM) and frequency-axis processing
- **Causal Convolutions**: Support for streaming inference
- **Learnable Sigmoid**: Frequency-dependent mask scaling

### Variants

| Model | Time Module | Params | RF (frames) | RF (samples) |
|-------|-------------|--------|-------------|--------------|
| PrimeKnet | LKFCA (conv) | ~2M | 317 | 32,000 |
| PrimeKnet-GRU | GRU | ~2.5M | Variable | Variable |
| PrimeKnet-LSTM | LSTM | ~3M | Variable | Variable |

**Receptive Field (RF)** depends on kernel configurations. See model source code comments for detailed calculations.

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
python train.py +model=primeknet_gru_masking model.param.gru_layers=2

# Override training settings
python train.py +dset=taps batch_size=8 lr=1e-3

# Combine configs
python train.py +model=primeknet_lstm_masking +dset=vibravox epochs=100
```

Configuration files are in `conf/`:
- `config.yaml`: Base configuration
- `model/*.yaml`: Model architecture configs
- `dset/*.yaml`: Dataset-specific configs

## Project Structure

```
BAFNet-plus/
├── conf/                   # Hydra configurations
│   ├── config.yaml         # Main config
│   ├── dset/               # Dataset configs
│   └── model/              # Model configs
├── models/                 # Model implementations
│   ├── primeknet.py        # Base PrimeKnet
│   ├── primeknet_gru.py    # GRU variant
│   ├── primeknet_lstm.py   # LSTM variant
│   └── discriminator.py    # MetricGAN discriminator
├── dataset/                # Dataset file lists
├── outputs/                # Training outputs (auto-generated)
├── docs/                   # Documentation
├── train.py                # Training entry point
├── evaluate.py             # Evaluation script
├── enhance.py              # Inference script
├── data.py                 # Dataset implementation
├── solver.py               # Training loop
├── stft.py                 # STFT utilities
└── utils.py                # Helper functions
```

## Evaluation Metrics

- **PESQ**: Perceptual Evaluation of Speech Quality
- **STOI**: Short-Time Objective Intelligibility
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio
- **WER** (optional): Word Error Rate with ASR model

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bafnet_plus_2025,
  author = {Your Name},
  title = {BAFNet-plus: Semi-Real-Time Speech Enhancement},
  year = {2025},
  url = {https://github.com/your-username/BAFNet-plus}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- PrimeKnet architecture inspiration
- MetricGAN training framework
- Hydra configuration management
- TAPS and Vibravox dataset providers

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].
