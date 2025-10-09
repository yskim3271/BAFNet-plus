# Project Plan: BAFNet-plus

## Project Scope

### Objective
Develop and improve semi-real-time speech enhancement models based on PrimeKnet architecture, targeting high performance with real-time processing capability.

### Core Components
1. **Model Architecture**: PrimeKnet with variants (GRU, LSTM, Mamba)
   - Encoder: DS_DDB (Dilated-Separable Dense Blocks)
   - Middle: Two-Stage Blocks (time-frequency processing)
   - Decoder: Masking/mapping for magnitude and phase reconstruction

2. **Training Pipeline**:
   - Multi-loss training (magnitude, phase, complex, consistency, metric)
   - MetricGAN discriminator for perceptual quality
   - Noise augmentation with SNR control and optional reverberation

3. **Data Processing**:
   - STFT-based feature extraction
   - Dynamic noise augmentation
   - Support for throat microphone and acoustic microphone inputs

## File Tree

```
BAFNet-plus/
├── conf/                      # Hydra configurations
│   ├── config.yaml           # Main config
│   ├── dset/                 # Dataset configs
│   │   ├── taps.yaml
│   │   └── vibravox.yaml
│   └── model/                # Model configs
│       ├── primeknet_masking.yaml
│       ├── primeknet_gru_masking.yaml
│       └── primeknet_lstm_masking.yaml
├── models/                    # Model implementations
│   ├── primeknet.py          # Base PrimeKnet
│   ├── primeknet_gru.py      # GRU variant
│   ├── primeknet_lstm.py     # LSTM variant
│   ├── bafnet.py             # BAFNet implementation
│   └── discriminator.py      # MetricGAN discriminator
├── dataset/                   # Dataset file lists
├── outputs/                   # Training outputs (Hydra managed)
├── docs/                      # Documentation
│   └── PLAN.md               # This file
├── tests/                     # Unit tests (to be added)
├── train.py                   # Training entry point
├── evaluate.py                # Evaluation script
├── enhance.py                 # Inference script
├── data.py                    # Dataset implementation
├── solver.py                  # Training loop
├── stft.py                    # STFT utilities
├── utils.py                   # Helper functions
├── compute_metrics.py         # Metrics computation
├── requirements.txt           # Python dependencies
├── run.sh                     # Training launcher
├── eval.sh                    # Evaluation launcher
├── README.md                  # Project documentation
└── CLAUDE.md                  # Development guide
```

## Milestones

### Phase 1: Setup & Documentation ✓
- [x] Repository structure analysis
- [x] Create PLAN.md
- [ ] Generate comprehensive README.md
- [ ] Create CLAUDE.md for development guidelines

### Phase 2: Development Infrastructure
- [ ] Add Python linting (Black, isort, flake8)
- [ ] Configure pre-commit hooks
- [ ] Add unit tests for core modules
- [ ] Setup GitHub Actions CI

### Phase 3: Model Development
- [ ] Baseline model validation
- [ ] Architecture experiments
- [ ] Real-time optimization
- [ ] Performance benchmarking

### Phase 4: Evaluation & Deployment
- [ ] Comprehensive evaluation pipeline
- [ ] Model export (ONNX/TorchScript)
- [ ] Real-time inference demo
- [ ] Documentation and examples

## Technical Specifications

### Model Architecture Details

**PrimeKnet-GRU Configuration:**
- Dense channels: 64
- Number of TS blocks: 4
- GRU layers: 1-2
- Time blocks per TS: 2
- Frequency blocks per TS: 2
- FFT length: 400 (16kHz SR)

**Receptive Field Analysis:**
- Time-axis RF: ~317 frames
- Sample RF: 32,000 samples (2.0s @ 16kHz)
- Suitable for semi-real-time processing

### Training Configuration

**Loss Weights:**
- Magnitude: 0.9
- Phase: 0.3
- Complex: 0.1
- Consistency: 0.05
- Metric (GAN): 0.05

**Optimization:**
- Optimizer: AdamW
- Learning rate: 5e-4
- LR decay: 0.99 (exponential)
- Batch size: 4
- Epochs: 200

**Data Augmentation:**
- SNR range: [-15, 20] dB (training)
- Reverb proportion: 0-50%
- Target dB FS: -25 ± 10 dB

## Development Workflow

### Training a New Model
```bash
# Basic training
CUDA_VISIBLE_DEVICES=0 python train.py +model=primeknet_gru_masking +dset=taps

# Resume from checkpoint
python train.py +model=primeknet_gru_masking +dset=taps continue_from=outputs/checkpoint_dir
```

### Evaluation
```bash
# Run evaluation script
bash eval.sh
```

### Inference
```bash
python enhance.py --chkpt_dir outputs/model_dir --chkpt_file best.th \
  --noise_dir path/to/noise --rir_dir path/to/rir \
  --snr 0 --output_dir samples
```

## Next Steps

1. **Immediate (Week 1)**:
   - Complete documentation (README.md, CLAUDE.md)
   - Add basic unit tests
   - Setup linting and formatting

2. **Short-term (Weeks 2-4)**:
   - Validate baseline model performance
   - Experiment with architecture variants
   - Optimize for real-time inference

3. **Long-term (Months 2-3)**:
   - Comprehensive evaluation on multiple datasets
   - Model compression and acceleration
   - Deploy real-time demo application

## References & Resources

- **PrimeKnet**: Base architecture with large kernel feature attention
- **MetricGAN**: Discriminator for perceptual quality improvement
- **Hydra**: Configuration management framework
- **Datasets**: TAPS (yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset), Vibravox
