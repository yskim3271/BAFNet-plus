# CLAUDE.md - Development Guide

## Quick Start Commands

### Build/Run
```bash
# Install dependencies
pip install -r requirements.txt
# Or with dev dependencies
pip install -e ".[dev]"

# Train model
CUDA_VISIBLE_DEVICES=0 python -m src.train +model=primeknet_gru_masking +dset=taps

# Resume training
python -m src.train +model=primeknet_gru_masking +dset=taps \
  continue_from=outputs/checkpoint_dir

# Run inference
python -m src.enhance --chkpt_dir outputs/model_dir --chkpt_file best.th \
  --noise_dir /path/to/noise --rir_dir /path/to/rir \
  --snr 0 --output_dir samples
```

### Linting and Testing
```bash
# Run all tests
make test
# Or: pytest tests/ -v

# Run single test file
pytest tests/test_models.py -v

# Run single test
pytest tests/test_models.py::TestPrimeKnet::test_primeknet_forward -v

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
  - Classes: `PascalCase` (e.g., `PrimeKnet`, `DenseEncoder`)
  - Functions/methods: `snake_case` (e.g., `mag_pha_stft`, `tailor_dB_FS`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SR`)
  - Private methods: `_leading_underscore` (e.g., `_serialize`)

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
- `src/solver.py`: Training loop implementation (Solver class)
- `src/data.py`: Dataset and data augmentation logic
- `src/stft.py`: STFT/iSTFT utilities
- `src/utils.py`: Utility functions
- `src/models/primeknet.py`: Model architectures
- `tools/track_experiment.py`: Experiment tracking tool

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
- Example: `feat: add PrimeKnet-Mamba variant`

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
pytest tests/test_models.py::TestPrimeKnet -v

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
├── WORKFLOW.md            # Experiment tracking guide
├── requirements.txt       # Python dependencies
├── src/                   # Source code (library + scripts)
│   ├── data.py           # Dataset and data augmentation
│   ├── solver.py         # Training loop (Solver class)
│   ├── stft.py           # STFT/iSTFT utilities
│   ├── utils.py          # Utility functions
│   ├── train.py          # Training entry point
│   ├── enhance.py        # Inference script
│   ├── evaluate.py       # Evaluation script
│   ├── compute_metrics.py # Metric computation
│   └── models/           # Model architectures
├── tools/                 # Utility tools
│   └── track_experiment.py # Experiment tracking
├── conf/                  # Hydra configurations
├── dataset/               # Dataset file lists
├── tests/                 # Unit tests
├── outputs/               # Training outputs (auto-generated)
└── results/               # Experiment tracking results
    ├── experiments.csv   # All experiment data
    ├── EXPERIMENTS.md    # Summary tables
    └── plots/            # Visualization plots
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

### Tracking Experiments
1. After training completes, run: `python tools/track_experiment.py --update`
2. Check results in `results/EXPERIMENTS.md` and `results/plots/`
3. See `WORKFLOW.md` for detailed experiment tracking guide

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

## Resources

- **Hydra Docs**: https://hydra.cc/docs/intro/
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **Black Formatting**: https://black.readthedocs.io/
- **Pytest Guide**: https://docs.pytest.org/
