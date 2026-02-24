#!/usr/bin/env python
"""
PrimeKnet InstanceNorm2d calibration script.

This script collects channel-wise statistics from InstanceNorm2d layers
by running calibration data through a trained model. The statistics can
then be used for frozen normalization or conv folding during inference.

Usage:
    # Basic usage (TAPS dataset)
    python src/calibrate.py \
        --chkpt_dir results/experiments/prk_taps_mask \
        --num_batches 100

    # Full options
    python src/calibrate.py \
        --chkpt_dir results/experiments/prk_taps_mask \
        --chkpt_file best.th \
        --num_batches 100 \
        --batch_size 4 \
        --output model.normstats.pt \
        --device cuda

Output:
    Saves calibration statistics to <chkpt_dir>/model.normstats.pt (default)
    or specified output path.
"""

import argparse
import sys
import importlib.util
from pathlib import Path
from typing import Callable, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


# Direct module loading to avoid src/__init__.py which has many dependencies
def _load_module_direct(module_name: str, file_path: Path):
    """Load a module directly from file path without triggering __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get project root
_project_root = Path(__file__).parent.parent

# Load required modules directly
_stft = _load_module_direct('src.stft', _project_root / 'src' / 'stft.py')
mag_pha_stft = _stft.mag_pha_stft

_primeknet = _load_module_direct('src.models.primeknet', _project_root / 'src' / 'models' / 'primeknet.py')
PrimeKnet = _primeknet.PrimeKnet

_data = _load_module_direct('src.data', _project_root / 'src' / 'data.py')
Noise_Augmented_Dataset = _data.Noise_Augmented_Dataset

_accumulator = _load_module_direct(
    'src.calibration.accumulator', _project_root / 'src' / 'calibration' / 'accumulator.py'
)
_calibrator = _load_module_direct(
    'src.calibration.calibrator', _project_root / 'src' / 'calibration' / 'calibrator.py'
)
INCalibrator = _calibrator.INCalibrator


def load_model_and_config(chkpt_dir: str, chkpt_file: str = 'best.th') -> Tuple[nn.Module, dict]:
    """
    Load model and configuration from checkpoint directory.

    Args:
        chkpt_dir: Directory containing checkpoint and .hydra/config.yaml
        chkpt_file: Checkpoint filename

    Returns:
        Tuple of (model, config)
    """
    chkpt_dir = Path(chkpt_dir)

    # Load Hydra config
    config_path = chkpt_dir / '.hydra' / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Create model - use cfg.model.param for model parameters
    if hasattr(cfg.model, 'param'):
        model_params = OmegaConf.to_container(cfg.model.param, resolve=True)
    else:
        # Fallback for old config format
        model_params = OmegaConf.to_container(cfg.model, resolve=True)
        # Remove non-model params
        for key in ['model_lib', 'model_class', 'input_type']:
            model_params.pop(key, None)

    model = PrimeKnet(**model_params)

    # Load weights
    chkpt_path = chkpt_dir / chkpt_file
    if not chkpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {chkpt_path}")

    state = torch.load(chkpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])

    return model, cfg


def get_model_param(cfg, key: str, default=None):
    """Get model parameter from config, handling both old and new formats."""
    if hasattr(cfg.model, 'param') and hasattr(cfg.model.param, key):
        return getattr(cfg.model.param, key)
    elif hasattr(cfg.model, key):
        return getattr(cfg.model, key)
    elif hasattr(cfg, key):
        return getattr(cfg, key)
    return default


def create_input_transform(cfg) -> Callable:
    """
    Create a function that transforms DataLoader batches to model input.

    Args:
        cfg: Hydra config with model and STFT parameters

    Returns:
        Transform function: batch -> model_input
    """
    n_fft = get_model_param(cfg, 'fft_len') or cfg.get('n_fft', 400)
    hop_size = get_model_param(cfg, 'hop_len') or cfg.get('hop_size', 100)
    win_size = get_model_param(cfg, 'win_len') or cfg.get('win_size', 400)
    compress_factor = get_model_param(cfg, 'compress_factor') or cfg.get('compress_factor', 0.3)

    def transform(batch):
        # Noise_Augmented_Dataset returns (bcs, noisy_acs, clean_acs) or with id/text
        if len(batch) >= 3:
            bcs, noisy_acs, clean_acs = batch[0], batch[1], batch[2]
        else:
            noisy_acs = batch[0]
            clean_acs = batch[1] if len(batch) > 1 else noisy_acs

        # STFT transform on noisy audio
        _, _, noisy_com = mag_pha_stft(
            noisy_acs,
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            center=True
        )
        return noisy_com  # [B, F, T, 2]

    return transform


def load_file_list(filepath: str) -> list:
    """Load file list from text file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File list not found: {filepath}")

    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def create_taps_dataloader(cfg, batch_size: int, num_workers: int = 4, split: str = 'validation') -> DataLoader:
    """
    Create DataLoader for TAPS dataset calibration.

    Args:
        cfg: Hydra config with dataset parameters
        batch_size: Batch size for calibration
        num_workers: Number of data loading workers
        split: Dataset split ('validation' or 'test')

    Returns:
        DataLoader instance
    """
    from datasets import load_dataset, Audio

    # Load TAPS dataset from HuggingFace
    # Map split names: TAPS uses 'dev' instead of 'validation'
    hf_split = 'dev' if split == 'validation' else split
    print(f"Loading TAPS {hf_split} dataset from HuggingFace...")
    taps_dataset = load_dataset(
        "yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset",
        split=hf_split
    )

    # Cast audio columns to ensure proper decoding at 16kHz
    taps_dataset = taps_dataset.cast_column("audio.throat_microphone", Audio(sampling_rate=16000, decode=True))
    taps_dataset = taps_dataset.cast_column("audio.acoustic_microphone", Audio(sampling_rate=16000, decode=True))

    # Load noise and RIR file lists
    if split == 'validation':
        noise_file = cfg.dset.get('noise_valid', cfg.dset.get('noise_dev', None))
        rir_file = cfg.dset.get('rir_valid', cfg.dset.get('rir_dev', None))
        noise_cfg = cfg.get('valid_noise', cfg.get('train_noise', {}))
    elif split == 'test':
        noise_file = cfg.dset.get('noise_test', None)
        rir_file = cfg.dset.get('rir_test', None)
        noise_cfg = cfg.get('test_noise', {})
    else:
        noise_file = cfg.dset.get('noise_train', None)
        rir_file = cfg.dset.get('rir_train', None)
        noise_cfg = cfg.get('train_noise', {})

    if noise_file is None:
        raise ValueError(f"No noise file configured for split: {split}")

    # Get noise and RIR directories
    noise_dir = cfg.dset.get('noise_dir', '')
    rir_dir = cfg.dset.get('rir_dir', '')

    # Load file lists and prepend directory paths
    noise_list = load_file_list(noise_file)
    if noise_dir:
        noise_list = [str(Path(noise_dir) / f) for f in noise_list]

    rir_list = load_file_list(rir_file) if rir_file else []
    if rir_dir and rir_list:
        rir_list = [str(Path(rir_dir) / f) for f in rir_list]

    print(f"  Loaded {len(noise_list)} noise files")
    print(f"  Loaded {len(rir_list)} RIR files")

    # Get noise config parameters
    snr_range = noise_cfg.get('snr_range', [-15, 10])
    reverb_proportion = noise_cfg.get('reverb_proportion', 0.0)
    target_dB_FS = noise_cfg.get('target_dB_FS', -25)
    target_dB_FS_floating_value = noise_cfg.get('target_dB_FS_floating_value', 0)
    silence_length = noise_cfg.get('silence_length', 0.2)
    # Force non-deterministic mode for calibration (avoids index out of range errors)
    deterministic = False

    # Create Noise_Augmented_Dataset
    dataset = Noise_Augmented_Dataset(
        datapair_list=taps_dataset,
        noise_list=noise_list,
        rir_list=rir_list,
        snr_range=snr_range,
        reverb_proportion=reverb_proportion,
        target_dB_FS=target_dB_FS,
        target_dB_FS_floating_value=target_dB_FS_floating_value,
        silence_length=silence_length,
        sampling_rate=cfg.get('sampling_rate', 16000),
        segment=cfg.get('segment', 64000),
        stride=cfg.get('stride', 32000),
        shift=None,  # No shift for calibration
        with_id=False,
        with_text=False,
        deterministic=deterministic,
        bcs_only=False,
    )

    print(f"  Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate InstanceNorm2d statistics for frozen inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--chkpt_dir',
        type=str,
        required=True,
        help='Checkpoint directory (must contain .hydra/config.yaml)'
    )
    parser.add_argument(
        '--chkpt_file',
        type=str,
        default='best.th',
        help='Checkpoint filename'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=100,
        help='Number of batches for calibration'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size (smaller is more stable)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['train', 'validation', 'test'],
        help='Dataset split to use for calibration'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path (default: <chkpt_dir>/model.normstats.pt)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )

    args = parser.parse_args()

    # Set output path
    if args.output is None:
        args.output = str(Path(args.chkpt_dir) / 'model.normstats.pt')

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and config
    print(f"\nLoading model from: {args.chkpt_dir}")
    model, cfg = load_model_and_config(args.chkpt_dir, args.chkpt_file)
    model.to(device)
    model.eval()

    # Print model info
    dense_channel = get_model_param(cfg, 'dense_channel')
    dense_depth = get_model_param(cfg, 'dense_depth', 4)
    print(f"Model: {cfg.model.get('model_class', 'PrimeKnet')}")
    print(f"  dense_channel: {dense_channel}")
    print(f"  dense_depth: {dense_depth}")

    # Create DataLoader
    print(f"\nCreating dataloader (split={args.split})...")
    dataset_type = cfg.dset.get('dataset', 'unknown')

    if dataset_type == 'TAPS':
        dataloader = create_taps_dataloader(cfg, args.batch_size, args.num_workers, args.split)
    else:
        print(f"Warning: Unknown dataset type '{dataset_type}', using dummy data")
        # Fallback: create dummy data
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, length=1000, segment=64000):
                self.length = length
                self.segment_samples = segment

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                bcs = torch.randn(self.segment_samples)
                noisy = torch.randn(self.segment_samples)
                clean = torch.randn(self.segment_samples)
                return bcs, noisy, clean

        segment = cfg.get('segment', 64000)
        dataloader = DataLoader(
            DummyDataset(1000, segment),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        print("  Using dummy data for calibration")

    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches: {min(args.num_batches, len(dataloader))}")

    # Create calibrator
    print(f"\nInitializing calibrator...")
    calibrator = INCalibrator(model, device=device)

    # Create input transform
    input_transform = create_input_transform(cfg)

    # Run calibration
    print(f"\nRunning calibration ({args.num_batches} batches)...")
    stats = calibrator.calibrate(
        dataloader=dataloader,
        num_batches=args.num_batches,
        input_transform=input_transform,
        show_progress=True,
    )

    # Save with metadata
    calibration_config = {
        'num_batches': args.num_batches,
        'batch_size': args.batch_size,
        'split': args.split,
        'n_fft': get_model_param(cfg, 'fft_len') or cfg.get('n_fft', 400),
        'hop_size': get_model_param(cfg, 'hop_len') or cfg.get('hop_size', 100),
        'win_size': get_model_param(cfg, 'win_len') or cfg.get('win_size', 400),
        'compress_factor': get_model_param(cfg, 'compress_factor') or cfg.get('compress_factor', 0.3),
        'segment': cfg.get('segment', None),
        'sample_rate': cfg.get('sampling_rate', 16000),
    }

    calibrator.save(args.output, calibration_config=calibration_config)

    print(f"\n{'='*60}")
    print(f"Calibration complete!")
    print(f"Stats saved to: {args.output}")
    print(f"{'='*60}")

    # Verify output file
    print("\nVerifying saved file...")
    loaded_stats, meta = INCalibrator.load(args.output)
    print(f"  Modules: {len(loaded_stats)}")
    print(f"  Model: {meta.get('model_class', 'unknown')}")
    print(f"  Timestamp: {meta.get('timestamp', 'unknown')}")


if __name__ == '__main__':
    main()
