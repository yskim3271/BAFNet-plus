import torch
import torch.nn as nn
import argparse
import os
import sys
import yaml
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock torchaudio if not present
try:
    import torchaudio
except ImportError:
    import sys
    from unittest.mock import MagicMock
    sys.modules["torchaudio"] = MagicMock()

from src.models.primeknet import PrimeKnet

def load_config_from_checkpoint(checkpoint_path):
    """Load model configuration from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / ".hydra" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            f"Make sure the checkpoint directory contains .hydra/config.yaml"
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def export_onnx(output_path, checkpoint_path=None, opset_version=17):
    print(f"Exporting PrimeKnet to {output_path}...")

    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for automatic config loading")

    # Load configuration from checkpoint directory
    print(f"Loading config from checkpoint directory...")
    config = load_config_from_checkpoint(checkpoint_path)

    # Extract model parameters
    model_params = config['model']['param']
    print(f"Model configuration:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")

    # Create model with loaded parameters
    model = PrimeKnet(**model_params)

    # Load checkpoint weights
    print(f"Loading checkpoint from {checkpoint_path}")
    # Note: weights_only=False is used because checkpoint contains OmegaConf objects
    # This is safe if the checkpoint is from a trusted source (your own training)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats:
    # Format 1: {'model': state_dict, 'args': ...}  (BAFNet-plus format)
    # Format 2: {'state_dict': state_dict, ...}     (common format)
    # Format 3: state_dict directly
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle state dict keys if they start with 'module.' (DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    # Load with strict=True to catch any mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {len(missing)}")
        if len(missing) <= 5:
            for k in missing:
                print(f"  - {k}")
    if unexpected:
        print(f"Warning: Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 5:
            for k in unexpected:
                print(f"  - {k}")

    model.eval()

    # Dummy input: [Batch, Freq, Time, 2]
    # Freq = fft_len // 2 + 1
    # Time = 100 (arbitrary)
    fft_len = model_params['fft_len']
    freq_bins = fft_len // 2 + 1
    print(f"Creating dummy input with shape: [1, {freq_bins}, 100, 2]")
    dummy_input = torch.randn(1, freq_bins, 100, 2)

    # Dynamic axes for variable sequence length
    dynamic_axes = {
        'noisy_com': {0: 'batch_size', 2: 'time'},
        'denoised_mag': {0: 'batch_size', 2: 'time'},
        'denoised_pha': {0: 'batch_size', 2: 'time'},
        'denoised_com': {0: 'batch_size', 2: 'time'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['noisy_com'],
        output_names=['denoised_mag', 'denoised_pha', 'denoised_com'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        verbose=False
    )
    print("Export complete.")

    # Verify
    import onnxruntime as ort
    import numpy as np

    print("Verifying with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # PyTorch output
    with torch.no_grad():
        pt_outs = model(dummy_input)
    
    # Compare (slightly relaxed tolerance for float32 precision)
    for i, (pt_out, ort_out) in enumerate(zip(pt_outs, ort_outs)):
        np.testing.assert_allclose(pt_out.numpy(), ort_out, rtol=1e-02, atol=1e-03)
        print(f"Output {i} matches (max diff: {np.abs(pt_out.numpy() - ort_out).max():.6f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export PrimeKnet model to ONNX format with automatic config loading"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., results/experiments/prk_taps_mask/best.th)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path (default: <checkpoint_name>.onnx)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        experiment_name = checkpoint_path.parent.name
        args.output = f"{experiment_name}.onnx"
        print(f"Output filename not specified, using: {args.output}")

    export_onnx(args.output, args.checkpoint, args.opset)
