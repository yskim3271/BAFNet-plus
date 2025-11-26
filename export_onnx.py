import torch
import torch.nn as nn
import argparse
import os
import sys
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

def export_onnx(output_path, checkpoint_path=None, opset_version=17):
    print(f"Exporting PrimeKnet to {output_path}...")

    # Default parameters based on primeknet_lora_track2.yaml
    # Note: You might need to adjust these if your model config differs
    model = PrimeKnet(
        win_len=400,
        hop_len=100,
        fft_len=400,
        dense_channel=64,
        sigmoid_beta=2.0,
        compress_factor=0.3,
        dense_depth=4,
        num_tsblock=4,
        time_dw_kernel_size=3,
        time_block_kernel=[3, 11, 23, 31],
        freq_block_kernel=[3, 11, 23, 31],
        time_block_num=2,
        freq_block_num=2,
        infer_type='mapping',
        causal=False, # Set to True if you want to export a causal version for streaming
        encoder_padding_ratio=(0.5, 0.5),
        decoder_padding_ratio=(0.5, 0.5)
    )

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Handle state dict keys if they start with 'module.' or similar
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("No checkpoint provided, using random weights.")

    model.eval()

    # Dummy input: [Batch, Freq, Time, 2]
    # Freq = fft_len // 2 + 1 = 201
    # Time = 100 (arbitrary)
    dummy_input = torch.randn(1, 201, 100, 2)

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
    
    # Compare
    for i, (pt_out, ort_out) in enumerate(zip(pt_outs, ort_outs)):
        np.testing.assert_allclose(pt_out.numpy(), ort_out, rtol=1e-03, atol=1e-04)
        print(f"Output {i} matches.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="primeknet.onnx", help="Output ONNX file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_onnx(args.output, args.checkpoint, args.opset)
