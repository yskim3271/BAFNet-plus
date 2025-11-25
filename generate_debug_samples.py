#!/usr/bin/env python
"""
Generate debug samples for ONNX mobile inference debugging.

This script generates sample data with multiple noise types at different SNR levels
and saves intermediate results (STFT, ONNX inference, iSTFT) to help debug audio
quality issues in mobile ONNX inference.

Usage:
    python generate_debug_samples.py --onnx prk_taps_mask.onnx --output-dir debug_samples

Output structure:
    debug_samples/
    ├── README.md                 # Documentation for all generated files
    ├── clean_audio.wav           # Original clean speech
    ├── metadata.npy              # Global metadata
    ├── noise_00_filename/
    │   ├── noise_audio.wav       # Noise sample used
    │   ├── snr_-5dB/
    │   │   ├── noisy.wav         # Noisy input audio
    │   │   ├── stft_input.npz    # STFT output (mag, pha, com)
    │   │   ├── onnx_output.npz   # ONNX model output (mag, pha, com)
    │   │   ├── denoised.wav      # Final denoised audio
    │   │   └── clean_reference.wav
    │   ├── snr_0dB/
    │   └── snr_5dB/
    ├── noise_01_filename/
    │   └── ...
    └── ...
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.stft import mag_pha_stft, mag_pha_istft, complex_to_mag_pha


def load_taps_sample(split="test", index=0):
    """Load a sample from TAPS dataset."""
    print(f"Loading TAPS {split} dataset...")
    ds = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split=split)

    sample = ds[index]
    # Get acoustic microphone audio (clean speech)
    clean_audio = sample["audio.acoustic_microphone"]["array"].astype("float32")
    sample_rate = sample["audio.acoustic_microphone"]["sampling_rate"]
    speaker_id = sample["speaker_id"]
    sentence_id = sample["sentence_id"]
    text = sample.get("text", "")

    print(f"  Sample: {speaker_id}_{sentence_id}")
    print(f"  Text: {text}")
    print(f"  Duration: {len(clean_audio) / sample_rate:.2f}s")
    print(f"  Sample rate: {sample_rate}Hz")

    return torch.tensor(clean_audio), sample_rate, f"{speaker_id}_{sentence_id}", text


def load_noise_sample(noise_dir, noise_file, target_length, sample_rate=16000):
    """Load a noise sample and match it to target length."""
    noise_path = os.path.join(noise_dir, noise_file)
    print(f"  Loading noise: {noise_file}")

    noise, sr = torchaudio.load(noise_path)
    assert sr == sample_rate, f"Sample rate mismatch: {sr} vs {sample_rate}"

    # Convert to mono if needed
    if noise.ndim > 1 and noise.shape[0] > 1:
        noise = noise[0, :]
    else:
        noise = noise.squeeze()

    noise = noise.to(dtype=torch.float32)

    # Repeat or truncate to match target length
    if noise.shape[-1] < target_length:
        repeat_times = int(np.ceil(target_length / noise.shape[-1]))
        noise = noise.repeat(repeat_times)
    noise = noise[:target_length]

    return noise


def norm_amplitude(y, eps=1e-6):
    """Normalize amplitude to [-1, 1]."""
    scalar = torch.max(torch.abs(y)) + eps
    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    """Scale signal to target dB FS."""
    rms = torch.sqrt(torch.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y = y * scalar
    return y, rms, scalar


def snr_mix(clean, noise, snr, eps=1e-6):
    """Mix clean signal with noise at specified SNR."""
    clean_rms = torch.sqrt(torch.mean(clean ** 2))
    noise_rms = torch.sqrt(torch.mean(noise ** 2))
    snr_scalar = (clean_rms / (10 ** (snr / 20))) / (noise_rms + eps)
    noise_scaled = noise * snr_scalar
    noisy = clean + noise_scaled
    return noisy, noise_scaled


def save_audio(path, audio, sample_rate, verbose=True):
    """Save audio to wav file."""
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(path, audio, sample_rate)
    if verbose:
        print(f"    Saved: {path}")


def generate_readme(output_dir, metadata, noise_files, snr_levels):
    """Generate README.md documentation for the debug samples."""

    readme_content = f"""# Debug Samples for ONNX Mobile Inference

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

This directory contains debug samples for diagnosing audio quality issues in mobile ONNX inference.
Each processing stage (STFT, ONNX inference, iSTFT) has its intermediate outputs saved for comparison.

## Sample Information

- **Clean Speech Sample**: `{metadata['sample_id']}`
- **Text**: "{metadata['sample_text']}"
- **Duration**: {metadata['clean_duration_sec']:.2f} seconds
- **Sample Rate**: {metadata['sample_rate']} Hz
- **Total Samples**: {metadata['clean_length_samples']}

## STFT Parameters

| Parameter | Value |
|-----------|-------|
| n_fft | {metadata['n_fft']} |
| hop_size | {metadata['hop_size']} |
| win_size | {metadata['win_size']} |
| compress_factor | {metadata['compress_factor']} |
| center | True |

## Directory Structure

```
{os.path.basename(output_dir)}/
├── README.md                 # This file
├── clean_audio.wav           # Original clean speech (normalized to -25 dB FS)
├── metadata.npy              # Python dict with all parameters
"""

    for i, noise_file in enumerate(noise_files):
        noise_name = Path(noise_file).stem
        safe_name = f"noise_{i:02d}_{noise_name[:20]}"
        readme_content += f"""├── {safe_name}/
│   ├── noise_audio.wav       # Noise sample used
"""
        for j, snr in enumerate(snr_levels):
            prefix = "│   ├──" if j < len(snr_levels) - 1 else "│   └──"
            readme_content += f"""{prefix} snr_{snr}dB/
│   │   ├── noisy.wav             # Noisy input (clean + noise at {snr}dB SNR)
│   │   ├── stft_input.npz        # STFT output
│   │   ├── onnx_output.npz       # ONNX model output
│   │   ├── denoised.wav          # Reconstructed audio from ONNX output
│   │   └── clean_reference.wav   # Clean audio (same scaling as noisy)
"""

    readme_content += """```

## File Descriptions

### Audio Files (.wav)

| File | Description |
|------|-------------|
| `clean_audio.wav` | Original clean speech, normalized to -25 dB FS |
| `noise_audio.wav` | Noise sample used for mixing, normalized to -25 dB FS |
| `noisy.wav` | Input audio with noise mixed at specified SNR level |
| `denoised.wav` | Output audio reconstructed from ONNX model output via iSTFT |
| `clean_reference.wav` | Clean audio with same scaling as noisy (for quality comparison) |

### NumPy Files (.npz)

#### stft_input.npz
STFT output that becomes the input to the ONNX model.

```python
import numpy as np
data = np.load('stft_input.npz')
magnitude = data['magnitude']  # Shape: (1, 201, T) - Compressed magnitude
phase = data['phase']          # Shape: (1, 201, T) - Phase in radians [-pi, pi]
complex = data['complex']      # Shape: (1, 201, T, 2) - [real, imag], THIS IS ONNX INPUT
```

**Note**: The `complex` array is the actual input to the ONNX model.
- Dimension 0: Batch size (always 1)
- Dimension 1: Frequency bins (n_fft/2 + 1 = 201)
- Dimension 2: Time frames (variable)
- Dimension 3: Real and imaginary parts [real, imag]

#### onnx_output.npz
Output from the ONNX model inference.

```python
data = np.load('onnx_output.npz')
magnitude = data['magnitude']  # Shape: (1, 201, T) - Denoised magnitude
phase = data['phase']          # Shape: (1, 201, T) - Denoised phase
complex = data['complex']      # Shape: (1, 201, T, 2) - Denoised [real, imag]
```

## Debugging Guide

### Step 1: Verify STFT
Compare your mobile STFT output with `stft_input.npz`:
```python
import numpy as np

# Load reference
ref = np.load('stft_input.npz')
ref_complex = ref['complex']  # Shape: (1, 201, T, 2)

# Load your mobile output
mobile_complex = ...  # Your mobile STFT output

# Compare
diff = np.abs(ref_complex - mobile_complex)
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
```

### Step 2: Verify ONNX Inference
Compare your mobile ONNX output with `onnx_output.npz`:
```python
ref = np.load('onnx_output.npz')
ref_mag = ref['magnitude']
ref_pha = ref['phase']

# Compare with your mobile output
```

### Step 3: Verify iSTFT
Compare your mobile reconstructed audio with `denoised.wav`:
```python
import soundfile as sf

ref_audio, sr = sf.read('denoised.wav')
mobile_audio = ...  # Your mobile iSTFT output

diff = np.abs(ref_audio - mobile_audio)
print(f"Max diff: {diff.max():.6f}")
```

## Noise Types Used

| Index | Noise File |
|-------|------------|
"""

    for i, noise_file in enumerate(noise_files):
        readme_content += f"| {i} | `{noise_file}` |\n"

    readme_content += f"""
## SNR Levels

The following SNR (Signal-to-Noise Ratio) levels are included:
- **{', '.join([f'{snr} dB' for snr in snr_levels])}**

Lower SNR means more noise (harder denoising task).

## Technical Notes

1. **Magnitude Compression**: Magnitude is compressed using power-law compression:
   - Forward: `mag_compressed = mag ^ compress_factor`
   - Inverse: `mag = mag_compressed ^ (1 / compress_factor)`

2. **Phase Representation**: Phase is in radians, range [-pi, pi]

3. **Complex Representation**: The complex spectrogram is stored as:
   - `real = magnitude * cos(phase)`
   - `imag = magnitude * sin(phase)`

4. **ONNX Model I/O**:
   - Input: `noisy_com` - Shape (batch, 201, time, 2)
   - Outputs: `denoised_mag`, `denoised_pha`, `denoised_com`
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  Saved: {readme_path}")

    return readme_path


def process_single_noise(
    noise_file,
    noise_index,
    noise_dir,
    output_dir,
    clean_audio,
    sample_rate,
    snr_levels,
    ort_session,
    input_name,
    n_fft,
    hop_size,
    win_size,
    compress_factor,
):
    """Process a single noise file with all SNR levels."""

    # Create noise directory
    noise_name = Path(noise_file).stem
    safe_name = f"noise_{noise_index:02d}_{noise_name[:20]}"
    noise_output_dir = os.path.join(output_dir, safe_name)
    os.makedirs(noise_output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Noise {noise_index}: {noise_file}")
    print(f"{'#'*60}")

    # Load noise
    noise = load_noise_sample(noise_dir, noise_file, len(clean_audio), sample_rate)

    # Normalize and save noise
    noise_norm, _ = norm_amplitude(noise)
    noise_norm, _, _ = tailor_dB_FS(noise_norm, target_dB_FS=-25)
    save_audio(os.path.join(noise_output_dir, "noise_audio.wav"), noise_norm, sample_rate)

    # Generate samples for each SNR level
    for snr in snr_levels:
        print(f"\n  [SNR {snr}dB]")

        snr_dir = os.path.join(noise_output_dir, f"snr_{snr}dB")
        os.makedirs(snr_dir, exist_ok=True)

        # Mix clean and noise at target SNR
        clean_for_mix, _ = norm_amplitude(clean_audio.clone())
        clean_for_mix, _, _ = tailor_dB_FS(clean_for_mix, target_dB_FS=-25)

        noise_for_mix, _ = norm_amplitude(noise.clone())
        noise_for_mix, _, _ = tailor_dB_FS(noise_for_mix, target_dB_FS=-25)

        noisy, noise_scaled = snr_mix(clean_for_mix, noise_for_mix, snr)

        # Prevent clipping
        if torch.max(torch.abs(noisy)) > 0.99:
            scale = 0.99 / torch.max(torch.abs(noisy))
            noisy = noisy * scale
            clean_for_mix = clean_for_mix * scale

        # Save noisy audio
        save_audio(os.path.join(snr_dir, "noisy.wav"), noisy, sample_rate)

        # ===== STFT =====
        noisy_input = noisy.unsqueeze(0)
        mag, pha, com = mag_pha_stft(
            noisy_input,
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            center=True,
            stack_dim=-1
        )

        # Save STFT output
        stft_data = {
            "magnitude": mag.numpy(),
            "phase": pha.numpy(),
            "complex": com.numpy(),
        }
        np.savez(os.path.join(snr_dir, "stft_input.npz"), **stft_data)
        print(f"    STFT: mag shape {mag.shape}, range [{mag.min():.4f}, {mag.max():.4f}]")

        # ===== ONNX Inference =====
        onnx_input = com.numpy()
        ort_outputs = ort_session.run(None, {input_name: onnx_input})
        out_mag, out_pha, out_com = ort_outputs

        # Save ONNX output
        onnx_data = {
            "magnitude": out_mag,
            "phase": out_pha,
            "complex": out_com,
        }
        np.savez(os.path.join(snr_dir, "onnx_output.npz"), **onnx_data)
        print(f"    ONNX: mag range [{out_mag.min():.4f}, {out_mag.max():.4f}]")

        # ===== iSTFT =====
        out_mag_t = torch.tensor(out_mag)
        out_pha_t = torch.tensor(out_pha)

        denoised = mag_pha_istft(
            out_mag_t.squeeze(0),
            out_pha_t.squeeze(0),
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            center=True
        )

        # Normalize to prevent clipping
        if torch.max(torch.abs(denoised)) > 0.99:
            denoised = denoised * (0.99 / torch.max(torch.abs(denoised)))

        # Save denoised audio
        save_audio(os.path.join(snr_dir, "denoised.wav"), denoised, sample_rate)

        # Save clean reference
        save_audio(os.path.join(snr_dir, "clean_reference.wav"), clean_for_mix, sample_rate)

        print(f"    iSTFT: denoised range [{denoised.min():.4f}, {denoised.max():.4f}]")

    return safe_name


def generate_debug_samples(
    onnx_path: str,
    output_dir: str,
    noise_dir: str,
    noise_files: list = None,
    num_noises: int = 5,
    snr_levels: list = [-5, 0, 5],
    sample_index: int = 0,
    n_fft: int = 400,
    hop_size: int = 100,
    win_size: int = 400,
    compress_factor: float = 0.3,
):
    """Generate debug samples with multiple noise types."""

    os.makedirs(output_dir, exist_ok=True)

    # Load ONNX model
    print(f"\nLoading ONNX model: {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {ort_session.get_inputs()[0].shape}")

    # Load clean speech from TAPS
    clean_audio, sample_rate, sample_id, sample_text = load_taps_sample(
        split="test", index=sample_index
    )

    # Load noise file list
    noise_list_path = project_root / "dataset" / "taps" / "noise_test.txt"
    with open(noise_list_path, "r") as f:
        all_noise_files = [line.strip() for line in f.readlines() if line.strip()]

    # Select noise files
    if noise_files is None:
        noise_files = all_noise_files[:num_noises]

    print(f"\nUsing {len(noise_files)} noise files:")
    for i, nf in enumerate(noise_files):
        print(f"  {i}: {nf}")

    # Normalize and save clean audio
    clean_norm, _ = norm_amplitude(clean_audio)
    clean_norm, _, _ = tailor_dB_FS(clean_norm, target_dB_FS=-25)
    save_audio(os.path.join(output_dir, "clean_audio.wav"), clean_norm, sample_rate)

    # Create metadata
    metadata = {
        "sample_id": sample_id,
        "sample_text": sample_text,
        "sample_index": sample_index,
        "noise_files": noise_files,
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_size": hop_size,
        "win_size": win_size,
        "compress_factor": compress_factor,
        "snr_levels": snr_levels,
        "clean_length_samples": len(clean_audio),
        "clean_duration_sec": len(clean_audio) / sample_rate,
        "onnx_model": onnx_path,
        "generated_at": datetime.now().isoformat(),
    }

    np.save(os.path.join(output_dir, "metadata.npy"), metadata)
    print(f"  Saved: {output_dir}/metadata.npy")

    # Process each noise file
    processed_noise_dirs = []
    for i, noise_file in enumerate(noise_files):
        noise_dir_name = process_single_noise(
            noise_file=noise_file,
            noise_index=i,
            noise_dir=noise_dir,
            output_dir=output_dir,
            clean_audio=clean_audio,
            sample_rate=sample_rate,
            snr_levels=snr_levels,
            ort_session=ort_session,
            input_name=input_name,
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
        )
        processed_noise_dirs.append(noise_dir_name)

    # Generate README
    print(f"\n{'='*60}")
    print("Generating README.md...")
    generate_readme(output_dir, metadata, noise_files, snr_levels)

    # Summary
    print(f"\n{'='*60}")
    print(f"Debug samples generation complete!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Total noise types: {len(noise_files)}")
    print(f"SNR levels: {snr_levels}")
    print(f"Total sample sets: {len(noise_files) * len(snr_levels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate debug samples for ONNX mobile inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default settings (5 noises, SNR -5/0/5 dB)
  python generate_debug_samples.py --onnx prk_taps_mask.onnx

  # Use specific noise files
  python generate_debug_samples.py --onnx model.onnx --noise-files noise1.wav noise2.wav

  # Custom SNR levels
  python generate_debug_samples.py --onnx model.onnx --snr-levels -10 -5 0 5 10
        """
    )
    parser.add_argument(
        "--onnx", type=str, default="prk_taps_mask.onnx",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="debug_samples",
        help="Output directory for debug samples"
    )
    parser.add_argument(
        "--noise-dir", type=str,
        default="/home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k",
        help="Directory containing noise files"
    )
    parser.add_argument(
        "--noise-files", type=str, nargs="+", default=None,
        help="Specific noise files to use (default: first 5 from noise_test.txt)"
    )
    parser.add_argument(
        "--num-noises", type=int, default=5,
        help="Number of noise files to use (default: 5, ignored if --noise-files is set)"
    )
    parser.add_argument(
        "--snr-levels", type=int, nargs="+", default=[-5, 0, 5],
        help="SNR levels to generate (default: -5, 0, 5)"
    )
    parser.add_argument(
        "--sample-index", type=int, default=0,
        help="Index of TAPS test sample to use"
    )
    parser.add_argument(
        "--n-fft", type=int, default=400,
        help="FFT size"
    )
    parser.add_argument(
        "--hop-size", type=int, default=100,
        help="Hop size"
    )
    parser.add_argument(
        "--win-size", type=int, default=400,
        help="Window size"
    )
    parser.add_argument(
        "--compress-factor", type=float, default=0.3,
        help="Magnitude compression factor"
    )

    args = parser.parse_args()

    generate_debug_samples(
        onnx_path=args.onnx,
        output_dir=args.output_dir,
        noise_dir=args.noise_dir,
        noise_files=args.noise_files,
        num_noises=args.num_noises,
        snr_levels=args.snr_levels,
        sample_index=args.sample_index,
        n_fft=args.n_fft,
        hop_size=args.hop_size,
        win_size=args.win_size,
        compress_factor=args.compress_factor,
    )
