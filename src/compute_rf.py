#!/usr/bin/env python3
"""
Receptive Field and Algorithmic Latency Calculator for PrimeKnet

This script computes the receptive field size and algorithmic latency
of the PrimeKnet model based on its hyperparameters.

Usage:
    # Direct parameters
    python src/compute_rf.py --dense_depth 4 --num_tsblock 4 \\
        --time_block_kernel 3 5 7 11 --causal True

    # From experiment in CSV
    python src/compute_rf.py --experiment prk_1117_1 \\
        --csv results/experiments.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


# ============================================================================
# Receptive Field Calculation Functions
# ============================================================================

def compute_conv_rf(rf_in: int, kernel: int, stride: int = 1, dilation: int = 1) -> Tuple[int, int]:
    """
    Compute receptive field after a convolution layer.

    Args:
        rf_in: Input receptive field size
        kernel: Kernel size
        stride: Stride (default: 1)
        dilation: Dilation rate (default: 1)

    Returns:
        Tuple of (rf_out, stride_cumulative)

    Formula:
        rf_out = rf_in + (kernel - 1) * dilation * stride_cumulative
    """
    # For single layer: stride_cumulative is just stride
    # RF increases by the effective kernel size
    effective_kernel = (kernel - 1) * dilation
    rf_out = rf_in + effective_kernel
    return rf_out, stride


def compute_dense_encoder_rf(depth: int, padding_ratio: Tuple[float, float] = (0.5, 0.5)) -> Dict[str, int]:
    """
    Compute receptive field for DenseEncoder block.

    DenseEncoder structure:
      1. Conv2d(in, dense, (1,1))  - no RF change
      2. DS_DDB (depth layers):
         - Each layer: Dilated Conv2d(*, *, (3,3), dilation=2^i)
         - Layer i uses dilation = 2^i (1, 2, 4, 8 for depth=4)
      3. Conv2d(dense, dense, (1,3), (1,2))  - freq downsampling

    Args:
        depth: Number of DS_DDB layers (default: 4)
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding

    Returns:
        Dictionary with 'time' and 'freq' receptive fields

    Example (depth=4):
        Layer 0 (dil=1): RF += (3-1)*1 = 2  → RF = 1 + 2 = 3
        Layer 1 (dil=2): RF += (3-1)*2 = 4  → RF = 3 + 4 = 7
        Layer 2 (dil=4): RF += (3-1)*4 = 8  → RF = 7 + 8 = 15
        Layer 3 (dil=8): RF += (3-1)*8 = 16 → RF = 15 + 16 = 31
    """
    rf_time = 1
    rf_freq = 1

    # Conv2d(in, dense, (1,1)) - no change

    # DS_DDB block: depth layers of dilated (3,3) convolutions
    for i in range(depth):
        dilation_time = 2 ** i  # 1, 2, 4, 8, ...
        # Time axis: dilated convolution
        rf_time, _ = compute_conv_rf(rf_time, kernel=3, dilation=dilation_time)
        # Freq axis: no dilation (dilation=1)
        rf_freq, _ = compute_conv_rf(rf_freq, kernel=3, dilation=1)

    # Conv2d(dense, dense, (1,3), (1,2))
    # Time: kernel=1, no change
    # Freq: kernel=3, stride=2
    rf_freq, _ = compute_conv_rf(rf_freq, kernel=3, stride=2)

    return {'time': rf_time, 'freq': rf_freq}


def compute_tsblock_rf(
    time_block_num: int,
    freq_block_num: int,
    time_dw_kernel: int,
    time_block_kernels: List[int],
    freq_block_kernels: List[int],
    causal: bool = False
) -> Dict[str, int]:
    """
    Compute receptive field for a single TS_BLOCK.

    TS_BLOCK structure:
      - time_block_num x Group_Prime_Kernel_FFN (time)
      - freq_block_num x Group_Prime_Kernel_FFN (freq)
      - Channel_Attention_Block (time, kernel=time_dw_kernel)

    Args:
        time_block_num: Number of time processing blocks
        freq_block_num: Number of frequency processing blocks
        time_dw_kernel: Depthwise kernel for channel attention
        time_block_kernels: List of kernel sizes for time blocks
        freq_block_kernels: List of kernel sizes for freq blocks
        causal: Whether model is causal

    Returns:
        Dictionary with 'time' and 'freq' receptive fields
    """
    rf_time = 1
    rf_freq = 1

    # Time blocks
    for _ in range(time_block_num):
        # Each Group_Prime_Kernel_FFN uses max kernel from kernel_list
        max_time_kernel = max(time_block_kernels)
        rf_time, _ = compute_conv_rf(rf_time, kernel=max_time_kernel)

    # Frequency blocks
    for _ in range(freq_block_num):
        max_freq_kernel = max(freq_block_kernels)
        rf_freq, _ = compute_conv_rf(rf_freq, kernel=max_freq_kernel)

    # Channel Attention Block (time dimension)
    rf_time, _ = compute_conv_rf(rf_time, kernel=time_dw_kernel)

    return {'time': rf_time, 'freq': rf_freq}


def compute_mask_decoder_rf(depth: int, padding_ratio: Tuple[float, float] = (0.5, 0.5)) -> Dict[str, int]:
    """
    Compute receptive field for MaskDecoder block.

    MaskDecoder structure:
      1. Conv2d(dense, dense, (1,2)) - freq upsampling
      2. DS_DDB (depth layers):
         - Each layer: Dilated Conv2d(*, *, (3,3), dilation=2^i)
         - Layer i uses dilation = 2^i (1, 2, 4, 8 for depth=4)
      3. Conv2d(dense, out, (1,1)) - no RF change

    Args:
        depth: Number of DS_DDB layers
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding

    Returns:
        Dictionary with 'time' and 'freq' receptive fields

    Example (depth=4):
        Same as DenseEncoder: RF = 31 frames
    """
    rf_time = 1
    rf_freq = 1

    # Conv2d(dense, dense, (1,2)) - freq upsampling
    # Time: kernel=1, no change
    # Freq: kernel=2, stride=1
    rf_freq, _ = compute_conv_rf(rf_freq, kernel=2)

    # DS_DDB block: depth layers of dilated (3,3) convolutions
    for i in range(depth):
        dilation_time = 2 ** i  # 1, 2, 4, 8, ...
        # Time axis: dilated convolution
        rf_time, _ = compute_conv_rf(rf_time, kernel=3, dilation=dilation_time)
        # Freq axis: no dilation (dilation=1)
        rf_freq, _ = compute_conv_rf(rf_freq, kernel=3, dilation=1)

    # Conv2d(dense, out, (1,1)) - no change

    return {'time': rf_time, 'freq': rf_freq}


def compute_primeknet_rf(params: Dict) -> Dict[str, int]:
    """
    Compute total receptive field for PrimeKnet model.

    Args:
        params: Dictionary of model hyperparameters
            - dense_depth: Depth of DenseEncoder/MaskDecoder
            - num_tsblock: Number of TS_BLOCK modules
            - time_block_num: Time blocks per TS_BLOCK
            - freq_block_num: Freq blocks per TS_BLOCK
            - time_dw_kernel_size: DW kernel for channel attention
            - time_block_kernel: List of kernel sizes for time
            - freq_block_kernel: List of kernel sizes for freq
            - causal: Whether model is causal
            - encoder_padding_ratio: Encoder padding (left, right)
            - decoder_padding_ratio: Decoder padding (left, right)

    Returns:
        Dictionary with total 'time' and 'freq' receptive fields
    """
    # Extract parameters
    dense_depth = params.get('dense_depth', 4)
    num_tsblock = params.get('num_tsblock', 4)
    time_block_num = params.get('time_block_num', 2)
    freq_block_num = params.get('freq_block_num', 2)
    time_dw_kernel = params.get('time_dw_kernel_size', 3)
    time_block_kernel = params.get('time_block_kernel', [3, 11, 23, 31])
    freq_block_kernel = params.get('freq_block_kernel', [3, 11, 23, 31])
    causal = params.get('causal', False)
    encoder_padding_ratio = params.get('encoder_padding_ratio', (0.5, 0.5))
    decoder_padding_ratio = params.get('decoder_padding_ratio', (0.5, 0.5))

    # Initialize RF
    rf_time = 1
    rf_freq = 1

    # DenseEncoder
    encoder_rf = compute_dense_encoder_rf(dense_depth, encoder_padding_ratio)
    rf_time += encoder_rf['time'] - 1
    rf_freq += encoder_rf['freq'] - 1

    # TS_BLOCK x num_tsblock
    for _ in range(num_tsblock):
        tsblock_rf = compute_tsblock_rf(
            time_block_num, freq_block_num,
            time_dw_kernel, time_block_kernel, freq_block_kernel,
            causal
        )
        rf_time += tsblock_rf['time'] - 1
        rf_freq += tsblock_rf['freq'] - 1

    # MaskDecoder
    decoder_rf = compute_mask_decoder_rf(dense_depth, decoder_padding_ratio)
    rf_time += decoder_rf['time'] - 1
    rf_freq += decoder_rf['freq'] - 1

    return {
        'time': rf_time,
        'freq': rf_freq,
        'encoder_rf_time': encoder_rf['time'],
        'encoder_rf_freq': encoder_rf['freq'],
        'tsblock_rf_time': tsblock_rf['time'] * num_tsblock,
        'tsblock_rf_freq': tsblock_rf['freq'] * num_tsblock,
        'decoder_rf_time': decoder_rf['time'],
        'decoder_rf_freq': decoder_rf['freq'],
    }


# ============================================================================
# Latency Calculation Functions
# ============================================================================

def compute_algorithmic_latency(
    rf_time: int,
    hop_len: int,
    sample_rate: int = 16000,
    encoder_padding_ratio: Tuple[float, float] = (0.5, 0.5)
) -> Dict[str, float]:
    """
    Compute algorithmic latency based on look-ahead frames (single module).

    DEPRECATED: Use compute_total_latency() for full model latency calculation.

    Total Algorithmic Latency = STFT overhead + Model look-ahead
                              = (R + 1) × hop_len

    Where R = number of future frames (right padding / look-ahead)

    The "+1" accounts for STFT overhead: must wait at least 1 hop to produce output.

    For fully causal (1.0, 0.0): R=0, latency = 1 hop = 6.25ms
    For symmetric (0.5, 0.5): R=15 (for encoder RF=31), latency = 16 hops = 100ms

    Args:
        rf_time: Time-axis receptive field size (in frames) - should be ENCODER RF
        hop_len: STFT hop size (in samples)
        sample_rate: Audio sample rate (default: 16000 Hz)
        encoder_padding_ratio: (left_ratio, right_ratio) tuple

    Returns:
        Dictionary with latency in samples, ms, and frames, plus look-ahead info
    """
    left_ratio, right_ratio = encoder_padding_ratio

    # R = number of future frames (look-ahead)
    look_ahead_frames = (rf_time - 1) * right_ratio

    # Past context frames (for reference)
    past_context_frames = (rf_time - 1) * left_ratio

    # Total latency = STFT overhead (1 hop) + model look-ahead (R hops)
    latency_frames = look_ahead_frames + 1

    # Latency in samples
    latency_samples = latency_frames * hop_len

    # Latency in milliseconds
    latency_ms = (latency_samples / sample_rate) * 1000

    return {
        'latency_frames': latency_frames,
        'latency_samples': latency_samples,
        'latency_ms': latency_ms,
        'look_ahead_frames': look_ahead_frames,
        'past_context_frames': past_context_frames
    }


def compute_total_latency(
    encoder_rf_time: int,
    tsblock_rf_time: int,
    decoder_rf_time: int,
    hop_len: int,
    sample_rate: int = 16000,
    encoder_padding_ratio: Tuple[float, float] = (0.5, 0.5),
    decoder_padding_ratio: Tuple[float, float] = (0.5, 0.5),
    tsblock_causal: bool = True
) -> Dict[str, float]:
    """
    Compute total algorithmic latency considering all modules.

    Look-ahead accumulates through the model pipeline:
      - Encoder look-ahead: R_enc
      - TS_BLOCK look-ahead: R_ts (0 if causal, else accumulated)
      - Decoder look-ahead: R_dec
      - Total look-ahead: R_total = R_enc + R_ts + R_dec

    Total Latency = (R_total + 1) × hop_len
    The "+1" is STFT overhead (must wait at least 1 hop).

    Args:
        encoder_rf_time: Encoder receptive field (time axis)
        tsblock_rf_time: TS_BLOCK receptive field (time axis) per block
        decoder_rf_time: Decoder receptive field (time axis)
        hop_len: STFT hop size in samples
        sample_rate: Audio sample rate (default: 16000 Hz)
        encoder_padding_ratio: (left, right) padding ratio for encoder
        decoder_padding_ratio: (left, right) padding ratio for decoder
        tsblock_causal: Whether TS_BLOCK is causal (True = no look-ahead)

    Returns:
        Dictionary with cumulative latency information
    """
    # Encoder look-ahead
    # Must use round() to ensure integer frames, as padding is applied in integer units
    encoder_left, encoder_right = encoder_padding_ratio
    total_pad_enc = encoder_rf_time - 1
    left_pad_enc = round(total_pad_enc * encoder_left)
    R_enc = total_pad_enc - left_pad_enc  # right_pad = total - left (ensures integer)

    # TS_BLOCK look-ahead
    if tsblock_causal:
        R_ts = 0
    else:
        # TS_BLOCK uses symmetric padding when non-causal
        # Symmetric: left_ratio = right_ratio = 0.5
        total_pad_ts = tsblock_rf_time - 1
        left_pad_ts = round(total_pad_ts * 0.5)
        R_ts = total_pad_ts - left_pad_ts

    # Decoder look-ahead
    decoder_left, decoder_right = decoder_padding_ratio
    total_pad_dec = decoder_rf_time - 1
    left_pad_dec = round(total_pad_dec * decoder_left)
    R_dec = total_pad_dec - left_pad_dec

    # Total look-ahead (cumulative)
    R_total = R_enc + R_ts + R_dec

    # Total latency = STFT overhead + total look-ahead
    latency_frames = R_total + 1

    # Convert to samples and milliseconds
    latency_samples = latency_frames * hop_len
    latency_ms = (latency_samples / sample_rate) * 1000

    return {
        'latency_frames': latency_frames,
        'latency_samples': latency_samples,
        'latency_ms': latency_ms,
        'look_ahead_frames_total': R_total,
        'look_ahead_frames_encoder': R_enc,
        'look_ahead_frames_tsblock': R_ts,
        'look_ahead_frames_decoder': R_dec,
        'past_context_frames_encoder': left_pad_enc,  # Use calculated integer value
        'past_context_frames_decoder': left_pad_dec,  # Use calculated integer value
    }


# ============================================================================
# CLI and Formatting Functions
# ============================================================================

def format_summary(params: Dict, rf_result: Dict, latency_result: Dict) -> str:
    """
    Format results as a summary table.

    Args:
        params: Model parameters
        rf_result: Receptive field results
        latency_result: Latency results

    Returns:
        Formatted string
    """
    output = []
    output.append("=" * 70)
    output.append("PrimeKnet Receptive Field & Latency Analysis")
    output.append("=" * 70)
    output.append("")

    # Configuration
    output.append("Configuration:")
    output.append(f"  Model: PrimeKnet")
    output.append(f"  Causal: {params.get('causal', False)}")
    output.append(f"  Dense Depth: {params.get('dense_depth', 4)}")
    output.append(f"  Num TS_BLOCK: {params.get('num_tsblock', 4)}")
    output.append(f"  Encoder Padding: {params.get('encoder_padding_ratio', (0.5, 0.5))}")
    output.append(f"  Decoder Padding: {params.get('decoder_padding_ratio', (0.5, 0.5))}")
    output.append("")

    # Receptive Field
    output.append("Receptive Field:")
    hop_len = params.get('hop_len', 100)
    sr = params.get('sampling_rate', 16000)
    time_ms = (rf_result['time'] * hop_len / sr) * 1000

    output.append(f"  Time-axis RF: {rf_result['time']} frames ({time_ms:.1f}ms @ {sr}Hz)")
    output.append(f"  Frequency-axis RF: {rf_result['freq']} bins")
    output.append("")

    # Algorithmic Latency
    output.append("Algorithmic Latency:")
    output.append(f"  STFT hop size: {hop_len} samples ({hop_len/sr*1000:.2f}ms)")
    output.append("")
    output.append("  Look-ahead Breakdown:")

    # Check if we have the new detailed look-ahead data or old format
    if 'look_ahead_frames_total' in latency_result:
        # New format with detailed breakdown
        R_enc = latency_result['look_ahead_frames_encoder']
        R_ts = latency_result['look_ahead_frames_tsblock']
        R_dec = latency_result['look_ahead_frames_decoder']
        R_total = latency_result['look_ahead_frames_total']

        output.append(f"    Encoder:   {int(R_enc)} frames ({R_enc*hop_len/sr*1000:.2f}ms)")
        output.append(f"    TS_BLOCK:  {int(R_ts)} frames ({R_ts*hop_len/sr*1000:.2f}ms)")
        output.append(f"    Decoder:   {int(R_dec)} frames ({R_dec*hop_len/sr*1000:.2f}ms)")
        output.append(f"    Total:     {int(R_total)} frames ({R_total*hop_len/sr*1000:.2f}ms)")
    else:
        # Old format (single look-ahead value)
        R_total = latency_result.get('look_ahead_frames', 0)
        output.append(f"    Total:     {R_total:.1f} frames ({R_total*hop_len/sr*1000:.2f}ms)")

    output.append("")
    output.append(f"  Total Latency: {latency_result['latency_frames']:.1f} frames " +
                  f"(STFT overhead + look-ahead)")
    output.append(f"  Total Latency: {latency_result['latency_samples']:.1f} samples")
    output.append(f"  Total Latency: {latency_result['latency_ms']:.2f}ms")
    output.append("")

    # Layer-wise Breakdown
    output.append("Layer-wise Breakdown:")
    output.append(f"  DenseEncoder:    RF_time={rf_result['encoder_rf_time']}, " +
                  f"RF_freq={rf_result['encoder_rf_freq']}")
    output.append(f"  TS_BLOCK x{params.get('num_tsblock', 4)}:    " +
                  f"RF_time=+{rf_result['tsblock_rf_time']}, " +
                  f"RF_freq=+{rf_result['tsblock_rf_freq']}")
    output.append(f"  MaskDecoder:     RF_time={rf_result['decoder_rf_time']}, " +
                  f"RF_freq={rf_result['decoder_rf_freq']}")
    output.append("")
    output.append("=" * 70)

    return "\n".join(output)


def parse_experiment_from_csv(exp_name: str, csv_path: Path) -> Optional[Dict]:
    """
    Load experiment parameters from CSV file.

    Args:
        exp_name: Experiment name
        csv_path: Path to experiments.csv

    Returns:
        Dictionary of parameters or None if not found
    """
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Find experiment
    exp_rows = df[df['exp_name'] == exp_name]
    if exp_rows.empty:
        print(f"Error: Experiment '{exp_name}' not found in CSV")
        return None

    row = exp_rows.iloc[0]

    # Extract parameters
    params = {}

    # Required params (handle NaN values)
    def get_int_param(key, default):
        val = row.get(key, default)
        return int(val) if not pd.isna(val) else default

    def get_bool_param(key, default):
        val = row.get(key, default)
        return bool(val) if not pd.isna(val) else default

    params['dense_depth'] = get_int_param('dense_depth', 4)
    params['num_tsblock'] = get_int_param('num_tsblock', 4)
    params['time_block_num'] = get_int_param('time_block_num', 2)
    params['freq_block_num'] = get_int_param('freq_block_num', 2)
    params['time_dw_kernel_size'] = get_int_param('time_dw_kernel_size', 3)
    params['hop_len'] = get_int_param('hop_len', 100)
    params['causal'] = get_bool_param('causal', False)

    # Parse list parameters
    import ast
    params['time_block_kernel'] = ast.literal_eval(row.get('time_block_kernel', '[3, 11, 23, 31]'))
    params['freq_block_kernel'] = ast.literal_eval(row.get('freq_block_kernel', '[3, 11, 23, 31]'))

    # Parse padding ratios
    encoder_padding = row.get('encoder_padding_ratio', None)
    if encoder_padding and encoder_padding != 'N/A' and not pd.isna(encoder_padding):
        params['encoder_padding_ratio'] = tuple(ast.literal_eval(encoder_padding))
    else:
        params['encoder_padding_ratio'] = (0.5, 0.5)

    decoder_padding = row.get('decoder_padding_ratio', None)
    if decoder_padding and decoder_padding != 'N/A' and not pd.isna(decoder_padding):
        params['decoder_padding_ratio'] = tuple(ast.literal_eval(decoder_padding))
    else:
        params['decoder_padding_ratio'] = (0.5, 0.5)

    return params


def main():
    parser = argparse.ArgumentParser(
        description="Compute receptive field and latency for PrimeKnet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From experiment CSV
  python src/compute_rf.py --experiment prk_1117_1 --csv results/experiments.csv

  # Direct parameters
  python src/compute_rf.py --dense_depth 4 --num_tsblock 4 \\
      --time_block_kernel 3 5 7 11 --causal True
        """
    )

    # Experiment mode
    parser.add_argument('--experiment', type=str, help='Experiment name from CSV')
    parser.add_argument('--csv', type=str, default='results/experiments.csv',
                        help='Path to experiments CSV file')

    # Direct parameter mode
    parser.add_argument('--dense_depth', type=int, default=4, help='Dense encoder/decoder depth')
    parser.add_argument('--num_tsblock', type=int, default=4, help='Number of TS_BLOCK modules')
    parser.add_argument('--time_block_num', type=int, default=2, help='Time blocks per TS_BLOCK')
    parser.add_argument('--freq_block_num', type=int, default=2, help='Freq blocks per TS_BLOCK')
    parser.add_argument('--time_dw_kernel_size', type=int, default=3, help='DW kernel size')
    parser.add_argument('--time_block_kernel', type=int, nargs='+', default=[3, 11, 23, 31],
                        help='Time block kernel sizes')
    parser.add_argument('--freq_block_kernel', type=int, nargs='+', default=[3, 11, 23, 31],
                        help='Freq block kernel sizes')
    parser.add_argument('--hop_len', type=int, default=100, help='STFT hop length')
    parser.add_argument('--causal', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether model is causal')
    parser.add_argument('--encoder_padding_ratio', type=float, nargs=2, default=[0.5, 0.5],
                        help='Encoder padding ratio (left right)')
    parser.add_argument('--decoder_padding_ratio', type=float, nargs=2, default=[0.5, 0.5],
                        help='Decoder padding ratio (left right)')

    args = parser.parse_args()

    # Load parameters
    if args.experiment:
        params = parse_experiment_from_csv(args.experiment, Path(args.csv))
        if params is None:
            return
    else:
        params = {
            'dense_depth': args.dense_depth,
            'num_tsblock': args.num_tsblock,
            'time_block_num': args.time_block_num,
            'freq_block_num': args.freq_block_num,
            'time_dw_kernel_size': args.time_dw_kernel_size,
            'time_block_kernel': args.time_block_kernel,
            'freq_block_kernel': args.freq_block_kernel,
            'hop_len': args.hop_len,
            'causal': args.causal,
            'encoder_padding_ratio': tuple(args.encoder_padding_ratio),
            'decoder_padding_ratio': tuple(args.decoder_padding_ratio),
        }

    # Compute RF and latency
    rf_result = compute_primeknet_rf(params)

    # Compute total latency considering all modules (encoder, tsblock, decoder)
    # Determine tsblock_causal from params
    tsblock_causal = params.get('causal', False)

    latency_result = compute_total_latency(
        encoder_rf_time=rf_result['encoder_rf_time'],
        tsblock_rf_time=rf_result['tsblock_rf_time'],
        decoder_rf_time=rf_result['decoder_rf_time'],
        hop_len=params['hop_len'],
        encoder_padding_ratio=params['encoder_padding_ratio'],
        decoder_padding_ratio=params.get('decoder_padding_ratio', (0.5, 0.5)),
        tsblock_causal=tsblock_causal
    )

    # Print summary
    summary = format_summary(params, rf_result, latency_result)
    print(summary)


if __name__ == '__main__':
    main()
