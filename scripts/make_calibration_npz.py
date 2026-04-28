"""Generate TAPS-aware calibration .npz set for export_onnx.py --quantize_qdq.

quantize_onnx_qdq_for_htp's CalibrationDataReader expects each .npz to
contain ``mag`` and ``pha`` arrays shaped [1, F, export_time_frames]
matching the streaming model inputs. Missing ``state_*`` keys are zero-
filled by the reader (matching real cold-start behaviour). Without a
calibration_dir the reader falls back to random data, which produces
quantization scales that are well outside the actual BCS distribution
and inflates HTP-side numerical drift; this script provides the real
distribution so the quantized model's accuracy approaches the FP32 one.

The chunking sequence is identical to make_enhancer_golden.py /
StreamingEnhancer.processChunk on device — 200-sample left context +
1200-sample chunk → STFT center=False → 11 frames — so the calibration
distribution matches what the device actually feeds into the model at
inference time.

Usage (from repo root):

    python BAFNetPlus/scripts/make_calibration_npz.py \\
        --output_dir /tmp/bafnet_calib_taps \\
        --num_utts 5 --chunks_per_utt 20

Outputs ``num_utts × chunks_per_utt`` files named
``calib_<utt>_<chunk>.npz`` under output_dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BAFNETPLUS_ROOT = REPO_ROOT / "BAFNetPlus"
if str(BAFNETPLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(BAFNETPLUS_ROOT))

from src.stft import mag_pha_stft  # noqa: E402

# Streaming export constants (bm_map_50ms) — mirror make_enhancer_golden.py.
CHUNK_FRAMES = 8
EXPORT_FRAMES = 11
N_FFT = 400
HOP = 100
WIN = 400
COMPRESS = 0.3
SR = 16000
SAMPLES_PER_CHUNK = (EXPORT_FRAMES - 1) * HOP + WIN // 2  # 1200
OUTPUT_SAMPLES = CHUNK_FRAMES * HOP  # 800


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True, type=str,
                   help="Directory to write calib_<utt>_<chunk>.npz files into.")
    p.add_argument("--num_utts", type=int, default=5,
                   help="Number of TAPS test utterances to draw from (default 5).")
    p.add_argument("--chunks_per_utt", type=int, default=20,
                   help="Number of streaming chunks to extract per utterance (default 20).")
    p.add_argument("--start_utt", type=int, default=0,
                   help="First TAPS test utt_idx (default 0).")
    return p.parse_args()


def load_bcs(utt_idx: int) -> np.ndarray:
    """Load one TAPS test BCS sample at 16 kHz."""
    from datasets import load_dataset
    ds = load_dataset(
        "yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test"
    )
    item = ds[utt_idx]
    bcs = np.asarray(item["audio.throat_microphone"]["array"], dtype=np.float32)
    sr = item["audio.throat_microphone"]["sampling_rate"]
    if sr != SR:
        raise RuntimeError(f"utt_idx={utt_idx} sample rate {sr} != {SR}")
    return bcs


def stft_chunk(chunk_with_context: torch.Tensor):
    mag, pha, _ = mag_pha_stft(
        chunk_with_context.unsqueeze(0),
        n_fft=N_FFT, hop_size=HOP, win_size=WIN,
        compress_factor=COMPRESS, center=False,
    )
    return mag, pha  # [1, F, 11]


def stream_chunks(audio: np.ndarray, max_chunks: int):
    """Replicate StreamingEnhancer.processChunk's STFT inputs and yield (push_idx, mag, pha)."""
    audio_t = torch.from_numpy(audio.astype(np.float32))
    n_pushes = len(audio_t) // OUTPUT_SAMPLES
    if n_pushes < 2:
        raise RuntimeError(
            f"audio too short: {len(audio_t)} samples → {n_pushes} pushes (need ≥ 2)."
        )
    stft_context = torch.zeros(WIN // 2)
    input_buffer = torch.tensor([], dtype=torch.float32)
    emitted = 0
    for push_idx in range(n_pushes):
        new_chunk = audio_t[push_idx * OUTPUT_SAMPLES:(push_idx + 1) * OUTPUT_SAMPLES]
        input_buffer = torch.cat([input_buffer, new_chunk])
        if len(input_buffer) < SAMPLES_PER_CHUNK:
            continue
        chunk_samples = input_buffer[:SAMPLES_PER_CHUNK]
        chunk_with_context = torch.cat([stft_context, chunk_samples])
        mag, pha = stft_chunk(chunk_with_context)
        adv = OUTPUT_SAMPLES
        ctx_size = WIN // 2
        if adv >= ctx_size:
            stft_context = input_buffer[adv - ctx_size:adv].clone()
        else:
            need = ctx_size - adv
            stft_context = torch.cat([stft_context[-need:], input_buffer[:adv]]).clone()
        input_buffer = input_buffer[OUTPUT_SAMPLES:]
        yield push_idx, mag.numpy().astype(np.float32), pha.numpy().astype(np.float32)
        emitted += 1
        if emitted >= max_chunks:
            return


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"make_calibration_npz.py")
    print(f"  output_dir     : {out_dir}")
    print(f"  num_utts       : {args.num_utts}")
    print(f"  chunks_per_utt : {args.chunks_per_utt}")
    print(f"  start_utt      : {args.start_utt}")

    total_written = 0
    for k in range(args.num_utts):
        utt_idx = args.start_utt + k
        bcs = load_bcs(utt_idx)
        print(f"  utt_idx={utt_idx} bcs={len(bcs)} samples ({len(bcs) / SR:.2f} s)")
        for chunk_idx, mag, pha in stream_chunks(bcs, max_chunks=args.chunks_per_utt):
            assert mag.shape == (1, N_FFT // 2 + 1, EXPORT_FRAMES), \
                f"unexpected mag shape: {mag.shape}"
            out_path = out_dir / f"calib_{utt_idx:03d}_{chunk_idx:03d}.npz"
            np.savez(out_path, mag=mag, pha=pha)
            total_written += 1

    print(f"\nWrote {total_written} calibration npz files to {out_dir}")


if __name__ == "__main__":
    main()
