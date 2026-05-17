"""Generate TAPS-aware calibration .npz set for the S8 functional-stateful
BAFNet+ INT8 QDQ export (S17, post-S9.6).

The S17 INT8 QDQ branch of ``src.models.streaming.onnx.export`` consumes a
directory of ``.npz`` files via :class:`onnxruntime.quantization.CalibrationDataReader`.
Each file must contain the four host-side audio inputs the S8 graph expects:

    bcs_mag, bcs_pha, acs_mag, acs_pha   — shape ``[1, F=201, T=14]``

with the host's STFT contract (``n_fft=400``, ``hop=100``, ``win=400``,
``compress_factor=0.3``, ``center=True``). State tensors are zero-filled by
the reader (mirrors real cold-start activations).

The chunking sequence mirrors the host wrapper's first-chunk behaviour:
each ``.npz`` represents a single ``T_export=14`` slice from a longer
streaming session, so the calibrator sees the same activation distribution
the deployed Galaxy device produces on chunk 0.

Two distributions are emitted (Defect D recipe — see
``docs/wiki/concepts/android-streaming-deployment.md`` → § Pipeline
Foot-Guns → Defect D):

  * **voiced** — paired (BCS, ACS) chunks drawn from TAPS test
    ``throat_microphone`` (BCS) + ``acoustic_microphone`` (ACS) at 16 kHz;
    the same modality the device feeds at inference time. Default
    ``20 utts × 20 chunks = 400`` files.
  * **silence** — strictly-zero audio on both channels. Without these the
    QDQ activation zero-points lock to the voiced mean and silence input
    propagates through the model as a deterministic non-zero "bias
    signal", producing the Defect D first-chunk onset click + sustained
    tonal noise floor on the deployed device. Default ``50`` files.

Usage (from BAFNetPlus repo root):

    python scripts/make_calibration_npz.py \\
        --output_dir /tmp/bafnet_calib_taps_v3 \\
        --num_utts 20 --chunks_per_utt 20 --num_silence_chunks 50

Outputs ``num_utts × chunks_per_utt`` voiced files named
``calib_voiced_<utt>_<chunk>.npz`` plus ``num_silence_chunks`` silence
files named ``calib_silence_<idx>.npz`` under ``output_dir``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stft import mag_pha_stft  # noqa: E402

# S8 (50 ms variant) geometry — anchored from
# results/onnx/bafnetplus_50ms_fp32.onnx.json. Re-confirm by reading the
# sidecar in code if a future variant lands.
N_FFT = 400
HOP = 100
WIN = 400
COMPRESS = 0.3
SR = 16000
T_EXPORT = 14                        # chunk + L_enc + L_dec = 8 + 3 + 3.
TOTAL_FRAMES_NEEDED = 11             # chunk + L_enc — Python streaming wrapper.
SAMPLES_PER_CHUNK = (TOTAL_FRAMES_NEEDED - 1) * HOP + WIN // 2  # = 1200
OUTPUT_SAMPLES = 8 * HOP             # chunk_size_frames * hop = 800.
FREQ_BINS = N_FFT // 2 + 1           # = 201


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to write calib_*.npz files into.",
    )
    p.add_argument(
        "--num_utts",
        type=int,
        default=20,
        help="Number of TAPS test utterances to draw from (default 20).",
    )
    p.add_argument(
        "--chunks_per_utt",
        type=int,
        default=20,
        help="Number of T_export=14 streaming chunks to extract per utterance (default 20).",
    )
    p.add_argument(
        "--start_utt",
        type=int,
        default=0,
        help="First TAPS test utt_idx (default 0).",
    )
    p.add_argument(
        "--num_silence_chunks",
        type=int,
        default=50,
        help=(
            "Number of pure-silence chunks to emit (default 50). Required to keep INT8 "
            "zero-points anchored at audio=0; see Defect D in "
            "docs/wiki/concepts/android-streaming-deployment.md."
        ),
    )
    return p.parse_args()


def load_bcs_acs(utt_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one TAPS test (BCS, ACS) pair at 16 kHz."""
    from datasets import load_dataset

    ds = load_dataset(
        "yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test"
    )
    item = ds[utt_idx]
    bcs = np.asarray(item["audio.throat_microphone"]["array"], dtype=np.float32)
    acs = np.asarray(item["audio.acoustic_microphone"]["array"], dtype=np.float32)
    bcs_sr = item["audio.throat_microphone"]["sampling_rate"]
    acs_sr = item["audio.acoustic_microphone"]["sampling_rate"]
    if bcs_sr != SR or acs_sr != SR:
        raise RuntimeError(
            f"utt_idx={utt_idx} sample rate mismatch: bcs_sr={bcs_sr}, acs_sr={acs_sr} (expected {SR})"
        )
    if bcs.size != acs.size:
        # TAPS pairs are equal-length by dataset contract — fail fast if not.
        n = int(min(bcs.size, acs.size))
        bcs = bcs[:n]
        acs = acs[:n]
    return bcs, acs


def _stft_chunk(audio_with_context: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Run the host STFT on a ``T_export=14`` worth of audio (with context).

    Returns ``(mag, pha)`` each shaped ``[1, F=201, T=14]`` in float32 numpy.
    Matches the host wrapper's ``center=True`` path used by
    :class:`~src.models.streaming.onnx.ort_wrapper.BAFNetPlusOrtStreaming`.
    """
    mag, pha, _com = mag_pha_stft(
        audio_with_context.unsqueeze(0),
        n_fft=N_FFT, hop_size=HOP, win_size=WIN,
        compress_factor=COMPRESS, center=True,
    )
    # mag_pha_stft returns [B, F, T] when input is [B, samples] under center=True.
    # We crop to T_export=14 frames so the calibrator sees the exact shape the
    # ORT graph consumes; with center=True + len(audio) chosen below, the STFT
    # produces exactly T_export frames so no crop is needed in practice.
    if mag.shape[2] != T_EXPORT:
        # Defensive crop: take leading T_export frames.
        mag = mag[:, :, :T_EXPORT]
        pha = pha[:, :, :T_EXPORT]
    return mag.numpy().astype(np.float32), pha.numpy().astype(np.float32)


def _t_export_samples_needed() -> int:
    """Sample count required for ``center=True`` STFT to emit ``T_export`` frames.

    Under ``center=True`` (reflect-pad by ``n_fft // 2`` on each side), the
    frame count is ``floor(len / hop) + 1`` for ``len >= n_fft // 2``. So
    ``len = (T_export - 1) * hop = 13 * 100 = 1300`` samples gives exactly 14
    frames. We use ``1300`` samples (no extra context required because
    ``center=True`` reflects internally).
    """
    return (T_EXPORT - 1) * HOP  # = 1300


def stream_chunks_for_calibration(
    bcs_audio: np.ndarray,
    acs_audio: np.ndarray,
    max_chunks: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Yield ``T_export=14`` STFT slices stepping by ``OUTPUT_SAMPLES = 800`` per chunk.

    Mirrors the device's matured-chunk cadence — each chunk slot advances by
    one ``output_samples_per_chunk = 800`` step (50 ms @ 16 kHz). The host
    STFT uses ``center=True`` so the calibration NPZ format matches the
    runtime ORT graph's input layout exactly.
    """
    n_samples_needed = _t_export_samples_needed()
    bcs_t = torch.from_numpy(bcs_audio.astype(np.float32))
    acs_t = torch.from_numpy(acs_audio.astype(np.float32))
    n_total = int(min(bcs_t.numel(), acs_t.numel()))
    if n_total < n_samples_needed:
        raise RuntimeError(
            f"audio too short: {n_total} samples — need at least {n_samples_needed} for one T_export chunk."
        )
    emitted = 0
    step = OUTPUT_SAMPLES
    for start in range(0, n_total - n_samples_needed + 1, step):
        end = start + n_samples_needed
        bcs_slice = bcs_t[start:end]
        acs_slice = acs_t[start:end]
        bcs_mag, bcs_pha = _stft_chunk(bcs_slice)
        acs_mag, acs_pha = _stft_chunk(acs_slice)
        yield emitted, bcs_mag, bcs_pha, acs_mag, acs_pha
        emitted += 1
        if emitted >= max_chunks:
            return


def silence_mag_pha() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mag, pha)`` for one strictly-zero audio chunk.

    Both context and chunk samples are zero. :func:`src.stft.mag_pha_stft`
    adds ``1e-9`` inside ``sqrt`` and ``1e-8`` inside ``atan2`` for numerical
    stability, so the resulting tensors are not strictly zero — they sit at
    the deterministic floor ``mag ≈ (1e-9)^(compress/2) ≈ 0.04467`` and
    ``pha = atan2(1e-8, 1e-8) = π/4`` across every frame/bin. This is the
    exact same floor a device-side silent input would produce, so feeding
    these to the QDQ calibrator anchors activation zero-points at the real
    silence regime — Defect D requires this.
    """
    n_samples = _t_export_samples_needed()
    zero_audio = torch.zeros(n_samples, dtype=torch.float32)
    mag, pha = _stft_chunk(zero_audio)
    return mag, pha


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("make_calibration_npz.py (S17 — functional-stateful BAFNet+ 190-state)")
    print(f"  output_dir          : {out_dir}")
    print(f"  num_utts            : {args.num_utts}")
    print(f"  chunks_per_utt      : {args.chunks_per_utt}")
    print(f"  start_utt           : {args.start_utt}")
    print(f"  num_silence_chunks  : {args.num_silence_chunks}")
    print(f"  T_export            : {T_EXPORT}, freq_bins: {FREQ_BINS}")
    print(f"  samples per T_export chunk : {_t_export_samples_needed()} (center=True STFT)")

    voiced_written = 0
    for k in range(args.num_utts):
        utt_idx = args.start_utt + k
        bcs_audio, acs_audio = load_bcs_acs(utt_idx)
        print(
            f"  voiced utt_idx={utt_idx} bcs={len(bcs_audio)} samples "
            f"({len(bcs_audio) / SR:.2f} s) -> {args.chunks_per_utt} chunks"
        )
        for chunk_idx, bcs_mag, bcs_pha, acs_mag, acs_pha in stream_chunks_for_calibration(
            bcs_audio, acs_audio, max_chunks=args.chunks_per_utt
        ):
            assert bcs_mag.shape == (1, FREQ_BINS, T_EXPORT), f"unexpected bcs_mag shape: {bcs_mag.shape}"
            assert acs_mag.shape == (1, FREQ_BINS, T_EXPORT), f"unexpected acs_mag shape: {acs_mag.shape}"
            out_path = out_dir / f"calib_voiced_{utt_idx:03d}_{chunk_idx:03d}.npz"
            np.savez(out_path, bcs_mag=bcs_mag, bcs_pha=bcs_pha, acs_mag=acs_mag, acs_pha=acs_pha)
            voiced_written += 1

    silence_written = 0
    if args.num_silence_chunks > 0:
        sil_mag, sil_pha = silence_mag_pha()
        # Sanity: deterministic
        sil_mag_2, sil_pha_2 = silence_mag_pha()
        assert np.array_equal(sil_mag, sil_mag_2) and np.array_equal(sil_pha, sil_pha_2), (
            "silence mag/pha is not deterministic across two calls"
        )
        sil_mag_floor = float(sil_mag.max())
        sil_pha_floor = float(sil_pha.max())
        # Sanity: all frames/bins sit at the floor (constant across spectrogram).
        assert np.allclose(sil_mag, sil_mag_floor), (
            f"silence mag has non-uniform values: min={sil_mag.min()} max={sil_mag.max()}"
        )
        assert np.allclose(sil_pha, sil_pha_floor), (
            f"silence pha has non-uniform values: min={sil_pha.min()} max={sil_pha.max()}"
        )
        for idx in range(args.num_silence_chunks):
            out_path = out_dir / f"calib_silence_{idx:03d}.npz"
            # Both channels are silent — bcs and acs share the same floor tensors.
            np.savez(
                out_path,
                bcs_mag=sil_mag,
                bcs_pha=sil_pha,
                acs_mag=sil_mag,
                acs_pha=sil_pha,
            )
            silence_written += 1
        print(
            f"  silence chunks written: {silence_written} "
            f"(mag floor={sil_mag_floor:.6f}, pha floor={sil_pha_floor:.6f}; "
            f"deterministic — duplicates anchor activation zero-points to the device-side silence regime)"
        )

    print(f"\nWrote {voiced_written + silence_written} calibration npz files to {out_dir}")
    print(f"  voiced  : {voiced_written}")
    print(f"  silence : {silence_written}")


if __name__ == "__main__":
    main()
