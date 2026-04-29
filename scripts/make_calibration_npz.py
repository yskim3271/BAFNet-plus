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

Two distributions are emitted:

  * **voiced** — TAPS test BCS (throat_microphone) recordings, the same
    modality the device feeds at inference time.
  * **silence** — strictly zero audio chunks. Without these, the QDQ
    activation zero-points lock to the voiced mean and silence input
    propagates through the model as a deterministic non-zero "bias
    signal" (Defect D — see
    docs/wiki/concepts/android-streaming-deployment.md).

Usage (from repo root):

    python BAFNetPlus/scripts/make_calibration_npz.py \\
        --output_dir /tmp/bafnet_calib_taps_v3 \\
        --num_utts 100 --chunks_per_utt 30 --num_silence_chunks 100

Outputs ``num_utts × chunks_per_utt`` voiced files named
``calib_voiced_<utt>_<chunk>.npz`` plus ``num_silence_chunks`` silence
files named ``calib_silence_<idx>.npz`` under output_dir. The voiced
chunks per utt are stratified — sampled at evenly-spaced positions
across the full utt — so phoneme-region diversity (vowels,
consonants, breaths, pauses) is captured rather than just the first
``chunks_per_utt × 50 ms`` of each utt.
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
                   help="Directory to write calib_*.npz files into.")
    p.add_argument("--num_utts", type=int, default=100,
                   help="Number of TAPS test utterances to draw from (default 100). "
                        "TAPS test split has 1000 utts so this is a 10% sample.")
    p.add_argument("--chunks_per_utt", type=int, default=30,
                   help="Number of streaming chunks to extract per utterance (default 30). "
                        "Stratified across the utt so phoneme-region diversity is captured.")
    p.add_argument("--start_utt", type=int, default=0,
                   help="First TAPS test utt_idx (default 0).")
    p.add_argument("--num_silence_chunks", type=int, default=100,
                   help="Number of pure-silence chunks to emit (default 100). "
                        "Required to keep INT8 zero-points anchored at audio=0; "
                        "see Defect D in docs/wiki/concepts/android-streaming-deployment.md.")
    p.add_argument("--no_stratified", action="store_true",
                   help="Disable stratified chunk sampling and revert to taking "
                        "the first --chunks_per_utt chunks of each utt (legacy behaviour).")
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


def stream_chunks(audio: np.ndarray, max_chunks: int, stratified: bool = True):
    """Replicate StreamingEnhancer.processChunk's STFT inputs and yield (push_idx, mag, pha).

    When ``stratified`` is True, the function first runs the full streaming
    sequence to collect every (mag, pha) emission the utt produces, then
    yields ``max_chunks`` of those at evenly-spaced offsets. This captures
    phoneme-region diversity (vowels at one offset, consonants/silences at
    another) rather than only the first ``max_chunks × 50 ms`` of the utt.
    The streaming context (`stft_context`, `input_buffer`) is preserved
    sequentially so each emission's mag/pha is byte-identical to what the
    device would feed at that audio offset.
    """
    audio_t = torch.from_numpy(audio.astype(np.float32))
    n_pushes = len(audio_t) // OUTPUT_SAMPLES
    if n_pushes < 2:
        raise RuntimeError(
            f"audio too short: {len(audio_t)} samples → {n_pushes} pushes (need ≥ 2)."
        )
    stft_context = torch.zeros(WIN // 2)
    input_buffer = torch.tensor([], dtype=torch.float32)
    emissions: list[tuple[int, np.ndarray, np.ndarray]] = []
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
        emissions.append((
            push_idx,
            mag.numpy().astype(np.float32),
            pha.numpy().astype(np.float32),
        ))

    n_avail = len(emissions)
    if n_avail == 0:
        return
    n_take = min(max_chunks, n_avail)
    if not stratified or n_take >= n_avail:
        for e in emissions[:n_take]:
            yield e
        return
    # Even-spaced indices spanning [0, n_avail-1]; dedupe in case of rounding
    # collisions when n_take is close to n_avail.
    target = np.linspace(0, n_avail - 1, n_take).round().astype(int)
    seen: set[int] = set()
    for i in target:
        ii = int(i)
        if ii in seen:
            continue
        seen.add(ii)
        yield emissions[ii]


def silence_mag_pha() -> tuple[np.ndarray, np.ndarray]:
    """Return (mag, pha) for one strictly-zero audio chunk.

    Both context and chunk samples are zero. ``mag_pha_stft`` adds 1e-9
    inside ``sqrt`` and 1e-8 inside ``atan2`` for numerical stability, so
    the resulting tensors are not strictly zero — they sit at the
    deterministic floor ``mag ≈ (1e-9)^(compress/2) ≈ 0.04467`` and
    ``pha = atan2(1e-8, 1e-8) = π/4`` across every frame/bin. This is the
    exact same floor a device-side silent input would produce (raw int16
    BT noise floor ≈ 3 → float² ≈ 8e-9 ≲ 1e-9 epsilon), so feeding these
    values to the QDQ calibrator anchors activation zero-points at the
    real silence regime — that is what Defect D requires.
    """
    chunk_with_context = torch.zeros(WIN // 2 + SAMPLES_PER_CHUNK, dtype=torch.float32)
    mag, pha = stft_chunk(chunk_with_context)
    return mag.numpy().astype(np.float32), pha.numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stratified = not args.no_stratified
    print(f"make_calibration_npz.py")
    print(f"  output_dir          : {out_dir}")
    print(f"  num_utts            : {args.num_utts}")
    print(f"  chunks_per_utt      : {args.chunks_per_utt}")
    print(f"  start_utt           : {args.start_utt}")
    print(f"  num_silence_chunks  : {args.num_silence_chunks}")
    print(f"  stratified          : {stratified}")

    voiced_written = 0
    for k in range(args.num_utts):
        utt_idx = args.start_utt + k
        bcs = load_bcs(utt_idx)
        if (k % 10) == 0 or k == args.num_utts - 1:
            print(f"  voiced utt_idx={utt_idx} ({k+1}/{args.num_utts}) "
                  f"bcs={len(bcs)} samples ({len(bcs) / SR:.2f} s)")
        for chunk_idx, mag, pha in stream_chunks(
            bcs, max_chunks=args.chunks_per_utt, stratified=stratified
        ):
            assert mag.shape == (1, N_FFT // 2 + 1, EXPORT_FRAMES), \
                f"unexpected mag shape: {mag.shape}"
            out_path = out_dir / f"calib_voiced_{utt_idx:03d}_{chunk_idx:03d}.npz"
            np.savez(out_path, mag=mag, pha=pha)
            voiced_written += 1

    silence_written = 0
    if args.num_silence_chunks > 0:
        sil_mag, sil_pha = silence_mag_pha()
        sil_mag_2, sil_pha_2 = silence_mag_pha()
        assert np.array_equal(sil_mag, sil_mag_2) and np.array_equal(sil_pha, sil_pha_2), \
            "silence mag/pha is not deterministic across two calls"
        sil_mag_floor = float(sil_mag.max())
        sil_pha_floor = float(sil_pha.max())
        assert np.allclose(sil_mag, sil_mag_floor), \
            f"silence mag has non-uniform values: min={sil_mag.min()} max={sil_mag.max()}"
        assert np.allclose(sil_pha, sil_pha_floor), \
            f"silence pha has non-uniform values: min={sil_pha.min()} max={sil_pha.max()}"
        for idx in range(args.num_silence_chunks):
            out_path = out_dir / f"calib_silence_{idx:03d}.npz"
            np.savez(out_path, mag=sil_mag, pha=sil_pha)
            silence_written += 1
        print(f"  silence chunks written: {silence_written} "
              f"(mag floor={sil_mag_floor:.6f}, pha floor={sil_pha_floor:.6f}; "
              f"deterministic — duplicates anchor activation zero-points "
              f"to the device-side silence regime)")

    print(f"\nWrote {voiced_written + silence_written} calibration npz files to {out_dir}")
    print(f"  voiced  : {voiced_written}")
    print(f"  silence : {silence_written}")


if __name__ == "__main__":
    main()
