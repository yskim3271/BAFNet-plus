"""Generate a TAPS-aware BAFNet+ INT8 calibration fixture (4-key chunk format).

The :func:`quantize_bafnetplus_qdq_for_htp` reader at
``src/models/streaming/onnx/export_bafnetplus_onnx.py:1037-1148`` expects a
fixture directory with the following layout:

    <fixture_dir>/
        manifest.json
        chunk_000/{bcs_mag,bcs_pha,acs_mag,acs_pha}.bin
        chunk_001/...
        ...

Each ``.bin`` is raw float32 with shape ``(1, 201, 11)`` matching the streaming
ONNX model inputs (1 batch × ``n_fft//2+1`` freq bins × ``export_time_frames``).

This script is the BAFNet+ counterpart of ``scripts/make_calibration_npz.py``
(which only emits a single-channel ``mag/pha`` ``.npz`` and is for a different
ONNX export). It produces a **silence-enriched in-distribution TAPS BCS+ACS
fixture**, addressing M2 in the D5b plan review:

  - The existing fixture at
    ``Android_projects/benchmark-app/src/androidTest/assets/bafnetplus_fixtures/``
    was generated from random-seeded synthetic audio (``input_scale=0.1,
    seed=42``), which does NOT match the paper prose claim that calibration
    used "silence-enriched in-distribution TAPS subset"
    (``latex/main.tex:1670-1671``).

  - This script regenerates the fixture from real TAPS test BCS+ACS pairs +
    deterministic silence anchors, restoring consistency with the paper.

Usage (from BAFNetPlus repo root):
    python scripts/make_bafnetplus_calibration_fixture.py \\
        --output_dir results/onnx/calibration_fixture_taps \\
        --num_utts 20 --chunks_per_utt 20 --num_silence_chunks 50

Defaults reproduce the published ``num_calibration_samples = 40`` budget
(20 voiced + 20 silence chunks at minimum).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BAFNETPLUS_ROOT = REPO_ROOT / "BAFNetPlus"
if str(BAFNETPLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(BAFNETPLUS_ROOT))

from src.stft import mag_pha_stft  # noqa: E402

# Streaming export constants. Defaults match the published 50ms/B-tier export
# (chunk_size=8 + lookahead=3 → export_time_frames=11). For D5d's per-tier
# isolated exports (chunk_size_frames ∈ {2,4,8,12,16}, lookahead=0), pass
# ``--chunk_frames N --lookahead 0`` to produce [1, 201, N] chunks; the
# largest N=16 fixture also serves all smaller-N tiers via slicing in the
# QDQ calibration reader.
N_FFT = 400
HOP = 100
WIN = 400
COMPRESS = 0.3
SR = 16000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True, type=str,
                   help="Directory to write chunk_NNN/*.bin + manifest.json.")
    p.add_argument("--num_utts", type=int, default=20)
    p.add_argument("--chunks_per_utt", type=int, default=20)
    p.add_argument("--start_utt", type=int, default=0)
    p.add_argument("--num_silence_chunks", type=int, default=50)
    p.add_argument("--chunk_frames", type=int, default=8,
                   help="output chunk_size_frames (advance per push)")
    p.add_argument("--lookahead", type=int, default=3,
                   help="encoder_lookahead frames buffered with each chunk; "
                        "export_time_frames = chunk_frames + lookahead")
    return p.parse_args()


def _load_taps_pair(utt_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load (bcs, acs) pair at 16 kHz for one TAPS test utterance."""
    from datasets import load_dataset
    ds = load_dataset(
        "yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test",
    )
    item = ds[utt_idx]
    bcs = np.asarray(item["audio.throat_microphone"]["array"], dtype=np.float32)
    acs = np.asarray(item["audio.acoustic_microphone"]["array"], dtype=np.float32)
    bcs_sr = item["audio.throat_microphone"]["sampling_rate"]
    acs_sr = item["audio.acoustic_microphone"]["sampling_rate"]
    if bcs_sr != SR or acs_sr != SR:
        raise RuntimeError(
            f"TAPS utt_idx={utt_idx} sr mismatch (bcs={bcs_sr}, acs={acs_sr}, want={SR})",
        )
    return bcs, acs


def _stft_chunk(chunk_with_context: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    mag, pha, _ = mag_pha_stft(
        chunk_with_context.unsqueeze(0),
        n_fft=N_FFT, hop_size=HOP, win_size=WIN,
        compress_factor=COMPRESS, center=False,
    )
    return mag.numpy().astype(np.float32), pha.numpy().astype(np.float32)


def _stream_voiced(
    bcs: np.ndarray, acs: np.ndarray, max_chunks: int,
    samples_per_chunk: int, output_samples: int,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Replicate StreamingEnhancer.processChunk's chunk geometry on (bcs, acs).

    Yields ``(chunk_idx, bcs_mag, bcs_pha, acs_mag, acs_pha)`` per emitted chunk.
    """
    SAMPLES_PER_CHUNK = samples_per_chunk
    OUTPUT_SAMPLES = output_samples
    n_pushes = min(len(bcs), len(acs)) // OUTPUT_SAMPLES
    if n_pushes < 2:
        return

    bcs_t = torch.from_numpy(bcs.astype(np.float32))
    acs_t = torch.from_numpy(acs.astype(np.float32))
    bcs_ctx = torch.zeros(WIN // 2)
    acs_ctx = torch.zeros(WIN // 2)
    bcs_buf = torch.tensor([], dtype=torch.float32)
    acs_buf = torch.tensor([], dtype=torch.float32)

    emitted = 0
    for push_idx in range(n_pushes):
        s = push_idx * OUTPUT_SAMPLES
        e = s + OUTPUT_SAMPLES
        bcs_buf = torch.cat([bcs_buf, bcs_t[s:e]])
        acs_buf = torch.cat([acs_buf, acs_t[s:e]])
        if len(bcs_buf) < SAMPLES_PER_CHUNK:
            continue

        bcs_chunk = bcs_buf[:SAMPLES_PER_CHUNK]
        acs_chunk = acs_buf[:SAMPLES_PER_CHUNK]
        bcs_mag, bcs_pha = _stft_chunk(torch.cat([bcs_ctx, bcs_chunk]))
        acs_mag, acs_pha = _stft_chunk(torch.cat([acs_ctx, acs_chunk]))

        adv = OUTPUT_SAMPLES
        ctx_size = WIN // 2
        if adv >= ctx_size:
            bcs_ctx = bcs_buf[adv - ctx_size:adv].clone()
            acs_ctx = acs_buf[adv - ctx_size:adv].clone()
        else:
            need = ctx_size - adv
            bcs_ctx = torch.cat([bcs_ctx[-need:], bcs_buf[:adv]]).clone()
            acs_ctx = torch.cat([acs_ctx[-need:], acs_buf[:adv]]).clone()
        bcs_buf = bcs_buf[OUTPUT_SAMPLES:]
        acs_buf = acs_buf[OUTPUT_SAMPLES:]

        yield emitted, bcs_mag, bcs_pha, acs_mag, acs_pha
        emitted += 1
        if emitted >= max_chunks:
            return


def _silence_chunk(samples_per_chunk: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bcs_mag, bcs_pha, acs_mag, acs_pha) for a strictly-zero chunk.

    Both BCS and ACS spectrograms are computed from the same zero waveform, so
    they sit at the deterministic ``mag_pha_stft`` epsilon floor (mag ≈ 0.045,
    pha = π/4) — the exact regime a device-side silent input produces (per
    ``make_calibration_npz.py:silence_mag_pha`` notes).
    """
    chunk = torch.zeros(WIN // 2 + samples_per_chunk, dtype=torch.float32)
    bcs_mag, bcs_pha = _stft_chunk(chunk)
    acs_mag, acs_pha = _stft_chunk(chunk)
    return bcs_mag, bcs_pha, acs_mag, acs_pha


def _write_chunk_dir(out_dir: Path, idx: int,
                     bcs_mag: np.ndarray, bcs_pha: np.ndarray,
                     acs_mag: np.ndarray, acs_pha: np.ndarray) -> Path:
    chunk_dir = out_dir / f"chunk_{idx:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    bcs_mag.astype(np.float32).tofile(chunk_dir / "bcs_mag.bin")
    bcs_pha.astype(np.float32).tofile(chunk_dir / "bcs_pha.bin")
    acs_mag.astype(np.float32).tofile(chunk_dir / "acs_mag.bin")
    acs_pha.astype(np.float32).tofile(chunk_dir / "acs_pha.bin")
    return chunk_dir


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_frames = int(args.chunk_frames)
    lookahead = int(args.lookahead)
    export_frames = chunk_frames + lookahead
    samples_per_chunk = (export_frames - 1) * HOP + WIN // 2
    output_samples = chunk_frames * HOP

    print(f"make_bafnetplus_calibration_fixture.py")
    print(f"  output_dir          : {out_dir}")
    print(f"  num_utts            : {args.num_utts}")
    print(f"  chunks_per_utt      : {args.chunks_per_utt}")
    print(f"  start_utt           : {args.start_utt}")
    print(f"  num_silence_chunks  : {args.num_silence_chunks}")
    print(f"  chunk_frames        : {chunk_frames}")
    print(f"  lookahead           : {lookahead}")
    print(f"  export_frames       : {export_frames}  (each .bin shape [1, 201, {export_frames}])")
    print(f"  samples_per_chunk   : {samples_per_chunk}")
    print(f"  output_samples      : {output_samples}")

    chunk_idx = 0
    chunks_meta = []

    for k in range(args.num_utts):
        utt_idx = args.start_utt + k
        bcs, acs = _load_taps_pair(utt_idx)
        n = min(len(bcs), len(acs))
        bcs = bcs[:n]
        acs = acs[:n]
        emitted = 0
        for _local_idx, bm, bp, am, ap in _stream_voiced(
            bcs, acs, args.chunks_per_utt,
            samples_per_chunk=samples_per_chunk,
            output_samples=output_samples,
        ):
            _write_chunk_dir(out_dir, chunk_idx, bm, bp, am, ap)
            chunks_meta.append({"chunk": chunk_idx, "kind": "voiced", "utt_idx": utt_idx})
            chunk_idx += 1
            emitted += 1
        print(f"  voiced utt_idx={utt_idx}: {emitted} chunks (total now {chunk_idx})")

    if args.num_silence_chunks > 0:
        sil_bm, sil_bp, sil_am, sil_ap = _silence_chunk(samples_per_chunk)
        for _ in range(args.num_silence_chunks):
            _write_chunk_dir(out_dir, chunk_idx, sil_bm, sil_bp, sil_am, sil_ap)
            chunks_meta.append({"chunk": chunk_idx, "kind": "silence"})
            chunk_idx += 1
        print(f"  silence chunks      : {args.num_silence_chunks} "
              f"(deterministic mag/pha epsilon floor)")

    manifest = {
        "version": 1,
        "generator": "BAFNetPlus/scripts/make_bafnetplus_calibration_fixture.py",
        "generator_config": {
            "num_utts": args.num_utts,
            "chunks_per_utt": args.chunks_per_utt,
            "start_utt": args.start_utt,
            "num_silence_chunks": args.num_silence_chunks,
            "chunk_frames": chunk_frames,
            "lookahead": lookahead,
        },
        "model": "BAFNetPlus",
        "ablation_mode": "full",
        "calibration_distribution": "silence-enriched in-distribution TAPS BCS+ACS",
        "stft_config": {
            "n_fft": N_FFT, "hop_size": HOP, "win_length": WIN,
            "sample_rate": SR, "center": False, "compress_factor": COMPRESS,
        },
        "chunk_geometry": {
            "samples_per_chunk": samples_per_chunk,
            "output_samples_per_chunk": output_samples,
            "input_lookahead_frames": lookahead,
            "total_frames_needed": export_frames,
            "freq_bins": N_FFT // 2 + 1,
        },
        "chunks": chunks_meta,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {chunk_idx} chunk dirs + manifest.json to {out_dir}")
    print(f"  voiced  : {chunk_idx - args.num_silence_chunks}")
    print(f"  silence : {args.num_silence_chunks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
