#!/usr/bin/env python3
"""Generate golden fixtures for Python<->Kotlin streaming parity tests.

Runs the streaming inference pipeline against the bundled ONNX model using
Python-reference STFT/iSTFT (torch-equivalent numpy port), dumping per-chunk
intermediate tensors as raw little-endian float32 binaries plus a JSON manifest.

This is the REFERENCE pipeline that Kotlin should match once A1-A7 fixes land.
Current Kotlin has structural STFT bugs that will cause all parity tests to
fail — that is the expected behavior of Stage 2.

Output layout under <output_dir>/:
    manifest.json                   schema + per-chunk file listing
    input_audio.bin                 [duration_samples] raw pcm
    chunk_<i>/
        input_samples.bin           [samples_per_chunk=1200]
        stft_context_in.bin         [win_size//2=200]
        stft_input.bin              [samples_per_chunk + 200 = 1400]
        stft_mag.bin                [freq_bins=201, export_time_frames=11]
        stft_pha.bin                [201, 11]
        model_mag_in.bin            [1, 201, 11]
        model_pha_in.bin            [1, 201, 11]
        states_in.bin               concat of 80 state tensors in state_order
        est_mask.bin                [1, 201, 11]
        phase_real.bin              [1, 201, 11]
        phase_imag.bin              [1, 201, 11]
        next_states.bin             concat of 80 next_state tensors
        est_mag.bin                 [1, 201, 11] pre-crop
        est_pha.bin                 [1, 201, 11] = atan2(imag+eps, real+eps)
        est_mag_crop.bin            [1, 201, 8] after keep-first-chunk_size
        est_pha_crop.bin            [1, 201, 8]
        ola_buffer_in.bin           [win_size-hop_size=300]
        ola_norm_in.bin             [300]
        ola_buffer_out.bin          [300]
        ola_norm_out.bin            [300]
        istft_output.bin            [output_samples_per_chunk=800]

Usage:
    python scripts/make_streaming_golden.py \\
        --model android/benchmark-app/src/main/assets/model.onnx \\
        --config android/benchmark-app/src/main/assets/streaming_config.json \\
        --output_dir android/benchmark-app/src/androidTest/assets/fixtures \\
        --duration_sec 2.0 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def periodic_hann(win_size: int) -> np.ndarray:
    """torch.hann_window default (periodic): 0.5*(1 - cos(2*pi*n / N))."""
    n = np.arange(win_size, dtype=np.float64)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / win_size))
    return w.astype(np.float32)


def stft_frames(
    audio: np.ndarray,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """center=False STFT matching src/stft.py::mag_pha_stft with center=False.

    Returns (mag, pha) of shape [freq_bins, num_frames] with compressed magnitude.
    """
    window = periodic_hann(win_size)
    num_frames = (len(audio) - win_size) // hop_size + 1
    if num_frames <= 0:
        raise ValueError(f"audio too short: {len(audio)} < {win_size}")
    freq_bins = n_fft // 2 + 1

    mag = np.empty((freq_bins, num_frames), dtype=np.float32)
    pha = np.empty((freq_bins, num_frames), dtype=np.float32)

    for t in range(num_frames):
        frame = audio[t * hop_size : t * hop_size + win_size].astype(np.float64) * window
        spec = np.fft.rfft(frame, n=n_fft)
        real = spec.real.astype(np.float32)
        imag = spec.imag.astype(np.float32)
        m_raw = np.sqrt(real * real + imag * imag + 1e-9).astype(np.float32)
        p = np.arctan2(imag + 1e-8, real + 1e-8).astype(np.float32)
        mag[:, t] = np.power(m_raw, compress_factor, dtype=np.float32)
        pha[:, t] = p

    return mag, pha


def manual_istft_ola(
    mag: np.ndarray,
    pha: np.ndarray,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float,
    ola_buffer: np.ndarray,
    ola_norm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy port of src/stft.py::manual_istft_ola (batch=1 squeezed).

    Args:
        mag: [freq_bins, T] compressed
        pha: [freq_bins, T]
        ola_buffer, ola_norm: carry-over [win_size-hop_size]

    Returns:
        output: [T * hop_size]
        new_ola_buffer, new_ola_norm: carry-over [win_size-hop_size]
    """
    T = mag.shape[1]
    mag_dec = np.power(mag, 1.0 / compress_factor, dtype=np.float32)
    real = mag_dec * np.cos(pha)
    imag = mag_dec * np.sin(pha)
    spec = real.astype(np.complex64) + 1j * imag.astype(np.complex64)

    # irfft over freq axis — spec is [F, T], irfft wants [..., F] so transpose
    frames = np.fft.irfft(spec.T, n=n_fft).astype(np.float32)  # [T, n_fft]
    frames = frames[:, :win_size]

    window = periodic_hann(win_size)
    frames = frames * window
    window_sq = window * window

    output_samples = T * hop_size
    tail_size = win_size - hop_size
    total_size = output_samples + tail_size

    buf = np.zeros(total_size, dtype=np.float32)
    norm = np.zeros(total_size, dtype=np.float32)

    carry = min(len(ola_buffer), tail_size)
    buf[:carry] = ola_buffer[:carry]
    norm[:carry] = ola_norm[:carry]

    for t in range(T):
        start = t * hop_size
        buf[start : start + win_size] += frames[t]
        norm[start : start + win_size] += window_sq

    output = buf[:output_samples].copy()
    output_norm = norm[:output_samples]
    safe = output_norm > 1e-8
    output[safe] = output[safe] / output_norm[safe]

    new_ola_buffer = buf[output_samples:].copy()
    new_ola_norm = norm[output_samples:].copy()

    return output, new_ola_buffer, new_ola_norm


def save_bin(arr: np.ndarray, path: Path) -> int:
    """Write float32 array as raw little-endian bytes. Returns byte count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(arr, dtype=np.float32)
    # numpy.tofile uses host byte order. We enforce little-endian explicitly.
    data = data.astype("<f4", copy=False)
    data.tofile(path)
    return data.nbytes


def bin_descriptor(arr: np.ndarray, rel_path: str) -> Dict:
    return {
        "file": rel_path,
        "shape": list(arr.shape),
        "dtype": "float32",
        "bytes": int(np.prod(arr.shape) * 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model",
        default="android/benchmark-app/src/main/assets/model.onnx",
        help="ONNX model path (relative to repo root)",
    )
    parser.add_argument(
        "--config",
        default="android/benchmark-app/src/main/assets/streaming_config.json",
    )
    parser.add_argument(
        "--output_dir",
        default="android/benchmark-app/src/androidTest/assets/fixtures",
    )
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input_scale",
        type=float,
        default=0.1,
        help="Scale for gaussian input (0.1 ~= typical speech RMS)",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=-1,
        help="Dump at most this many model chunks (-1 = all)",
    )
    parser.add_argument(
        "--dump_states",
        action="store_true",
        help=(
            "Include full states_in/next_states binaries per chunk (~24 MB/chunk). "
            "Off by default — parity tests replay sequentially from zero-state."
        ),
    )
    args = parser.parse_args()

    model_path = (PROJECT_ROOT / args.model).resolve()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"model       : {model_path}")
    print(f"config      : {config_path}")
    print(f"output_dir  : {output_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    stft_cfg = cfg["stft_config"]
    streaming_cfg = cfg["streaming_config"]
    state_info = cfg["state_info"]

    n_fft = int(stft_cfg["n_fft"])
    hop_size = int(stft_cfg["hop_size"])
    win_size = int(stft_cfg["win_length"])
    sample_rate = int(stft_cfg["sample_rate"])
    compress_factor = float(stft_cfg["compress_factor"])

    chunk_size = int(streaming_cfg["chunk_size_frames"])
    encoder_lookahead = int(streaming_cfg["encoder_lookahead"])
    decoder_lookahead = int(streaming_cfg["decoder_lookahead"])
    export_time_frames = int(streaming_cfg["export_time_frames"])
    freq_bins = int(streaming_cfg["freq_bins"])

    state_names: List[str] = list(state_info["state_names"])

    # Reference-pipeline derived constants (Python semantics; NOT current Kotlin)
    stft_future_samples = win_size // 2  # 200
    input_lookahead_frames = encoder_lookahead  # 3 (matches lacosenet.py:162)
    total_frames_needed = chunk_size + input_lookahead_frames  # 11
    assert total_frames_needed == export_time_frames, (
        f"Derived total_frames_needed={total_frames_needed} "
        f"!= export_time_frames={export_time_frames}"
    )
    samples_per_chunk = (total_frames_needed - 1) * hop_size + stft_future_samples  # 1200
    output_samples_per_chunk = chunk_size * hop_size  # 800
    ola_tail_size = win_size - hop_size  # 300

    # Load ORT session
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_input_names = [i.name for i in sess.get_inputs()]
    onnx_output_names = [o.name for o in sess.get_outputs()]

    # Cross-check state registry against ONNX
    ort_state_names = sorted(n for n in onnx_input_names if n.startswith("state_"))
    if ort_state_names != sorted(state_names):
        raise RuntimeError(
            "State names mismatch between streaming_config.json and ONNX inputs"
        )

    state_shapes: Dict[str, Tuple[int, ...]] = {}
    for inp in sess.get_inputs():
        if inp.name.startswith("state_"):
            state_shapes[inp.name] = tuple(int(d) for d in inp.shape)

    # Determine next_state mapping
    next_state_names = {n: f"next_{n}" for n in state_names}
    missing_next = [ns for ns in next_state_names.values() if ns not in onnx_output_names]
    if missing_next:
        raise RuntimeError(f"Missing next_state outputs in ONNX: {missing_next[:3]}...")

    # Fixed-seed input audio (unit-variance gaussian scaled to --input_scale)
    rng = np.random.default_rng(args.seed)
    duration_samples = int(args.duration_sec * sample_rate)
    audio = (
        rng.standard_normal(duration_samples, dtype=np.float64).astype(np.float32)
        * args.input_scale
    )

    # Pre-compute state layout (concatenation order = sorted state_names)
    sorted_state_names = sorted(state_names)
    state_layout = []
    offset = 0
    for name in sorted_state_names:
        shape = state_shapes[name]
        size = int(np.prod(shape))
        state_layout.append(
            {
                "name": name,
                "shape": list(shape),
                "offset_floats": offset,
                "size_floats": size,
            }
        )
        offset += size
    total_state_floats = offset

    # Streaming state
    states: Dict[str, np.ndarray] = {
        n: np.zeros(state_shapes[n], dtype=np.float32) for n in sorted_state_names
    }
    ola_buffer = np.zeros(ola_tail_size, dtype=np.float32)
    ola_norm = np.zeros(ola_tail_size, dtype=np.float32)
    stft_context = np.zeros(stft_future_samples, dtype=np.float32)

    # Cumulative input buffer — accepts slices of 800 samples at a time from
    # `audio`, flushes in samples_per_chunk=1200 blocks. This mirrors what
    # Kotlin's AudioBuffer/processChunk path should do.
    input_buffer = np.zeros(0, dtype=np.float32)

    # Feed audio in 800-sample increments; pad with zeros at the end so the
    # stream can flush every chunk the real audio contributed to.
    n_feed_calls = (duration_samples + output_samples_per_chunk - 1) // output_samples_per_chunk
    flush_calls = (total_frames_needed + chunk_size - 1) // chunk_size + 1
    padded = np.concatenate(
        [audio, np.zeros(flush_calls * output_samples_per_chunk, dtype=np.float32)]
    )

    chunks_meta: List[Dict] = []
    chunk_idx = 0

    save_bin(audio, output_dir / "input_audio.bin")

    for call in range(n_feed_calls + flush_calls):
        start = call * output_samples_per_chunk
        incoming = padded[start : start + output_samples_per_chunk]
        input_buffer = np.concatenate([input_buffer, incoming])

        if len(input_buffer) < samples_per_chunk:
            continue

        if args.max_chunks >= 0 and chunk_idx >= args.max_chunks:
            break

        # Pull one processing window
        chunk_samples = input_buffer[:samples_per_chunk].copy()

        # Snapshot states + stft_context + ola state BEFORE advancing
        states_in_snapshot = {n: states[n].copy() for n in sorted_state_names}
        stft_context_snapshot = stft_context.copy()
        ola_buffer_in_snapshot = ola_buffer.copy()
        ola_norm_in_snapshot = ola_norm.copy()

        # STFT input = prev context (zeros on first chunk) + chunk samples
        stft_input = np.concatenate([stft_context, chunk_samples])
        assert len(stft_input) == samples_per_chunk + stft_future_samples  # 1400
        mag, pha = stft_frames(stft_input, n_fft, hop_size, win_size, compress_factor)
        # mag/pha shape: [freq_bins, export_time_frames] = [201, 11]
        assert mag.shape == (freq_bins, export_time_frames), f"mag shape {mag.shape}"

        # Update stft_context for next call (mirrors lacosenet.py:559-566)
        advance = output_samples_per_chunk  # 800
        context_size = stft_future_samples  # 200
        # advance=800 >= context_size=200 → take last 200 of the consumed span
        next_stft_context = input_buffer[advance - context_size : advance].copy()

        # Build ONNX feed
        model_mag_in = mag[np.newaxis, :, :].astype(np.float32)  # [1, 201, 11]
        model_pha_in = pha[np.newaxis, :, :].astype(np.float32)

        feed = {"mag": model_mag_in, "pha": model_pha_in}
        for name in sorted_state_names:
            feed[name] = states[name]

        out_arrays = sess.run(onnx_output_names, feed)
        out_map = dict(zip(onnx_output_names, out_arrays))

        est_mask = out_map["est_mask"].astype(np.float32)  # [1, 201, 11]
        phase_real = out_map["phase_real"].astype(np.float32)
        phase_imag = out_map["phase_imag"].astype(np.float32)

        # Compute est_mag (mag * mask, matches infer_type=masking path)
        est_mag = model_mag_in * est_mask  # [1, 201, 11]
        # Compute est_pha via atan2 with same epsilon as StatefulInference.kt:328
        est_pha = np.arctan2(phase_imag + 1e-8, phase_real + 1e-8).astype(np.float32)

        # Crop to chunk_size=8 (keep first 8 frames only)
        est_mag_crop = est_mag[:, :, :chunk_size].copy()  # [1, 201, 8]
        est_pha_crop = est_pha[:, :, :chunk_size].copy()

        # iSTFT with OLA carry-over
        istft_out, new_ola_buffer, new_ola_norm = manual_istft_ola(
            est_mag_crop[0],
            est_pha_crop[0],
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            ola_buffer=ola_buffer,
            ola_norm=ola_norm,
        )
        assert istft_out.shape == (output_samples_per_chunk,)

        # Persist chunk files
        chunk_dir = output_dir / f"chunk_{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        rel_prefix = f"chunk_{chunk_idx:03d}"

        # Concatenate states into a single buffer (in sorted_state_names order)
        states_in_concat = np.concatenate(
            [states_in_snapshot[n].reshape(-1) for n in sorted_state_names]
        ).astype(np.float32)
        next_states_concat = np.concatenate(
            [out_map[next_state_names[n]].reshape(-1) for n in sorted_state_names]
        ).astype(np.float32)
        assert states_in_concat.size == total_state_floats
        assert next_states_concat.size == total_state_floats

        # Dump individual binaries. State tensors (~24 MB/chunk) are opt-in via
        # --dump_states because parity tests replay sequentially from zero state.
        files: Dict[str, Dict] = {}
        tensors_to_dump = [
            ("input_samples", chunk_samples),
            ("stft_context_in", stft_context_snapshot),
            ("stft_input", stft_input.astype(np.float32)),
            ("stft_mag", mag),
            ("stft_pha", pha),
            ("model_mag_in", model_mag_in),
            ("model_pha_in", model_pha_in),
            ("est_mask", est_mask),
            ("phase_real", phase_real),
            ("phase_imag", phase_imag),
            ("est_mag", est_mag),
            ("est_pha", est_pha),
            ("est_mag_crop", est_mag_crop),
            ("est_pha_crop", est_pha_crop),
            ("ola_buffer_in", ola_buffer_in_snapshot),
            ("ola_norm_in", ola_norm_in_snapshot),
            ("ola_buffer_out", new_ola_buffer),
            ("ola_norm_out", new_ola_norm),
            ("istft_output", istft_out),
        ]
        if args.dump_states:
            tensors_to_dump.extend(
                [
                    ("states_in", states_in_concat),
                    ("next_states", next_states_concat),
                ]
            )
        for key, arr in tensors_to_dump:
            rel = f"{rel_prefix}/{key}.bin"
            save_bin(arr, output_dir / rel)
            files[key] = bin_descriptor(arr, rel)

        chunks_meta.append({"idx": chunk_idx, "files": files})

        # Advance streaming state
        states = {n: out_map[next_state_names[n]].copy() for n in sorted_state_names}
        ola_buffer = new_ola_buffer
        ola_norm = new_ola_norm
        stft_context = next_stft_context
        input_buffer = input_buffer[output_samples_per_chunk:]

        chunk_idx += 1

    manifest = {
        "version": 1,
        "generator": "scripts/make_streaming_golden.py",
        "generator_config": {
            "seed": args.seed,
            "duration_sec": args.duration_sec,
            "input_scale": args.input_scale,
            "max_chunks": args.max_chunks,
        },
        "streaming_config_path": str(args.config),
        "model_path": str(args.model),
        "derived": {
            "samples_per_chunk": samples_per_chunk,
            "output_samples_per_chunk": output_samples_per_chunk,
            "stft_future_samples": stft_future_samples,
            "input_lookahead_frames": input_lookahead_frames,
            "total_frames_needed": total_frames_needed,
            "ola_tail_size": ola_tail_size,
        },
        "stft_config": stft_cfg,
        "streaming_config": streaming_cfg,
        "state_layout": state_layout,
        "state_order": sorted_state_names,
        "input_audio": bin_descriptor(audio, "input_audio.bin"),
        "num_chunks": len(chunks_meta),
        "chunks": chunks_meta,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_bytes = sum(
        f.get("bytes", 0)
        for chunk in chunks_meta
        for f in chunk["files"].values()
    ) + manifest["input_audio"]["bytes"]

    print(
        f"\nDumped {len(chunks_meta)} chunks, "
        f"{total_bytes / 1024 / 1024:.2f} MB total, "
        f"manifest at {output_dir / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
