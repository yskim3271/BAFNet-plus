#!/usr/bin/env python3
"""Generate golden fixtures for BAFNetPlus streaming Kotlin parity tests.

Dual-input variant of ``scripts/make_streaming_golden.py``. Runs
:class:`BAFNetPlusStreaming` on synthetic BCS + ACS streams and dumps per-chunk
intermediate tensors as raw little-endian float32 binaries plus a JSON manifest.

Key differences vs LaCoSENet fixture generator:
- Two synchronized inputs (bcs_samples, acs_samples) with correlated base
  noise and BCS lowpass + time-varying envelope (so the calibration path
  actually produces non-trivial common_log_gain; see plan §8 S2 R2-2).
- Uses :class:`BAFNetPlusStreaming` (PyTorch reference) directly — no ONNX model
  yet, Stage 3 builds it.
- Captures ~30 intermediate tensors per chunk, including fusion internals
  (calibration_feat/hidden, common/relative log gains, calibrated complex,
  alpha softmax, fused complex, final est_mag/est_pha).

Output layout under <output_dir>/:
    manifest.json
    input_audio_bcs.bin                 [duration_samples]
    input_audio_acs.bin                 [duration_samples]
    chunk_<i>/
        input_samples_bcs.bin           [samples_per_chunk=1200]
        input_samples_acs.bin
        stft_context_bcs_in.bin         [win_size//2=200]
        stft_context_acs_in.bin
        stft_input_bcs.bin              [context+samples=1400]
        stft_input_acs.bin
        bcs_mag.bin                     [1, 201, 11]
        bcs_pha.bin
        acs_mag.bin
        acs_pha.bin
        bcs_est_mag.bin                 [1, 201, 8] (post chunk_size crop)
        bcs_est_pha.bin
        bcs_com_out.bin                 [1, 201, 8, 2]
        acs_est_mag.bin
        acs_est_pha.bin
        acs_com_out.bin
        acs_mask.bin                    [1, 201, 8] (masking-branch raw mask)
        calibration_feat.bin            [1, 5, 8]
        calibration_hidden.bin          [1, 16, 8]
        common_log_gain.bin             [1, 1, 8]
        relative_log_gain.bin           [1, 1, 8]
        bcs_com_cal.bin                 [1, 201, 8, 2]
        acs_com_cal.bin
        alpha_softmax.bin               [1, 2, 201, 8]
        est_mag.bin                     [1, 201, 8]
        est_pha.bin
        ola_buffer_in.bin               [win_size-hop_size=300]
        ola_norm_in.bin
        ola_buffer_out.bin
        ola_norm_out.bin
        istft_output.bin                [output_samples_per_chunk=800]

Usage:
    python scripts/make_bafnetplus_streaming_golden.py \\
        --output_dir android/benchmark-app/src/androidTest/assets/bafnetplus_fixtures \\
        --duration_sec 2.0 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.signal import butter, filtfilt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.streaming.bafnetplus_streaming import BAFNetPlusStreaming  # noqa: E402
from src.models.streaming.utils import StateFramesContext  # noqa: E402
from src.stft import complex_to_mag_pha, mag_pha_stft  # noqa: E402


# ---------------------------------------------------------------------------
# Audio synthesis (R2-2: make calibration path non-trivially active)
# ---------------------------------------------------------------------------
def _bandpass(x: np.ndarray, lo_hz: float, hi_hz: float, fs: int) -> np.ndarray:
    b, a = butter(4, [lo_hz / (fs / 2), hi_hz / (fs / 2)], btype="band")
    return filtfilt(b, a, x).astype(np.float32)


def generate_dual_audio(
    seed: int,
    duration_samples: int,
    sample_rate: int,
    input_scale: float,
    correlation: float = 0.7,
    bcs_band_hz: Tuple[float, float] = (200.0, 2500.0),
    env_hz: float = 1.5,
    env_depth: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthesize BCS + ACS with correlated base, BCS bandpass + strong envelope.

    BCS simulates body-conduction: bandlimited (~200 Hz–2.5 kHz, keeping the
    voicing band) with a strong low-rate amplitude envelope so BCS/ACS log
    energy ratio sweeps through a wide range each chunk. Envelope depth 0.95
    takes BCS energy from ~5 % to ~100 % of its bandpass baseline, pulling the
    tanh-bounded calibration common_log_gain out of saturation (plan R2-2).

    ACS simulates air-conduction: wideband Gaussian. Both share ``correlation``
    of their total variance via a common base signal (correlation drops post
    envelope but the structure remains).
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(duration_samples, dtype=np.float64).astype(np.float32)

    acs_indep = rng.standard_normal(duration_samples, dtype=np.float64).astype(np.float32)
    bcs_indep = rng.standard_normal(duration_samples, dtype=np.float64).astype(np.float32)

    w_shared = math.sqrt(correlation)
    w_indep = math.sqrt(1.0 - correlation)
    acs = w_shared * base + w_indep * acs_indep
    bcs_raw = w_shared * base + w_indep * bcs_indep

    bcs_bp = _bandpass(bcs_raw, bcs_band_hz[0], bcs_band_hz[1], sample_rate)
    # Renormalize post-bandpass so BCS baseline RMS ≈ ACS RMS (avoids
    # systematic log_E_diff sign that would saturate tanh).
    bcs_bp = bcs_bp * (float(np.std(acs)) / (float(np.std(bcs_bp)) + 1e-8))

    t = np.arange(duration_samples, dtype=np.float32) / sample_rate
    env = (1.0 - env_depth) + env_depth * (np.sin(2.0 * math.pi * env_hz * t) * 0.5 + 0.5)
    bcs = bcs_bp * env

    acs = acs * input_scale
    bcs = bcs * input_scale

    return torch.from_numpy(bcs.astype(np.float32)), torch.from_numpy(acs.astype(np.float32))


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def save_bin(arr: np.ndarray, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(arr, dtype=np.float32).astype("<f4", copy=False)
    data.tofile(path)
    return int(data.nbytes)


def as_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().numpy().astype(np.float32)


def bin_descriptor(arr: np.ndarray, rel_path: str, sha256_enabled: bool) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "file": rel_path,
        "shape": list(arr.shape),
        "dtype": "float32",
        "bytes": int(np.prod(arr.shape) * 4),
    }
    if sha256_enabled:
        entry["sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
    return entry


# ---------------------------------------------------------------------------
# Per-chunk capture (dual-branch encoder → decoder → calibration → fusion → iSTFT)
# ---------------------------------------------------------------------------
def capture_chunk(
    streaming: BAFNetPlusStreaming,
    bcs_spec: torch.Tensor,
    acs_spec: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run one streaming step on pre-computed spectrograms and collect every
    intermediate tensor the fixture exports.

    Returns dict with keys enumerated in the file-header docstring. Raises
    ValueError if the decoder cannot run yet (feature buffer under-filled).
    """
    sm = streaming
    model = sm.model
    out: Dict[str, torch.Tensor] = {}

    _, _, T_spec, _ = bcs_spec.shape
    valid_frames = min(T_spec, sm.chunk_size)

    with StateFramesContext(None if sm.disable_state_guard else valid_frames):
        bcs_mag, bcs_ts_out, _ = sm._process_encoder(
            bcs_spec,
            model.mapping.dense_encoder,
            sm.streaming_tsblocks_mapping,
            sm._tsblock_states_mapping,
        )
        acs_mag, acs_ts_out, _ = sm._process_encoder(
            acs_spec,
            model.masking.dense_encoder,
            sm.streaming_tsblocks_masking,
            sm._tsblock_states_masking,
        )
    # Re-derive phase for manifest convenience (both branches).
    bcs_pha = torch.atan2(bcs_spec[..., 1] + 1e-8, bcs_spec[..., 0] + 1e-8)
    acs_pha = torch.atan2(acs_spec[..., 1] + 1e-8, acs_spec[..., 0] + 1e-8)
    out["bcs_mag"] = bcs_mag
    out["bcs_pha"] = bcs_pha
    out["acs_mag"] = acs_mag
    out["acs_pha"] = acs_pha

    sm._buffer_features(bcs_ts_out, bcs_mag, valid_frames, sm.bcs_feature_buffer)
    sm._buffer_features(acs_ts_out, acs_mag, valid_frames, sm.acs_feature_buffer)
    sm._buffered_frames += valid_frames

    total_needed = sm.chunk_size + sm.decoder_lookahead
    if sm._buffered_frames < total_needed:
        raise ValueError("decoder not ready (pre-warm)")

    bcs_ext_feat, bcs_ext_mag = sm._extract_extended(sm.bcs_feature_buffer)
    acs_ext_feat, acs_ext_mag = sm._extract_extended(sm.acs_feature_buffer)

    with StateFramesContext(None if sm.disable_state_guard else sm.chunk_size):
        bcs_est_mag, bcs_est_pha, bcs_com_out = sm._decode_branch(
            bcs_ext_feat,
            bcs_ext_mag,
            model.mapping.mask_decoder,
            model.mapping.phase_decoder,
            infer_type="mapping",
            return_mask=False,
        )
        acs_est_mag, acs_est_pha, acs_com_out, acs_mask = sm._decode_branch(
            acs_ext_feat,
            acs_ext_mag,
            model.masking.mask_decoder,
            model.masking.phase_decoder,
            infer_type="masking",
            return_mask=True,
        )
        out["bcs_est_mag"] = bcs_est_mag
        out["bcs_est_pha"] = bcs_est_pha
        out["bcs_com_out"] = bcs_com_out
        out["acs_est_mag"] = acs_est_mag
        out["acs_est_pha"] = acs_est_pha
        out["acs_com_out"] = acs_com_out
        out["acs_mask"] = acs_mask

        # Calibration (inline — mirrors _apply_calibration_streaming so we can
        # capture each sub-tensor). calibration_encoder is already stateful.
        eps = 1e-8
        bcs_log_E = torch.log(bcs_est_mag.pow(2).mean(dim=1, keepdim=True) + eps)
        acs_log_E = torch.log(acs_est_mag.pow(2).mean(dim=1, keepdim=True) + eps)
        log_E_diff = bcs_log_E - acs_log_E
        mask_mean = acs_mask.mean(dim=1, keepdim=True)
        mask_var = acs_mask.var(dim=1, keepdim=True, unbiased=False)
        calibration_feat = torch.cat(
            [bcs_log_E, acs_log_E, log_E_diff, mask_mean, mask_var], dim=1
        )  # [B, 5, T]
        calibration_hidden = model.calibration_encoder(calibration_feat)  # [B, 16, T]
        common_log_gain = torch.tanh(model.common_gain_head(calibration_hidden))
        common_log_gain = common_log_gain * model.calibration_max_common_log_gain
        if model.use_relative_gain:
            relative_log_gain = torch.tanh(model.relative_gain_head(calibration_hidden))
            relative_log_gain = relative_log_gain * model.calibration_max_relative_log_gain
            bcs_gain = torch.exp(common_log_gain - 0.5 * relative_log_gain)
            acs_gain = torch.exp(common_log_gain + 0.5 * relative_log_gain)
        else:
            relative_log_gain = torch.zeros_like(common_log_gain)
            bcs_gain = acs_gain = torch.exp(common_log_gain)

        bcs_gain_b = bcs_gain.transpose(1, 2).unsqueeze(1)
        acs_gain_b = acs_gain.transpose(1, 2).unsqueeze(1)
        bcs_com_cal = bcs_com_out * bcs_gain_b
        acs_com_cal = acs_com_out * acs_gain_b

        out["calibration_feat"] = calibration_feat
        out["calibration_hidden"] = calibration_hidden
        out["common_log_gain"] = common_log_gain
        out["relative_log_gain"] = relative_log_gain
        out["bcs_com_cal"] = bcs_com_cal
        out["acs_com_cal"] = acs_com_cal

        # Fusion (inline — mirrors _fuse so we capture alpha_softmax).
        bcs_mag_cal = torch.sqrt(bcs_com_cal[..., 0] ** 2 + bcs_com_cal[..., 1] ** 2 + eps)
        acs_mag_cal = torch.sqrt(acs_com_cal[..., 0] ** 2 + acs_com_cal[..., 1] ** 2 + eps)
        alpha_feat = torch.stack([bcs_mag_cal, acs_mag_cal, acs_mask], dim=1).transpose(2, 3)
        for block in model.alpha_convblocks:
            alpha_feat = block(alpha_feat)
        alpha = model.alpha_out(alpha_feat).transpose(2, 3)  # [B, 2, F, T]
        alpha_softmax = torch.softmax(alpha, dim=1)
        alpha_bcs = alpha_softmax[:, 0].unsqueeze(-1)
        alpha_acs = alpha_softmax[:, 1].unsqueeze(-1)
        est_com = bcs_com_cal * alpha_bcs + acs_com_cal * alpha_acs
        est_mag, est_pha = complex_to_mag_pha(est_com, stack_dim=-1)

        out["alpha_softmax"] = alpha_softmax
        out["est_mag"] = est_mag
        out["est_pha"] = est_pha

    removed = sm._slide_feature_buffer(sm.bcs_feature_buffer, sm.chunk_size)
    sm._slide_feature_buffer(sm.acs_feature_buffer, sm.chunk_size)
    sm._buffered_frames -= removed

    # Snapshot OLA state BEFORE iSTFT (input) then advance (output).
    ola_buffer_in = sm._ola_buffer.clone()
    ola_norm_in = sm._ola_norm.clone()
    istft_output = sm._manual_istft_ola(est_mag, est_pha)
    out["ola_buffer_in"] = ola_buffer_in
    out["ola_norm_in"] = ola_norm_in
    out["ola_buffer_out"] = sm._ola_buffer.clone()
    out["ola_norm_out"] = sm._ola_norm.clone()
    out["istft_output"] = istft_output

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--chkpt_dir_mapping",
        default="results/experiments/bm_map_50ms",
        help="Mapping Backbone checkpoint dir (relative to repo root).",
    )
    parser.add_argument(
        "--chkpt_dir_masking",
        default="results/experiments/bm_mask_50ms",
    )
    parser.add_argument("--chkpt_file", default="best.th")
    parser.add_argument(
        "--output_dir",
        default="android/benchmark-app/src/androidTest/assets/bafnetplus_fixtures",
    )
    parser.add_argument("--duration_sec", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_scale", type=float, default=0.1)
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=-1,
        help="Dump at most this many model chunks (-1 = all).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no_sha256",
        action="store_true",
        help="Skip SHA256 per-tensor (speeds up generation, reduces manifest size).",
    )
    args = parser.parse_args()

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sha256_enabled = not args.no_sha256

    chkpt_map = (PROJECT_ROOT / args.chkpt_dir_mapping).resolve()
    chkpt_mask = (PROJECT_ROOT / args.chkpt_dir_masking).resolve()
    print(f"mapping ckpt: {chkpt_map}")
    print(f"masking ckpt: {chkpt_mask}")
    print(f"output_dir  : {output_dir}")

    torch.manual_seed(args.seed)

    streaming = BAFNetPlusStreaming.from_checkpoint(
        chkpt_dir_mapping=str(chkpt_map),
        chkpt_dir_masking=str(chkpt_mask),
        chkpt_file=args.chkpt_file,
        device=args.device,
        verbose=False,
    )
    streaming.eval()
    streaming.reset_state()

    sample_rate = streaming.sample_rate
    hop_size = streaming.hop_size
    win_size = streaming.win_size
    n_fft = streaming.n_fft
    chunk_size = streaming.chunk_size
    samples_per_chunk = streaming.samples_per_chunk
    output_samples_per_chunk = streaming.output_samples_per_chunk
    ola_tail_size = streaming.ola_tail_size
    stft_future_samples = streaming.stft_future_samples
    input_lookahead_frames = streaming.input_lookahead_frames
    total_frames_needed = streaming.total_frames_needed
    freq_bins = n_fft // 2 + 1

    duration_samples = int(args.duration_sec * sample_rate)
    bcs_audio, acs_audio = generate_dual_audio(
        seed=args.seed,
        duration_samples=duration_samples,
        sample_rate=sample_rate,
        input_scale=args.input_scale,
    )
    bcs_audio = bcs_audio.to(args.device)
    acs_audio = acs_audio.to(args.device)

    save_bin(as_np(bcs_audio), output_dir / "input_audio_bcs.bin")
    save_bin(as_np(acs_audio), output_dir / "input_audio_acs.bin")

    # Feed in output_samples_per_chunk=800 increments; pad with zeros at the
    # tail so every residual chunk inside the stream is flushed.
    flush_calls = (total_frames_needed + chunk_size - 1) // chunk_size + 1
    pad_len = flush_calls * output_samples_per_chunk
    bcs_padded = torch.cat([bcs_audio, torch.zeros(pad_len, device=args.device)])
    acs_padded = torch.cat([acs_audio, torch.zeros(pad_len, device=args.device)])

    n_feed_calls = (duration_samples + output_samples_per_chunk - 1) // output_samples_per_chunk
    total_calls = n_feed_calls + flush_calls

    # Streaming state — independent from streaming's input buffers so we can
    # snapshot stft_context_in BEFORE advancing. We only use streaming's
    # ``_ola_buffer``, ``_tsblock_states_*``, ``bcs/acs_feature_buffer``.
    bcs_buf = torch.zeros(0, device=args.device)
    acs_buf = torch.zeros(0, device=args.device)
    bcs_stft_ctx = torch.zeros(stft_future_samples, device=args.device)
    acs_stft_ctx = torch.zeros(stft_future_samples, device=args.device)

    chunks_meta: List[Dict[str, Any]] = []
    chunk_idx = 0

    for call in range(total_calls):
        start = call * output_samples_per_chunk
        bcs_buf = torch.cat([bcs_buf, bcs_padded[start : start + output_samples_per_chunk]])
        acs_buf = torch.cat([acs_buf, acs_padded[start : start + output_samples_per_chunk]])

        if len(bcs_buf) < samples_per_chunk:
            continue
        if args.max_chunks >= 0 and chunk_idx >= args.max_chunks:
            break

        bcs_chunk = bcs_buf[:samples_per_chunk].clone()
        acs_chunk = acs_buf[:samples_per_chunk].clone()

        # Snapshot STFT contexts (used as iSTFT-like ring buffer)
        stft_ctx_bcs_in = bcs_stft_ctx.clone()
        stft_ctx_acs_in = acs_stft_ctx.clone()

        if streaming.stft_center:
            bcs_full = torch.cat([stft_ctx_bcs_in, bcs_chunk])
            acs_full = torch.cat([stft_ctx_acs_in, acs_chunk])
        else:
            bcs_full = bcs_chunk
            acs_full = acs_chunk

        # STFT
        _, _, bcs_spec = mag_pha_stft(
            bcs_full.unsqueeze(0),
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=streaming.compress_factor,
            center=False,
        )
        _, _, acs_spec = mag_pha_stft(
            acs_full.unsqueeze(0),
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=streaming.compress_factor,
            center=False,
        )

        try:
            captured = capture_chunk(streaming, bcs_spec, acs_spec)
        except ValueError:
            # Pre-warm: advance buffers but emit nothing yet.
            advance = output_samples_per_chunk
            ctx_size = stft_future_samples
            if streaming.stft_center:
                if advance >= ctx_size:
                    bcs_stft_ctx = bcs_buf[advance - ctx_size : advance].clone()
                    acs_stft_ctx = acs_buf[advance - ctx_size : advance].clone()
                else:
                    need = ctx_size - advance
                    bcs_stft_ctx = torch.cat(
                        [bcs_stft_ctx[len(bcs_stft_ctx) - need :], bcs_buf[:advance]]
                    ).clone()
                    acs_stft_ctx = torch.cat(
                        [acs_stft_ctx[len(acs_stft_ctx) - need :], acs_buf[:advance]]
                    ).clone()
            bcs_buf = bcs_buf[advance:]
            acs_buf = acs_buf[advance:]
            continue

        # Persist binaries
        chunk_dir = output_dir / f"chunk_{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        rel_prefix = f"chunk_{chunk_idx:03d}"

        tensors_to_dump: List[Tuple[str, np.ndarray]] = [
            ("input_samples_bcs", as_np(bcs_chunk)),
            ("input_samples_acs", as_np(acs_chunk)),
            ("stft_context_bcs_in", as_np(stft_ctx_bcs_in)),
            ("stft_context_acs_in", as_np(stft_ctx_acs_in)),
            ("stft_input_bcs", as_np(bcs_full)),
            ("stft_input_acs", as_np(acs_full)),
            ("bcs_mag", as_np(captured["bcs_mag"])),
            ("bcs_pha", as_np(captured["bcs_pha"])),
            ("acs_mag", as_np(captured["acs_mag"])),
            ("acs_pha", as_np(captured["acs_pha"])),
            ("bcs_est_mag", as_np(captured["bcs_est_mag"])),
            ("bcs_est_pha", as_np(captured["bcs_est_pha"])),
            ("bcs_com_out", as_np(captured["bcs_com_out"])),
            ("acs_est_mag", as_np(captured["acs_est_mag"])),
            ("acs_est_pha", as_np(captured["acs_est_pha"])),
            ("acs_com_out", as_np(captured["acs_com_out"])),
            ("acs_mask", as_np(captured["acs_mask"])),
            ("calibration_feat", as_np(captured["calibration_feat"])),
            ("calibration_hidden", as_np(captured["calibration_hidden"])),
            ("common_log_gain", as_np(captured["common_log_gain"])),
            ("relative_log_gain", as_np(captured["relative_log_gain"])),
            ("bcs_com_cal", as_np(captured["bcs_com_cal"])),
            ("acs_com_cal", as_np(captured["acs_com_cal"])),
            ("alpha_softmax", as_np(captured["alpha_softmax"])),
            ("est_mag", as_np(captured["est_mag"])),
            ("est_pha", as_np(captured["est_pha"])),
            ("ola_buffer_in", as_np(captured["ola_buffer_in"])),
            ("ola_norm_in", as_np(captured["ola_norm_in"])),
            ("ola_buffer_out", as_np(captured["ola_buffer_out"])),
            ("ola_norm_out", as_np(captured["ola_norm_out"])),
            ("istft_output", as_np(captured["istft_output"])),
        ]

        files: Dict[str, Dict[str, Any]] = {}
        for key, arr in tensors_to_dump:
            rel = f"{rel_prefix}/{key}.bin"
            save_bin(arr, output_dir / rel)
            files[key] = bin_descriptor(arr, rel, sha256_enabled)
        chunks_meta.append({"idx": chunk_idx, "files": files})

        # Advance streaming state
        advance = output_samples_per_chunk
        ctx_size = stft_future_samples
        if streaming.stft_center:
            if advance >= ctx_size:
                bcs_stft_ctx = bcs_buf[advance - ctx_size : advance].clone()
                acs_stft_ctx = acs_buf[advance - ctx_size : advance].clone()
            else:
                need = ctx_size - advance
                bcs_stft_ctx = torch.cat(
                    [bcs_stft_ctx[len(bcs_stft_ctx) - need :], bcs_buf[:advance]]
                ).clone()
                acs_stft_ctx = torch.cat(
                    [acs_stft_ctx[len(acs_stft_ctx) - need :], acs_buf[:advance]]
                ).clone()
        bcs_buf = bcs_buf[advance:]
        acs_buf = acs_buf[advance:]
        chunk_idx += 1

    # Diagnostics: verify calibration path is exercised (plan R2-2).
    # NOTE: common_log_gain saturates at -0.5 for any synthetic input because
    # the trained model's tanh response is dominated by acs_mask_mean and the
    # Gaussian-based mask floor > 0.1 keeps tanh near -1. relative_log_gain
    # and alpha_softmax are the practical exercise signals for synthetic
    # fixtures (parity tests still cover common_log_gain byte-wise).
    def _concat_chunk_tensor(key: str, reshape: Tuple[int, ...]) -> np.ndarray:
        arrs = []
        for c in chunks_meta:
            arr = np.fromfile(output_dir / c["files"][key]["file"], dtype="<f4").reshape(reshape)
            arrs.append(arr)
        return np.stack(arrs) if arrs else np.zeros((0,))

    if chunks_meta:
        common_gain = _concat_chunk_tensor("common_log_gain", (1, 1, 8))
        relative_gain = _concat_chunk_tensor("relative_log_gain", (1, 1, 8))
        alpha = _concat_chunk_tensor("alpha_softmax", (1, 2, freq_bins, 8))
        common_overall_std = float(common_gain.std())
        relative_overall_std = float(relative_gain.std())
        alpha_overall_std = float(alpha.std())
    else:
        common_overall_std = 0.0
        relative_overall_std = 0.0
        alpha_overall_std = 0.0
    exercised = (relative_overall_std > 0.01) or (alpha_overall_std > 0.01)

    manifest = {
        "version": 1,
        "generator": "scripts/make_bafnetplus_streaming_golden.py",
        "generator_config": {
            "seed": args.seed,
            "duration_sec": args.duration_sec,
            "input_scale": args.input_scale,
            "max_chunks": args.max_chunks,
            "chkpt_dir_mapping": str(args.chkpt_dir_mapping),
            "chkpt_dir_masking": str(args.chkpt_dir_masking),
            "chkpt_file": args.chkpt_file,
            "device": args.device,
        },
        "model": "BAFNetPlus",
        "ablation_mode": streaming._streaming_config.get("ablation_mode", "full"),
        "derived": {
            "samples_per_chunk": samples_per_chunk,
            "output_samples_per_chunk": output_samples_per_chunk,
            "stft_future_samples": stft_future_samples,
            "input_lookahead_frames": input_lookahead_frames,
            "total_frames_needed": total_frames_needed,
            "ola_tail_size": ola_tail_size,
            "freq_bins": freq_bins,
        },
        "stft_config": {
            "n_fft": n_fft,
            "hop_size": hop_size,
            "win_length": win_size,
            "sample_rate": sample_rate,
            "center": True,
            "compress_factor": streaming.compress_factor,
        },
        "streaming_config": streaming.streaming_config,
        "calibration_diagnostics": {
            "common_log_gain_overall_std": common_overall_std,
            "relative_log_gain_overall_std": relative_overall_std,
            "alpha_softmax_overall_std": alpha_overall_std,
            "exercised": exercised,
            "note": (
                "common_log_gain saturates near -0.5 for synthetic Gaussian inputs "
                "(trained model's tanh head is driven by acs_mask_mean > 0.1). "
                "relative_log_gain and alpha_softmax are the practical exercise signals; "
                "Stage 4 parity still compares common_log_gain byte-wise."
            ),
        },
        "input_audio_bcs": bin_descriptor(
            as_np(bcs_audio), "input_audio_bcs.bin", sha256_enabled
        ),
        "input_audio_acs": bin_descriptor(
            as_np(acs_audio), "input_audio_acs.bin", sha256_enabled
        ),
        "num_chunks": len(chunks_meta),
        "chunks": chunks_meta,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_bytes = (
        sum(f.get("bytes", 0) for chunk in chunks_meta for f in chunk["files"].values())
        + manifest["input_audio_bcs"]["bytes"]
        + manifest["input_audio_acs"]["bytes"]
    )
    print(
        f"\nDumped {len(chunks_meta)} chunks, "
        f"{total_bytes / 1024 / 1024:.2f} MB total, "
        f"manifest at {output_dir / 'manifest.json'}"
    )
    print(
        f"Calibration diagnostics: common_log_gain std={common_overall_std:.4f}, "
        f"relative_log_gain std={relative_overall_std:.4f}, "
        f"alpha_softmax std={alpha_overall_std:.4f} (exercised={exercised})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
