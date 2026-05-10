# Copyright (c) POSTECH, and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: yunsik kim
"""Streaming-ONNX BAFNetPlus evaluator (D5b — host-side INT8 vs FP32 quality delta).

Drives a stateful streaming BAFNet+ ONNX graph over the TAPS test loader produced
by :func:`src.runtime_common.prepare_evaluation_runtime`, mirroring the on-device
chunk sequence (200-sample STFT context + 1200-sample chunk → 11 frames →
``bcs_mag/pha + acs_mag/pha + 166 states`` → ``est_mag, est_com_real,
est_com_imag, 166 next_states`` → manual iSTFT OLA), then calls
:func:`src.compute_metrics.compute_metrics` on the reconstructed waveform versus
the clean ACS reference.

Used by D5b (Paper TASLP §VIII Tab VI INT8 columns). Reuses the chunk
arithmetic from ``BAFNetPlus/scripts/make_enhancer_golden.py`` but for the
**dual-input** BAFNet+ graph (bcs+acs) and the host-side ``atan2`` convention.

The session is constructed with the CPU EP — INT8 graphs run on the CPU EP via
the ORT QDQ kernels and produce numerics within HTP tolerance for accuracy
validation purposes. (HTP execution is exercised on-device by D5a/c, not here.)

**Important — fusion weight provenance**: The BAFNet+ ONNX export rebuilds the
model from two backbone checkpoints (``bm_map_<T>ms`` + ``bm_mask_<T>ms``) and
**re-initialises the fusion weights via Kaiming-random under a fixed seed**
(see :class:`src.models.streaming.bafnetplus_streaming.BAFNetPlusStreaming`
docstring §"Stage 1 scope"). The trained fusion weights stored in the unified
``bafnetplus_<T>ms/best.th`` checkpoint are *not* loaded into the export
pipeline. As a result, ONNX-side metrics (FP32 or INT8) are not directly
comparable to the FP32 PyTorch reference produced from ``best.th``: the gap
mixes (a) untrained-fusion regression with (b) INT8 quantization error. The
INT8-vs-FP32-ONNX delta isolates pure quantization error; the
INT8-vs-FP32-PyTorch delta measures the full deployment delta. Aggregator
:mod:`Paper_TASLP.tables.scripts.aggregate_d5b` reports both.

**Time alignment**: Streaming output has a small algorithmic-latency offset
(≤ ``encoder_lookahead × hop`` ≈ a few hundred samples / ~25 ms at 16 kHz)
relative to the non-streaming PyTorch baseline. PESQ has internal
cross-correlation alignment and STOI uses TF-magnitude; both are robust to
this offset. The min-length trim in :func:`evaluate_onnx` handles the trailing
edge mismatch but does not phase-align the leading edge — expected absolute
ONNX-vs-PyTorch drift is small but non-zero.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import onnxruntime as ort  # noqa: E402

from src.compute_metrics import compute_metrics  # noqa: E402
from src.stft import mag_pha_stft, manual_istft_ola  # noqa: E402
from src.utils import LogProgress, bold  # noqa: E402


def _load_streaming_config(config_path: str) -> Dict[str, Any]:
    """Load the bafnetplus_streaming_config.json sibling produced by export.

    The config is the source of truth for STFT params, chunk geometry, and the
    166 state names — all of which must agree with the ONNX graph's I/O.
    """
    with open(config_path, "r") as f:
        return json.load(f)


def _derive_streaming_geometry(cfg: Dict[str, Any]) -> Dict[str, int]:
    """Compute samples-per-chunk / output-samples / context derived from cfg."""
    stft = cfg["stft_config"]
    streaming = cfg["streaming_config"]
    n_fft = int(stft["n_fft"])
    hop = int(stft["hop_size"])
    win = int(stft["win_length"])
    chunk_frames = int(streaming["chunk_size_frames"])
    export_frames = int(streaming["export_time_frames"])
    return {
        "n_fft": n_fft,
        "hop": hop,
        "win": win,
        "compress_factor": float(stft["compress_factor"]),
        "chunk_frames": chunk_frames,
        "export_frames": export_frames,
        "samples_per_chunk": (export_frames - 1) * hop + win // 2,
        "output_samples": chunk_frames * hop,
    }


def _stft_chunk(
    chunk_with_context: torch.Tensor,
    n_fft: int,
    hop: int,
    win: int,
    compress_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mag/pha STFT (center=False) and return numpy arrays for ORT."""
    mag, pha, _ = mag_pha_stft(
        chunk_with_context.unsqueeze(0),
        n_fft=n_fft, hop_size=hop, win_size=win,
        compress_factor=compress_factor, center=False,
    )
    return mag.numpy().astype(np.float32), pha.numpy().astype(np.float32)


def _stream_one_utterance(
    bcs: np.ndarray,
    acs: np.ndarray,
    sess: ort.InferenceSession,
    state_names: List[str],
    state_shapes: Dict[str, Tuple[int, ...]],
    geom: Dict[str, int],
) -> np.ndarray:
    """Stream one (bcs, acs) pair through the ONNX graph and return enhanced PCM.

    ``bcs`` and ``acs`` are 16 kHz mono float32 arrays with the same length.
    Returns the ``chunk_frames * hop * n_pushes_emitted`` mature samples.
    """
    if len(bcs) != len(acs):
        raise ValueError(f"bcs/acs length mismatch: {len(bcs)} vs {len(acs)}")

    n_pushes = len(bcs) // geom["output_samples"]
    if n_pushes < 2:
        raise RuntimeError(
            f"audio too short: {len(bcs)} samples → {n_pushes} pushes "
            f"(need ≥ 2 to produce any enhanced output).",
        )

    bcs_t = torch.from_numpy(bcs.astype(np.float32))
    acs_t = torch.from_numpy(acs.astype(np.float32))

    onnx_states: Dict[str, np.ndarray] = {
        name: np.zeros(state_shapes[name], dtype=np.float32) for name in state_names
    }

    half_win = geom["win"] // 2
    bcs_stft_ctx = torch.zeros(half_win)
    acs_stft_ctx = torch.zeros(half_win)
    bcs_buf = torch.tensor([], dtype=torch.float32)
    acs_buf = torch.tensor([], dtype=torch.float32)

    ola_buf = torch.zeros(geom["win"] - geom["hop"])
    ola_norm = torch.zeros(geom["win"] - geom["hop"])

    enhanced_chunks: List[np.ndarray] = []

    for push_idx in range(n_pushes):
        s = push_idx * geom["output_samples"]
        e = s + geom["output_samples"]
        bcs_buf = torch.cat([bcs_buf, bcs_t[s:e]])
        acs_buf = torch.cat([acs_buf, acs_t[s:e]])

        if len(bcs_buf) < geom["samples_per_chunk"]:
            continue

        bcs_chunk = bcs_buf[: geom["samples_per_chunk"]]
        acs_chunk = acs_buf[: geom["samples_per_chunk"]]
        bcs_with_ctx = torch.cat([bcs_stft_ctx, bcs_chunk])
        acs_with_ctx = torch.cat([acs_stft_ctx, acs_chunk])

        bcs_mag, bcs_pha = _stft_chunk(
            bcs_with_ctx,
            geom["n_fft"], geom["hop"], geom["win"], geom["compress_factor"],
        )
        acs_mag, acs_pha = _stft_chunk(
            acs_with_ctx,
            geom["n_fft"], geom["hop"], geom["win"], geom["compress_factor"],
        )

        adv = geom["output_samples"]
        if adv >= half_win:
            bcs_stft_ctx = bcs_buf[adv - half_win:adv].clone()
            acs_stft_ctx = acs_buf[adv - half_win:adv].clone()
        else:
            need = half_win - adv
            bcs_stft_ctx = torch.cat([bcs_stft_ctx[-need:], bcs_buf[:adv]]).clone()
            acs_stft_ctx = torch.cat([acs_stft_ctx[-need:], acs_buf[:adv]]).clone()
        bcs_buf = bcs_buf[geom["output_samples"]:]
        acs_buf = acs_buf[geom["output_samples"]:]

        ort_inputs: Dict[str, np.ndarray] = {
            "bcs_mag": bcs_mag, "bcs_pha": bcs_pha,
            "acs_mag": acs_mag, "acs_pha": acs_pha,
        }
        ort_inputs.update(onnx_states)
        outs = sess.run(None, ort_inputs)
        est_mag = outs[0]
        est_com_real = outs[1]
        est_com_imag = outs[2]
        next_states = outs[3:]
        if len(next_states) != len(state_names):
            raise RuntimeError(
                f"output state count mismatch: got {len(next_states)} "
                f"vs {len(state_names)} state inputs.",
            )
        onnx_states = {n: next_states[i] for i, n in enumerate(state_names)}

        # Match the on-device convention: phase reconstructed via atan2 with the
        # 1e-8 epsilons used in src.stft (numerical-stability twin of mag_pha_stft).
        est_pha = np.arctan2(est_com_imag + 1e-8, est_com_real + 1e-8)
        est_mag_t = torch.from_numpy(est_mag[:, :, : geom["chunk_frames"]])
        est_pha_t = torch.from_numpy(est_pha[:, :, : geom["chunk_frames"]])

        out, ola_buf, ola_norm = manual_istft_ola(
            est_mag_t, est_pha_t,
            n_fft=geom["n_fft"], hop_size=geom["hop"], win_size=geom["win"],
            compress_factor=geom["compress_factor"],
            ola_buffer=ola_buf, ola_norm=ola_norm,
        )
        # manual_istft_ola already returns ``[T*hop_size]`` mature samples
        # (== ``output_samples`` for our chunk_frames=8, hop=100); the slice
        # is defensive for chunk_frames mismatch in future configs.
        enhanced_chunks.append(out[: geom["output_samples"]].numpy())

    if not enhanced_chunks:
        raise RuntimeError("No enhanced chunks produced; audio too short.")
    return np.concatenate(enhanced_chunks).astype(np.float32)


def evaluate_onnx(
    onnx_path: str,
    config_path: str,
    data_loader_list: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    bcs_gain_db: float = 0.0,
    acs_gain_db: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Streaming ONNX evaluator. Returns the same shape as ``src.evaluate.evaluate``.

    Args:
        onnx_path: Path to the streaming BAFNet+ ONNX graph (FP32 or INT8 QDQ).
        config_path: Path to the sibling ``bafnetplus_streaming_config.json``.
        data_loader_list: ``{snr_str: DataLoader}`` produced by
            :func:`src.runtime_common.build_eval_loaders`.
        logger: Logger instance.
        bcs_gain_db: Independent BCS-channel gain perturbation in dB (mirrors
            :func:`src.evaluate.evaluate` so ONNX-path numbers can be compared
            against the PyTorch path under the same gain regime). Default 0.0.
        acs_gain_db: Independent ACS-channel gain perturbation in dB. Applied
            to both ``noisy_acs`` (model input) and ``clean_acs`` (reference)
            so the SE target shifts with the input. Default 0.0.

    Returns:
        ``{"<snr>dB": {"pesq": ..., "stoi": ..., "csig": ..., "cbak": ...,
        "covl": ..., "segSNR": ...}, ...}``.
    """
    log = logger or logging.getLogger(__name__)
    cfg = _load_streaming_config(config_path)
    geom = _derive_streaming_geometry(cfg)
    state_names: List[str] = list(cfg["state_info"]["state_names"])

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_inputs = {inp.name: tuple(inp.shape) for inp in sess.get_inputs()}
    state_shapes: Dict[str, Tuple[int, ...]] = {}
    for name in state_names:
        if name not in onnx_inputs:
            raise RuntimeError(f"ONNX missing state input: {name!r}")
        state_shapes[name] = tuple(int(d) if isinstance(d, int) else 1 for d in onnx_inputs[name])

    log.info(
        f"Loaded ONNX: {onnx_path} (states={len(state_names)}, "
        f"chunk_frames={geom['chunk_frames']}, export_frames={geom['export_frames']})",
    )

    bcs_scalar = 10.0 ** (bcs_gain_db / 20.0) if bcs_gain_db != 0.0 else 1.0
    acs_scalar = 10.0 ** (acs_gain_db / 20.0) if acs_gain_db != 0.0 else 1.0
    if bcs_scalar != 1.0 or acs_scalar != 1.0:
        log.info(
            f"Gain perturbation: BCS={bcs_gain_db:+.1f}dB ({bcs_scalar:.4f}x), "
            f"ACS={acs_gain_db:+.1f}dB ({acs_scalar:.4f}x)",
        )

    metrics: Dict[str, Dict[str, float]] = {}
    for snr, data_loader in data_loader_list.items():
        iterator = LogProgress(log, data_loader, name=f"Evaluate ONNX on {snr}dB")
        results: List[Tuple[float, float, float, float, float, float]] = []
        for data in iterator:
            bcs, noisy_acs, clean_acs, _utt_id, _text = data
            bcs_np = bcs.squeeze().detach().cpu().numpy().astype(np.float32)
            acs_np = noisy_acs.squeeze().detach().cpu().numpy().astype(np.float32)
            clean_np = clean_acs.squeeze().detach().cpu().numpy().astype(np.float32)

            if bcs_scalar != 1.0:
                bcs_np = bcs_np * bcs_scalar
            if acs_scalar != 1.0:
                acs_np = acs_np * acs_scalar
                clean_np = clean_np * acs_scalar

            enhanced = _stream_one_utterance(
                bcs_np, acs_np, sess, state_names, state_shapes, geom,
            )

            if len(enhanced) != len(clean_np):
                length = min(len(enhanced), len(clean_np))
                enhanced = enhanced[:length]
                clean_np = clean_np[:length]

            results.append(compute_metrics(clean_np, enhanced))

        pesq, csig, cbak, covl, segSNR, stoi = np.mean(results, axis=0)
        snr_key = "native" if str(snr) == "native" else f"{snr}dB"
        metrics[snr_key] = {
            "pesq": float(pesq),
            "stoi": float(stoi),
            "csig": float(csig),
            "cbak": float(cbak),
            "covl": float(covl),
            "segSNR": float(segSNR),
        }
        log.info(bold(
            f"ONNX Performance on {snr}dB: PESQ={pesq:.4f}, STOI={stoi:.4f}, "
            f"CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}",
        ))

    return metrics
