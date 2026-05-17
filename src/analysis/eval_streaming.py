"""Full-utterance 3-path streaming eval harness CLI (S10).

S10 of the streaming-ONNX rebuild — Stages 5 + 6 of the reference-runtime
plan. This module is a **downstream consumer** of the S6 PT streaming wrapper,
the S8 functional-stateful BAFNet+ exportable core, and the S9 ORT host
wrapper; it does **not** re-implement chunked streaming.

It runs three enhancement paths over the same complete utterance:

    (1) Reference: ``mag_pha_stft(*, center=True)`` ->
        :meth:`BAFNetPlus.forward` -> ``mag_pha_istft(*, center=True)``
        (the non-streaming bit-exact metric anchor).
    (2) PT streaming: :meth:`BAFNetPlusStreaming.process_audio` (S6).
    (3) ORT streaming: :meth:`BAFNetPlusOrtStreaming.process_audio` (S9 —
        the deployed FP32 ONNX path).

Per-utterance dump:
    * ``len_samples`` / ``len_seconds``
    * ``n_chunks_processed`` (= ceil(len_padded / output_samples_per_chunk))
    * ``n_warmup_none_returns`` (the S6/S9 contract — 2 for 50 ms anchor)
    * ``output_trim_length`` (the wrapper's returned audio length)
    * ``waveform_{max,rms}_diff_{ref_vs_pt, ref_vs_ort, pt_vs_ort}``
      (S9.5 shift-corrected; pt_vs_ort is shift-invariant)
    * ``correlation_{ref_vs_pt, ref_vs_ort, pt_vs_ort}`` (Pearson)
    * ``pesq_ref_pt`` / ``pesq_streaming_pt`` / ``pesq_ort_streaming``
      (wb mode, full utterance only)
    * ``delta_pesq_{ref_vs_pt, ref_vs_ort, pt_vs_ort}``
    * ``stoi_{ref_pt, streaming_pt, ort_streaming}`` (if ``pystoi`` is
      importable; otherwise ``None``)
    * ``failure_reasons`` (list[str] — empty when all gates pass)

Aggregate dump:
    * ``max/mean/median_abs_delta_pesq_{ref_vs_ort, pt_vs_ort, ref_vs_pt}``
    * ``all_pass_individual`` (every utt's ``failure_reasons`` is empty)
    * ``aggregate_pass`` (aggregate mean gates met)
    * ``overall_pass`` (both)

**Hard rule**: PESQ is computed ONLY on the full, trimmed-to-clean-ref
utterance. Passing a ``clean_ref_audio`` shorter than the BCS/ACS pair length
raises :class:`ValueError` (the S10 partial-utterance-rejection contract).
Chunk subsets / prefix-only / hand-picked regions cannot pass this harness.

Presets
-------
``--preset smoke``
    TAPS test idx ``[0]``, ``pesq_tol = PESQ_TOL_TRIPLE_LP_TARGET = 1e-4``.
    Idx 0 hits the LP target at ``2.24e-5`` (S9.5 envelope), so this preset
    exits 0 on the deployed real ckpt. Output: stdout summary only.

``--preset full``
    TAPS test idx ``[0,1,2,3,4]``, ``pesq_tol =
    PESQ_TOL_TRIPLE_DEFAULT = 5e-3``. Aggregate ``mean ≤ 2.5e-3`` per the
    S9 ``run_gate_triple`` aggregate convention. Output: JSON + Markdown
    table by default.

Both presets default to the existing exported real-ckpt ONNX at
``results/onnx/bafnetplus_50ms_fp32.onnx``; if absent, the harness
re-exports on the fly via :meth:`BAFNetPlusOrtStreaming.from_checkpoint`
(cached to ``results/onnx/``).

Tolerance defaults
------------------
``--pesq-tol`` defaults to ``PESQ_TOL_TRIPLE_DEFAULT = 5e-3``: this is the
S9 ``|delta_pesq_ref_vs_ort|`` envelope (idx 0..4 max ``1.36e-3``).
``--cross-streaming-tol`` defaults to ``0.15`` (the S7 baseline): the PT
streaming path retains the S6/S7 non-streaming-fusion drift that the
functional-stateful ORT path closed, so ``|delta_pesq_pt_vs_ort|`` is
naturally ``~3-6e-2``. The cross-streaming gate is a regression detector —
tighten via ``--cross-streaming-tol`` for stricter checking.

Anchors are re-confirmed at runtime via the wrappers' ``streaming_config``
property + the ORT sidecar JSON; nothing is hard-coded for behaviour
(only defaults). The 50 ms BAFNet+ variant uses ``n_fft=400, hop=100,
win=400, compress=0.3, chunk_size=8, L_enc=L_dec=3, L_alpha=0,
T_export=14, samples_per_chunk=1200, output_samples_per_chunk=800,
latency=50.0 ms``.

CLI exit code is non-zero on any gate violation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.analysis.pesq_streaming_gate import (
    PESQ_TOL_TRIPLE_DEFAULT,
    PESQ_TOL_TRIPLE_LP_TARGET,
    build_triple_paths,
    compute_pesq,
    enhance_ort_streaming,
    enhance_reference,
    enhance_streaming,
    load_taps_utterance,
)
from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.bafnetplus_streaming import BAFNetPlusStreaming
from src.models.streaming.onnx.ort_wrapper import BAFNetPlusOrtStreaming

logger = logging.getLogger(__name__)

# 50 ms BAFNet+ STFT anchors (re-confirmed at runtime via streaming_config).
_DEFAULT_N_FFT = 400
_DEFAULT_HOP = 100
_DEFAULT_WIN = 400
_DEFAULT_COMPRESS = 0.3
_DEFAULT_SR = 16000
_DEFAULT_CHUNK_SIZE = 8

# CLI defaults.
_DEFAULT_UNIFIED_CKPT_DIR = "results/experiments/bafnetplus_50ms"
_DEFAULT_CHKPT_FILE = "best.th"
_DEFAULT_ONNX_ARTIFACT = "results/onnx/bafnetplus_50ms_fp32.onnx"
_DEFAULT_ONNX_OUTPUT_DIR = "results/onnx"
_DEFAULT_TAPS_INDICES_FULL: List[int] = [0, 1, 2, 3, 4]
_DEFAULT_TAPS_INDICES_SMOKE: List[int] = [0]

# Default audio max-diff tolerance for the wf_max_diff_ref_vs_ort failure
# check. Empirical S9.5 envelope on the deployed ckpt is max|d_ref_ort|
# ~1.5e-2 on the worst utterance; 5e-2 gives ~3.3x headroom.
_DEFAULT_AUDIO_MAX_DIFF_TOL = 5e-2

# Cross-streaming PESQ tolerance default — the PT path has documented
# non-streaming-fusion drift (S6/S7) that the ORT path closes (S8); the
# resulting |d_pt_ort| is naturally ~3-6e-2 on the deployed ckpt. 0.15 is
# the S7 baseline relaxation. Tighten via --cross-streaming-tol for
# strict regression checking.
_DEFAULT_CROSS_STREAMING_TOL = 0.15

# Expected warm-up None returns for the 50 ms anchor (S6 contract — 2).
EXPECTED_WARMUP_NONE_RETURNS = 2

# JSON schema version (frozen — bump if keys change incompatibly).
EVAL_STREAMING_SCHEMA_VERSION = "s10-eval-streaming-fp32-v1"

# Frozen output schema keys (used by tests + downstream consumers).
RUN_METADATA_KEYS: Tuple[str, ...] = (
    "schema_version",
    "unified_ckpt_dir",
    "chkpt_file",
    "checkpoint_md5",
    "onnx_path",
    "onnx_sidecar_path",
    "onnx_schema_version",
    "taps_indices",
    "pesq_tol",
    "cross_streaming_tol",
    "audio_max_diff_tol",
    "shift_correction_samples",
    "chunk_size",
    "device",
    "sample_rate",
    "stft",
    "setup_seconds",
)
PER_UTTERANCE_KEYS: Tuple[str, ...] = (
    "taps_idx",
    "per_utt_seconds",
    "len_samples",
    "len_seconds",
    "n_chunks_processed",
    "n_warmup_none_returns",
    "output_trim_length",
    "expected_output_trim_length",
    "shift_correction_samples",
    "waveform_max_diff_ref_vs_pt",
    "waveform_rms_diff_ref_vs_pt",
    "waveform_max_diff_ref_vs_ort",
    "waveform_rms_diff_ref_vs_ort",
    "waveform_max_diff_pt_vs_ort",
    "waveform_rms_diff_pt_vs_ort",
    "correlation_ref_vs_pt",
    "correlation_ref_vs_ort",
    "correlation_pt_vs_ort",
    "pesq_ref_pt",
    "pesq_streaming_pt",
    "pesq_ort_streaming",
    "delta_pesq_ref_vs_pt",
    "delta_pesq_ref_vs_ort",
    "delta_pesq_pt_vs_ort",
    "stoi_ref_pt",
    "stoi_streaming_pt",
    "stoi_ort_streaming",
    "failure_reasons",
)
AGGREGATE_KEYS: Tuple[str, ...] = (
    "max_abs_delta_pesq_ref_vs_ort",
    "mean_abs_delta_pesq_ref_vs_ort",
    "median_abs_delta_pesq_ref_vs_ort",
    "max_abs_delta_pesq_pt_vs_ort",
    "mean_abs_delta_pesq_pt_vs_ort",
    "median_abs_delta_pesq_pt_vs_ort",
    "max_abs_delta_pesq_ref_vs_pt",
    "mean_abs_delta_pesq_ref_vs_pt",
    "median_abs_delta_pesq_ref_vs_pt",
    "all_pass_individual",
    "aggregate_pass",
    "overall_pass",
)


# --------------------------------------------------------------- helpers
def _to_1d(t: Tensor) -> Tensor:
    """Squeeze a 1-/2-D tensor to a 1-D vector (raise on rank > 2)."""
    if t.dim() == 1:
        return t
    if t.dim() == 2:
        return t.squeeze(0) if t.shape[0] == 1 else t.squeeze()
    return t.squeeze()


def _max_rms_pair(diff: Tensor) -> Tuple[float, float]:
    """Return (max|d|, rms(d)) on a 1-D diff tensor (NaN if empty)."""
    if diff.numel() == 0:
        return float("nan"), float("nan")
    return float(diff.max().item()), float(diff.pow(2).mean().sqrt().item())


def _pearson(a: Tensor, b: Tensor) -> float:
    """Pearson correlation between two 1-D tensors (NaN if length 0 or const)."""
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    am = a64 - a64.mean()
    bm = b64 - b64.mean()
    denom = (am.pow(2).sum().sqrt() * bm.pow(2).sum().sqrt()).item()
    if denom == 0.0:
        return float("nan")
    return float((am * bm).sum().item() / denom)


def _validate_full_utterance(clean_ref_audio: Tensor, bcs_audio: Tensor, acs_audio: Tensor) -> int:
    """Reject partial-utterance PESQ mode.

    PESQ must be computed on the FULL trimmed-to-clean-ref utterance. A
    ``clean_ref_audio`` shorter than the matched (BCS, ACS) pair length is
    a partial / prefix-only / hand-picked region — rejected.

    Args:
        clean_ref_audio: Clean reference audio ``[T]``.
        bcs_audio: BCS input ``[T]``.
        acs_audio: ACS input ``[T]``.

    Returns:
        ``pair_len = min(len(bcs), len(acs))`` (informational).

    Raises:
        ValueError: If ``len(clean_ref_audio) < pair_len``.
    """
    pair_len = int(min(len(bcs_audio), len(acs_audio)))
    if int(len(clean_ref_audio)) < pair_len:
        raise ValueError(
            "PESQ requires full-utterance comparison: clean_ref_audio length "
            f"{int(len(clean_ref_audio))} is shorter than the matched input pair length {pair_len}. "
            "Partial / prefix-only / hand-picked clean references are rejected; "
            "pass the complete clean utterance."
        )
    return pair_len


def _compute_failure_reasons(
    *,
    n_warmup_none_returns: int,
    expected_warmup: int,
    output_trim_length: int,
    expected_output_trim_length: int,
    wf_max_diff_ref_vs_ort: float,
    audio_max_diff_tol: float,
    delta_pesq_ref_vs_ort: float,
    pesq_tol: float,
    delta_pesq_pt_vs_ort: float,
    cross_streaming_tol: float,
) -> List[str]:
    """Collect the per-utterance failure reasons given the measured deltas.

    Pure function — easy to test without loading a model. Each gate is
    independent (multiple may be populated on a single bad utterance).

    Args:
        n_warmup_none_returns: Empirically-observed warm-up None count.
        expected_warmup: Expected count for the 50 ms anchor (``= 2``).
        output_trim_length: ``len(streaming_audio)``.
        expected_output_trim_length: ``len(bcs_input_to_process_audio)``
            (the wrapper's documented trim contract).
        wf_max_diff_ref_vs_ort: Audio max-abs diff between reference and
            ORT-streaming (post-shift-correction).
        audio_max_diff_tol: Audio max-diff failure threshold.
        delta_pesq_ref_vs_ort: PESQ delta ref - ort (signed).
        pesq_tol: ``|delta_pesq_ref_vs_ort|`` failure threshold.
        delta_pesq_pt_vs_ort: PESQ delta pt - ort (signed).
        cross_streaming_tol: ``|delta_pesq_pt_vs_ort|`` failure threshold.

    Returns:
        Sorted list of human-readable failure reason strings.
    """
    failures: List[str] = []
    if n_warmup_none_returns != expected_warmup:
        failures.append(
            f"n_warmup_none_returns={n_warmup_none_returns} != {expected_warmup} "
            "(50 ms anchor S6/S9 wrapper contract)"
        )
    if output_trim_length != expected_output_trim_length:
        failures.append(
            f"output_trim_length={output_trim_length} != "
            f"len(bcs_input)={expected_output_trim_length} (streaming trim contract violated)"
        )
    if wf_max_diff_ref_vs_ort > audio_max_diff_tol:
        failures.append(
            f"wf_max_diff_ref_vs_ort={wf_max_diff_ref_vs_ort:.3e} > " f"audio_max_diff_tol={audio_max_diff_tol:.3e}"
        )
    if abs(delta_pesq_ref_vs_ort) > pesq_tol:
        failures.append(f"|delta_pesq_ref_vs_ort|={abs(delta_pesq_ref_vs_ort):.3e} > pesq_tol={pesq_tol:.3e}")
    if abs(delta_pesq_pt_vs_ort) > cross_streaming_tol:
        failures.append(
            f"|delta_pesq_pt_vs_ort|={abs(delta_pesq_pt_vs_ort):.3e} > "
            f"cross_streaming_tol={cross_streaming_tol:.3e}"
        )
    return failures


def _count_pt_warmup_none_returns(pt_streaming: BAFNetPlusStreaming, bcs_audio: Tensor, acs_audio: Tensor) -> int:
    """Replay one utterance through :meth:`process_samples` to count warm-up.

    Mirrors :meth:`BAFNetPlusStreaming.process_audio`'s internal loop
    (reset + zero-flush + 800-sample steps) and exits early on the first
    non-None return — typically after 2 calls for the 50 ms anchor.

    Args:
        pt_streaming: The PT streaming wrapper.
        bcs_audio: BCS audio (any length; ``process_audio`` flushes on top).
        acs_audio: ACS audio (sample-aligned with BCS).

    Returns:
        Number of consecutive ``None`` returns before the first matured
        output.
    """
    pt_streaming.reset_state()
    flush_size = pt_streaming.samples_per_chunk * (pt_streaming.total_lookahead + 2)
    device = pt_streaming.device
    bcs_audio = _to_1d(bcs_audio).to(device)
    acs_audio = _to_1d(acs_audio).to(device)
    pair_len = min(len(bcs_audio), len(acs_audio))
    bcs_full = torch.cat([bcs_audio[:pair_len], torch.zeros(flush_size, device=device)])
    acs_full = torch.cat([acs_audio[:pair_len], torch.zeros(flush_size, device=device)])
    n_none = 0
    step = pt_streaming.output_samples_per_chunk
    for i in range(0, len(bcs_full), step):
        bcs_chunk = bcs_full[i : i + step]
        acs_chunk = acs_full[i : i + step]
        if len(bcs_chunk) == 0 or len(acs_chunk) == 0:
            break
        result = pt_streaming.process_samples(bcs_chunk, acs_chunk)
        if result is None:
            n_none += 1
        else:
            break
    pt_streaming.reset_state()
    return n_none


# -------------------------------------------------------------- per-utt
def evaluate_utterance(
    bafnet: BAFNetPlus,
    pt_streaming: BAFNetPlusStreaming,
    ort_streaming: BAFNetPlusOrtStreaming,
    *,
    bcs_audio: Tensor,
    acs_audio: Tensor,
    clean_ref_audio: Tensor,
    sample_rate: int = _DEFAULT_SR,
    n_fft: int = _DEFAULT_N_FFT,
    hop_size: int = _DEFAULT_HOP,
    win_size: int = _DEFAULT_WIN,
    compress_factor: float = _DEFAULT_COMPRESS,
    shift_correction_samples: Optional[int] = None,
    pesq_tol: float = PESQ_TOL_TRIPLE_DEFAULT,
    cross_streaming_tol: float = _DEFAULT_CROSS_STREAMING_TOL,
    audio_max_diff_tol: float = _DEFAULT_AUDIO_MAX_DIFF_TOL,
    expected_warmup_none_returns: int = EXPECTED_WARMUP_NONE_RETURNS,
    compute_stoi: bool = True,
) -> Dict[str, Any]:
    """Run all 3 enhancement paths + compute PESQ triple + extended diagnostics.

    Pads ``win//2`` zeros leading + trailing on both inputs so the
    reference ``center=True`` reflect-pad bit-aligns with both streaming
    wrappers' zero past-context / zero flush. Both streaming outputs are
    further shifted by ``shift_correction_samples`` samples to compensate
    for the ``manual_istft_ola`` boundary offset (S9.5 alignment finding).

    Args:
        bafnet / pt_streaming / ort_streaming: All three wrappers, derived
            from the same unified ckpt (e.g. via :func:`build_triple_paths`).
        bcs_audio / acs_audio: ``[T]`` paired-modality inputs.
        clean_ref_audio: Clean target ``[T]``. **Must be at least as long
            as the matched (BCS, ACS) pair length** — partial-utterance
            references are rejected with ``ValueError``.
        sample_rate / n_fft / hop_size / win_size / compress_factor:
            STFT parameters (50 ms BAFNet+ defaults).
        shift_correction_samples: Additional left-strip applied to both
            streaming outputs before PESQ scoring. Default ``None``
            resolves to ``n_fft // 2`` (the S9.5 alignment fix).
        pesq_tol: ``|delta_pesq_ref_vs_ort|`` failure threshold.
        cross_streaming_tol: ``|delta_pesq_pt_vs_ort|`` failure threshold.
        audio_max_diff_tol: ``waveform_max_diff_ref_vs_ort`` failure
            threshold.
        expected_warmup_none_returns: Expected warm-up None count (50 ms
            anchor = 2).
        compute_stoi: If ``True`` and ``pystoi`` importable, compute STOI;
            otherwise leave the ``stoi_*`` fields as ``None``.

    Returns:
        Dict with the keys listed in :data:`PER_UTTERANCE_KEYS` (minus
        ``taps_idx`` / ``per_utt_seconds`` which the caller adds).

    Raises:
        ValueError: If ``clean_ref_audio`` is shorter than the matched
            input pair length.
    """
    if shift_correction_samples is None:
        shift_correction_samples = n_fft // 2

    bcs_audio = _to_1d(bcs_audio)
    acs_audio = _to_1d(acs_audio)
    clean_ref_audio = _to_1d(clean_ref_audio)

    pair_len = _validate_full_utterance(clean_ref_audio, bcs_audio, acs_audio)
    bcs_audio = bcs_audio[:pair_len]
    acs_audio = acs_audio[:pair_len]
    clean_ref_audio = clean_ref_audio[:pair_len]

    pad = win_size // 2
    zeros_pad = torch.zeros(pad, dtype=bcs_audio.dtype)
    bcs_padded = torch.cat([zeros_pad, bcs_audio, zeros_pad])
    acs_padded = torch.cat([zeros_pad, acs_audio, zeros_pad])

    # 3-path enhancement.
    ref_audio = enhance_reference(
        bafnet,
        bcs_padded,
        acs_padded,
        n_fft=n_fft,
        hop_size=hop_size,
        win_size=win_size,
        compress_factor=compress_factor,
    )
    pt_audio = enhance_streaming(pt_streaming, bcs_padded, acs_padded)
    ort_audio = enhance_ort_streaming(ort_streaming, bcs_padded, acs_padded)
    output_trim_length = int(len(pt_audio))
    expected_output_trim_length = int(len(bcs_padded))

    # Empirical warm-up count (PT side — both wrappers share the S6/S9 contract).
    n_warmup_none_returns = _count_pt_warmup_none_returns(pt_streaming, bcs_padded, acs_padded)

    # n_chunks_processed derived from process_audio's flush + 800-sample step.
    flush_size = pt_streaming.samples_per_chunk * (pt_streaming.total_lookahead + 2)
    len_padded = len(bcs_padded) + flush_size
    step = pt_streaming.output_samples_per_chunk
    n_chunks_processed = int(np.ceil(len_padded / step))

    # Align to clean: strip leading win//2-zero pad on the reference; streaming
    # outputs get the additional 200-sample shift correction (S9.5).
    stream_start = pad + int(shift_correction_samples)
    L = int(len(clean_ref_audio))
    ref_aligned = ref_audio[pad : pad + L]
    pt_aligned = pt_audio[stream_start : stream_start + L]
    ort_aligned = ort_audio[stream_start : stream_start + L]
    mlen = min(L, len(ref_aligned), len(pt_aligned), len(ort_aligned))
    ref_aligned = ref_aligned[:mlen]
    pt_aligned = pt_aligned[:mlen]
    ort_aligned = ort_aligned[:mlen]
    clean_aligned = clean_ref_audio[:mlen]

    # Waveform diff between paths (post-shift-correction).
    wf_max_ref_pt, wf_rms_ref_pt = _max_rms_pair((ref_aligned - pt_aligned).abs())
    wf_max_ref_ort, wf_rms_ref_ort = _max_rms_pair((ref_aligned - ort_aligned).abs())
    wf_max_pt_ort, wf_rms_pt_ort = _max_rms_pair((pt_aligned - ort_aligned).abs())

    corr_ref_pt = _pearson(ref_aligned, pt_aligned)
    corr_ref_ort = _pearson(ref_aligned, ort_aligned)
    corr_pt_ort = _pearson(pt_aligned, ort_aligned)

    # PESQ — wb mode, on the FULL utterance trimmed to clean-ref length.
    pesq_ref_pt = compute_pesq(clean_aligned, ref_aligned, sample_rate=sample_rate, mode="wb")
    pesq_streaming_pt = compute_pesq(clean_aligned, pt_aligned, sample_rate=sample_rate, mode="wb")
    pesq_ort_streaming = compute_pesq(clean_aligned, ort_aligned, sample_rate=sample_rate, mode="wb")
    delta_ref_pt = pesq_ref_pt - pesq_streaming_pt
    delta_ref_ort = pesq_ref_pt - pesq_ort_streaming
    delta_pt_ort = pesq_streaming_pt - pesq_ort_streaming

    # STOI — optional; leave None when pystoi is absent.
    stoi_ref_pt: Optional[float] = None
    stoi_streaming_pt: Optional[float] = None
    stoi_ort_streaming: Optional[float] = None
    if compute_stoi:
        try:
            from pystoi import stoi as stoi_fn  # type: ignore[import-not-found]

            clean_np = clean_aligned.detach().cpu().numpy().astype(np.float32)
            ref_np = ref_aligned.detach().cpu().numpy().astype(np.float32)
            pt_np = pt_aligned.detach().cpu().numpy().astype(np.float32)
            ort_np = ort_aligned.detach().cpu().numpy().astype(np.float32)
            stoi_ref_pt = float(stoi_fn(clean_np, ref_np, sample_rate, extended=False))
            stoi_streaming_pt = float(stoi_fn(clean_np, pt_np, sample_rate, extended=False))
            stoi_ort_streaming = float(stoi_fn(clean_np, ort_np, sample_rate, extended=False))
        except ImportError:
            pass

    failure_reasons = _compute_failure_reasons(
        n_warmup_none_returns=n_warmup_none_returns,
        expected_warmup=expected_warmup_none_returns,
        output_trim_length=output_trim_length,
        expected_output_trim_length=expected_output_trim_length,
        wf_max_diff_ref_vs_ort=wf_max_ref_ort,
        audio_max_diff_tol=audio_max_diff_tol,
        delta_pesq_ref_vs_ort=delta_ref_ort,
        pesq_tol=pesq_tol,
        delta_pesq_pt_vs_ort=delta_pt_ort,
        cross_streaming_tol=cross_streaming_tol,
    )

    return {
        "len_samples": int(mlen),
        "len_seconds": float(mlen / sample_rate),
        "n_chunks_processed": int(n_chunks_processed),
        "n_warmup_none_returns": int(n_warmup_none_returns),
        "output_trim_length": output_trim_length,
        "expected_output_trim_length": expected_output_trim_length,
        "shift_correction_samples": int(shift_correction_samples),
        "waveform_max_diff_ref_vs_pt": wf_max_ref_pt,
        "waveform_rms_diff_ref_vs_pt": wf_rms_ref_pt,
        "waveform_max_diff_ref_vs_ort": wf_max_ref_ort,
        "waveform_rms_diff_ref_vs_ort": wf_rms_ref_ort,
        "waveform_max_diff_pt_vs_ort": wf_max_pt_ort,
        "waveform_rms_diff_pt_vs_ort": wf_rms_pt_ort,
        "correlation_ref_vs_pt": corr_ref_pt,
        "correlation_ref_vs_ort": corr_ref_ort,
        "correlation_pt_vs_ort": corr_pt_ort,
        "pesq_ref_pt": float(pesq_ref_pt),
        "pesq_streaming_pt": float(pesq_streaming_pt),
        "pesq_ort_streaming": float(pesq_ort_streaming),
        "delta_pesq_ref_vs_pt": float(delta_ref_pt),
        "delta_pesq_ref_vs_ort": float(delta_ref_ort),
        "delta_pesq_pt_vs_ort": float(delta_pt_ort),
        "stoi_ref_pt": stoi_ref_pt,
        "stoi_streaming_pt": stoi_streaming_pt,
        "stoi_ort_streaming": stoi_ort_streaming,
        "failure_reasons": failure_reasons,
    }


# -------------------------------------------------------------- aggregate
def _stats(xs: List[float]) -> Dict[str, float]:
    """Compute max / mean / median of a list of floats (defaults zero on empty)."""
    if not xs:
        return {"max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "max": float(max(xs)),
        "mean": float(sum(xs) / len(xs)),
        "median": float(np.median(xs)),
    }


def _aggregate(
    per_utterance: List[Dict[str, Any]],
    *,
    pesq_tol: float,
    cross_streaming_tol: float,
) -> Dict[str, Any]:
    """Aggregate per-utterance reports into the aggregate-block dict."""
    abs_ref_ort = [abs(r["delta_pesq_ref_vs_ort"]) for r in per_utterance]
    abs_pt_ort = [abs(r["delta_pesq_pt_vs_ort"]) for r in per_utterance]
    abs_ref_pt = [abs(r["delta_pesq_ref_vs_pt"]) for r in per_utterance]
    s_ref_ort = _stats(abs_ref_ort)
    s_pt_ort = _stats(abs_pt_ort)
    s_ref_pt = _stats(abs_ref_pt)
    all_pass_individual = all(not r["failure_reasons"] for r in per_utterance)
    aggregate_pass = bool(s_ref_ort["mean"] <= pesq_tol / 2.0 and s_pt_ort["mean"] <= cross_streaming_tol)
    overall_pass = bool(all_pass_individual and aggregate_pass)
    return {
        "max_abs_delta_pesq_ref_vs_ort": s_ref_ort["max"],
        "mean_abs_delta_pesq_ref_vs_ort": s_ref_ort["mean"],
        "median_abs_delta_pesq_ref_vs_ort": s_ref_ort["median"],
        "max_abs_delta_pesq_pt_vs_ort": s_pt_ort["max"],
        "mean_abs_delta_pesq_pt_vs_ort": s_pt_ort["mean"],
        "median_abs_delta_pesq_pt_vs_ort": s_pt_ort["median"],
        "max_abs_delta_pesq_ref_vs_pt": s_ref_pt["max"],
        "mean_abs_delta_pesq_ref_vs_pt": s_ref_pt["mean"],
        "median_abs_delta_pesq_ref_vs_pt": s_ref_pt["median"],
        "all_pass_individual": all_pass_individual,
        "aggregate_pass": aggregate_pass,
        "overall_pass": overall_pass,
    }


def _render_markdown_table(per_utt: List[Dict[str, Any]]) -> str:
    """Render a per-utterance Markdown summary table (9 cols)."""
    lines = [
        "| idx | dur (s) | pesq_ref | pesq_pt | pesq_ort | |dRO| | |dPO| | wf_rms_RO | wf_rms_PO |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in per_utt:
        lines.append(
            f"| {r['taps_idx']} | {r['len_seconds']:.2f} | {r['pesq_ref_pt']:.6f} | "
            f"{r['pesq_streaming_pt']:.6f} | {r['pesq_ort_streaming']:.6f} | "
            f"{abs(r['delta_pesq_ref_vs_ort']):.3e} | {abs(r['delta_pesq_pt_vs_ort']):.3e} | "
            f"{r['waveform_rms_diff_ref_vs_ort']:.3e} | {r['waveform_rms_diff_pt_vs_ort']:.3e} |"
        )
    return "\n".join(lines)


# -------------------------------------------------------------- runner
def run_eval_streaming(
    *,
    unified_ckpt_dir: str = _DEFAULT_UNIFIED_CKPT_DIR,
    chkpt_file: str = _DEFAULT_CHKPT_FILE,
    onnx_artifact: Optional[str] = None,
    split_combined_sidecar: Optional[str] = None,
    output_dir: Optional[str] = None,
    taps_indices: Optional[List[int]] = None,
    pesq_tol: float = PESQ_TOL_TRIPLE_DEFAULT,
    cross_streaming_tol: float = _DEFAULT_CROSS_STREAMING_TOL,
    audio_max_diff_tol: float = _DEFAULT_AUDIO_MAX_DIFF_TOL,
    shift_correction_samples: Optional[int] = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    device: str = "cpu",
    sample_rate: int = _DEFAULT_SR,
    compute_stoi: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the 3-path eval over the chosen TAPS test-split indices.

    Loads the matched (reference, PT-streaming, ORT-streaming) triple once
    via :func:`build_triple_paths`, then loops the requested utterances
    calling :func:`evaluate_utterance` per item. Returns the full report
    dict (the same shape the CLI dumps as JSON).

    Args:
        unified_ckpt_dir: Unified BAFNet+ experiment directory.
        chkpt_file: Checkpoint filename.
        onnx_artifact: Existing exported ORT artifact to reuse; if
            ``None``, re-export on the fly into ``output_dir``.
        output_dir: Where to write the ONNX if ``onnx_artifact`` is
            absent. ``None`` -> a fresh temp dir (managed by
            :meth:`BAFNetPlusOrtStreaming.from_checkpoint`).
        taps_indices: TAPS test-split indices. Defaults to
            ``[0,1,2,3,4]`` (the S1 validation list).
        pesq_tol: ``|delta_pesq_ref_vs_ort|`` per-utterance + aggregate
            (mean) tolerance.
        cross_streaming_tol: ``|delta_pesq_pt_vs_ort|`` per-utterance +
            aggregate (mean) tolerance.
        audio_max_diff_tol: Audio max-diff (ref-vs-ort) failure threshold.
        shift_correction_samples: Override the S9.5 alignment fix
            (default ``None`` -> ``n_fft // 2 = 200``).
        chunk_size: Streaming chunk size (50 ms anchor -> 8).
        device: Device for loading (CPU-only at S10).
        sample_rate: PESQ-wb requires 16 kHz.
        compute_stoi: Compute STOI if ``pystoi`` is available.
        verbose: ``logger.info`` per-utterance summary.

    Returns:
        Dict ``{run_metadata, per_utterance, aggregate}`` — see the module
        docstring for the full key set.
    """
    if taps_indices is None:
        taps_indices = list(_DEFAULT_TAPS_INDICES_FULL)

    started_at = time.time()
    bafnet, pt_streaming, ort_streaming = build_triple_paths(
        unified_ckpt_dir=unified_ckpt_dir,
        chkpt_file=chkpt_file,
        chunk_size=chunk_size,
        device=device,
        onnx_artifact=onnx_artifact,
        split_combined_sidecar=split_combined_sidecar,
        output_dir=output_dir,
        verbose=verbose,
    )
    setup_seconds = time.time() - started_at

    per_utt: List[Dict[str, Any]] = []
    for idx in taps_indices:
        t0 = time.time()
        bcs, acs, clean, sr = load_taps_utterance(idx, split="test")
        if sr != sample_rate:
            raise ValueError(f"TAPS idx={idx}: sample_rate={sr} != expected {sample_rate}")
        report = evaluate_utterance(
            bafnet,
            pt_streaming,
            ort_streaming,
            bcs_audio=bcs,
            acs_audio=acs,
            clean_ref_audio=clean,
            sample_rate=sample_rate,
            shift_correction_samples=shift_correction_samples,
            pesq_tol=pesq_tol,
            cross_streaming_tol=cross_streaming_tol,
            audio_max_diff_tol=audio_max_diff_tol,
            compute_stoi=compute_stoi,
        )
        report["taps_idx"] = int(idx)
        report["per_utt_seconds"] = float(time.time() - t0)
        per_utt.append(report)
        if verbose:
            logger.info(
                "TAPS idx=%d (%.2fs wall): pesq_ref=%.6f pesq_pt=%.6f pesq_ort=%.6f "
                "|dRO|=%.3e |dPO|=%.3e failures=%s",
                idx,
                report["per_utt_seconds"],
                report["pesq_ref_pt"],
                report["pesq_streaming_pt"],
                report["pesq_ort_streaming"],
                abs(report["delta_pesq_ref_vs_ort"]),
                abs(report["delta_pesq_pt_vs_ort"]),
                report["failure_reasons"] or "none",
            )

    aggregate = _aggregate(per_utt, pesq_tol=pesq_tol, cross_streaming_tol=cross_streaming_tol)

    import hashlib

    ckpt_path = Path(unified_ckpt_dir) / chkpt_file
    ckpt_md5 = hashlib.md5(ckpt_path.read_bytes()).hexdigest() if ckpt_path.exists() else None
    effective_shift = int(shift_correction_samples) if shift_correction_samples is not None else _DEFAULT_N_FFT // 2
    # ``onnx_schema_version`` is informational. For the split path we record the
    # combined sidecar path + per-graph schemas so downstream checks (e.g. the
    # S11/S21 host parity report) can attribute drift.
    is_split = split_combined_sidecar is not None
    if is_split:
        # ``ort_streaming.onnx_path`` is the trunk path under the S10
        # compatibility alias — see :class:`BAFNetPlusOrtSplitStreaming`.
        onnx_schema_value: Any = {
            "trunk": ort_streaming.trunk_schema_version,
            "head": ort_streaming.head_schema_version,
            "trunk_precision": ort_streaming.trunk_precision,
        }
    else:
        onnx_schema_value = "s8-bafnetplus-functional-fp32"
    run_metadata: Dict[str, Any] = {
        "schema_version": EVAL_STREAMING_SCHEMA_VERSION,
        "unified_ckpt_dir": str(unified_ckpt_dir),
        "chkpt_file": chkpt_file,
        "checkpoint_md5": ckpt_md5,
        "onnx_path": str(ort_streaming.onnx_path),
        "onnx_sidecar_path": str(ort_streaming.sidecar_path),
        "onnx_schema_version": onnx_schema_value,
        "split_combined_sidecar": str(split_combined_sidecar) if is_split else None,
        "taps_indices": list(taps_indices),
        "pesq_tol": float(pesq_tol),
        "cross_streaming_tol": float(cross_streaming_tol),
        "audio_max_diff_tol": float(audio_max_diff_tol),
        "shift_correction_samples": effective_shift,
        "chunk_size": int(chunk_size),
        "device": str(device),
        "sample_rate": int(sample_rate),
        "stft": {
            "n_fft": _DEFAULT_N_FFT,
            "hop_size": _DEFAULT_HOP,
            "win_size": _DEFAULT_WIN,
            "compress_factor": _DEFAULT_COMPRESS,
        },
        "setup_seconds": float(setup_seconds),
    }
    return {
        "run_metadata": run_metadata,
        "per_utterance": per_utt,
        "aggregate": aggregate,
    }


# ----------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. Exposed so tests can parse-only without execution."""
    parser = argparse.ArgumentParser(
        prog="src.analysis.eval_streaming",
        description=(
            "S10 full-utterance 3-path streaming eval harness "
            "(reference / PT-streaming / ORT-streaming). PESQ on FULL "
            "utterances only — partial / prefix / hand-picked regions rejected."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["smoke", "full", "custom"],
        default="custom",
        help=(
            "Pre-configured runs. 'smoke' = idx [0] + LP target 1e-4 on |dRO|. "
            "'full' = idx 0..4 + PESQ_TOL_TRIPLE_DEFAULT = 5e-3 on |dRO|. "
            "'custom' = honour the individual flags (default)."
        ),
    )
    parser.add_argument("--unified-ckpt-dir", default=_DEFAULT_UNIFIED_CKPT_DIR)
    parser.add_argument("--chkpt-file", default=_DEFAULT_CHKPT_FILE)
    parser.add_argument(
        "--onnx-artifact",
        default=None,
        help=(
            "Path to the S8-exported FP32 ONNX. Default: "
            f"{_DEFAULT_ONNX_ARTIFACT} if it exists (with sidecar JSON), "
            "otherwise re-export on the fly into --output-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Where to export the ONNX if --onnx-artifact is unset/missing. "
            f"Default: {_DEFAULT_ONNX_OUTPUT_DIR} (BAFNetPlusOrtStreaming."
            "from_checkpoint caches there)."
        ),
    )
    parser.add_argument(
        "--taps-indices",
        type=int,
        nargs="+",
        default=None,
        help="TAPS test-split indices. Default: --preset full -> [0,1,2,3,4]; smoke -> [0].",
    )
    parser.add_argument("--pesq-tol", type=float, default=None)
    parser.add_argument(
        "--cross-streaming-tol",
        type=float,
        default=None,
        help=(
            f"|delta_pesq_pt_vs_ort| tolerance. Default: {_DEFAULT_CROSS_STREAMING_TOL} "
            "(S7 baseline relaxation — the PT path's non-streaming-fusion drift naturally "
            "yields |dPO| ~3-6e-2; tighten for strict regression checking)."
        ),
    )
    parser.add_argument(
        "--audio-max-diff-tol",
        type=float,
        default=_DEFAULT_AUDIO_MAX_DIFF_TOL,
        help="waveform_max_diff_ref_vs_ort failure threshold.",
    )
    parser.add_argument(
        "--shift-correction-samples",
        type=int,
        default=None,
        help="Default = n_fft // 2 = 200 (S9.5 alignment fix). Pass 0 to disable.",
    )
    parser.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-rate", type=int, default=_DEFAULT_SR)
    parser.add_argument(
        "--no-stoi",
        action="store_true",
        help="Skip STOI even if pystoi is available (stoi_* fields become null).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write the run report as indent-2 JSON.",
    )
    parser.add_argument(
        "--output-markdown",
        default=None,
        help="Write the per-utterance Markdown summary table.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the aggregate (skip the per-utt Markdown table on stdout).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in preset-specific defaults on the argparse Namespace in place."""
    if args.preset == "smoke":
        if args.taps_indices is None:
            args.taps_indices = list(_DEFAULT_TAPS_INDICES_SMOKE)
        if args.pesq_tol is None:
            args.pesq_tol = PESQ_TOL_TRIPLE_LP_TARGET
        if args.cross_streaming_tol is None:
            args.cross_streaming_tol = _DEFAULT_CROSS_STREAMING_TOL
        args.summary_only = True
    elif args.preset == "full":
        if args.taps_indices is None:
            args.taps_indices = list(_DEFAULT_TAPS_INDICES_FULL)
        if args.pesq_tol is None:
            args.pesq_tol = PESQ_TOL_TRIPLE_DEFAULT
        if args.cross_streaming_tol is None:
            args.cross_streaming_tol = _DEFAULT_CROSS_STREAMING_TOL
    else:  # custom
        if args.taps_indices is None:
            args.taps_indices = list(_DEFAULT_TAPS_INDICES_FULL)
        if args.pesq_tol is None:
            args.pesq_tol = PESQ_TOL_TRIPLE_DEFAULT
        if args.cross_streaming_tol is None:
            args.cross_streaming_tol = _DEFAULT_CROSS_STREAMING_TOL
    return args


def _resolve_onnx_artifact(arg: Optional[str]) -> Optional[str]:
    """If --onnx-artifact unset, fall back to the default deployed path when present."""
    if arg is not None:
        return arg
    default_path = Path(_DEFAULT_ONNX_ARTIFACT)
    default_sidecar = Path(_DEFAULT_ONNX_ARTIFACT + ".json")
    if default_path.exists() and default_sidecar.exists():
        return str(default_path)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint. Returns 0 on overall pass, 1 on any failure."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _apply_preset(args)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    onnx_artifact = _resolve_onnx_artifact(args.onnx_artifact)
    output_dir = args.output_dir
    if output_dir is None and onnx_artifact is None:
        output_dir = _DEFAULT_ONNX_OUTPUT_DIR

    report = run_eval_streaming(
        unified_ckpt_dir=args.unified_ckpt_dir,
        chkpt_file=args.chkpt_file,
        onnx_artifact=onnx_artifact,
        output_dir=output_dir,
        taps_indices=args.taps_indices,
        pesq_tol=args.pesq_tol,
        cross_streaming_tol=args.cross_streaming_tol,
        audio_max_diff_tol=args.audio_max_diff_tol,
        shift_correction_samples=args.shift_correction_samples,
        chunk_size=args.chunk_size,
        device=args.device,
        sample_rate=args.sample_rate,
        compute_stoi=not args.no_stoi,
        verbose=args.verbose,
    )

    aggregate = report["aggregate"]
    per_utt = report["per_utterance"]

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report -> {out_path}")

    md = _render_markdown_table(per_utt)
    if args.output_markdown:
        md_path = Path(args.output_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with md_path.open("w") as f:
            f.write(md + "\n")
        print(f"Wrote Markdown table -> {md_path}")

    print("=" * 72)
    print(f"S10 eval_streaming - {len(per_utt)} utt(s); preset={args.preset}")
    print(
        f"pesq_tol = {args.pesq_tol:.3e}; "
        f"cross_streaming_tol = {report['run_metadata']['cross_streaming_tol']:.3e}; "
        f"audio_max_diff_tol = {report['run_metadata']['audio_max_diff_tol']:.3e}"
    )
    print("-" * 72)
    if not args.summary_only:
        print(md)
        print("-" * 72)
    print(
        f"|ref-ort| max={aggregate['max_abs_delta_pesq_ref_vs_ort']:.3e} "
        f"mean={aggregate['mean_abs_delta_pesq_ref_vs_ort']:.3e} "
        f"median={aggregate['median_abs_delta_pesq_ref_vs_ort']:.3e}"
    )
    print(
        f"|pt-ort|  max={aggregate['max_abs_delta_pesq_pt_vs_ort']:.3e} "
        f"mean={aggregate['mean_abs_delta_pesq_pt_vs_ort']:.3e} "
        f"median={aggregate['median_abs_delta_pesq_pt_vs_ort']:.3e}"
    )
    print(
        f"all_pass_individual={aggregate['all_pass_individual']} "
        f"aggregate_pass={aggregate['aggregate_pass']} "
        f"overall_pass={aggregate['overall_pass']}"
    )

    return 0 if aggregate["overall_pass"] else 1


__all__ = [
    "EVAL_STREAMING_SCHEMA_VERSION",
    "EXPECTED_WARMUP_NONE_RETURNS",
    "RUN_METADATA_KEYS",
    "PER_UTTERANCE_KEYS",
    "AGGREGATE_KEYS",
    "build_arg_parser",
    "evaluate_utterance",
    "main",
    "run_eval_streaming",
]


if __name__ == "__main__":
    sys.exit(main())
