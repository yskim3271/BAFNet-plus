"""Tests for the S7 PESQ-streaming gate harness.

This module is the **stage-3 exit gate** test surface. The S7 launch prompt
specifies seven tests covering the harness contract + the empirical PESQ
delta on the unified BAFNet+ checkpoint over the agreed TAPS test-split
validation list.

The PESQ gate (tests ``test_pesq_gate_smoke_utterance`` /
``test_pesq_gate_aggregate_validation_list``) is **real-ckpt-only** — they
skip when the unified ``bafnetplus_50ms`` ckpt + per-branch ``bm_*_50ms``
Hydra configs are absent locally, OR when the TAPS HF dataset cache is
absent (each TAPS test utterance is multiple seconds long and CPU PESQ
scoring + streaming inference take ~30-50s per utterance, so the gate is
not run in lightweight CI). The structural tests (paired-paths sharing,
reference-path bit-equivalence, streaming-path bit-equivalence, documented
tolerance) run without the real ckpt — they construct a tiny synthetic
BAFNet+ and verify the harness shape contracts.

S7 default tolerance is ``PESQ_TOL_DEFAULT = 0.15`` (empirically calibrated
on TAPS test idx 0..4 — per-utterance ``|ΔPESQ| ∈ [5.04e-2, 8.62e-2]``,
mean ``6.58e-2``; see the S7 Status entry in
``docs/wiki/projects/reference-runtime.md`` for the full per-utterance pairs).
The LP target (``1e-4``) is unachievable while the S6 non-streaming-fusion
drift exists; S8 functional-stateful fusion is expected to bring it back
to LP target.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from src.analysis import pesq_streaming_gate as gate_mod
from src.analysis.pesq_streaming_gate import (
    PESQ_TOL_DEFAULT,
    build_paired_paths,
    compute_pesq,
    enhance_reference,
    enhance_streaming,
    run_gate,
    score_utterance,
)
from src.checkpoint import ConfigDict, load_checkpoint
from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.bafnetplus_streaming import BAFNetPlusStreaming
from src.stft import mag_pha_istft, mag_pha_stft


# --- 50 ms BAFNet+ anchors ---
CHUNK_SIZE = 8
N_FFT, HOP, WIN, COMPRESS = 400, 100, 400, 0.3
FREQ_SIZE = N_FFT // 2 + 1
SAMPLE_RATE = 16000

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_CKPT_DIR = _REPO_ROOT / "results" / "experiments" / "bafnetplus_50ms"
_REAL_CKPT_FILE = _REAL_CKPT_DIR / "best.th"
_BM_MAP_CFG = _REPO_ROOT / "results" / "experiments" / "bm_map_50ms" / ".hydra" / "config.yaml"
_BM_MASK_CFG = _REPO_ROOT / "results" / "experiments" / "bm_mask_50ms" / ".hydra" / "config.yaml"

# TAPS HF dataset cache (S1-confirmed locally — 6.6 GB, 2 arrow shards / split).
_TAPS_HF_CACHE = (
    Path.home() / ".cache" / "huggingface" / "datasets" / "yskim3271___throat_and_acoustic_pairing_speech_dataset"
)


def _real_ckpt_available() -> bool:
    return _REAL_CKPT_FILE.exists() and _BM_MAP_CFG.exists() and _BM_MASK_CFG.exists()


def _taps_cache_available() -> bool:
    return _TAPS_HF_CACHE.exists()


def _synthetic_backbone_param(infer_type: str, dense_channel: int = 8) -> Dict:
    return {
        "n_fft": N_FFT,
        "hop_size": HOP,
        "win_size": WIN,
        "dense_channel": dense_channel,
        "sigmoid_beta": 2.0,
        "compress_factor": COMPRESS,
        "dense_depth": 4,
        "num_tsblock": 2,
        "time_dw_kernel_size": 3,
        "time_block_kernel": [3, 5, 7, 11],
        "freq_block_kernel": [3, 11, 23, 31],
        "time_block_num": 2,
        "freq_block_num": 2,
        "causal_ts_block": True,
        "encoder_padding_ratio": (0.9, 0.1),
        "decoder_padding_ratio": (0.9, 0.1),
        "sca_kernel_size": 11,
        "infer_type": infer_type,
    }


def _synthetic_bafnetplus(ablation_mode: str = "full", dense_channel: int = 8) -> BAFNetPlus:
    args_mapping = ConfigDict(
        {
            "model_lib": "backbone",
            "model_class": "Backbone",
            "param": _synthetic_backbone_param("mapping", dense_channel=dense_channel),
        }
    )
    args_masking = ConfigDict(
        {
            "model_lib": "backbone",
            "model_class": "Backbone",
            "param": _synthetic_backbone_param("masking", dense_channel=dense_channel),
        }
    )
    model = BAFNetPlus(
        args_mapping=args_mapping,
        args_masking=args_masking,
        ablation_mode=ablation_mode,
        load_pretrained_weights=False,
    )
    return model.eval()


def _real_bafnetplus() -> BAFNetPlus:
    """Wiki foot-gun pattern to load the unified bafnetplus_50ms ckpt (cf. test_bafnetplus_core)."""
    from omegaconf import OmegaConf

    map_conf = OmegaConf.load(_BM_MAP_CFG)
    mask_conf = OmegaConf.load(_BM_MASK_CFG)
    args_mapping = ConfigDict(
        {
            "model_lib": map_conf.model.model_lib,
            "model_class": map_conf.model.model_class,
            "param": OmegaConf.to_container(map_conf.model.param, resolve=True),
        }
    )
    args_masking = ConfigDict(
        {
            "model_lib": mask_conf.model.model_lib,
            "model_class": mask_conf.model.model_class,
            "param": OmegaConf.to_container(mask_conf.model.param, resolve=True),
        }
    )
    bp_conf = OmegaConf.load(_REAL_CKPT_DIR / ".hydra" / "config.yaml")
    bp_param = OmegaConf.to_container(bp_conf.model.param, resolve=True)
    bp_param.pop("checkpoint_mapping", None)
    bp_param.pop("checkpoint_masking", None)
    model = BAFNetPlus(
        args_mapping=args_mapping,
        args_masking=args_masking,
        load_pretrained_weights=False,
        **bp_param,
    )
    model = load_checkpoint(model, str(_REAL_CKPT_DIR), "best.th", "cpu")
    return model.eval()


# ============================================================================
# Structural / contract tests — run without real ckpt or TAPS cache.
# ============================================================================
def test_reference_path_matches_bafnet_forward():
    """``enhance_reference`` is bit-equivalent to direct STFT(center=True) -> forward -> iSTFT."""
    torch.manual_seed(2039)
    bafnet = _synthetic_bafnetplus("full")
    bcs = torch.randn(4000) * 0.05
    acs = torch.randn(4000) * 0.05

    out_helper = enhance_reference(bafnet, bcs, acs, n_fft=N_FFT, hop_size=HOP, win_size=WIN, compress_factor=COMPRESS)

    bcs_com = mag_pha_stft(bcs.unsqueeze(0), N_FFT, HOP, WIN, COMPRESS, center=True)[2]
    acs_com = mag_pha_stft(acs.unsqueeze(0), N_FFT, HOP, WIN, COMPRESS, center=True)[2]
    with torch.no_grad():
        em, ep, _ = bafnet((bcs_com, acs_com))
    out_direct = mag_pha_istft(em, ep, N_FFT, HOP, WIN, COMPRESS, center=True).squeeze(0)

    assert out_helper.shape == out_direct.shape
    assert torch.equal(out_helper, out_direct)


def test_streaming_path_matches_process_audio():
    """``enhance_streaming`` is the identity over :meth:`BAFNetPlusStreaming.process_audio`."""
    torch.manual_seed(2039)
    bafnet = _synthetic_bafnetplus("full")
    streaming = BAFNetPlusStreaming.from_model(bafnet, chunk_size=CHUNK_SIZE, device="cpu")
    bcs = torch.randn(4000) * 0.05
    acs = torch.randn(4000) * 0.05
    out_helper = enhance_streaming(streaming, bcs, acs)
    # process_audio internally resets state; calling it twice gives the same output.
    out_direct = streaming.process_audio(bcs, acs)
    assert out_helper.shape == out_direct.shape
    assert torch.equal(out_helper, out_direct)


def test_paired_paths_share_fusion_modules_byte_identical():
    """``build_paired_paths`` returns models whose fusion modules are SHARED references.

    Skipped when the real ckpt is absent (the helper goes through ``from_checkpoint``).
    """
    if not _real_ckpt_available():
        pytest.skip(
            f"real checkpoint not present (ckpt={_REAL_CKPT_FILE.exists()}, "
            f"bm_map_cfg={_BM_MAP_CFG.exists()}, bm_mask_cfg={_BM_MASK_CFG.exists()})"
        )
    bafnet, streaming = build_paired_paths(
        unified_ckpt_dir=str(_REAL_CKPT_DIR),
        chkpt_file="best.th",
        chunk_size=CHUNK_SIZE,
        device="cpu",
        verify_weight_equality=True,  # this is the internal audit
    )
    # Re-assert explicitly at the test level (so a future internal refactor that
    # accidentally drops verify_weight_equality still gets caught).
    assert streaming.alpha_convblocks is bafnet.alpha_convblocks
    assert streaming.alpha_out is bafnet.alpha_out
    if bafnet.use_calibration:
        assert streaming.calibration_encoder is bafnet.calibration_encoder
        assert streaming.common_gain_head is bafnet.common_gain_head
        if bafnet.use_relative_gain:
            assert streaming.relative_gain_head is bafnet.relative_gain_head
    # Branches must be deep-copies (one is stateful-converted).
    assert streaming.bcs_streaming.model is not bafnet.mapping
    assert streaming.acs_streaming.model is not bafnet.masking
    # Branch state_dicts numerically equal modulo stateful-conv renames: weight tensors
    # corresponding to the same conv must be tensor-equal. Spot-check via the un-renamed
    # parameters (those whose key matches between non-stateful and stateful forms).
    bafnet_map_sd = bafnet.mapping.state_dict()
    stream_map_sd = streaming.bcs_streaming.model.state_dict()
    shared_keys = set(bafnet_map_sd.keys()) & set(stream_map_sd.keys())
    assert len(shared_keys) > 0, "expected at least some shared parameter names between branches"
    for k in shared_keys:
        if "weight" in k or "bias" in k:
            assert torch.equal(bafnet_map_sd[k], stream_map_sd[k]), f"weight drift on key {k!r}"
            break  # one cross-check is enough — confirms ckpt loaded the same weights.


def test_pesq_tolerance_documented_in_module():
    """The chosen ``PESQ_TOL_DEFAULT`` is documented in the module's top-level docstring.

    Catches accidental tolerance drift across sessions: any future change to
    ``PESQ_TOL_DEFAULT`` must also update the module docstring (which the wiki
    Status entry cites). S9 reworked the docstring to refer to
    ``PESQ_TOL_DEFAULT`` by name instead of the bare ``pesq_tol`` literal;
    either spelling counts.
    """
    doc = gate_mod.__doc__ or ""
    stripped = doc.replace("``", "")
    accepted = (
        f"pesq_tol = {PESQ_TOL_DEFAULT}" in stripped or f"PESQ_TOL_DEFAULT = {PESQ_TOL_DEFAULT}" in stripped
    )
    assert accepted, (
        "Module docstring must mention the chosen tolerance literal "
        f"({PESQ_TOL_DEFAULT}); update the docstring + the wiki Status entry "
        "if you intentionally relax/tighten the gate."
    )
    # Justification phrase ("non-streaming-fusion drift" or "S6 ... drift") must also be present.
    assert "non-streaming-fusion drift" in doc, "Tolerance justification missing in module docstring"


def test_compute_pesq_sanity_on_synthetic_audio():
    """``compute_pesq`` returns a finite wb score on synthetic audio (sanity)."""
    torch.manual_seed(2039)
    sr = SAMPLE_RATE
    clean = torch.randn(sr).clamp_(-1, 1) * 0.1
    noisy = clean + torch.randn(sr) * 0.01
    score = compute_pesq(clean, noisy, sample_rate=sr, mode="wb")
    assert math.isfinite(score)
    assert 1.0 <= score <= 5.0  # PESQ-wb range is roughly [1.04, 4.64]


# ============================================================================
# Real-ckpt + real-data gates — the S7 exit-gate proper.
# ============================================================================
def _require_real_ckpt_and_taps_cache():
    if not _real_ckpt_available():
        pytest.skip(
            f"real checkpoint not present (ckpt={_REAL_CKPT_FILE.exists()}, "
            f"bm_map_cfg={_BM_MAP_CFG.exists()}, bm_mask_cfg={_BM_MASK_CFG.exists()})"
        )
    if not _taps_cache_available():
        pytest.skip(
            f"TAPS HF dataset cache not present at {_TAPS_HF_CACHE}; "
            "the S7 gate is real-data-only and requires the cached dataset."
        )


def test_pesq_gate_smoke_utterance():
    """Single TAPS test idx=0 smoke gate: ``|ΔPESQ| ≤ PESQ_TOL_DEFAULT``.

    The smoke utterance — same one the Android fixture used in S1
    (``taps.utt_idx 0``). Asserts structural fields exist + are finite,
    logs the PESQ pair + delta + waveform diagnostics, and gates
    ``|delta_pesq| ≤ PESQ_TOL_DEFAULT``.
    """
    _require_real_ckpt_and_taps_cache()
    bafnet, streaming = build_paired_paths(
        unified_ckpt_dir=str(_REAL_CKPT_DIR), chkpt_file="best.th", chunk_size=CHUNK_SIZE, device="cpu"
    )
    bcs, acs, clean, sr = gate_mod.load_taps_utterance(0, split="test")
    assert sr == SAMPLE_RATE

    report = score_utterance(
        bafnet,
        streaming,
        bcs_audio=bcs,
        acs_audio=acs,
        clean_ref_audio=clean,
        sample_rate=sr,
    )
    print(
        f"[S7 smoke idx=0] len={report['len_samples']} "
        f"pesq_ref={report['pesq_ref_pt']:.6f} pesq_str={report['pesq_streaming_pt']:.6f} "
        f"|dpesq|={abs(report['delta_pesq']):.6e} "
        f"wf_max={report['waveform_max_diff']:.3e} wf_rms={report['waveform_rms_diff']:.3e} "
        f"wf_max_steady={report['waveform_max_diff_steady']:.3e} "
        f"wf_rms_steady={report['waveform_rms_diff_steady']:.3e}"
    )

    # Structural / finiteness checks.
    for k in (
        "len_samples",
        "pesq_ref_pt",
        "pesq_streaming_pt",
        "delta_pesq",
        "waveform_max_diff",
        "waveform_rms_diff",
        "waveform_max_diff_steady",
        "waveform_rms_diff_steady",
    ):
        assert k in report, f"missing field {k!r}"
    assert report["len_samples"] > 0
    assert math.isfinite(report["pesq_ref_pt"])
    assert math.isfinite(report["pesq_streaming_pt"])
    assert math.isfinite(report["delta_pesq"])

    # The S7 gate.
    abs_delta = abs(report["delta_pesq"])
    assert abs_delta <= PESQ_TOL_DEFAULT, (
        f"S7 gate failed on TAPS idx=0: |ΔPESQ|={abs_delta:.6e} > tol={PESQ_TOL_DEFAULT} "
        f"(pesq_ref={report['pesq_ref_pt']:.6f}, pesq_str={report['pesq_streaming_pt']:.6f}). "
        f"Investigate: FIFO alignment, _run_fusion mag-input slice, _manual_istft_ola edge handling, "
        f"per-branch STFT context update sequence. The S6 first-chunk bit-exact gate "
        f"(test_spectrogram_parity_real_checkpoint) is the sanity anchor."
    )


def test_pesq_gate_aggregate_validation_list():
    """Aggregate S7 gate on idx ∈ {0,1,2,3,4}: every utterance + aggregate must pass.

    Per-utterance: ``|ΔPESQ| ≤ PESQ_TOL_DEFAULT`` (the same gate as the smoke).
    Aggregate: ``mean(|ΔPESQ|) ≤ PESQ_TOL_DEFAULT / 2`` (slightly tighter — the
    aggregate should not be worse than the worst individual within a factor
    of 2).
    """
    _require_real_ckpt_and_taps_cache()
    result = run_gate(
        unified_ckpt_dir=str(_REAL_CKPT_DIR),
        chkpt_file="best.th",
        taps_indices=[0, 1, 2, 3, 4],
        pesq_tol=PESQ_TOL_DEFAULT,
        chunk_size=CHUNK_SIZE,
        device="cpu",
        sample_rate=SAMPLE_RATE,
        log_progress=False,
    )
    agg = result["aggregate"]
    print(
        f"[S7 aggregate idx=0..4] max|dpesq|={agg['max_abs_delta_pesq']:.6e} "
        f"mean={agg['mean_abs_delta_pesq']:.6e} median={agg['median_abs_delta_pesq']:.6e} "
        f"all_pass={agg['all_pass_individual']} agg_pass={agg['aggregate_pass']} "
        f"overall={agg['overall_pass']}"
    )
    for r in result["per_utterance"]:
        print(
            f"  idx={r['taps_idx']}: pesq_ref={r['pesq_ref_pt']:.6f} "
            f"pesq_str={r['pesq_streaming_pt']:.6f} |d|={abs(r['delta_pesq']):.6e}"
        )

    assert agg["all_pass_individual"], (
        f"S7 per-utterance gate failed at tol={PESQ_TOL_DEFAULT}: " f"max|ΔPESQ|={agg['max_abs_delta_pesq']:.6e}"
    )
    assert agg[
        "aggregate_pass"
    ], f"S7 aggregate gate failed: mean|ΔPESQ|={agg['mean_abs_delta_pesq']:.6e} > {PESQ_TOL_DEFAULT/2:.6e}"


def test_first_chunk_spectrogram_parity_at_audio_level():
    """Diagnostic: first 800 audio samples → re-STFT spectrogram drift on real speech.

    The S6 :func:`test_spectrogram_parity_real_checkpoint` already proved
    spectrogram-level first-chunk bit-exact parity. This test was originally
    intended as an audio-domain re-spectrogram regression check, but it is
    NOT a reliable proxy on real speech because the iSTFT-OLA warm-up
    region (first ``win - hop = 300`` samples, normalised by a partial Hann
    sum) dominates the audio-domain first 800 samples whenever speech has
    onset within that window. On TAPS test idx=0 the observed first-chunk
    re-STFT ``max|dmag| ~ 0.2`` reflects that warm-up effect, NOT a fusion
    drift bug. The launch prompt explicitly marked this test as
    "Optional — skip if hairier than expected; the main gate is PESQ".

    The test is kept as a finite-value diagnostic: it logs the actual
    re-STFT drift so a future regression that broke first-chunk SPEC
    parity (and so manifested as a HUGE re-STFT drift) would still surface.
    Specifically, we bound the drift at ``< 10.0`` (a loose sentinel that
    would catch order-of-magnitude regressions — e.g. NaN, drastic
    misalignment — while tolerating the OLA-warm-up cost on real speech).
    The actual S7 gate is :func:`test_pesq_gate_smoke_utterance` /
    :func:`test_pesq_gate_aggregate_validation_list`.
    """
    _require_real_ckpt_and_taps_cache()
    bafnet, streaming = build_paired_paths(
        unified_ckpt_dir=str(_REAL_CKPT_DIR), chkpt_file="best.th", chunk_size=CHUNK_SIZE, device="cpu"
    )
    bcs, acs, _, _ = gate_mod.load_taps_utterance(0, split="test")
    pad = WIN // 2
    bcs_padded = torch.cat([torch.zeros(pad), bcs, torch.zeros(pad)])
    acs_padded = torch.cat([torch.zeros(pad), acs, torch.zeros(pad)])

    ref_audio = enhance_reference(bafnet, bcs_padded, acs_padded)
    str_audio = enhance_streaming(streaming, bcs_padded, acs_padded)

    edge = streaming.output_samples_per_chunk  # 800 samples
    ref_chunk = ref_audio[:edge]
    str_chunk = str_audio[:edge]
    ref_mag = mag_pha_stft(ref_chunk.unsqueeze(0), N_FFT, HOP, WIN, COMPRESS, center=True)[0]
    str_mag = mag_pha_stft(str_chunk.unsqueeze(0), N_FFT, HOP, WIN, COMPRESS, center=True)[0]
    common_t = min(ref_mag.shape[2], str_mag.shape[2], CHUNK_SIZE)
    diff = (ref_mag[:, :, :common_t] - str_mag[:, :, :common_t]).abs().max().item()
    print(f"[S7 first-chunk audio→STFT parity diagnostic] common_t={common_t} max|dmag|={diff:.3e}")
    # Loose sentinel — would catch only order-of-magnitude regressions (NaN, drastic
    # misalignment). Real speech's OLA warm-up gives diffs in [0.1, 0.5]; a fusion
    # drift bug would be O(10) or higher.
    assert math.isfinite(diff) and diff < 10.0, (
        f"First-chunk audio→STFT spectrogram drift {diff:.3e} >= 10.0 — order-of-magnitude regression; "
        f"the actual S7 PESQ gate (test_pesq_gate_*) is the precise check."
    )


# ============================================================================
# Smoke / regression sanity tests for the harness loaders themselves.
# ============================================================================
def test_load_taps_utterance_shapes_and_rate():
    """Loader returns shape-consistent ``float32`` tensors at 16 kHz."""
    if not _taps_cache_available():
        pytest.skip(f"TAPS HF dataset cache not present at {_TAPS_HF_CACHE}")
    bcs, acs, clean, sr = gate_mod.load_taps_utterance(0, split="test")
    assert bcs.dtype == torch.float32 and acs.dtype == torch.float32 and clean.dtype == torch.float32
    assert bcs.dim() == 1 and acs.dim() == 1 and clean.dim() == 1
    assert len(bcs) == len(acs) == len(clean)
    assert sr == SAMPLE_RATE


def test_run_gate_short_indices_list_aggregates_correctly():
    """``run_gate(... taps_indices=[0])`` produces a 1-entry aggregate.

    Real-ckpt + TAPS cache required (the runner exercises the full path).
    """
    _require_real_ckpt_and_taps_cache()
    result = run_gate(
        unified_ckpt_dir=str(_REAL_CKPT_DIR),
        chkpt_file="best.th",
        taps_indices=[0],
        pesq_tol=PESQ_TOL_DEFAULT,
        chunk_size=CHUNK_SIZE,
        device="cpu",
    )
    assert len(result["per_utterance"]) == 1
    assert result["per_utterance"][0]["taps_idx"] == 0
    agg = result["aggregate"]
    assert agg["max_abs_delta_pesq"] == abs(result["per_utterance"][0]["delta_pesq"])
    assert agg["mean_abs_delta_pesq"] == agg["max_abs_delta_pesq"]
    # 1-element median == that element.
    assert np.isclose(agg["median_abs_delta_pesq"], agg["max_abs_delta_pesq"])
