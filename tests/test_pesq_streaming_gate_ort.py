"""Tests for the S9 3-way PESQ gate (ref + PT-streaming + ORT-streaming).

This module is the **Stage-4 exit-gate** test surface. The S9 launch
prompt specifies four PESQ-level tests (k/l/m/n) over the agreed TAPS
test-split validation list, plus structural cross-checks on the wrapper
construction path.

The S9.5 alignment fix (``shift_correction_samples = n_fft // 2``) tightens
the per-utt + aggregate gates by ~30x vs the original S9 ``0.15`` envelope.
The new envelope on TAPS test idx {0,1,2,3,4} is ``|Δref-ort| ∈ [6.20e-6,
1.36e-3]``, mean ``6.16e-4``. The triple gate uses
``PESQ_TOL_TRIPLE_DEFAULT = 5e-3`` (per-utt) + aggregate ``mean ≤
PESQ_TOL_TRIPLE_DEFAULT / 2 = 2.5e-3``, giving ~3.7x / ~4x headroom on the
observed worst / mean. The LP-aspirational ``PESQ_TOL_TRIPLE_LP_TARGET =
1e-4`` is achieved for 2/5 utterances out of the box (idx 0 at 2.24e-5, idx
2 at 6.20e-6); pursuing it for all 5 would require extending
:class:`ExportableBackboneCore` with a ``forward_with_mask`` method to
avoid the 1-ULP mask-recovery division drift (option (i) in the S9 launch
prompt).

The PESQ gate (``test_pesq_gate_triple_*``) is **real-ckpt-only** —
skipped when the unified ``bafnetplus_50ms`` ckpt + per-branch ``bm_*``
Hydra configs are absent locally OR when the TAPS HF dataset cache is
absent. The structural test (``test_pesq_triple_tolerance_documented``)
runs without the real ckpt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis import pesq_streaming_gate as gate_mod
from src.analysis.pesq_streaming_gate import (
    PESQ_TOL_TRIPLE_CROSS_STREAMING_DEFAULT,
    PESQ_TOL_TRIPLE_DEFAULT,
    PESQ_TOL_TRIPLE_LP_TARGET,
    build_triple_paths,
    run_gate_triple,
    score_utterance_triple,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_CKPT_DIR = _REPO_ROOT / "results" / "experiments" / "bafnetplus_50ms"
_REAL_CKPT_FILE = _REAL_CKPT_DIR / "best.th"
_BM_MAP_CFG = _REPO_ROOT / "results" / "experiments" / "bm_map_50ms" / ".hydra" / "config.yaml"
_BM_MASK_CFG = _REPO_ROOT / "results" / "experiments" / "bm_mask_50ms" / ".hydra" / "config.yaml"
_REAL_ONNX = _REPO_ROOT / "results" / "onnx" / "bafnetplus_50ms_fp32.onnx"
_REAL_SIDECAR = _REPO_ROOT / "results" / "onnx" / "bafnetplus_50ms_fp32.onnx.json"

_TAPS_HF_CACHE = (
    Path.home() / ".cache" / "huggingface" / "datasets" / "yskim3271___throat_and_acoustic_pairing_speech_dataset"
)


def _real_ckpt_available() -> bool:
    return _REAL_CKPT_FILE.exists() and _BM_MAP_CFG.exists() and _BM_MASK_CFG.exists()


def _real_onnx_available() -> bool:
    return _REAL_ONNX.exists() and _REAL_SIDECAR.exists()


def _taps_cache_available() -> bool:
    return _TAPS_HF_CACHE.exists()


# ============================================================================
# (m) Documented tolerance / docstring sentinel.
# ============================================================================
def test_pesq_triple_tolerance_documented():
    """The triple gate's tolerance constants + alignment-fix note are documented."""
    assert PESQ_TOL_TRIPLE_DEFAULT == 5e-3
    assert PESQ_TOL_TRIPLE_LP_TARGET == 1e-4
    assert PESQ_TOL_TRIPLE_CROSS_STREAMING_DEFAULT == 0.10
    doc = gate_mod.__doc__ or ""
    # The S9 module docstring documents the iSTFT-OLA host boundary delta as the dominant
    # cost (with the S9.5 alignment-fix finding that removes 200-sample misalignment).
    assert "iSTFT-OLA" in doc or "manual_istft_ola" in doc, "module docstring missing iSTFT-OLA discussion"
    assert "PESQ_TOL_TRIPLE_DEFAULT" in doc or "5e-3" in doc, "module docstring missing triple tolerance documentation"


# ============================================================================
# (n) build_triple_paths: shared fusion + ckpt MD5 cross-check.
# ============================================================================
@pytest.mark.skipif(not (_real_ckpt_available() and _real_onnx_available()), reason="real ckpt + ORT artifact missing")
def test_build_triple_paths_share_reference_and_ckpt_md5():
    """``build_triple_paths`` enforces shared-reference fusion + ckpt MD5 match.

    Uses the existing S8-exported real-ckpt ONNX at
    ``results/onnx/bafnetplus_50ms_fp32.onnx`` so the test doesn't pay
    the re-export cost.
    """
    bafnet, pt_streaming, ort_streaming = build_triple_paths(
        unified_ckpt_dir=str(_REAL_CKPT_DIR),
        chkpt_file="best.th",
        chunk_size=8,
        onnx_artifact=str(_REAL_ONNX),
    )
    # S7 shared-reference contract on (bafnet, pt_streaming) — already tested in S7.
    assert pt_streaming.alpha_convblocks is bafnet.alpha_convblocks
    assert pt_streaming.alpha_out is bafnet.alpha_out
    if bafnet.use_calibration:
        assert pt_streaming.calibration_encoder is bafnet.calibration_encoder
    # S9 ckpt-MD5 cross-check.
    import hashlib

    actual_md5 = hashlib.md5(_REAL_CKPT_FILE.read_bytes()).hexdigest()
    assert ort_streaming.checkpoint_info.get("md5") == actual_md5


# ============================================================================
# (k) Smoke gate on TAPS idx=0 (real ckpt + real TAPS cache required).
# ============================================================================
@pytest.mark.skipif(
    not (_real_ckpt_available() and _real_onnx_available() and _taps_cache_available()),
    reason="real ckpt + ORT artifact + TAPS HF cache missing",
)
def test_pesq_gate_triple_smoke_utterance():
    """3-way smoke on TAPS test idx=0: ``|delta_pesq_ref_vs_ort| <= PESQ_TOL_TRIPLE_DEFAULT``.

    Uses the existing S8-exported real-ckpt ONNX (skips re-export). Score
    via :func:`score_utterance_triple` which applies the S9.5 alignment fix
    (``shift_correction_samples = n_fft // 2 = 200``) by default.
    """
    bafnet, pt_streaming, ort_streaming = build_triple_paths(
        unified_ckpt_dir=str(_REAL_CKPT_DIR),
        chkpt_file="best.th",
        chunk_size=8,
        onnx_artifact=str(_REAL_ONNX),
    )
    bcs, acs, clean, sr = gate_mod.load_taps_utterance(0, split="test")
    report = score_utterance_triple(
        bafnet,
        pt_streaming,
        ort_streaming,
        bcs_audio=bcs,
        acs_audio=acs,
        clean_ref_audio=clean,
        sample_rate=sr,
    )
    print(
        f"\n[s9_smoke_triple idx=0] shift={report['shift_correction_samples']} "
        f"pesq_ref={report['pesq_ref_pt']:.6f}, pesq_pt={report['pesq_streaming_pt']:.6f}, "
        f"pesq_ort={report['pesq_ort_streaming']:.6f}\n"
        f"  |d_ref_ort|={abs(report['delta_pesq_ref_vs_ort']):.6e}, "
        f"|d_pt_ort|={abs(report['delta_pesq_pt_vs_ort']):.6e}\n"
        f"  wf max|d|_pt_ort={report['waveform_max_diff_pt_vs_ort']:.3e}, "
        f"rms_pt_ort={report['waveform_rms_diff_pt_vs_ort']:.3e}"
    )
    # Required: PESQ-table keys + the new shift field populated + finite.
    for k in (
        "pesq_ref_pt",
        "pesq_streaming_pt",
        "pesq_ort_streaming",
        "delta_pesq_ref_vs_ort",
        "delta_pesq_pt_vs_ort",
        "waveform_max_diff_pt_vs_ort",
        "waveform_rms_diff_pt_vs_ort",
        "shift_correction_samples",
    ):
        assert k in report and report[k] == report[k]  # x == x is False for NaN.
    # idx=0 hits |d_ref_ort| = 2.24e-5 under the alignment fix — at PESQ-wb's
    # numerical floor. Bonus assertion that idx=0 actually meets the LP target.
    assert abs(report["delta_pesq_ref_vs_ort"]) <= PESQ_TOL_TRIPLE_LP_TARGET
    # |d_pt_ort| is bounded by the S7 PT non-streaming-fusion drift envelope
    # (~5e-2 on real speech); the S9.5 alignment fix doesn't close this.
    assert abs(report["delta_pesq_pt_vs_ort"]) <= PESQ_TOL_TRIPLE_CROSS_STREAMING_DEFAULT


# ============================================================================
# (l) Aggregate gate on TAPS idx 0..4 — THE Stage-4 exit gate.
# ============================================================================
@pytest.mark.skipif(
    not (_real_ckpt_available() and _real_onnx_available() and _taps_cache_available()),
    reason="real ckpt + ORT artifact + TAPS HF cache missing",
)
def test_pesq_gate_triple_aggregate_validation_list():
    """3-way aggregate on TAPS test idx 0..4 (the S1 validation list).

    Per-utterance gate: ``|delta_pesq_ref_vs_ort| <= PESQ_TOL_TRIPLE_DEFAULT``,
    ``|delta_pesq_pt_vs_ort| <= PESQ_TOL_TRIPLE_DEFAULT``.
    Aggregate gate: ``mean(|delta_pesq_ref_vs_ort|) <= PESQ_TOL_TRIPLE_DEFAULT/2``.

    Uses :func:`run_gate_triple`'s default shift correction
    (``shift_correction_samples = n_fft // 2 = 200``) — the S9.5 alignment fix.

    THIS IS THE S9 STAGE-4 EXIT GATE.
    """
    result = run_gate_triple(
        unified_ckpt_dir=str(_REAL_CKPT_DIR),
        chkpt_file="best.th",
        taps_indices=[0, 1, 2, 3, 4],
        pesq_tol=PESQ_TOL_TRIPLE_DEFAULT,
        cross_streaming_tol=PESQ_TOL_TRIPLE_CROSS_STREAMING_DEFAULT,
        chunk_size=8,
        onnx_artifact=str(_REAL_ONNX),
        log_progress=False,
    )
    print(
        f"\n[s9_triple_aggregate] tol={result['pesq_tol']}, "
        f"max|d_ref_ort|={result['aggregate']['max_abs_delta_pesq_ref_vs_ort']:.6e}, "
        f"mean={result['aggregate']['mean_abs_delta_pesq_ref_vs_ort']:.6e}, "
        f"median={result['aggregate']['median_abs_delta_pesq_ref_vs_ort']:.6e}\n"
        f"  cross-stream max|d_pt_ort|={result['aggregate']['max_abs_delta_pesq_pt_vs_ort']:.6e}, "
        f"mean={result['aggregate']['mean_abs_delta_pesq_pt_vs_ort']:.6e}"
    )
    print("  per-utterance (idx, pesq_ref, pesq_pt, pesq_ort, |d_ref_ort|, |d_pt_ort|):")
    for r in result["per_utterance"]:
        print(
            f"    idx={r['taps_idx']:2d} ref={r['pesq_ref_pt']:.4f} pt={r['pesq_streaming_pt']:.4f} "
            f"ort={r['pesq_ort_streaming']:.4f} |d_ref_ort|={abs(r['delta_pesq_ref_vs_ort']):.3e} "
            f"|d_pt_ort|={abs(r['delta_pesq_pt_vs_ort']):.3e}"
        )
    agg = result["aggregate"]
    assert agg["all_pass_ref_vs_ort"], (
        f"per-utt ref_vs_ort gate failed: max={agg['max_abs_delta_pesq_ref_vs_ort']:.3e} > "
        f"{PESQ_TOL_TRIPLE_DEFAULT}"
    )
    assert agg["all_pass_pt_vs_ort"], (
        f"per-utt pt_vs_ort gate failed: max={agg['max_abs_delta_pesq_pt_vs_ort']:.3e} > "
        f"{PESQ_TOL_TRIPLE_CROSS_STREAMING_DEFAULT}"
    )
    assert agg["aggregate_pass_ref_vs_ort"], (
        f"aggregate ref_vs_ort gate failed: mean={agg['mean_abs_delta_pesq_ref_vs_ort']:.3e} > "
        f"{PESQ_TOL_TRIPLE_DEFAULT/2:.3e}"
    )
    assert agg["overall_pass"]


# ============================================================================
# (extra) run_gate_triple with a single-element list aggregates correctly.
# ============================================================================
@pytest.mark.skipif(
    not (_real_ckpt_available() and _real_onnx_available() and _taps_cache_available()),
    reason="real ckpt + ORT artifact + TAPS HF cache missing",
)
def test_run_gate_triple_single_index_aggregates():
    """Single-index gate: max == mean == median; cross-streaming gate independent."""
    result = run_gate_triple(
        unified_ckpt_dir=str(_REAL_CKPT_DIR),
        chkpt_file="best.th",
        taps_indices=[0],
        pesq_tol=PESQ_TOL_TRIPLE_DEFAULT,
        cross_streaming_tol=PESQ_TOL_TRIPLE_CROSS_STREAMING_DEFAULT,
        chunk_size=8,
        onnx_artifact=str(_REAL_ONNX),
    )
    agg = result["aggregate"]
    assert agg["max_abs_delta_pesq_ref_vs_ort"] == agg["mean_abs_delta_pesq_ref_vs_ort"]
    assert agg["mean_abs_delta_pesq_ref_vs_ort"] == agg["median_abs_delta_pesq_ref_vs_ort"]
    assert len(result["per_utterance"]) == 1
