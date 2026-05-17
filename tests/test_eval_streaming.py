"""Tests for the S10 full-utterance streaming eval harness.

The CLI / aggregate test (``test_eval_streaming_cli_smoke``,
``test_eval_streaming_aggregate_matches_s9_envelope``) are real-ckpt + real
TAPS HF cache gated — they exercise the same path the deployed pipeline
will. The structural / contract tests run without any model.

Hard rule re-asserted by these tests: **PESQ is computed only on full,
trimmed-to-clean-ref utterances**. A partial / prefix-only / hand-picked
clean reference is rejected with :class:`ValueError`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from src.analysis import eval_streaming as eval_mod
from src.analysis.eval_streaming import (
    AGGREGATE_KEYS,
    EVAL_STREAMING_SCHEMA_VERSION,
    EXPECTED_WARMUP_NONE_RETURNS,
    PER_UTTERANCE_KEYS,
    RUN_METADATA_KEYS,
    _apply_preset,
    _compute_failure_reasons,
    _count_pt_warmup_none_returns,
    _validate_full_utterance,
    build_arg_parser,
)
from src.analysis.pesq_streaming_gate import (
    PESQ_TOL_TRIPLE_DEFAULT,
    PESQ_TOL_TRIPLE_LP_TARGET,
    build_triple_paths,
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
# (1) Partial-utterance PESQ rejection — the S10-spec'd hard contract.
# ============================================================================
def test_eval_streaming_harness_rejects_partial_utterance_pesq():
    """``_validate_full_utterance`` rejects clean refs shorter than the pair length.

    PESQ must run on the FULL trimmed-to-clean-ref utterance. Chunk subsets,
    prefix-only outputs, or hand-picked voiced regions cannot pass this gate.
    The harness raises ``ValueError`` with an explicit error message at the
    earliest possible point so a misuse fails fast.
    """
    bcs = torch.zeros(16000)  # 1 s
    acs = torch.zeros(16000)
    clean_partial = torch.zeros(8000)  # 0.5 s — prefix-only
    with pytest.raises(ValueError, match="full-utterance"):
        _validate_full_utterance(clean_partial, bcs, acs)

    # Exact-length clean ref is accepted.
    clean_full = torch.zeros(16000)
    assert _validate_full_utterance(clean_full, bcs, acs) == 16000

    # Longer clean is also accepted (the caller may have raw clean still
    # full-length; pair_len truncation happens downstream).
    clean_longer = torch.zeros(20000)
    assert _validate_full_utterance(clean_longer, bcs, acs) == 16000

    # Mismatched BCS / ACS — pair_len = min(BCS, ACS); clean must be at
    # least min(BCS, ACS).
    bcs_short = torch.zeros(8000)
    acs_long = torch.zeros(16000)
    assert _validate_full_utterance(torch.zeros(8000), bcs_short, acs_long) == 8000


# ============================================================================
# (2) failure_reasons computation — pure function, no model needed.
# ============================================================================
def test_eval_streaming_failure_reasons_populated_on_bad_input():
    """``_compute_failure_reasons`` populates every failing gate independently.

    Feed deliberately bad values into every gate; assert the resulting
    list mentions each violation. Pure-function test — the harness
    doesn't crash when many gates fail.
    """
    reasons = _compute_failure_reasons(
        n_warmup_none_returns=5,
        expected_warmup=2,
        output_trim_length=100,
        expected_output_trim_length=200,
        wf_max_diff_ref_vs_ort=1.0,
        audio_max_diff_tol=0.01,
        delta_pesq_ref_vs_ort=0.5,
        pesq_tol=0.01,
        delta_pesq_pt_vs_ort=0.5,
        cross_streaming_tol=0.01,
    )
    # All 5 gates fire.
    joined = " | ".join(reasons)
    assert "n_warmup_none_returns=5" in joined
    assert "output_trim_length=100" in joined
    assert "wf_max_diff_ref_vs_ort" in joined
    assert "|delta_pesq_ref_vs_ort|" in joined
    assert "|delta_pesq_pt_vs_ort|" in joined
    assert len(reasons) == 5

    # Healthy inputs -> empty.
    reasons_ok = _compute_failure_reasons(
        n_warmup_none_returns=2,
        expected_warmup=2,
        output_trim_length=200,
        expected_output_trim_length=200,
        wf_max_diff_ref_vs_ort=0.001,
        audio_max_diff_tol=0.05,
        delta_pesq_ref_vs_ort=1e-5,
        pesq_tol=5e-3,
        delta_pesq_pt_vs_ort=0.03,
        cross_streaming_tol=0.15,
    )
    assert reasons_ok == []


# ============================================================================
# (3) JSON schema keys frozen — protects downstream consumers.
# ============================================================================
def test_eval_streaming_json_schema_keys_frozen():
    """The harness's frozen output schema keys match the documented set.

    Downstream consumers (the S11 Android binder, future regression tools)
    rely on the JSON keys being stable. If a key is added or removed, the
    schema_version must be bumped and this test updated deliberately.
    """
    expected_run_metadata = {
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
    }
    assert set(RUN_METADATA_KEYS) == expected_run_metadata

    expected_per_utt = {
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
    }
    assert set(PER_UTTERANCE_KEYS) == expected_per_utt

    expected_aggregate = {
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
    }
    assert set(AGGREGATE_KEYS) == expected_aggregate

    # Schema version literal is frozen.
    assert EVAL_STREAMING_SCHEMA_VERSION == "s10-eval-streaming-fp32-v1"


# ============================================================================
# (4) Smoke preset uses LP target on |dRO|.
# ============================================================================
def test_eval_streaming_preset_smoke_uses_lp_target():
    """``--preset smoke`` resolves pesq_tol to ``PESQ_TOL_TRIPLE_LP_TARGET = 1e-4``."""
    parser = build_arg_parser()
    args = parser.parse_args(["--preset", "smoke"])
    _apply_preset(args)
    assert args.pesq_tol == PESQ_TOL_TRIPLE_LP_TARGET
    assert args.taps_indices == [0]
    assert args.summary_only is True

    args_full = parser.parse_args(["--preset", "full"])
    _apply_preset(args_full)
    assert args_full.pesq_tol == PESQ_TOL_TRIPLE_DEFAULT
    assert args_full.taps_indices == [0, 1, 2, 3, 4]

    # Explicit override wins over the preset default.
    args_override = parser.parse_args(["--preset", "full", "--pesq-tol", "1e-2"])
    _apply_preset(args_override)
    assert args_override.pesq_tol == 1e-2

    args_custom = parser.parse_args(["--preset", "custom", "--taps-indices", "3"])
    _apply_preset(args_custom)
    assert args_custom.taps_indices == [3]
    assert args_custom.pesq_tol == PESQ_TOL_TRIPLE_DEFAULT


# ============================================================================
# (5) warm-up count matches the S6 50 ms anchor contract (== 2). Real ckpt.
# ============================================================================
@pytest.mark.skipif(
    not (_real_ckpt_available() and _real_onnx_available()),
    reason="real ckpt + ORT artifact missing",
)
def test_eval_streaming_warmup_count_matches_S6_contract():
    """``_count_pt_warmup_none_returns`` returns 2 for the 50 ms anchor.

    Verifies the S6/S9 wrapper contract holds at the harness layer:
    process_samples returns ``None`` twice before producing the first
    matured 800-sample chunk for a 1-second sample of zero audio.
    """
    _, pt_streaming, _ort = build_triple_paths(
        unified_ckpt_dir=str(_REAL_CKPT_DIR),
        chkpt_file="best.th",
        chunk_size=8,
        onnx_artifact=str(_REAL_ONNX),
    )
    audio = torch.zeros(16000)
    n_none = _count_pt_warmup_none_returns(pt_streaming, audio, audio)
    assert n_none == EXPECTED_WARMUP_NONE_RETURNS == 2


# ============================================================================
# (6) Aggregate gate matches the S9.5 envelope on TAPS test idx 0..4 (heavy).
# ============================================================================
@pytest.mark.skipif(
    not (_real_ckpt_available() and _real_onnx_available() and _taps_cache_available()),
    reason="real ckpt + ORT artifact + TAPS HF cache missing",
)
def test_eval_streaming_aggregate_matches_s9_envelope():
    """End-to-end: the harness's aggregate |dRO| matches the S9.5 envelope.

    Per the S9.5 Status entry (TAPS test idx 0..4 with shift correction):
    ``|Δref-ort| max = 1.36e-3, mean = 6.16e-4, median = 4.67e-4``.

    Asserts conservative bounds with ~10% headroom on max + mean so a
    future ORT-numerical-noise tightening doesn't break the test. Uses
    permissive ``cross_streaming_tol = 0.15`` (S7 baseline) so the
    structural cross-streaming gate doesn't fire on the documented PT
    non-streaming-fusion drift.
    """
    report = eval_mod.run_eval_streaming(
        unified_ckpt_dir=str(_REAL_CKPT_DIR),
        chkpt_file="best.th",
        onnx_artifact=str(_REAL_ONNX),
        taps_indices=[0, 1, 2, 3, 4],
        pesq_tol=PESQ_TOL_TRIPLE_DEFAULT,
        cross_streaming_tol=0.15,
        compute_stoi=False,  # speed
        verbose=False,
    )
    agg = report["aggregate"]
    print(
        "\n[s10_aggregate] |ref-ort| "
        f"max={agg['max_abs_delta_pesq_ref_vs_ort']:.6e} "
        f"mean={agg['mean_abs_delta_pesq_ref_vs_ort']:.6e} "
        f"median={agg['median_abs_delta_pesq_ref_vs_ort']:.6e}; "
        f"|pt-ort| max={agg['max_abs_delta_pesq_pt_vs_ort']:.6e} "
        f"mean={agg['mean_abs_delta_pesq_pt_vs_ort']:.6e}"
    )
    # S9.5 envelope with a generous bound (max 1.5e-3 vs observed 1.36e-3,
    # mean 7e-4 vs observed 6.16e-4 — ~10% headroom on both).
    assert agg["max_abs_delta_pesq_ref_vs_ort"] <= 1.5e-3
    assert agg["mean_abs_delta_pesq_ref_vs_ort"] <= 7e-4
    # All five per-utt |dRO| pass the 5e-3 envelope.
    for r in report["per_utterance"]:
        assert abs(r["delta_pesq_ref_vs_ort"]) <= PESQ_TOL_TRIPLE_DEFAULT, (
            f"TAPS idx={r['taps_idx']}: |dRO|={abs(r['delta_pesq_ref_vs_ort']):.3e} "
            f"> tol {PESQ_TOL_TRIPLE_DEFAULT:.3e}"
        )
    # The aggregate (|dRO|) mean gate must pass at pesq_tol/2.
    assert agg["aggregate_pass"], (
        f"aggregate_pass=False; mean|dRO|={agg['mean_abs_delta_pesq_ref_vs_ort']:.3e} > "
        f"{PESQ_TOL_TRIPLE_DEFAULT/2:.3e} OR mean|dPO|={agg['mean_abs_delta_pesq_pt_vs_ort']:.3e} > 0.15"
    )
    # idx 0 and idx 2 hit the LP target (S9.5 finding).
    by_idx: Dict[int, Dict[str, Any]] = {r["taps_idx"]: r for r in report["per_utterance"]}
    assert abs(by_idx[0]["delta_pesq_ref_vs_ort"]) <= PESQ_TOL_TRIPLE_LP_TARGET
    assert abs(by_idx[2]["delta_pesq_ref_vs_ort"]) <= PESQ_TOL_TRIPLE_LP_TARGET


# ============================================================================
# (7) CLI smoke — subprocess invocation with --preset smoke.
# ============================================================================
@pytest.mark.skipif(
    not (_real_ckpt_available() and _real_onnx_available() and _taps_cache_available()),
    reason="real ckpt + ORT artifact + TAPS HF cache missing",
)
def test_eval_streaming_cli_smoke(tmp_path: Path):
    """End-to-end CLI smoke via ``subprocess.run``.

    Runs ``python -m src.analysis.eval_streaming --preset smoke`` on TAPS
    idx=0 (the LP target hits 2.24e-5 << 1e-4); asserts exit code 0 + a
    JSON file with the expected schema is written.
    """
    out_json = tmp_path / "smoke_run.json"
    cmd: List[str] = [
        sys.executable,
        "-m",
        "src.analysis.eval_streaming",
        "--preset",
        "smoke",
        "--unified-ckpt-dir",
        str(_REAL_CKPT_DIR),
        "--onnx-artifact",
        str(_REAL_ONNX),
        "--no-stoi",
        "--output-json",
        str(out_json),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert (
        result.returncode == 0
    ), f"smoke CLI exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert out_json.exists(), f"--output-json file not written: {result.stdout}\n{result.stderr}"
    report = json.loads(out_json.read_text())
    assert set(report.keys()) == {"run_metadata", "per_utterance", "aggregate"}
    assert report["run_metadata"]["schema_version"] == EVAL_STREAMING_SCHEMA_VERSION
    assert report["run_metadata"]["taps_indices"] == [0]
    assert report["run_metadata"]["pesq_tol"] == PESQ_TOL_TRIPLE_LP_TARGET
    assert len(report["per_utterance"]) == 1
    utt = report["per_utterance"][0]
    # Required schema keys.
    for k in PER_UTTERANCE_KEYS:
        assert k in utt, f"smoke JSON per-utt missing key {k!r}"
    # The LP target is met on idx=0.
    assert abs(utt["delta_pesq_ref_vs_ort"]) <= PESQ_TOL_TRIPLE_LP_TARGET
    # Warm-up + trim contracts hold.
    assert utt["n_warmup_none_returns"] == EXPECTED_WARMUP_NONE_RETURNS == 2
    assert utt["output_trim_length"] == utt["expected_output_trim_length"]
    # No failure reasons.
    assert utt["failure_reasons"] == [], utt["failure_reasons"]
    # Aggregate sanity.
    agg = report["aggregate"]
    assert set(agg.keys()) == set(AGGREGATE_KEYS)
    assert agg["overall_pass"] is True
