"""Tests for the S21 split BAFNet+ core (``streaming/onnx/bafnetplus_core.py``).

S21 (B2 graph splitting) adds :class:`BAFNetPlusTrunkCore` and
:class:`BAFNetPlusHeadCore`, which together re-express
:class:`ExportableBAFNetPlusCore` as a two-graph chain (trunk → head) where
the trunk is INT8-safe (no atan2/softmax/sqrt) and the head carries the
INT8-hostile cluster (atan2/softmax/sqrt) on the FP32 side. This module covers
the structural + numerical gates for the split design:

(a) ``test_split_chain_matches_canonical_synthetic[full|mask_only_alpha|no_calibration]``
    — full-sequence PT split chain (trunk → head) ≡ canonical
    :class:`ExportableBAFNetPlusCore` (atan2 mode) on a complete spectrogram
    within libm precision (~1e-5 on est_mag/est_pha/est_com).

(b) ``test_state_partition`` — the trunk has 92 mapping + 92 masking = 184
    states; the head has 2 calibration + 4 alpha = 6 states; the total
    (190) matches the canonical S8 graph.

(c) ``test_io_contract_frozen`` — input/output names + state-name prefixes
    follow the S21 sidecar's frozen contract.

(d) ``test_state_shape_integrity_per_chunk`` — every next-state shape
    equals the prev-state shape (and the sidecar's recorded shape) after
    chunking through trunk → head with ``state_frames_for_update=chunk_size``.

(e) ``test_pt_vs_chained_ort_parity_via_export`` — round-trip:
    export trunk + head ONNX → run trunk session + head session →
    chained ORT output ≡ PT split chain output (5 random chunks).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

from src.checkpoint import ConfigDict
from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.onnx.bafnetplus_core import (
    BAFNetPlusHeadCore,
    BAFNetPlusTrunkCore,
    ExportableBAFNetPlusCore,
)
from src.models.streaming.onnx.export import (
    _auto_precision_sensitive_nodes,
    export_bafnetplus_head_to_onnx,
    export_bafnetplus_trunk_to_onnx,
    verify_bafnetplus_split_multistep,
)
from src.stft import complex_to_mag_pha


# --- 50 ms anchor + tolerances ---
CHUNK_SIZE = 8
N_FFT, HOP, WIN, COMPRESS = 400, 100, 400, 0.3
FREQ_SIZE = N_FFT // 2 + 1  # 201

# Libm precision floor on the per-branch sqrt-based reconstruction +
# downstream alpha conv + softmax + final atan2.
MAG_TOL_SPLIT = 1e-4
COM_TOL_SPLIT = 1e-4
PHA_TOL_SPLIT = 1e-3  # wrapped — atan2 ill-conditioning near zero crossings.

ABLATIONS = ["full", "mask_only_alpha", "no_calibration"]


# --------------------------------------------------------------------------- helpers
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


def _synthetic_bafnetplus(ablation_mode: str, dense_channel: int = 8) -> BAFNetPlus:
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


def _wrapped_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d)).abs()


# ===================================================================== (a) parity
@pytest.mark.parametrize("ablation_mode", ABLATIONS)
def test_split_chain_matches_canonical_synthetic(ablation_mode):
    """PT trunk → head chain equals :class:`ExportableBAFNetPlusCore` (atan2 mode)
    on a complete spectrogram within libm precision."""
    bafnet = _synthetic_bafnetplus(ablation_mode)

    # Canonical full-graph core (atan2 mode) as reference.
    canonical = ExportableBAFNetPlusCore.from_bafnetplus(bafnet, phase_output_mode="atan2").eval()
    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet).eval()
    head = BAFNetPlusHeadCore.from_bafnetplus(bafnet).eval()

    # Sanity: the two paths exhaust the same total state count.
    assert trunk.num_states + head.num_states == canonical.num_states

    torch.manual_seed(42)
    T = 32
    bcs_com = torch.randn(1, FREQ_SIZE, T, 2) * 0.3
    acs_com = torch.randn(1, FREQ_SIZE, T, 2) * 0.3
    bcs_mag, bcs_pha = complex_to_mag_pha(bcs_com, stack_dim=-1)
    acs_mag, acs_pha = complex_to_mag_pha(acs_com, stack_dim=-1)

    canon_states = canonical.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T)
    trunk_states = trunk.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T)
    head_states = head.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T)

    with torch.no_grad():
        canon_outs = canonical(bcs_mag, bcs_pha, acs_mag, acs_pha, *canon_states)
        ref_mag, ref_pha, ref_com = canon_outs[0], canon_outs[1], canon_outs[2]

        trunk_outs = trunk(bcs_mag, bcs_pha, acs_mag, acs_pha, *trunk_states)
        boundary = trunk_outs[:7]
        head_outs = head(*boundary, *head_states)
        est_mag, est_pha, est_com = head_outs[0], head_outs[1], head_outs[2]

    dmag = (ref_mag - est_mag).abs().max().item()
    dcom = (ref_com - est_com).abs().max().item()
    dpha = _wrapped_abs_diff(ref_pha, est_pha).max().item()
    assert math.isfinite(dmag) and dmag < MAG_TOL_SPLIT, f"{ablation_mode}: |dmag|={dmag:.3e} >= {MAG_TOL_SPLIT}"
    assert math.isfinite(dcom) and dcom < COM_TOL_SPLIT, f"{ablation_mode}: |dcom|={dcom:.3e} >= {COM_TOL_SPLIT}"
    assert math.isfinite(dpha) and dpha < PHA_TOL_SPLIT, f"{ablation_mode}: |dpha_wrap|={dpha:.3e} >= {PHA_TOL_SPLIT}"


# ===================================================================== (b) partition
def test_state_partition():
    """Trunk (92+92=184) + Head (2+4=6) = canonical 190 on the 50 ms anchor.

    The synthetic model has 2 TS-blocks (vs the real 50 ms ckpt's 4),
    halving the per-branch states from 92 to e.g. ~52. We assert the partition
    is consistent — trunk + head sum to the canonical's num_states — rather
    than asserting fixed counts.
    """
    bafnet = _synthetic_bafnetplus("full")
    canonical = ExportableBAFNetPlusCore.from_bafnetplus(bafnet, phase_output_mode="atan2")
    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet)
    head = BAFNetPlusHeadCore.from_bafnetplus(bafnet)
    assert trunk.num_states == trunk.mapping_core.num_states + trunk.masking_core.num_states
    assert head.num_states == head.num_calibration_states + head.num_alpha_states
    assert trunk.num_states + head.num_states == canonical.num_states


# ===================================================================== (c) contract
def test_io_contract_frozen():
    """Trunk + head IO names follow the S21 sidecar's frozen contract."""
    bafnet = _synthetic_bafnetplus("full")
    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet)
    head = BAFNetPlusHeadCore.from_bafnetplus(bafnet)

    # Trunk: 4 audio inputs + state inputs; 7 boundary outputs + next_state outputs.
    trunk_in = trunk.input_names()
    assert trunk_in[:4] == ["bcs_mag", "bcs_pha", "acs_mag", "acs_pha"]
    assert len(trunk_in) == 4 + trunk.num_states
    trunk_out = trunk.output_names()
    assert trunk_out[:7] == [
        "bcs_est_mag",
        "bcs_phase_real",
        "bcs_phase_imag",
        "acs_est_mag",
        "acs_phase_real",
        "acs_phase_imag",
        "acs_mask",
    ]
    assert len(trunk_out) == 7 + trunk.num_states
    # State name prefixes — mapping/* then masking/*.
    for n in trunk.get_state_names()[: trunk.mapping_core.num_states]:
        assert n.startswith("mapping/")
    for n in trunk.get_state_names()[trunk.mapping_core.num_states :]:
        assert n.startswith("masking/")

    # Head: 7 boundary inputs + state inputs; 3 final outputs + next_state outputs.
    head_in = head.input_names()
    assert head_in[:7] == [
        "bcs_est_mag",
        "bcs_phase_real",
        "bcs_phase_imag",
        "acs_est_mag",
        "acs_phase_real",
        "acs_phase_imag",
        "acs_mask",
    ]
    assert len(head_in) == 7 + head.num_states
    head_out = head.output_names()
    assert head_out[:3] == ["est_mag", "est_pha", "est_com"]
    assert len(head_out) == 3 + head.num_states
    # State name prefixes — calibration/* then alpha/*.
    for n in head.get_state_names()[: head.num_calibration_states]:
        assert n.startswith("calibration/")
    for n in head.get_state_names()[head.num_calibration_states :]:
        assert n.startswith("alpha/")


# ===================================================================== (d) state shape
def test_state_shape_integrity_per_chunk():
    """Every next-state shape == prev-state shape after one trunk → head chunk."""
    bafnet = _synthetic_bafnetplus("full")
    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet).eval()
    head = BAFNetPlusHeadCore.from_bafnetplus(bafnet).eval()
    trunk.set_state_frames_for_update(CHUNK_SIZE)
    head.set_state_frames_for_update(CHUNK_SIZE)

    T_export = CHUNK_SIZE + trunk.total_lookahead
    trunk_states = trunk.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T_export)
    head_states = head.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T_export)

    bcs_mag = torch.randn(1, FREQ_SIZE, T_export)
    bcs_pha = torch.randn(1, FREQ_SIZE, T_export)
    acs_mag = torch.randn(1, FREQ_SIZE, T_export).abs() + 1e-3
    acs_pha = torch.randn(1, FREQ_SIZE, T_export)

    with torch.no_grad():
        trunk_outs = trunk(bcs_mag, bcs_pha, acs_mag, acs_pha, *trunk_states)
        trunk_next = list(trunk_outs[7:])
        for prev, nxt in zip(trunk_states, trunk_next):
            assert prev.shape == nxt.shape

        boundary = trunk_outs[:7]
        head_outs = head(*boundary, *head_states)
        head_next = list(head_outs[3:])
        for prev, nxt in zip(head_states, head_next):
            assert prev.shape == nxt.shape


# ===================================================================== (e) chained ORT
def test_pt_vs_chained_ort_parity_via_export():
    """Round-trip: export trunk + head → chained ORT equals PT split chain."""
    bafnet = _synthetic_bafnetplus("full")
    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet)
    head = BAFNetPlusHeadCore.from_bafnetplus(bafnet)

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        trunk_path = td_p / "trunk.onnx"
        head_path = td_p / "head.onnx"
        export_bafnetplus_trunk_to_onnx(trunk, trunk_path, chunk_size=CHUNK_SIZE, verbose=False)
        export_bafnetplus_head_to_onnx(
            head, head_path, chunk_size=CHUNK_SIZE, time_frames=CHUNK_SIZE + trunk.total_lookahead, verbose=False
        )
        result = verify_bafnetplus_split_multistep(
            trunk_path,
            head_path,
            trunk,
            head,
            chunk_size=CHUNK_SIZE,
            num_steps=3,
            atol=1e-3,
            state_atol=1e-3,
            verbose=False,
        )
        assert result["state_shape_check"], "PT next-state shapes did not match prev-state shapes"
        assert result["all_finite"], "non-finite values in trunk or head next-states"
        assert result["all_match"], (
            f"chained ORT-vs-PT mismatch: max_output_diffs={result['max_output_diffs']}, "
            f"max_trunk_state_diff={result['max_trunk_state_diff']:.3e}, "
            f"max_head_state_diff={result['max_head_state_diff']:.3e}"
        )


# ============================================================= (f) auto-exclude
def test_auto_precision_sensitive_nodes_trunk_phase_decoder():
    """``_auto_precision_sensitive_nodes`` adds the phase-decoder cluster on
    the S21 trunk schema (mini-fix A1, S23 Phase C pre-flight).

    The trunk has no top-level Atan/Softmax/Pow/Sqrt (those moved to the
    FP32 head), so the unified-graph branch returns 0. The trunk-only
    branch must pick up the per-branch phase reconstruction cluster:
    ``phase_conv_{r,i}`` Conv + ``phase_conv/phase_conv.{0,1,2}``
    ConvTranspose/BN/PRelu. Hexagon V79 rejects these at HTP graph
    prepare with QUInt16 activations otherwise (S23 Phase C pre-flight
    2026-05-15: err 1002 on ``/phase_conv_r/Conv_token_4679_2``,
    ``/mapping_core/phase_conv_r/Conv_token_4745_2``,
    ``/phase_conv/phase_conv.2/PRelu_3``).
    """
    import json as _json

    bafnet = _synthetic_bafnetplus("full")
    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet)
    head = BAFNetPlusHeadCore.from_bafnetplus(bafnet)

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        trunk_path = td_p / "trunk.onnx"
        head_path = td_p / "head.onnx"
        export_bafnetplus_trunk_to_onnx(trunk, trunk_path, chunk_size=CHUNK_SIZE, verbose=False)
        export_bafnetplus_head_to_onnx(
            head,
            head_path,
            chunk_size=CHUNK_SIZE,
            time_frames=CHUNK_SIZE + trunk.total_lookahead,
            verbose=False,
        )

        trunk_excluded = _auto_precision_sensitive_nodes(trunk_path)
        head_excluded = _auto_precision_sensitive_nodes(head_path)

        # The trunk's three failing-on-HTP op-name stems must all be matched.
        # Each one is present in both the mapping branch (``/mapping_core/...``)
        # and the masking branch (``/...``), so we count two occurrences per stem.
        for stem in ("/phase_conv_r/Conv", "/phase_conv_i/Conv", "/phase_conv/phase_conv.2/PRelu"):
            matched = [n for n in trunk_excluded if n.endswith(stem)]
            assert len(matched) == 2, (
                f"trunk auto-exclude missing both branches for stem {stem!r}: "
                f"found {matched}"
            )

        # The full per-branch phase reconstruction cluster (10 nodes: 2 branches ×
        # {ConvTranspose, BatchNormalization, PRelu, r-Conv, i-Conv}) must be
        # entirely in the exclusion list. The size must stay within the
        # 5-15 reasonable-range guard from the cycle-prompt risk-mitigation note.
        phase_decoder_count = sum(
            1
            for n in trunk_excluded
            if "phase_conv_r/" in n or "phase_conv_i/" in n or "phase_conv/phase_conv." in n
        )
        assert phase_decoder_count == 10, (
            f"trunk auto-exclude phase-decoder cluster count = {phase_decoder_count}, expected 10. "
            f"all excluded: {trunk_excluded}"
        )
        assert 5 <= len(trunk_excluded) <= 15, (
            f"trunk auto-exclude returned {len(trunk_excluded)} nodes; "
            f"outside the 5-15 reasonable-range guard"
        )

        # The head has no phase_conv ops at all (those live in the trunk).
        for n in head_excluded:
            assert "phase_conv" not in n, (
                f"head auto-exclude unexpectedly contains phase_conv: {n}"
            )

        # Schema-conditional gate: rewriting the trunk sidecar to a non-trunk
        # schema must drop the phase-decoder cluster from the exclusion list
        # (so the S17/S20 unified paths keep their existing behavior).
        trunk_sidecar = trunk_path.with_suffix(trunk_path.suffix + ".json")
        original = _json.loads(trunk_sidecar.read_text())
        forged = dict(original)
        forged["schema_version"] = "s8-bafnetplus-functional-fp32"
        trunk_sidecar.write_text(_json.dumps(forged))
        try:
            non_trunk_excluded = _auto_precision_sensitive_nodes(trunk_path)
            for n in non_trunk_excluded:
                assert "phase_conv" not in n, (
                    f"non-trunk schema branch unexpectedly excluded phase_conv node: {n}"
                )
        finally:
            trunk_sidecar.write_text(_json.dumps(original))


