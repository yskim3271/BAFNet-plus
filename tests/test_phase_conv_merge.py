"""Tests for the Path E-4 surgery ``merge_phase_conv_ri_inplace``.

Covers:

(a) ``test_merge_numerical_equivalence_fp32`` — at the
    :class:`ExportableBackboneCore` level, the merged forward (single
    ``phase_conv_ri`` + channel slice) is bit-identical to the original two
    parallel 1×1 Convs on a random input. FP32 linear algebra guarantees ULP
    equivalence; we assert ``max|d|`` below 1e-6.

(b) ``test_state_count_unchanged`` — surgery does not change ``num_states``;
    the dead ``phase_conv_r`` / ``phase_conv_i`` modules remain attached but
    contribute no functional state.

(c) ``test_double_apply_raises`` — a second call after surgery raises, and
    the auxiliary guard for missing submodules raises with a clear message.

(d) ``test_trunk_level_canonical_parity`` — at the
    :class:`BAFNetPlusTrunkCore` level (both branches' cores merged), the
    trunk → head chain still matches the canonical full-graph atan2 reference
    within libm precision (same tolerance as ``test_split_chain_matches_
    canonical_synthetic``).

(e) ``test_onnx_graph_drops_phase_conv_r_i`` — after exporting the
    merged trunk to FP32 ONNX, the graph contains a ``phase_conv_ri`` node and
    no ``phase_conv_r`` / ``phase_conv_i`` nodes for either branch.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import torch

from src.checkpoint import ConfigDict
from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.onnx.backbone_core import ExportableBackboneCore
from src.models.streaming.onnx.bafnetplus_core import (
    BAFNetPlusHeadCore,
    BAFNetPlusTrunkCore,
    ExportableBAFNetPlusCore,
)
from src.models.streaming.onnx.export import export_bafnetplus_trunk_to_onnx
from src.models.streaming.onnx.phase_conv_merge import merge_phase_conv_ri_inplace
from src.stft import complex_to_mag_pha


# --- 50 ms anchor + tolerances (matches test_bafnetplus_split_core.py) ---
CHUNK_SIZE = 8
N_FFT, HOP, WIN, COMPRESS = 400, 100, 400, 0.3
FREQ_SIZE = N_FFT // 2 + 1  # 201

# Libm precision floor on the per-branch sqrt-based reconstruction + downstream
# alpha conv + softmax + final atan2 (same bounds as the v1 split-core test).
MAG_TOL_SPLIT = 1e-4
COM_TOL_SPLIT = 1e-4
PHA_TOL_SPLIT = 1e-3


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


def _wrapped_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d)).abs()


# ===================================================================== (a) numerical equivalence
@pytest.mark.parametrize("phase_output_mode", ["atan2", "complex"])
def test_merge_numerical_equivalence_fp32(phase_output_mode):
    """Merged forward ≡ original (bit-identical FP32) on random input.

    The merged ``phase_conv_ri`` weight is the row-concatenation of the
    originals; ``Conv(x) | Conv(x) | Slice`` ≡ ``[Conv_r(x); Conv_i(x)]`` in
    FP32 arithmetic up to ULP. We compare across **both** phase output modes
    (the merged forward routes through the same atan2 / complex branch).
    """
    bafnet = _synthetic_bafnetplus("full")

    # Two independent cores from the same backbone (deep-copies internally).
    core_orig = ExportableBackboneCore.from_backbone(
        bafnet.mapping, phase_output_mode=phase_output_mode
    ).eval()
    core_e4 = ExportableBackboneCore.from_backbone(
        bafnet.mapping, phase_output_mode=phase_output_mode
    ).eval()
    merge_phase_conv_ri_inplace(core_e4)

    torch.manual_seed(0)
    T = 32
    mag = torch.randn(1, FREQ_SIZE, T) * 0.5
    pha = torch.randn(1, FREQ_SIZE, T)
    states_orig = core_orig.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T)
    states_e4 = core_e4.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T)

    with torch.no_grad():
        out_orig = core_orig(mag, pha, *states_orig)
        out_e4 = core_e4(mag, pha, *states_e4)

    assert len(out_orig) == len(out_e4), (
        f"output tuple length mismatch: {len(out_orig)} vs {len(out_e4)}"
    )
    # Bit-identical on every output (audio + state). FP32 ULP bound.
    for k, (o, e) in enumerate(zip(out_orig, out_e4)):
        d = (o - e).abs().max().item()
        assert math.isfinite(d) and d < 1e-6, (
            f"phase_output_mode={phase_output_mode}, output[{k}] |d|={d:.3e} >= 1e-6"
        )


# ===================================================================== (b) state count
def test_state_count_unchanged():
    """``num_states`` and state names are unchanged by the surgery.

    ``phase_conv_r`` / ``phase_conv_i`` are stateless ``nn.Conv2d``; the new
    ``phase_conv_ri`` is also stateless. So neither the count nor the DFS
    order of functional stateful convs should shift.
    """
    bafnet = _synthetic_bafnetplus("full")
    core_orig = ExportableBackboneCore.from_backbone(bafnet.mapping)
    core_e4 = ExportableBackboneCore.from_backbone(bafnet.mapping)
    merge_phase_conv_ri_inplace(core_e4)

    assert core_orig.num_states == core_e4.num_states
    assert core_orig.get_state_names() == core_e4.get_state_names()


# ===================================================================== (c) guards
def test_double_apply_raises():
    """A second ``merge_phase_conv_ri_inplace`` call raises."""
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBackboneCore.from_backbone(bafnet.mapping)
    merge_phase_conv_ri_inplace(core)
    with pytest.raises(ValueError, match="already"):
        merge_phase_conv_ri_inplace(core)


def test_missing_phase_conv_raises():
    """Missing ``phase_conv_r`` / ``phase_conv_i`` surfaces a clear error."""
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBackboneCore.from_backbone(bafnet.mapping)
    # Remove one of the expected submodules to simulate an incompatible core.
    del core.phase_decoder._modules["phase_conv_i"]
    with pytest.raises(ValueError, match="phase_conv_r.*phase_conv_i"):
        merge_phase_conv_ri_inplace(core)


# ===================================================================== (d) trunk-level parity
def test_trunk_level_canonical_parity():
    """Merged trunk + canonical head ≡ canonical full graph within libm precision.

    Applies E-4 surgery to both ``mapping_core`` and ``masking_core`` inside a
    :class:`BAFNetPlusTrunkCore`, chains it with the canonical head, and
    asserts the (est_mag, est_pha, est_com) triple equals the canonical
    :class:`ExportableBAFNetPlusCore` (atan2 mode) reference on a full-sequence
    spectrogram within the same tolerance as the v1 split-core parity test.
    """
    bafnet = _synthetic_bafnetplus("full")

    canonical = ExportableBAFNetPlusCore.from_bafnetplus(bafnet, phase_output_mode="atan2").eval()
    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet).eval()
    head = BAFNetPlusHeadCore.from_bafnetplus(bafnet).eval()
    # E-4 surgery on each branch's core.
    merge_phase_conv_ri_inplace(trunk.mapping_core)
    merge_phase_conv_ri_inplace(trunk.masking_core)

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
    assert math.isfinite(dmag) and dmag < MAG_TOL_SPLIT, f"|dmag|={dmag:.3e} >= {MAG_TOL_SPLIT}"
    assert math.isfinite(dcom) and dcom < COM_TOL_SPLIT, f"|dcom|={dcom:.3e} >= {COM_TOL_SPLIT}"
    assert math.isfinite(dpha) and dpha < PHA_TOL_SPLIT, f"|dpha_wrap|={dpha:.3e} >= {PHA_TOL_SPLIT}"


# ===================================================================== (e) ONNX dead-code
def test_onnx_graph_drops_phase_conv_r_i():
    """Exported ONNX has ``phase_conv_ri`` nodes and no ``phase_conv_r`` /
    ``phase_conv_i`` nodes (both branches).

    The original :class:`PhaseDecoder` submodules are left attached after
    surgery but become unreachable from the rebound ``_forward_phase_decoder``;
    ``torch.onnx.export`` traces only the live path so the dead modules are not
    represented in the graph.
    """
    onnx = pytest.importorskip("onnx")
    bafnet = _synthetic_bafnetplus("full")
    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet).eval()
    merge_phase_conv_ri_inplace(trunk.mapping_core)
    merge_phase_conv_ri_inplace(trunk.masking_core)

    T_export = CHUNK_SIZE + trunk.total_lookahead
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "trunk_e4.onnx"
        export_bafnetplus_trunk_to_onnx(
            core=trunk,
            output_path=out_path,
            chunk_size=CHUNK_SIZE,
            freq_size=FREQ_SIZE,
            time_frames=T_export,
        )
        model = onnx.load(str(out_path))

    node_names = [n.name for n in model.graph.node]
    # phase_conv_r / phase_conv_i should be absent on both branches.
    bad = [n for n in node_names if "phase_conv_r/" in n or "phase_conv_i/" in n]
    assert not bad, f"phase_conv_r/i nodes still present after E-4 surgery: {bad[:8]}"
    # phase_conv_ri should be present (at least one per branch — both mapping & masking).
    ri = [n for n in node_names if "phase_conv_ri" in n]
    assert len(ri) >= 2, (
        f"expected >=2 phase_conv_ri nodes (mapping + masking branches); got {len(ri)}: {ri[:8]}"
    )
