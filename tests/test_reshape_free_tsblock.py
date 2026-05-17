"""Correctness tests for the reshape-free TSBlock — promoted to the only path
in cycle 13.

These tests pin the RF graph's numerical behaviour and state layout against
trusted PT references (the canonical non-streaming :class:`TSBlock` and the
1D per-F stateful conv). The cycle-11 ``default-vs-RF`` comparison tests were
retired together with the ``use_reshape_free_tsblock`` flag — the comparison
collapsed once the default-path was removed.

Covered cases:

1. ``test_tsblock_unit_parity`` — ``ReshapeFreeTSBlock`` vs canonical
   :class:`src.models.backbone.TSBlock` with the **canonical** algebra flags
   (``fold_residual_scale=False, post_norm=False``). FP32 ≤ 1e-5.
2. ``test_tsblock_unit_parity_algebra_variants`` — same module pair,
   parametrised over the four ``(fold_residual_scale, post_norm)`` combinations.
3. ``test_rf_state_layout`` — ``ExportableBackboneCore`` (RF as default) state
   count + per-state shape signature: 4 encoder DS_DDB (4D) + 40 TSBlock 4D
   ``[B, C, padding, F_enc]`` + 8 decoder DS_DDB (4D) for the test config
   (num_tsblock=2).
4. ``test_layernorm4dchannel_matches_layernorm1d`` — building-block unit test.
5. ``test_functional_stateful_conv2d_time_axis_matches_1d`` — building-block
   unit test (4D time-axis stateful conv ≡ per-F 1D stateful conv).
6. ``test_simple_gate_4d_matches_3d`` — building-block unit test.
"""

from __future__ import annotations

import zlib

import pytest
import torch

from src.models.backbone import Backbone, TSBlock
from src.models.streaming.onnx.backbone_core import ExportableBackboneCore
from src.models.streaming.onnx.reshape_free import (
    FunctionalStatefulConv2dTimeAxis,
    LayerNorm4dChannel,
    SimpleGate4d,
)
from src.models.streaming.onnx.reshape_free_tsblock import (
    convert_tsblock_to_reshape_free,
)

ATOL_FP32 = 1e-5


def _seed(name: str) -> None:
    torch.manual_seed(zlib.crc32(name.encode()))


def _make_canonical_tsblock(*, fold_residual_scale: bool = False, post_norm: bool = False) -> TSBlock:
    """Build a BAFNet+ canonical TSBlock (dense=64, kernels [3,5,7,9] / [3,11,23,31])."""
    return TSBlock(
        dense_channel=64,
        time_block_num=2,
        freq_block_num=2,
        time_dw_kernel_size=3,
        time_block_kernel=[3, 5, 7, 9],
        freq_block_kernel=[3, 11, 23, 31],
        causal=True,
        sca_kernel_size=11,
        fold_residual_scale=fold_residual_scale,
        post_norm=post_norm,
    ).eval()


def _make_canonical_backbone(*, num_tsblock: int = 2, infer_type: str = "masking") -> Backbone:
    """Build a Backbone matching the 50 ms BAFNet+ canonical config (smaller num_tsblock for test speed)."""
    return Backbone(
        n_fft=400,
        hop_size=100,
        win_size=400,
        dense_channel=64,
        sigmoid_beta=2.0,
        compress_factor=0.3,
        dense_depth=4,
        num_tsblock=num_tsblock,
        time_dw_kernel_size=3,
        time_block_kernel=[3, 5, 7, 9],
        freq_block_kernel=[3, 11, 23, 31],
        time_block_num=2,
        freq_block_num=2,
        causal_ts_block=True,
        encoder_padding_ratio=(0.8333, 0.1667),
        decoder_padding_ratio=(1.0, 0.0),
        sca_kernel_size=11,
        infer_type=infer_type,
    ).eval()


# ---------------------------------------------------------------------------
# (1) Single-block unit parity — canonical algebra
# ---------------------------------------------------------------------------


def test_tsblock_unit_parity():
    """Canonical TSBlock (fold_residual_scale=False, post_norm=False) parity ≤ 1e-5."""
    _seed("rf-unit-canonical")
    tsblock = _make_canonical_tsblock()
    rf = convert_tsblock_to_reshape_free(tsblock).eval()

    x = torch.randn(1, 64, 14, 100)
    with torch.no_grad():
        y_ref = tsblock(x)
        y_rf = rf(x)

    assert y_ref.shape == y_rf.shape == (1, 64, 14, 100)
    diff = (y_ref - y_rf).abs().max().item()
    assert diff <= ATOL_FP32, f"canonical TSBlock parity broken: max|Δ|={diff:.3e}"


# ---------------------------------------------------------------------------
# (2) Single-block unit parity — 4 algebra-flag combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fold_residual_scale", [False, True])
@pytest.mark.parametrize("post_norm", [False, True])
def test_tsblock_unit_parity_algebra_variants(fold_residual_scale, post_norm):
    """All four (fold_residual_scale, post_norm) combinations match within 1e-5."""
    _seed(f"rf-unit-{fold_residual_scale}-{post_norm}")
    tsblock = _make_canonical_tsblock(fold_residual_scale=fold_residual_scale, post_norm=post_norm)
    rf = convert_tsblock_to_reshape_free(tsblock).eval()

    x = torch.randn(1, 64, 14, 100)
    with torch.no_grad():
        y_ref = tsblock(x)
        y_rf = rf(x)
    diff = (y_ref - y_rf).abs().max().item()
    assert diff <= ATOL_FP32, (
        f"fold_residual_scale={fold_residual_scale} post_norm={post_norm}: "
        f"max|Δ|={diff:.3e}"
    )


# ---------------------------------------------------------------------------
# (3) RF state layout audit
# ---------------------------------------------------------------------------


def test_rf_state_layout():
    """``ExportableBackboneCore.from_backbone`` (RF as default) state contract.

    State order at num_tsblock=2 (52 total):
        4 dense_encoder DS_DDB (4D ``FunctionalStatefulConv2d``)
        40 sequence_block × 2 TSBlocks × 20 time-stage RF convs
           (4D ``[B, C, padding, F_enc=100]``)
        4 mask_decoder DS_DDB (4D)
        4 phase_decoder DS_DDB (4D)
    """
    _seed("rf-state-layout")
    backbone = _make_canonical_backbone()
    core = ExportableBackboneCore.from_backbone(backbone)

    assert core.num_states == 52, f"state count unexpected: {core.num_states}"

    shapes = core.get_state_shapes(batch_size=1, freq_size=201)

    enc_indices = list(range(4))
    ts_indices = list(range(4, 44))
    dec_indices = list(range(44, 52))

    for i in enc_indices + dec_indices:
        assert len(shapes[i]) == 4, f"DS_DDB state {i} should be 4D, got {shapes[i]}"

    for i in ts_indices:
        s = shapes[i]
        assert len(s) == 4, f"RF TSBlock state {i} should be 4D, got {s}"
        # Layout: [B=1, C, padding, F_enc=100].
        assert s[0] == 1, f"TSBlock state {i} batch dim should be 1, got {s}"
        assert s[3] == 100, f"TSBlock state {i} F_enc dim should be 100, got {s}"


# ---------------------------------------------------------------------------
# Internal unit checks for the 4D building blocks
# ---------------------------------------------------------------------------


def test_layernorm4dchannel_matches_layernorm1d():
    """``LayerNorm4dChannel`` on [B, C, T, F] ≡ ``LayerNorm1d`` on [B*F, C, T]."""
    _seed("ln4d")
    from src.models.backbone import LayerNorm1d

    C = 8
    ln1d = LayerNorm1d(C, eps=1e-6).eval()
    ln4d = LayerNorm4dChannel(C, eps=1e-6).eval()
    # Same weights/biases.
    ln4d.weight.data = ln1d.weight.data.clone()
    ln4d.bias.data = ln1d.bias.data.clone()

    B, T, F = 1, 5, 7
    x4 = torch.randn(B, C, T, F)
    with torch.no_grad():
        # 3D form: reshape to [B*F, C, T]
        x3 = x4.permute(0, 3, 1, 2).reshape(B * F, C, T)
        y3 = ln1d(x3)
        y4 = ln4d(x4)
        # Reshape y3 back to 4D for comparison.
        y3_4d = y3.reshape(B, F, C, T).permute(0, 2, 3, 1)
    diff = (y3_4d - y4).abs().max().item()
    assert diff <= 5e-6, f"LayerNorm4dChannel != LayerNorm1d: max|Δ|={diff:.3e}"


def test_functional_stateful_conv2d_time_axis_matches_1d():
    """Stateful 4D time-axis conv ≡ stateful 1D conv applied per-F."""
    from src.models.streaming.onnx.functional_stateful import FunctionalStatefulConv1d

    _seed("rf-time-axis-conv")
    in_C = 8
    out_C = 8
    K = 3
    padding = K // 2

    conv1d = FunctionalStatefulConv1d(in_C, out_C, kernel_size=K, padding=padding, groups=in_C).eval()
    conv2d = FunctionalStatefulConv2dTimeAxis(in_C, out_C, kernel_size=K, padding=padding, groups=in_C).eval()
    # Copy weights: [O, I, K] → [O, I, K, 1].
    conv2d.conv.weight.data = conv1d.conv.weight.data.unsqueeze(-1).clone()
    conv2d.conv.bias.data = conv1d.conv.bias.data.clone()

    B, T, F = 1, 14, 5
    x4 = torch.randn(B, in_C, T, F)
    state4 = conv2d.init_state(batch_size=B, freq_size=F)
    with torch.no_grad():
        y4, _ = conv2d(x4, state4, state_frames=None)
    # Reference: apply conv1d to each F column separately.
    refs = []
    for f in range(F):
        x3 = x4[:, :, :, f]  # [B, C, T]
        state3 = conv1d.init_state(batch_size=B)
        with torch.no_grad():
            y3, _ = conv1d(x3, state3, state_frames=None)
        refs.append(y3.unsqueeze(-1))  # [B, C, T, 1]
    y_ref = torch.cat(refs, dim=-1)  # [B, C, T, F]
    diff = (y_ref - y4).abs().max().item()
    assert diff <= 5e-6, f"Conv2dTimeAxis vs per-F Conv1d: max|Δ|={diff:.3e}"


def test_simple_gate_4d_matches_3d():
    """``SimpleGate4d`` on [B, C, T, F] ≡ original ``SimpleGate`` on [B*F, C, T]."""
    from src.models.backbone import SimpleGate

    _seed("sg4d")
    sg1d = SimpleGate().eval()
    sg4d = SimpleGate4d().eval()
    B, C, T, F = 1, 8, 5, 7  # C even
    x = torch.randn(B, C, T, F)
    with torch.no_grad():
        y4 = sg4d(x)
        # Equivalent 3D path: same chunk on dim=1.
        x3 = x.permute(0, 3, 1, 2).reshape(B * F, C, T)
        y3 = sg1d(x3)
        y3_4d = y3.reshape(B, F, C // 2, T).permute(0, 2, 3, 1)
    assert torch.allclose(y4, y3_4d, atol=0.0), "SimpleGate4d must match SimpleGate exactly"
