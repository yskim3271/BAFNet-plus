"""Tests for the algorithmic-lookahead calculator (``streaming/lookahead.py``).

Covers the S2 exit gate: the computed ``L_enc`` / ``L_dec`` equal the layer-wise
right-padding sum for the configured 50 ms BAFNet+ backbone, which must be 3 and
3. Also checks the fully-causal and symmetric configurations and the
right-padding breakdown.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from src.models.backbone import Backbone
from src.models.streaming.lookahead import (
    LookaheadInfo,
    compute_lookahead,
    right_padding_breakdown,
    sum_right_padding,
)


def _backbone(encoder_ratio, decoder_ratio, infer_type="mapping") -> Backbone:
    """A small Backbone with the requested padding ratios (TS-block size minimised)."""
    return Backbone(
        n_fft=400,
        hop_size=100,
        win_size=400,
        dense_channel=8,
        sigmoid_beta=2.0,
        compress_factor=0.3,
        dense_depth=4,
        num_tsblock=1,
        time_dw_kernel_size=3,
        time_block_kernel=[3, 5, 7, 11],
        freq_block_kernel=[3, 11, 23, 31],
        time_block_num=1,
        freq_block_num=1,
        causal_ts_block=True,
        encoder_padding_ratio=encoder_ratio,
        decoder_padding_ratio=decoder_ratio,
        sca_kernel_size=11,
        infer_type=infer_type,
    ).eval()


def test_lookahead_50ms_config_is_3_and_3():
    backbone = _backbone((0.9, 0.1), (0.9, 0.1))
    info = compute_lookahead(backbone)

    assert isinstance(info, LookaheadInfo)
    assert info.encoder_lookahead == 3
    assert info.decoder_lookahead == 3
    assert info.total_lookahead == 6

    # "layer-wise right-padding sum" cross-check (the calculator must agree with
    # a direct sum of AsymmetricConv2d.time_padding_right over each branch).
    assert sum_right_padding(backbone.dense_encoder) == 3
    assert sum_right_padding(backbone.mask_decoder) == 3
    assert sum_right_padding(backbone.phase_decoder) == 3
    assert info.encoder_lookahead == sum_right_padding(backbone.dense_encoder)
    assert info.decoder_lookahead == sum_right_padding(backbone.mask_decoder)

    # depth-4 DS_DDB, kernel 3, ratio (0.9, 0.1): per-layer right pads 0,0,1,2.
    assert [r for _, r in info.encoder_breakdown] == [0, 0, 1, 2]
    assert [r for _, r in info.decoder_breakdown] == [0, 0, 1, 2]
    assert [r for _, r in right_padding_breakdown(backbone.phase_decoder)] == [0, 0, 1, 2]


def test_lookahead_masking_variant_matches_mapping():
    # infer_type does not affect lookahead (only the AsymmetricConv2d padding does).
    info = compute_lookahead(_backbone((0.9, 0.1), (0.9, 0.1), infer_type="masking"))
    assert (info.encoder_lookahead, info.decoder_lookahead) == (3, 3)


def test_lookahead_fully_causal_is_zero():
    backbone = _backbone((1.0, 0.0), (1.0, 0.0))
    info = compute_lookahead(backbone)
    assert info.encoder_lookahead == 0
    assert info.decoder_lookahead == 0
    assert all(r == 0 for _, r in info.encoder_breakdown)


def test_lookahead_symmetric_sums_dilations():
    # ratio (0.5, 0.5), depth 4, kernel 3: per-layer right pad == dilation = 1,2,4,8.
    backbone = _backbone((0.5, 0.5), (0.5, 0.5))
    info = compute_lookahead(backbone)
    assert info.encoder_lookahead == 15
    assert info.decoder_lookahead == 15
    assert [r for _, r in info.encoder_breakdown] == [1, 2, 4, 8]


def test_compute_lookahead_rejects_non_backbone():
    with pytest.raises(AttributeError):
        compute_lookahead(nn.Linear(4, 4))


def test_compute_lookahead_rejects_mismatched_decoders():
    backbone = _backbone((0.9, 0.1), (0.9, 0.1))
    # Force the phase decoder's DS_DDB to use a different ratio so the two
    # decoders disagree on lookahead.
    backbone.phase_decoder.dense_block = _backbone((0.9, 0.1), (0.5, 0.5)).mask_decoder.dense_block
    assert sum_right_padding(backbone.mask_decoder) != sum_right_padding(backbone.phase_decoder)
    with pytest.raises(ValueError):
        compute_lookahead(backbone)
