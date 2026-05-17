"""Tests for the explicit-state ONNX export registry (``streaming/onnx/state_registry.py``).

Covers the S2 exit gate: ``collect_states_from_model`` returns the expected
number of state tensors for a small synthetic BAFNet+ backbone — one per
functional stateful conv (12 ``AsymmetricConv2d`` from the three depth-4
DS_DDBs + ``num_tsblock * time_block_num * (2 + 2*len(time_block_kernel))``
``CausalConv1d`` from the causal TS-block time stages; no ``CausalConv2d`` is
instantiated by ``Backbone``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbone import Backbone
from src.models.streaming.converters import convert_to_stateful, get_stateful_layer_count
from src.models.streaming.layers.stateful_conv import (
    StatefulAsymmetricConv2d,
    StatefulCausalConv1d,
    StatefulCausalConv2d,
)
from src.models.streaming.onnx.functional_stateful import (
    FunctionalStatefulCausalConv2d,
    FunctionalStatefulConv1d,
    FunctionalStatefulConv2d,
    convert_to_functional,
)
from src.models.streaming.onnx.state_registry import StateRegistry, collect_states_from_model

DENSE_DEPTH = 4
NUM_TSBLOCK = 1
TIME_BLOCK_NUM = 1
TIME_BLOCK_KERNEL = [3, 5, 7, 11]


def _small_backbone() -> Backbone:
    return Backbone(
        n_fft=400,
        hop_size=100,
        win_size=400,
        dense_channel=8,
        sigmoid_beta=2.0,
        compress_factor=0.3,
        dense_depth=DENSE_DEPTH,
        num_tsblock=NUM_TSBLOCK,
        time_dw_kernel_size=3,
        time_block_kernel=TIME_BLOCK_KERNEL,
        freq_block_kernel=[3, 11, 23, 31],
        time_block_num=TIME_BLOCK_NUM,
        freq_block_num=1,
        causal_ts_block=True,
        encoder_padding_ratio=(0.9, 0.1),
        decoder_padding_ratio=(0.9, 0.1),
        sca_kernel_size=11,
        infer_type="mapping",
    ).eval()


def _functionalize(module: nn.Module) -> int:
    """Recursively replace every Stateful* conv with its functional counterpart."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (StatefulCausalConv1d, StatefulAsymmetricConv2d, StatefulCausalConv2d)):
            setattr(module, name, convert_to_functional(child))
            replaced += 1
        else:
            replaced += _functionalize(child)
    return replaced


def test_collect_states_count_matches_stateful_layers():
    # 12 AsymmetricConv2d (3 depth-4 DS_DDBs: encoder + mask + phase) + 10 CausalConv1d (causal TS time stages).
    n_asym = 3 * DENSE_DEPTH
    n_causal1d = NUM_TSBLOCK * TIME_BLOCK_NUM * (2 + 2 * len(TIME_BLOCK_KERNEL))
    expected = n_asym + n_causal1d
    assert expected == 22  # sanity for this fixed config

    backbone = _small_backbone()
    convert_to_stateful(backbone, verbose=False, inplace=True)

    counts = get_stateful_layer_count(backbone)
    assert counts["StatefulAsymmetricConv2d"] == n_asym
    assert counts["StatefulCausalConv1d"] == n_causal1d
    assert counts["StatefulCausalConv2d"] == 0
    assert counts["total"] == expected

    n_func = _functionalize(backbone)
    assert n_func == expected

    registry, init_states = collect_states_from_model(backbone, batch_size=1, freq_size=100)
    assert isinstance(registry, StateRegistry)
    assert registry.num_states == expected
    assert len(init_states) == expected


def test_collect_states_metadata_and_roundtrip():
    backbone = _small_backbone()
    convert_to_stateful(backbone, verbose=False, inplace=True)
    _functionalize(backbone)
    registry, init_states = collect_states_from_model(backbone, batch_size=1, freq_size=100)

    names = registry.state_names
    assert len(names) == len(set(names)), "state names must be unique"
    assert all(n.startswith("state_") for n in names)

    func_types = (FunctionalStatefulConv1d, FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)
    for info, state in zip([registry.get_by_index(i) for i in range(registry.num_states)], init_states):
        assert state.dtype == torch.float32
        assert state.shape[0] == 1
        assert len(state.shape) in (3, 4)
        owner = dict(backbone.named_modules())[info.module_path]
        assert isinstance(owner, func_types)

    # list <-> dict round-trip preserves order/contents.
    as_dict = registry.to_dict(init_states)
    assert set(as_dict) == set(names)
    back = registry.from_dict(as_dict)
    assert len(back) == len(init_states)
    assert all(torch.equal(a, b) for a, b in zip(back, init_states))

    assert registry.summary().startswith(f"StateRegistry: {registry.num_states} states")


def test_init_all_states_respects_batch_size():
    backbone = _small_backbone()
    convert_to_stateful(backbone, verbose=False, inplace=True)
    _functionalize(backbone)
    registry, _ = collect_states_from_model(backbone, batch_size=1, freq_size=100)
    states_b3 = registry.init_all_states(batch_size=3)
    assert all(s.shape[0] == 3 for s in states_b3)
    assert all(torch.count_nonzero(s) == 0 for s in states_b3)
