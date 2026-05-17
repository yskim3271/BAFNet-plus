"""Parity tests: functional stateful == stateful == original padded conv.

Covers the S2 exit gate for ``onnx/functional_stateful.py``: the explicit-state
``FunctionalStateful*`` convs reproduce, chunk-by-chunk, the same output as both
the buffer-state ``Stateful*`` convs and the original zero-padded convs on
synthetic sequences (causal and lookahead cases).
"""

from __future__ import annotations

import zlib

import pytest
import torch

from src.models.backbone import AsymmetricConv2d, CausalConv1d, CausalConv2d
from src.models.streaming.context import StateFramesContext
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

ATOL = 2e-5
RTOL = 1e-4


def _seed(name: str) -> None:
    torch.manual_seed(zlib.crc32(name.encode()))


def _stream_stateful(conv, full, chunk_size, num_chunks, lookahead):
    """Chunk-by-chunk run of a buffer-state stateful conv (StateFramesContext-bounded)."""
    outs = []
    for c in range(num_chunks):
        chunk = full.narrow(2, c * chunk_size, chunk_size + lookahead)
        with StateFramesContext(chunk_size):
            out = conv(chunk)
        outs.append(out.narrow(2, 0, chunk_size))
    return torch.cat(outs, dim=2)


def _stream_functional(fconv, full, chunk_size, num_chunks, lookahead, state):
    """Chunk-by-chunk run of an explicit-state functional conv."""
    outs = []
    for c in range(num_chunks):
        chunk = full.narrow(2, c * chunk_size, chunk_size + lookahead)
        out, state = fconv(chunk, state, state_frames=chunk_size)
        outs.append(out.narrow(2, 0, chunk_size))
    return torch.cat(outs, dim=2), state


# ---------------------------------------------------------------------------
# 1D causal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kernel_size, padding, dilation, chunk_size, num_chunks",
    [
        (3, 1, 1, 8, 3),
        (5, 2, 1, 2, 6),
        (5, 2, 1, 1, 11),
        (3, 2, 2, 5, 4),
    ],
)
def test_functional_conv1d_matches_stateful_and_padded(kernel_size, padding, dilation, chunk_size, num_chunks):
    _seed(f"f1d-{kernel_size}-{padding}-{dilation}-{chunk_size}")
    channels = 6
    orig = CausalConv1d(channels, channels, kernel_size, padding=padding, dilation=dilation).eval()
    full = torch.randn(1, channels, num_chunks * chunk_size)
    with torch.no_grad():
        ref = orig(full)

    stateful = StatefulCausalConv1d.from_causal_conv(orig).eval()
    stateful.set_streaming(True)
    functional = convert_to_functional(stateful).eval()
    assert isinstance(functional, FunctionalStatefulConv1d)

    with torch.no_grad():
        out_stateful = _stream_stateful(stateful, full, chunk_size, num_chunks, lookahead=0)
        out_functional, last_state = _stream_functional(
            functional, full, chunk_size, num_chunks, lookahead=0, state=functional.init_state(1)
        )

    assert torch.allclose(out_stateful, ref, atol=ATOL, rtol=RTOL)
    assert torch.allclose(out_functional, ref, atol=ATOL, rtol=RTOL)
    assert last_state.shape == functional.init_state(1).shape


# ---------------------------------------------------------------------------
# 2D causal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kt, kf, pt, pf, chunk_size, num_chunks",
    [
        (3, 3, 1, 1, 4, 3),
        (5, 3, 2, 1, 2, 4),
        (5, 5, 2, 2, 3, 3),
    ],
)
def test_functional_causal_conv2d_matches_stateful_and_padded(kt, kf, pt, pf, chunk_size, num_chunks):
    _seed(f"f2d-causal-{kt}-{kf}-{pt}-{pf}-{chunk_size}")
    channels, freq = 5, 12
    orig = CausalConv2d(channels, channels, (kt, kf), padding=(pt, pf)).eval()
    full = torch.randn(1, channels, num_chunks * chunk_size, freq)
    with torch.no_grad():
        ref = orig(full)

    stateful = StatefulCausalConv2d.from_causal_conv2d(orig).eval()
    stateful.set_streaming(True)
    functional = convert_to_functional(stateful).eval()
    assert isinstance(functional, FunctionalStatefulCausalConv2d)

    with torch.no_grad():
        out_stateful = _stream_stateful(stateful, full, chunk_size, num_chunks, lookahead=0)
        out_functional, _ = _stream_functional(
            functional, full, chunk_size, num_chunks, lookahead=0, state=functional.init_state(1, freq)
        )

    assert torch.allclose(out_stateful, ref, atol=ATOL, rtol=RTOL)
    assert torch.allclose(out_functional, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# 2D asymmetric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kt, kf, pt, pf, dil, ratio, chunk_size, num_chunks",
    [
        (5, 3, 2, 1, (1, 1), (1.0, 0.0), 3, 4),  # causal (right=0)
        (3, 3, 1, 1, (1, 1), (1.0, 0.0), 8, 2),  # causal (right=0)
        (5, 3, 2, 1, (1, 1), (0.75, 0.25), 4, 4),  # lookahead=1, state_frames > left
        (5, 3, 2, 1, (1, 1), (0.75, 0.25), 2, 5),  # lookahead=1, state_frames < left
        (5, 3, 2, 1, (1, 1), (0.5, 0.5), 4, 3),  # lookahead=2
        (3, 3, 2, 1, (2, 1), (0.75, 0.25), 4, 4),  # dilated lookahead=1
    ],
)
def test_functional_asymmetric_conv2d_matches_stateful_and_padded(kt, kf, pt, pf, dil, ratio, chunk_size, num_chunks):
    _seed(f"f2d-asym-{kt}-{kf}-{pt}-{pf}-{dil}-{ratio}-{chunk_size}")
    channels, freq = 6, 13
    orig = AsymmetricConv2d(channels, channels, (kt, kf), padding=(pt, pf), padding_ratio=ratio, dilation=dil).eval()
    lookahead = orig.time_padding_right
    total_t = num_chunks * chunk_size + lookahead
    full = torch.randn(1, channels, total_t, freq)
    with torch.no_grad():
        ref = orig(full).narrow(2, 0, num_chunks * chunk_size)

    stateful = StatefulAsymmetricConv2d.from_asymmetric_conv(orig).eval()
    stateful.set_streaming(True)
    functional = convert_to_functional(stateful).eval()
    assert isinstance(functional, FunctionalStatefulConv2d)
    assert functional.time_padding_right == lookahead

    with torch.no_grad():
        out_stateful = _stream_stateful(stateful, full, chunk_size, num_chunks, lookahead=lookahead)
        out_functional, last_state = _stream_functional(
            functional, full, chunk_size, num_chunks, lookahead=lookahead, state=functional.init_state(1, freq)
        )

    assert out_stateful.shape == ref.shape
    assert torch.allclose(out_stateful, ref, atol=ATOL, rtol=RTOL)
    assert torch.allclose(out_functional, ref, atol=ATOL, rtol=RTOL)
    assert last_state.shape == functional.init_state(1, freq).shape
