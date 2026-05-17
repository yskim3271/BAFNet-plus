"""Parity tests: stateful conv (chunk-by-chunk) == original zero-padded conv.

Covers the S2 exit gates for ``layers/stateful_conv.py``:
    - ``StatefulCausalConv1d`` / ``StatefulCausalConv2d`` / ``StatefulAsymmetricConv2d``
      streamed output equals the original padded conv on synthetic sequences.
    - Both state-update branches are exercised: ``state_frames < padding_size``
      (tiny chunks) and ``state_frames > padding_size`` (large chunks).
    - The asymmetric (lookahead) path: extended chunks (current + lookahead)
      bounded by ``StateFramesContext(chunk_size)`` reconstruct the mid-sequence
      output exactly, and the explicit-``state_frames`` and context-manager paths
      agree.
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

ATOL = 2e-5
RTOL = 1e-4


def _make_seed(name: str) -> None:
    torch.manual_seed(zlib.crc32(name.encode()))


def _stream(conv, full: torch.Tensor, chunk_size: int, num_chunks: int, lookahead: int, mode: str) -> torch.Tensor:
    """Run ``conv`` chunk-by-chunk and concat the mature outputs.

    Args:
        conv: A stateful conv with streaming enabled.
        full: Full input ``[..., T, ...]`` with ``T == num_chunks*chunk_size + lookahead``
            (time axis is dim 2 for both the 1D ``[B,C,T]`` and 2D ``[B,C,T,F]`` convs).
        chunk_size: Frames advanced per chunk (== ``state_frames``).
        num_chunks: Number of chunks to process.
        lookahead: Extra trailing frames peeked per chunk (0 for causal convs).
        mode: ``"implicit"`` (no ``state_frames``; only valid when ``lookahead == 0``),
            ``"explicit"`` (pass ``state_frames=chunk_size``), or ``"context"``
            (wrap in ``StateFramesContext(chunk_size)``).

    Returns:
        Concatenated output ``[..., num_chunks*chunk_size, ...]``.
    """
    outs = []
    for c in range(num_chunks):
        chunk = full.narrow(2, c * chunk_size, chunk_size + lookahead)
        if mode == "context":
            with StateFramesContext(chunk_size):
                out = conv(chunk)
        elif mode == "explicit":
            out = conv(chunk, state_frames=chunk_size)
        elif mode == "implicit":
            assert lookahead == 0, "implicit mode is only meaningful without lookahead"
            out = conv(chunk)
        else:  # pragma: no cover - guard
            raise ValueError(f"unknown mode {mode}")
        outs.append(out.narrow(2, 0, chunk_size))
    return torch.cat(outs, dim=2)


# ---------------------------------------------------------------------------
# CausalConv1d (left-only padding)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kernel_size, padding, dilation, chunk_size, num_chunks",
    [
        (3, 1, 1, 8, 3),  # padding_size=2: state_frames(8) > padding_size(2)
        (5, 2, 1, 1, 12),  # padding_size=4: state_frames(1) < padding_size(4)
        (5, 2, 1, 2, 6),  # padding_size=4: state_frames(2) < padding_size(4)
        (3, 2, 2, 5, 4),  # dilated, padding_size=4: state_frames(5) > padding_size(4)
    ],
)
def test_stateful_causal_conv1d_matches_padded(kernel_size, padding, dilation, chunk_size, num_chunks):
    _make_seed(f"c1d-{kernel_size}-{padding}-{dilation}-{chunk_size}")
    channels = 6
    orig = CausalConv1d(channels, channels, kernel_size, padding=padding, dilation=dilation).eval()
    total_t = num_chunks * chunk_size
    full = torch.randn(1, channels, total_t)

    with torch.no_grad():
        ref = orig(full)

    stateful = StatefulCausalConv1d.from_causal_conv(orig).eval()
    stateful.set_streaming(True)
    with torch.no_grad():
        streamed = _stream(stateful, full, chunk_size, num_chunks, lookahead=0, mode="implicit")

    assert streamed.shape == ref.shape
    assert torch.allclose(streamed, ref, atol=ATOL, rtol=RTOL)


def test_stateful_causal_conv1d_state_frames_modes_agree():
    _make_seed("c1d-modes")
    channels = 5
    orig = CausalConv1d(channels, channels, 3, padding=1).eval()
    chunk_size, num_chunks = 4, 4
    full = torch.randn(1, channels, chunk_size * num_chunks)

    def run(mode):
        s = StatefulCausalConv1d.from_causal_conv(orig).eval()
        s.set_streaming(True)
        with torch.no_grad():
            return _stream(s, full, chunk_size, num_chunks, lookahead=0, mode=mode)

    out_implicit, out_explicit, out_context = run("implicit"), run("explicit"), run("context")
    assert torch.allclose(out_implicit, out_explicit, atol=ATOL, rtol=RTOL)
    assert torch.allclose(out_implicit, out_context, atol=ATOL, rtol=RTOL)


def test_stateful_causal_conv1d_non_streaming_is_original():
    _make_seed("c1d-nonstream")
    channels = 4
    orig = CausalConv1d(channels, channels, 5, padding=2).eval()
    x = torch.randn(2, channels, 13)
    stateful = StatefulCausalConv1d.from_causal_conv(orig).eval()  # streaming disabled by default
    with torch.no_grad():
        assert torch.allclose(stateful(x), orig(x), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# CausalConv2d (left-only time padding, symmetric freq padding)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kt, kf, pt, pf, chunk_size, num_chunks",
    [
        (3, 3, 1, 1, 4, 3),  # time_padding=2: state_frames(4) > 2
        (5, 3, 2, 1, 2, 4),  # time_padding=4: state_frames(2) < 4
        (5, 5, 2, 2, 3, 3),  # time_padding=4: state_frames(3) < 4
    ],
)
def test_stateful_causal_conv2d_matches_padded(kt, kf, pt, pf, chunk_size, num_chunks):
    _make_seed(f"c2d-{kt}-{kf}-{pt}-{pf}-{chunk_size}")
    channels, freq = 5, 12
    orig = CausalConv2d(channels, channels, (kt, kf), padding=(pt, pf)).eval()
    full = torch.randn(1, channels, num_chunks * chunk_size, freq)

    with torch.no_grad():
        ref = orig(full)

    stateful = StatefulCausalConv2d.from_causal_conv2d(orig).eval()
    stateful.set_streaming(True)
    with torch.no_grad():
        streamed = _stream(stateful, full, chunk_size, num_chunks, lookahead=0, mode="implicit")

    assert streamed.shape == ref.shape
    assert torch.allclose(streamed, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# AsymmetricConv2d, fully causal (padding_ratio=(1.0, 0.0) -> right=0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kt, kf, pt, pf, dil, chunk_size, num_chunks",
    [
        (5, 3, 2, 1, (1, 1), 3, 4),  # time total 4, left=4: state_frames(3) < 4
        (3, 3, 1, 1, (1, 1), 8, 2),  # time total 2, left=2: state_frames(8) > 2
        (3, 3, 2, 1, (2, 1), 5, 3),  # dilated; time total 4, left=4: state_frames(5) > 4
    ],
)
def test_stateful_asymmetric_conv2d_causal_matches_padded(kt, kf, pt, pf, dil, chunk_size, num_chunks):
    _make_seed(f"asym-causal-{kt}-{kf}-{pt}-{pf}-{dil}-{chunk_size}")
    channels, freq = 6, 14
    orig = AsymmetricConv2d(
        channels, channels, (kt, kf), padding=(pt, pf), padding_ratio=(1.0, 0.0), dilation=dil
    ).eval()
    assert orig.time_padding_right == 0
    full = torch.randn(1, channels, num_chunks * chunk_size, freq)

    with torch.no_grad():
        ref = orig(full)

    stateful = StatefulAsymmetricConv2d.from_asymmetric_conv(orig).eval()
    stateful.set_streaming(True)
    with torch.no_grad():
        streamed = _stream(stateful, full, chunk_size, num_chunks, lookahead=0, mode="implicit")

    assert streamed.shape == ref.shape
    assert torch.allclose(streamed, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# AsymmetricConv2d with lookahead (right > 0) -- buffered streaming path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kt, kf, pt, pf, dil, ratio, chunk_size, num_chunks",
    [
        (5, 3, 2, 1, (1, 1), (0.75, 0.25), 4, 4),  # left=3 right=1: state_frames(4) > left(3)
        (5, 3, 2, 1, (1, 1), (0.75, 0.25), 2, 5),  # left=3 right=1: state_frames(2) < left(3)
        (5, 3, 2, 1, (1, 1), (0.5, 0.5), 4, 3),  # left=2 right=2: lookahead=2
        (3, 3, 2, 1, (2, 1), (0.75, 0.25), 4, 4),  # dilated; left=3 right=1
    ],
)
def test_stateful_asymmetric_conv2d_lookahead_matches_padded(kt, kf, pt, pf, dil, ratio, chunk_size, num_chunks):
    _make_seed(f"asym-la-{kt}-{kf}-{pt}-{pf}-{dil}-{ratio}-{chunk_size}")
    channels, freq = 6, 13
    orig = AsymmetricConv2d(channels, channels, (kt, kf), padding=(pt, pf), padding_ratio=ratio, dilation=dil).eval()
    lookahead = orig.time_padding_right
    assert lookahead > 0
    # T must give every streamed output frame its real future context.
    total_t = num_chunks * chunk_size + lookahead
    full = torch.randn(1, channels, total_t, freq)

    with torch.no_grad():
        ref = orig(full)  # [1, C, total_t, freq]
    ref_streamed = ref.narrow(2, 0, num_chunks * chunk_size)

    def run(mode):
        s = StatefulAsymmetricConv2d.from_asymmetric_conv(orig).eval()
        s.set_streaming(True)
        with torch.no_grad():
            return _stream(s, full, chunk_size, num_chunks, lookahead=lookahead, mode=mode)

    out_context = run("context")
    out_explicit = run("explicit")
    assert out_context.shape == ref_streamed.shape
    assert torch.allclose(out_context, ref_streamed, atol=ATOL, rtol=RTOL)
    assert torch.allclose(out_explicit, ref_streamed, atol=ATOL, rtol=RTOL)


def test_reset_state_restarts_stream():
    _make_seed("c1d-reset")
    channels = 4
    orig = CausalConv1d(channels, channels, 5, padding=2).eval()
    chunk_size, num_chunks = 3, 5
    full = torch.randn(1, channels, chunk_size * num_chunks)
    with torch.no_grad():
        ref = orig(full)

    stateful = StatefulCausalConv1d.from_causal_conv(orig).eval()
    stateful.set_streaming(True)
    with torch.no_grad():
        first = _stream(stateful, full, chunk_size, num_chunks, lookahead=0, mode="implicit")
        assert torch.allclose(first, ref, atol=ATOL, rtol=RTOL)
        stateful.reset_state()
        second = _stream(stateful, full, chunk_size, num_chunks, lookahead=0, mode="implicit")
    assert torch.allclose(second, ref, atol=ATOL, rtol=RTOL)
