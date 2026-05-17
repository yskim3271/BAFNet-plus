"""Reshape-free 4D building blocks for the BAFNet+ TSBlock export-core rewrite (cycle 11).

D2 sweep row #7 Path β. The original :class:`src.models.backbone.TSBlock` runs its
time and freq stages over reshaped 3D tensors (``[B*F, C, T]`` then ``[B*T, C, F]``),
producing 32 explicit ``Transpose`` + 32 ``Reshape`` ops on the runtime-active path
of the D1 trunk_t2 ONNX graph (cf. ``results/profiling/d2_7_conv1d_2d_plan.md`` and
``results/profiling/d2_7_conv_neighborhood.json``). Path β replaces those reshapes
with 4D Conv2d kernels whose width/height degenerates to 1 on the inactive axis:

- Time-axis 1D conv on ``[B*F, C, T]`` (kernel ``K``) ≡ Conv2d on ``[B, C, T, F]``
  with kernel ``(K, 1)``.
- Freq-axis 1D conv on ``[B*T, C, F]`` (kernel ``K``) ≡ Conv2d on ``[B, C, T, F]``
  with kernel ``(1, K)``.

This module hosts the **axis-aware** building blocks (no module yet uses them — the
TSBlock variant lives in :mod:`reshape_free_tsblock`). All modules here are
ONNX-export-friendly (explicit state I/O, no ``view``/``permute`` outside their
declared role) and weight-compatible with the original BAFNet+ trained checkpoint
via the conversion helpers in :mod:`reshape_free_tsblock` (``weight.unsqueeze(-1)``
or ``weight.unsqueeze(2)`` depending on axis).

Adapted from LaCoSENet ``src/models/streaming/layers/reshape_free.py`` +
``reshape_free_stateful.py``; BAFNet+ deltas:

* ``LayerNorm4dChannel`` mirrors BAFNet+'s ``LayerNorm1d`` algebra
  (mean/var over ``dim=1``, ``eps=1e-6``, weight/bias stored as ``[C]`` so the
  checkpoint copy is a bare ``.clone()`` with no shape mutation).
* Only the time axis carries state in BAFNet+'s TSBlock (the freq stage is
  hard-coded ``causal=False`` and uses ``AdaptiveAvgPool1d(1)`` for SCA, with no
  causal SCA depthwise conv) — so the stateful 4D conv only needs the
  time-axis variant. Freq-axis convs stay stateless ``nn.Conv2d(kernel=(1, K))``.
* The stateful 4D conv's ``forward(x, state, state_frames=None) → (out, next)``
  signature matches :class:`FunctionalStatefulConv1d` so the existing
  :class:`~src.models.streaming.onnx.backbone_core.StateIterator` consumes it
  unchanged via ``state_iter.run_layer(...)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm4dChannel(nn.Module):
    """Channel-axis LayerNorm for 4D ``[B, C, T, F]`` tensors — BAFNet+ ``LayerNorm1d`` analog.

    BAFNet+'s :class:`src.models.backbone.LayerNorm1d` normalises over ``dim=1`` of
    a 3D ``[B*F, C, T]`` tensor (see its custom ``LayerNormFunction`` — biased
    variance, ``eps=1e-6``, learnable ``weight``/``bias`` per channel).
    Re-expressed over 4D ``[B, C, T, F]`` (channels still on ``dim=1``), the
    algebra is identical:

    ``y = (x - mean(dim=1)) / sqrt(var(dim=1, unbiased=False) + eps)``
    then ``y * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)``.

    Weights are stored as ``[C]`` (same as the original) so a converter can do
    ``dst.weight.data = src.weight.data.clone()`` without a shape mutation.

    Args:
        channels: Number of channels (``C``).
        eps: Numerical-stability floor matching ``LayerNorm1d`` (default ``1e-6``).
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        y = (x - mean) / (var + self.eps).sqrt()
        w = self.weight.view(1, self.channels, 1, 1)
        b = self.bias.view(1, self.channels, 1, 1)
        return y * w + b


class SimpleGate4d(nn.Module):
    """4D ``[B, C, T, F]`` analog of :class:`src.models.backbone.SimpleGate`.

    Splits the channel axis in half and elementwise-multiplies. Bit-identical to
    the 3D ``SimpleGate`` (which splits ``[B*F, 2C, T]`` on ``dim=1``).
    """

    def forward(self, x: Tensor) -> Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class FunctionalStatefulConv2dTimeAxis(nn.Module):
    """Reshape-free 4D analog of :class:`FunctionalStatefulConv1d` for the time axis.

    A height-only ``Conv2d(kernel=(K, 1))`` over ``[B, C, T, F]`` that propagates
    state through the time axis (= height, ``dim=2``). The padding semantics
    mirror :class:`src.models.backbone.CausalConv1d`'s
    ``padding_size = padding * 2`` left-only convention (i.e. the original
    BAFNet+ ``CausalConv1d`` receives ``padding=get_padding(K)=K//2`` and applies
    it doubled-and-left as ``F.pad(x, [self.padding, 0])``).

    Forward signature: ``y, next_state = forward(x, state, state_frames=None)`` —
    matches :class:`FunctionalStatefulConv1d` exactly so the existing
    :class:`StateIterator` can run it unchanged via ``state_iter.run_layer(...)``.

    State layout (zero-init): ``[B, in_channels, padding_size, F]``. Compared
    to the 3D :class:`FunctionalStatefulConv1d` state ``[B*F, C, padding]``, the
    4D variant keeps ``F`` as the trailing axis (no ``B*F`` collapse) so the
    enclosing reshape-free TSBlock never has to permute/reshape.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: 1D kernel size along the time axis (``K``). The Conv2d
            uses ``kernel=(K, 1)`` internally.
        padding: Original ``CausalConv1d`` padding argument (doubled internally
            into ``padding_size = padding * 2``).
        stride: Time-axis stride (always 1 in BAFNet+'s TSBlock, kept for API
            parity).
        dilation: Time-axis dilation (always 1 in BAFNet+'s TSBlock).
        groups: Grouped-convolution groups (``in_channels`` for depthwise convs).
        bias: Whether to use a bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.padding_size = int(padding) * 2
        self.in_channels = int(in_channels)
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(int(kernel_size), 1),
            stride=(int(stride), 1),
            dilation=(int(dilation), 1),
            padding=0,
            groups=int(groups),
            bias=bias,
        )

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 100,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Return a zero state ``[B, in_channels, padding_size, freq_size]``.

        Args:
            batch_size: Logical batch size (typically 1 for streaming).
            freq_size: Frequency axis size (``F`` of the input ``[B, C, T, F]``).
            device: State device (defaults to the conv's parameter device).
            dtype: State dtype (defaults to the conv's parameter dtype).

        Returns:
            Zero-initialised state tensor of shape ``[B, C, padding_size, F]``.
        """
        if device is None:
            device = self.conv.weight.device
        if dtype is None:
            dtype = self.conv.weight.dtype
        return torch.zeros(
            batch_size,
            self.in_channels,
            self.padding_size,
            int(freq_size),
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with explicit state I/O over the 4D time axis.

        Args:
            x: Input tensor ``[B, C, T, F]``.
            state: Previous state ``[B, C, padding_size, F]``.
            state_frames: Number of leading time frames allowed to update state
                (``None`` uses all ``T``). Matches
                :class:`FunctionalStatefulConv1d` semantics.

        Returns:
            ``(output [B, C_out, T, F], next_state [B, C, padding_size, F])``.
        """
        _, _, t, _ = x.shape
        x_padded = torch.cat([state, x], dim=2)
        output = self.conv(x_padded)

        effective_t = state_frames if state_frames is not None else t
        x_for_state = x[:, :, :effective_t, :]
        if effective_t >= self.padding_size:
            next_state = x_for_state[:, :, -self.padding_size :, :]
        else:
            keep = self.padding_size - effective_t
            next_state = torch.cat([state[:, :, -keep:, :], x_for_state], dim=2)
        return output, next_state


__all__ = [
    "LayerNorm4dChannel",
    "SimpleGate4d",
    "FunctionalStatefulConv2dTimeAxis",
]
