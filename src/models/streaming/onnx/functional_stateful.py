"""Functional (explicit-state) stateful convolutions for ONNX export.

ONNX-exportable counterparts of ``src.models.streaming.layers.stateful_conv``.
Differences:

1. State is an explicit forward argument and return value — no internal
   ``_state`` buffer.
2. No ``clone().detach()`` — the state tensors are part of the computation graph.
3. No ``StateFramesContext`` — ``state_frames`` is passed explicitly.
4. Designed for a step-graph export where every conv state is graph I/O.

Numerical behaviour matches the stateful layers (and therefore the original
zero-padded convs on a full sequence): ``AsymmetricConv2d``'s Python ``round()``
split and the ``actual_total != total`` right-padding re-derivation are
reproduced here too.

Ported from LaCoSENet ``src/models/onnx_export/layers/functional_stateful.py``;
adjusted for BAFNet+ module paths.

Example:
    >>> conv = FunctionalStatefulConv1d(64, 64, kernel_size=3, padding=1)
    >>> state = conv.init_state(batch_size=1)
    >>> for chunk in chunks:
    ...     out, state = conv(chunk, state)
"""

from __future__ import annotations

from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor


class FunctionalStatefulConv1d(nn.Module):
    """ONNX-exportable stateful ``CausalConv1d`` with explicit state I/O.

    Forward signature: ``y, next_state = forward(x, state, state_frames=None)``.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        padding: Original ``CausalConv1d`` padding argument (doubled internally).
        stride: Convolution stride.
        dilation: Dilation rate.
        groups: Grouped-convolution groups.
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
    ):
        super().__init__()
        self.padding_size = padding * 2
        self.in_channels = in_channels
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Return a zero state ``[B, C, padding_size]``.

        Args:
            batch_size: Batch size.
            device: State device (defaults to the conv's parameter device).
            dtype: State dtype (defaults to the conv's parameter dtype).

        Returns:
            Zero-initialised state tensor.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.in_channels, self.padding_size, device=device, dtype=dtype)

    def forward(self, x: Tensor, state: Tensor, state_frames: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass with explicit state I/O.

        Args:
            x: Input tensor ``[B, C, T]``.
            state: Previous state ``[B, C, padding_size]``.
            state_frames: Number of leading frames allowed to update state
                (``None`` uses all ``T``).

        Returns:
            ``(output [B, C_out, T], next_state [B, C, padding_size])``.
        """
        _, _, t = x.shape
        x_padded = torch.cat([state, x], dim=2)
        output = self.conv(x_padded)

        effective_t = state_frames if state_frames is not None else t
        x_for_state = x[:, :, :effective_t]
        if effective_t >= self.padding_size:
            next_state = x_for_state[:, :, -self.padding_size :]
        else:
            keep = self.padding_size - effective_t
            next_state = torch.cat([state[:, :, -keep:], x_for_state], dim=2)
        return output, next_state


class FunctionalStatefulConv2d(nn.Module):
    """ONNX-exportable stateful ``AsymmetricConv2d`` with explicit state I/O.

    Forward signature: ``y, next_state = forward(x, state, state_frames=None)``.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: ``(time, freq)`` kernel size.
        padding: ``(time_padding, freq_padding)`` (the original argument).
        padding_ratio: ``(left_ratio, right_ratio)`` for the asymmetric time split.
        stride: ``(time, freq)`` stride.
        dilation: ``(time, freq)`` dilation.
        groups: Grouped-convolution groups.
        bias: Whether to use a bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        padding_ratio: Tuple[float, float] = (1.0, 0.0),
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        time_padding_total = padding[0] * 2
        freq_padding = padding[1]
        left_ratio, right_ratio = padding_ratio

        self.time_padding_left = round(time_padding_total * left_ratio)
        self.time_padding_right = round(time_padding_total * right_ratio)
        if self.time_padding_left + self.time_padding_right != time_padding_total:
            self.time_padding_right = time_padding_total - self.time_padding_left

        self.freq_padding = freq_padding
        self.padding_ratio = padding_ratio
        self.in_channels = in_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 257,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Return a zero state ``[B, C, time_padding_left, freq_size + 2*freq_padding]``.

        Args:
            batch_size: Batch size.
            freq_size: Frequency dimension of the conv input (before freq padding).
            device: State device (defaults to the conv's parameter device).
            dtype: State dtype (defaults to the conv's parameter dtype).

        Returns:
            Zero-initialised state tensor.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        freq_padded = freq_size + 2 * self.freq_padding
        return torch.zeros(
            batch_size, self.in_channels, self.time_padding_left, freq_padded, device=device, dtype=dtype
        )

    def forward(self, x: Tensor, state: Tensor, state_frames: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass with explicit state I/O.

        Args:
            x: Input tensor ``[B, C, T, F]``.
            state: Previous state ``[B, C, time_padding_left, F_padded]``.
            state_frames: Number of leading frames allowed to update state
                (``None`` uses all ``T``).

        Returns:
            ``(output [B, C_out, T, F], next_state [B, C, time_padding_left, F_padded])``.
        """
        _, _, t, _ = x.shape

        # 1. Frequency padding (always symmetric, always zero).
        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        freq_padded = x.shape[3]

        # 2. Time padding: left = state, right = zeros (future frames not present).
        right_pad = x.new_zeros(x.shape[0], x.shape[1], self.time_padding_right, freq_padded)
        x_padded = torch.cat([state, x, right_pad], dim=2)

        # 3. Output.
        output = self.conv(x_padded)

        # 4. Next state (last time_padding_left freq-padded frames of the input).
        effective_t = state_frames if state_frames is not None else t
        x_for_state = x[:, :, :effective_t, :]
        if effective_t >= self.time_padding_left:
            next_state = x_for_state[:, :, -self.time_padding_left :, :]
        else:
            keep = self.time_padding_left - effective_t
            next_state = torch.cat([state[:, :, -keep:, :], x_for_state], dim=2)
        return output, next_state


class FunctionalStatefulCausalConv2d(nn.Module):
    """ONNX-exportable stateful ``CausalConv2d`` with explicit state I/O.

    Fully-causal version of :class:`FunctionalStatefulConv2d`
    (``padding_ratio == (1.0, 0.0)``; no right padding).

    Forward signature: ``y, next_state = forward(x, state, state_frames=None)``.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: ``(time, freq)`` kernel size.
        padding: ``(time_padding, freq_padding)`` (the original argument).
        stride: ``(time, freq)`` stride.
        dilation: ``(time, freq)`` dilation.
        groups: Grouped-convolution groups.
        bias: Whether to use a bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.time_padding = padding[0] * 2
        self.freq_padding = padding[1]
        self.in_channels = in_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 257,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Return a zero state ``[B, C, time_padding, freq_size + 2*freq_padding]``."""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        freq_padded = freq_size + 2 * self.freq_padding
        return torch.zeros(batch_size, self.in_channels, self.time_padding, freq_padded, device=device, dtype=dtype)

    def forward(self, x: Tensor, state: Tensor, state_frames: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass with explicit state I/O.

        Args:
            x: Input tensor ``[B, C, T, F]``.
            state: Previous state ``[B, C, time_padding, F_padded]``.
            state_frames: Number of leading frames allowed to update state
                (``None`` uses all ``T``).

        Returns:
            ``(output [B, C_out, T, F], next_state [B, C, time_padding, F_padded])``.
        """
        _, _, t, _ = x.shape

        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        x_padded = torch.cat([state, x], dim=2)
        output = self.conv(x_padded)

        effective_t = state_frames if state_frames is not None else t
        x_for_state = x[:, :, :effective_t, :]
        if effective_t >= self.time_padding:
            next_state = x_for_state[:, :, -self.time_padding :, :]
        else:
            keep = self.time_padding - effective_t
            next_state = torch.cat([state[:, :, -keep:, :], x_for_state], dim=2)
        return output, next_state


def convert_to_functional(stateful_conv: nn.Module) -> nn.Module:
    """Convert a stateful conv to its functional (explicit-state) counterpart.

    Args:
        stateful_conv: A ``StatefulCausalConv1d`` / ``StatefulAsymmetricConv2d`` /
            ``StatefulCausalConv2d``.

    Returns:
        The matching ``FunctionalStateful*`` module with copied weights.

    Raises:
        TypeError: If ``stateful_conv`` is not a recognised stateful conv.
    """
    from src.models.streaming.layers.stateful_conv import (
        StatefulAsymmetricConv2d,
        StatefulCausalConv1d,
        StatefulCausalConv2d,
    )

    if isinstance(stateful_conv, StatefulCausalConv1d):
        orig1 = stateful_conv.conv
        out1 = FunctionalStatefulConv1d(
            in_channels=orig1.in_channels,
            out_channels=orig1.out_channels,
            kernel_size=orig1.kernel_size[0],
            padding=stateful_conv.padding_size // 2,
            stride=orig1.stride[0],
            dilation=orig1.dilation[0],
            groups=orig1.groups,
            bias=orig1.bias is not None,
        )
        out1.conv.load_state_dict(orig1.state_dict())
        return out1

    if isinstance(stateful_conv, StatefulAsymmetricConv2d):
        orig2 = stateful_conv.conv
        total_time = stateful_conv.time_padding_left + stateful_conv.time_padding_right
        # torch stubs type Conv2d.{kernel_size,stride,dilation} as tuple[int, ...]; at
        # runtime they are 2-tuples, so the cast is exact (not a behavioural override).
        out2 = FunctionalStatefulConv2d(
            in_channels=orig2.in_channels,
            out_channels=orig2.out_channels,
            kernel_size=cast(Tuple[int, int], orig2.kernel_size),
            padding=(total_time // 2, stateful_conv.freq_padding),
            padding_ratio=stateful_conv.padding_ratio,
            stride=cast(Tuple[int, int], orig2.stride),
            dilation=cast(Tuple[int, int], orig2.dilation),
            groups=orig2.groups,
            bias=orig2.bias is not None,
        )
        out2.conv.load_state_dict(orig2.state_dict())
        return out2

    if isinstance(stateful_conv, StatefulCausalConv2d):
        orig3 = stateful_conv.conv
        out3 = FunctionalStatefulCausalConv2d(
            in_channels=orig3.in_channels,
            out_channels=orig3.out_channels,
            kernel_size=cast(Tuple[int, int], orig3.kernel_size),
            padding=(stateful_conv.time_padding // 2, stateful_conv.freq_padding),
            stride=cast(Tuple[int, int], orig3.stride),
            dilation=cast(Tuple[int, int], orig3.dilation),
            groups=orig3.groups,
            bias=orig3.bias is not None,
        )
        out3.conv.load_state_dict(orig3.state_dict())
        return out3

    raise TypeError(f"Unsupported type for convert_to_functional: {type(stateful_conv)}")


__all__ = [
    "FunctionalStatefulConv1d",
    "FunctionalStatefulConv2d",
    "FunctionalStatefulCausalConv2d",
    "convert_to_functional",
]
