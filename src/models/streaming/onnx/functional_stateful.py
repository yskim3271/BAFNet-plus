"""Functional Stateful Layers for ONNX Export (BAFNet+ port from LaCoSENet).

Counterpart to ``src.models.streaming.layers.stateful_conv`` but with state
passed as explicit input/output (rather than held in an internal buffer).
This is the form ONNX export needs so that streaming state becomes graph
I/O and the trace is byte-equivalent to the PyTorch streaming forward.

Key differences vs ``StatefulCausalConv*``:
  - state is a forward argument, not ``self._state``
  - no ``clone().detach()`` (state stays in the autograd-graph free path
    naturally because the conv has no internal state)
  - no ``StateFramesContext`` global — ``state_frames`` is an explicit arg
  - one-to-one match with ``StatefulCausalConv1d/2d/AsymmetricConv2d``,
    convertible via :func:`convert_to_functional`

Mirrors LaCoSENet's ``src/models/onnx_export/layers/functional_stateful.py``.
This BAFNet+ copy lives next to ``export_bafnetplus_onnx.py`` so it is
isolated from the existing backbone-only export path (``export_onnx.py``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import Tensor


class FunctionalStatefulConv1d(nn.Module):
    """Stateful 1D causal conv with explicit state I/O.

    Mirrors :class:`StatefulCausalConv1d` streaming forward exactly.
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
            in_channels, out_channels, kernel_size, padding=0,
            stride=stride, dilation=dilation, groups=groups, bias=bias,
        )

    def init_state(self, batch_size: int = 1, device=None, dtype=None) -> Tensor:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.in_channels, self.padding_size, device=device, dtype=dtype)

    def forward(self, x: Tensor, state: Tensor, state_frames: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        B, C, T = x.shape
        x_padded = torch.cat([state, x], dim=2)
        output = self.conv(x_padded)
        effective_T = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_T]
        if effective_T >= self.padding_size:
            next_state = x_for_state[:, :, -self.padding_size:]
        else:
            keep = self.padding_size - effective_T
            old_part = state[:, :, -keep:]
            next_state = torch.cat([old_part, x_for_state], dim=2)
        return output, next_state


class FunctionalStatefulConv2d(nn.Module):
    """Stateful 2D asymmetric-causal conv with explicit state I/O.

    Mirrors :class:`StatefulAsymmetricConv2d` streaming forward exactly.
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
            in_channels, out_channels, kernel_size, padding=0,
            stride=stride, dilation=dilation, groups=groups, bias=bias,
        )

    def init_state(self, batch_size: int = 1, freq_size: int = 201, device=None, dtype=None) -> Tensor:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        freq_padded = freq_size + 2 * self.freq_padding
        return torch.zeros(
            batch_size, self.in_channels, self.time_padding_left, freq_padded,
            device=device, dtype=dtype,
        )

    def forward(self, x: Tensor, state: Tensor, state_frames: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        B, C, T, F = x.shape
        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        freq_padded = x.shape[3]
        # Skip right_pad concat when fully causal (padding_ratio=(1.0, 0.0)) to avoid
        # producing a 0-element tensor that ORT QDQ calibration cannot reduce over
        # (ReduceMax of empty tensor → RUNTIME_EXCEPTION).
        if self.time_padding_right > 0:
            right_pad = x.new_zeros(B, C, self.time_padding_right, freq_padded)
            x_padded = torch.cat([state, x, right_pad], dim=2)
        else:
            x_padded = torch.cat([state, x], dim=2)
        output = self.conv(x_padded)
        effective_T = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_T, :]
        if effective_T >= self.time_padding_left:
            next_state = x_for_state[:, :, -self.time_padding_left:, :]
        else:
            keep = self.time_padding_left - effective_T
            old_part = state[:, :, -keep:, :]
            next_state = torch.cat([old_part, x_for_state], dim=2)
        return output, next_state


class FunctionalStatefulCausalConv2d(nn.Module):
    """Stateful 2D fully-causal conv with explicit state I/O.

    Mirrors :class:`StatefulCausalConv2d` streaming forward exactly.
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
            in_channels, out_channels, kernel_size, padding=0,
            stride=stride, dilation=dilation, groups=groups, bias=bias,
        )

    def init_state(self, batch_size: int = 1, freq_size: int = 201, device=None, dtype=None) -> Tensor:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        freq_padded = freq_size + 2 * self.freq_padding
        return torch.zeros(
            batch_size, self.in_channels, self.time_padding, freq_padded,
            device=device, dtype=dtype,
        )

    def forward(self, x: Tensor, state: Tensor, state_frames: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        B, C, T, F = x.shape
        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        x_padded = torch.cat([state, x], dim=2)
        output = self.conv(x_padded)
        effective_T = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_T, :]
        if effective_T >= self.time_padding:
            next_state = x_for_state[:, :, -self.time_padding:, :]
        else:
            keep = self.time_padding - effective_T
            old_part = state[:, :, -keep:, :]
            next_state = torch.cat([old_part, x_for_state], dim=2)
        return output, next_state


def convert_to_functional(stateful_conv: nn.Module) -> nn.Module:
    """Convert a ``StatefulCausalConv*`` instance to its Functional counterpart.

    Copies the underlying ``nn.Conv*`` weights byte-equivalently. Shape /
    padding parameters are read off the source module so the conversion is
    one-to-one with no information loss.
    """
    from src.models.streaming.layers.stateful_conv import (
        StatefulAsymmetricConv2d,
        StatefulCausalConv1d,
        StatefulCausalConv2d,
    )

    if isinstance(stateful_conv, StatefulCausalConv1d):
        orig = stateful_conv.conv
        f = FunctionalStatefulConv1d(
            in_channels=orig.in_channels,
            out_channels=orig.out_channels,
            kernel_size=orig.kernel_size[0],
            padding=stateful_conv.padding_size // 2,
            stride=orig.stride[0],
            dilation=orig.dilation[0],
            groups=orig.groups,
            bias=orig.bias is not None,
        )
        f.conv.load_state_dict(orig.state_dict())
        return f

    if isinstance(stateful_conv, StatefulAsymmetricConv2d):
        orig = stateful_conv.conv
        total_time = stateful_conv.time_padding_left + stateful_conv.time_padding_right
        f = FunctionalStatefulConv2d(
            in_channels=orig.in_channels,
            out_channels=orig.out_channels,
            kernel_size=orig.kernel_size,
            padding=(total_time // 2, stateful_conv.freq_padding),
            padding_ratio=stateful_conv.padding_ratio,
            stride=orig.stride,
            dilation=orig.dilation,
            groups=orig.groups,
            bias=orig.bias is not None,
        )
        f.conv.load_state_dict(orig.state_dict())
        return f

    if isinstance(stateful_conv, StatefulCausalConv2d):
        orig = stateful_conv.conv
        f = FunctionalStatefulCausalConv2d(
            in_channels=orig.in_channels,
            out_channels=orig.out_channels,
            kernel_size=orig.kernel_size,
            padding=(stateful_conv.time_padding // 2, stateful_conv.freq_padding),
            stride=orig.stride,
            dilation=orig.dilation,
            groups=orig.groups,
            bias=orig.bias is not None,
        )
        f.conv.load_state_dict(orig.state_dict())
        return f

    raise TypeError(f"Unsupported stateful conv type: {type(stateful_conv)}")


def convert_module_inplace(module: nn.Module) -> int:
    """Walk ``module`` and replace any StatefulCausal* children with Functional* variants.

    Returns the number of conversions performed.
    """
    from src.models.streaming.layers.stateful_conv import (
        StatefulAsymmetricConv2d,
        StatefulCausalConv1d,
        StatefulCausalConv2d,
    )
    stateful_types = (StatefulCausalConv1d, StatefulAsymmetricConv2d, StatefulCausalConv2d)

    n = 0
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, stateful_types):
                functional = convert_to_functional(child)
                setattr(parent, child_name, functional)
                n += 1
    return n


__all__ = [
    "FunctionalStatefulConv1d",
    "FunctionalStatefulConv2d",
    "FunctionalStatefulCausalConv2d",
    "convert_to_functional",
    "convert_module_inplace",
]
