"""
Native TSBlock layers for LaCoSENet streaming inference.

This module provides Conv2d-based TSBlock layers that operate directly on
4D tensors [B, C, T, F] without reshape operations. When batch_size=1,
this approach provides up to 193x speedup by avoiding memory copies.

Key insight:
    Conv1d on reshaped [B*F, C, T] ≡ Conv2d with kernel (K, 1) on [B, C, T, F]
    Conv1d on reshaped [B*T, C, F] ≡ Conv2d with kernel (1, K) on [B, C, T, F]

Sections:
    1. Building Blocks: ChannelLayerNorm2d, SimpleGate2d, CausalConv2dTime, FreqCAB, FreqGPKFFN
    2. Stateful Streaming Layers: StreamingConv2d, StreamingCAB, StreamingGPKFFN, StreamingTSBlock
    3. Weight Transfer Helpers (private): backbone → streaming conversion utilities
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1 — Building Blocks
# =============================================================================


class ChannelLayerNorm2d(nn.Module):
    """
    Channel-wise layer normalization for 4D tensors [B, C, T, F].

    Normalizes over the channel dimension (dim=1), matching the behavior of
    LayerNorm1d on reshaped 3D tensors:
        LayerNorm1d on [B*F, C, T]  ≡  ChannelLayerNorm2d on [B, C, T, F]

    Args:
        channels: Number of channels
        eps: Small constant for numerical stability
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps

        # Learnable parameters [1, C, 1, 1] for broadcasting
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [B, C, T, F]

        Returns:
            Normalized tensor [B, C, T, F]
        """
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, T, F]
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)  # [B, 1, T, F]
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight + self.bias


class SimpleGate2d(nn.Module):
    """SimpleGate for 4D tensors [B, C, T, F]."""

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class CausalConv2dTime(nn.Module):
    """Causal Conv2d for time axis with left-only padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pad left side of time axis only
        x = F.pad(x, (0, 0, self.padding, 0))
        return self.conv(x)


class FreqCAB(nn.Module):
    """
    Frequency-domain Channel Attention Block.

    Operates on 4D tensors [B, C, T, F] using Conv2d with axis-specific kernels.

    Original CAB flow (with reshape):
        [B,C,T,F] → reshape → [B*F,C,T] → Conv1d → reshape → [B,C,T,F]

    FreqCAB flow:
        [B,C,T,F] → Conv2d(kernel=(K,1)) → [B,C,T,F]  (no reshape!)

    Args:
        channels: Number of input/output channels
        kernel_size: Kernel size for depthwise conv
        axis: Processing axis ('time' or 'freq')
        causal: If True and axis='time', use causal (left-only) padding
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        axis: str = "time",
        causal: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.axis = axis
        self.kernel_size = kernel_size
        self.causal = causal and (axis == "time")  # causal only applies to time axis
        dw_channel = channels * 2

        self.norm = ChannelLayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, dw_channel, kernel_size=1)

        # Axis-specific depthwise conv
        if axis == "time":
            # Process along time axis: kernel (K, 1)
            # For causal: no padding in conv, manual left padding in forward
            padding = 0 if self.causal else kernel_size // 2
            self.dwconv = nn.Conv2d(
                dw_channel,
                dw_channel,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                groups=dw_channel,
            )
            # Causal padding amount: (kernel_size - 1) on left
            self._causal_pad = kernel_size - 1
        else:
            # Process along freq axis: kernel (1, K)
            self.dwconv = nn.Conv2d(
                dw_channel,
                dw_channel,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dw_channel,
            )

        self.sg = SimpleGate2d()

        # SCA: squeeze (mean) + channel attention conv
        self.sca_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.pwconv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [B, C, T, F]

        Returns:
            Output tensor [B, C, T, F]
        """
        skip = x

        x = self.norm(x)
        x = self.pwconv1(x)

        # Apply causal padding if needed
        if self.causal:
            # Pad on left side of time axis: F.pad uses (left, right, top, bottom) for last 2 dims
            # For [B, C, T, F], time is dim 2, so we pad (0, 0, _causal_pad, 0)
            x = F.pad(x, (0, 0, self._causal_pad, 0))

        x = self.dwconv(x)
        x = self.sg(x)

        # SCA: global average along processing axis
        if self.axis == "time":
            attn = x.mean(dim=2, keepdim=True)  # [B, C, 1, F]
        else:
            attn = x.mean(dim=3, keepdim=True)  # [B, C, T, 1]

        attn = self.sca_conv(attn)
        x = x * attn

        x = self.pwconv2(x)

        return skip + x * self.beta


class FreqGPKFFN(nn.Module):
    """
    Frequency-domain Group Prime Kernel FFN.

    Operates on 4D tensors [B, C, T, F] using Conv2d with axis-specific kernels
    for multi-scale processing.

    Original GPKFFN flow:
        [B,C,T,F] → reshape → [B*F,C,T] → multiple Conv1d → reshape

    FreqGPKFFN flow:
        [B,C,T,F] → multiple Conv2d(kernel=(K,1)) → [B,C,T,F]

    Args:
        channels: Number of input/output channels
        kernel_list: List of kernel sizes for multi-scale processing
        axis: Processing axis ('time' or 'freq')
        causal: If True and axis='time', use causal (left-only) padding
    """

    def __init__(
        self,
        channels: int,
        kernel_list: List[int] = [3, 5, 7, 11],
        axis: str = "time",
        causal: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.axis = axis
        self.kernel_list = kernel_list
        self.causal = causal and (axis == "time")
        self.expand_ratio = len(kernel_list)
        mid_channel = channels * self.expand_ratio

        self.norm = ChannelLayerNorm2d(channels)
        self.proj_first = nn.Conv2d(channels, mid_channel, kernel_size=1)
        self.proj_last = nn.Conv2d(mid_channel, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Create conv layers for each kernel size
        for k in kernel_list:
            if axis == "time":
                if self.causal:
                    # Causal: left-only padding
                    attn_conv = nn.Sequential(
                        CausalConv2dTime(channels, channels, k, groups=channels),
                        nn.Conv2d(channels, channels, kernel_size=1),
                    )
                    main_conv = CausalConv2dTime(channels, channels, k, groups=channels)
                else:
                    # Non-causal: symmetric padding
                    attn_conv = nn.Sequential(
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=(k, 1),
                            padding=(k // 2, 0),
                            groups=channels,
                        ),
                        nn.Conv2d(channels, channels, kernel_size=1),
                    )
                    main_conv = nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(k, 1),
                        padding=(k // 2, 0),
                        groups=channels,
                    )
            else:
                # Freq axis: kernel (1, K) - always symmetric (non-causal)
                attn_conv = nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(1, k),
                        padding=(0, k // 2),
                        groups=channels,
                    ),
                    nn.Conv2d(channels, channels, kernel_size=1),
                )
                main_conv = nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    groups=channels,
                )

            setattr(self, f"attn_{k}", attn_conv)
            setattr(self, f"conv_{k}", main_conv)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [B, C, T, F]

        Returns:
            Output tensor [B, C, T, F]
        """
        shortcut = x

        x = self.norm(x)
        x = self.proj_first(x)

        # Split into chunks for multi-scale processing
        chunks = x.chunk(self.expand_ratio, dim=1)
        outputs = []

        for i, k in enumerate(self.kernel_list):
            attn_module = getattr(self, f"attn_{k}")
            conv_module = getattr(self, f"conv_{k}")

            attn_out = attn_module(chunks[i])
            conv_out = conv_module(chunks[i])
            outputs.append(attn_out * conv_out)

        x = torch.cat(outputs, dim=1)
        x = self.proj_last(x) * self.scale + shortcut

        return x


# =============================================================================
# Section 2 — Stateful Streaming Layers
# =============================================================================


class StreamingConv2d(nn.Module):
    """
    Stateful Conv2d for streaming inference.

    Maintains state buffer for causal (time-axis) convolutions.
    For freq-axis convolutions, state is typically not needed since
    frequency dimension is fully available.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size tuple (K_t, K_f)
        padding: Padding tuple (pad_t, pad_f)
        axis: Processing axis ('time' or 'freq')
        groups: Number of groups for grouped conv
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        axis: str = "time",
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.axis = axis
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # For time axis, we need left padding only (causal)
        # For freq axis, we use symmetric padding (non-causal)
        if axis == "time":
            self.padding_size = padding[0]  # Time padding
            self.state_dim = 2  # T dimension
            conv_padding = (0, padding[1])  # No time padding, keep freq padding
        else:
            self.padding_size = 0  # No state needed for freq
            self.state_dim = 3
            conv_padding = padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=conv_padding,
            groups=groups,
            bias=bias,
        )

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 100,
        time_size: int = 40,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Initialize state tensor.

        For time axis: state shape [B, C, padding_t, F]
        For freq axis: no state needed (return empty)
        """
        if dtype is None:
            dtype = self.conv.weight.dtype
        if device is None:
            device = self.conv.weight.device

        if self.axis == "time" and self.padding_size > 0:
            return torch.zeros(
                batch_size,
                self.in_channels,
                self.padding_size,
                freq_size,
                device=device,
                dtype=dtype,
            )
        else:
            # No state needed for freq axis
            return torch.zeros(1, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with explicit state I/O.

        Args:
            x: Input tensor [B, C, T, F]
            state: Previous state tensor
            state_frames: Number of frames (from start of input) to use for
                state update. If None, uses all T frames. This enables
                processing extended (lookahead) inputs while limiting state
                updates to the causal chunk.

        Returns:
            Tuple of (output, next_state)
        """
        if self.axis == "time" and self.padding_size > 0:
            # Concatenate state (past frames) with input
            x_padded = torch.cat([state, x], dim=2)

            # Compute output
            out = self.conv(x_padded)

            # Compute next state with optional gating
            effective_T = state_frames if state_frames is not None else x.shape[2]
            x_for_state = x[:, :, :effective_T, :]

            if effective_T >= self.padding_size:
                new_state = x_for_state[:, :, -self.padding_size :, :]
            else:
                keep = self.padding_size - effective_T
                old_part = state[:, :, -keep:, :]
                new_state = torch.cat([old_part, x_for_state], dim=2)

            return out, new_state
        else:
            # Freq axis: no state management
            out = self.conv(x)
            return out, state


class StreamingCAB(nn.Module):
    """
    Streaming Channel Attention Block.

    Combines:
    - Stateful depthwise Conv2d for temporal causality
    - Stateful depthwise Conv2d for SCA (causal local pooling replacement)

    State components:
    - dwconv_state: [B, 2C, padding, F] for causal depthwise conv
    - sca_dwconv_state: [B, C, sca_kernel_size-1, F] for causal SCA depthwise conv
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        axis: str = "time",
        sca_kernel_size: int = 11,
    ):
        super().__init__()
        self.channels = channels
        self.axis = axis
        self.kernel_size = kernel_size
        self.sca_kernel_size = sca_kernel_size
        dw_channel = channels * 2

        self.norm = ChannelLayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, dw_channel, kernel_size=1)

        # Stateful depthwise conv (causal: padding = kernel_size - 1)
        if axis == "time":
            causal_padding = kernel_size - 1  # Full left padding for causality
            self.dwconv = StreamingConv2d(
                dw_channel,
                dw_channel,
                kernel_size=(kernel_size, 1),
                padding=(causal_padding, 0),
                axis="time",
                groups=dw_channel,
            )
            self.dwconv_padding = causal_padding
        else:
            # Freq axis: non-causal, no state needed
            self.dwconv = nn.Conv2d(
                dw_channel,
                dw_channel,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dw_channel,
            )
            self.dwconv_padding = 0

        self.sg = SimpleGate2d()

        # SCA: depthwise conv (causal) + pointwise conv
        if axis == "time":
            sca_causal_padding = sca_kernel_size - 1
            self.sca_dwconv = StreamingConv2d(
                channels,
                channels,
                kernel_size=(sca_kernel_size, 1),
                padding=(sca_causal_padding, 0),
                axis="time",
                groups=channels,
                bias=False,
            )
            self.sca_dwconv_padding = sca_causal_padding
        else:
            self.sca_dwconv = None
            self.sca_dwconv_padding = 0

        self.sca_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.pwconv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 100,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Tensor]:
        """
        Initialize all states.

        Returns dict with:
        - 'dwconv': Conv state [B, 2C, padding, F]
        - 'sca_dwconv': SCA depthwise conv state [B, C, sca_kernel_size-1, F]
        """
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if device is None:
            device = next(self.parameters()).device

        states = {}

        # Dwconv state (only for time axis)
        if self.axis == "time" and self.dwconv_padding > 0:
            states["dwconv"] = torch.zeros(
                batch_size,
                self.channels * 2,
                self.dwconv_padding,
                freq_size,
                device=device,
                dtype=dtype,
            )
        else:
            states["dwconv"] = torch.zeros(1, device=device, dtype=dtype)

        # SCA depthwise conv state (only for time axis)
        if self.axis == "time" and self.sca_dwconv is not None:
            states["sca_dwconv"] = self.sca_dwconv.init_state(
                batch_size, freq_size, device=device, dtype=dtype,
            )
        else:
            states["sca_dwconv"] = torch.zeros(1, device=device, dtype=dtype)

        return states

    def forward(
        self,
        x: Tensor,
        state: Dict[str, Tensor],
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with explicit state I/O.

        Args:
            x: Input tensor [B, C, T, F]
            state: Dict with 'dwconv' and 'sca_dwconv' states
            state_frames: Number of frames for state update gating.
                Passed through to dwconv and sca_dwconv.

        Returns:
            Tuple of (output, new_state_dict)
        """
        skip = x
        new_state = {}

        x = self.norm(x)
        x = self.pwconv1(x)

        # Stateful dwconv (time axis) or regular conv (freq axis)
        if self.axis == "time" and isinstance(self.dwconv, StreamingConv2d):
            x, new_state["dwconv"] = self.dwconv(x, state["dwconv"], state_frames=state_frames)
        else:
            x = self.dwconv(x)
            new_state["dwconv"] = state["dwconv"]

        x = self.sg(x)

        # SCA: depthwise conv (causal) + pointwise conv
        if self.axis == "time" and self.sca_dwconv is not None:
            sca_out, new_state["sca_dwconv"] = self.sca_dwconv(
                x, state["sca_dwconv"], state_frames=state_frames
            )
            attn = self.sca_conv(sca_out)
        else:
            # Freq axis: simple mean (no state needed for frequency)
            attn = x.mean(dim=3, keepdim=True)
            attn = self.sca_conv(attn)
            new_state["sca_dwconv"] = state["sca_dwconv"]

        x = x * attn
        x = self.pwconv2(x)

        return skip + x * self.beta, new_state


class StreamingGPKFFN(nn.Module):
    """
    Streaming Group Prime Kernel FFN.

    Maintains state for each causal Conv2d in the multi-scale processing.
    """

    def __init__(
        self,
        channels: int,
        kernel_list: List[int] = [3, 5, 7, 11],
        axis: str = "time",
    ):
        super().__init__()
        self.channels = channels
        self.axis = axis
        self.kernel_list = kernel_list
        self.expand_ratio = len(kernel_list)
        mid_channel = channels * self.expand_ratio

        self.norm = ChannelLayerNorm2d(channels)
        self.proj_first = nn.Conv2d(channels, mid_channel, kernel_size=1)
        self.proj_last = nn.Conv2d(mid_channel, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Create stateful conv layers for each kernel
        for k in kernel_list:
            if axis == "time":
                # Stateful for time axis (causal: padding = k - 1)
                causal_padding = k - 1
                attn_dw = StreamingConv2d(
                    channels,
                    channels,
                    kernel_size=(k, 1),
                    padding=(causal_padding, 0),
                    axis="time",
                    groups=channels,
                )
                attn_pw = nn.Conv2d(channels, channels, kernel_size=1)
                main_conv = StreamingConv2d(
                    channels,
                    channels,
                    kernel_size=(k, 1),
                    padding=(causal_padding, 0),
                    axis="time",
                    groups=channels,
                )
            else:
                # Non-stateful for freq axis
                attn_dw = nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    groups=channels,
                )
                attn_pw = nn.Conv2d(channels, channels, kernel_size=1)
                main_conv = nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    groups=channels,
                )

            setattr(self, f"attn_dw_{k}", attn_dw)
            setattr(self, f"attn_pw_{k}", attn_pw)
            setattr(self, f"conv_{k}", main_conv)

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 100,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Tensor]:
        """Initialize states for all causal convs."""
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if device is None:
            device = next(self.parameters()).device

        states = {}

        for k in self.kernel_list:
            causal_padding = k - 1  # Causal: full left padding
            if self.axis == "time" and causal_padding > 0:
                states[f"attn_dw_{k}"] = torch.zeros(
                    batch_size, self.channels, causal_padding, freq_size, device=device, dtype=dtype
                )
                states[f"conv_{k}"] = torch.zeros(
                    batch_size, self.channels, causal_padding, freq_size, device=device, dtype=dtype
                )
            else:
                states[f"attn_dw_{k}"] = torch.zeros(1, device=device, dtype=dtype)
                states[f"conv_{k}"] = torch.zeros(1, device=device, dtype=dtype)

        return states

    def forward(
        self,
        x: Tensor,
        state: Dict[str, Tensor],
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with explicit state I/O.

        Args:
            x: Input tensor [B, C, T, F]
            state: Dict of state tensors for each causal conv
            state_frames: Number of frames for state update gating.

        Returns:
            Tuple of (output, new_state_dict)
        """
        shortcut = x
        new_state = {}

        x = self.norm(x)
        x = self.proj_first(x)

        chunks = x.chunk(self.expand_ratio, dim=1)
        outputs = []

        for i, k in enumerate(self.kernel_list):
            attn_dw = getattr(self, f"attn_dw_{k}")
            attn_pw = getattr(self, f"attn_pw_{k}")
            conv = getattr(self, f"conv_{k}")

            # Attention path
            if isinstance(attn_dw, StreamingConv2d):
                attn_out, new_state[f"attn_dw_{k}"] = attn_dw(
                    chunks[i], state[f"attn_dw_{k}"], state_frames=state_frames
                )
            else:
                attn_out = attn_dw(chunks[i])
                new_state[f"attn_dw_{k}"] = state[f"attn_dw_{k}"]

            attn_out = attn_pw(attn_out)

            # Conv path
            if isinstance(conv, StreamingConv2d):
                conv_out, new_state[f"conv_{k}"] = conv(
                    chunks[i], state[f"conv_{k}"], state_frames=state_frames
                )
            else:
                conv_out = conv(chunks[i])
                new_state[f"conv_{k}"] = state[f"conv_{k}"]

            outputs.append(attn_out * conv_out)

        x = torch.cat(outputs, dim=1)
        x = self.proj_last(x) * self.scale + shortcut

        return x, new_state


class StreamingTSBlock(nn.Module):
    """
    Streaming TSBlock for LaCoSENet inference.

    Combines all stateful components:
    - Time stage: StreamingCAB + StreamingGPKFFN (causal)
    - Freq stage: Non-stateful FreqCAB + FreqGPKFFN (full context)
    """

    def __init__(
        self,
        dense_channel: int = 64,
        time_block_num: int = 2,
        freq_block_num: int = 2,
        time_dw_kernel_size: int = 3,
        time_block_kernel: List[int] = [3, 5, 7, 11],
        freq_block_kernel: List[int] = [3, 5, 7, 11],
        sca_kernel_size: int = 11,
    ):
        super().__init__()
        self.dense_channel = dense_channel
        self.time_block_num = time_block_num
        self.freq_block_num = freq_block_num

        # Time stage (stateful)
        self.time_cabs = nn.ModuleList()
        self.time_gpkffns = nn.ModuleList()
        for _ in range(time_block_num):
            self.time_cabs.append(
                StreamingCAB(
                    dense_channel,
                    kernel_size=time_dw_kernel_size,
                    axis="time",
                    sca_kernel_size=sca_kernel_size,
                )
            )
            self.time_gpkffns.append(
                StreamingGPKFFN(
                    dense_channel,
                    kernel_list=time_block_kernel,
                    axis="time",
                )
            )

        # Freq stage (non-stateful, uses full freq context)
        freq_blocks = []
        for _ in range(freq_block_num):
            freq_blocks.append(
                nn.Sequential(
                    FreqCAB(dense_channel, kernel_size=3, axis="freq"),
                    FreqGPKFFN(dense_channel, kernel_list=freq_block_kernel, axis="freq"),
                )
            )
        self.freq_stage = nn.Sequential(*freq_blocks)

        self.beta_t = nn.Parameter(torch.zeros(1, dense_channel, 1, 1))
        self.beta_f = nn.Parameter(torch.zeros(1, dense_channel, 1, 1))

    def init_state(
        self,
        batch_size: int = 1,
        freq_size: int = 100,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Dict[str, Tensor]]:
        """Initialize all states for time stage."""
        states = []

        for i in range(self.time_block_num):
            cab_state = self.time_cabs[i].init_state(batch_size, freq_size, device, dtype)
            gpkffn_state = self.time_gpkffns[i].init_state(batch_size, freq_size, device, dtype)
            states.append({"cab": cab_state, "gpkffn": gpkffn_state})

        return states

    def forward(
        self,
        x: Tensor,
        state: List[Dict[str, Tensor]],
        state_frames: Optional[int] = None,
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """
        Forward pass with explicit state I/O.

        Args:
            x: Input tensor [B, C, T, F]
            state: List of state dicts for each time block
            state_frames: Number of frames for state update gating.

        Returns:
            Tuple of (output, new_states)
        """
        new_states = []
        residual = x

        # Time stage (stateful)
        for i in range(self.time_block_num):
            x, cab_state = self.time_cabs[i](x, state[i]["cab"], state_frames=state_frames)
            x, gpkffn_state = self.time_gpkffns[i](x, state[i]["gpkffn"], state_frames=state_frames)
            new_states.append({"cab": cab_state, "gpkffn": gpkffn_state})

        x = x + residual * self.beta_t

        # Freq stage (non-stateful)
        x = self.freq_stage(x) + x * self.beta_f

        return x, new_states

    @classmethod
    def from_backbone_tsblock(cls, ts_block: nn.Module) -> "StreamingTSBlock":
        """
        Convert a Backbone TSBlock to StreamingTSBlock for streaming inference.

        Args:
            ts_block: Original TSBlock module from Backbone model

        Returns:
            StreamingTSBlock with transferred weights
        """
        dense_channel = ts_block.dense_channel

        time_block_num = len(ts_block.time_stage)
        freq_block_num = len(ts_block.freq_stage)

        first_time_block = ts_block.time_stage[0]
        cab = first_time_block[0]
        gpkffn = first_time_block[1]

        if hasattr(cab.dwconv, "conv"):
            time_dw_kernel_size = cab.dwconv.conv.kernel_size[0]
        elif hasattr(cab.dwconv, "kernel_size"):
            time_dw_kernel_size = cab.dwconv.kernel_size[0]
        else:
            time_dw_kernel_size = 3

        sca_kernel_size = 11
        if hasattr(cab, "sca") and isinstance(cab.sca, nn.Sequential):
            sca_first = cab.sca[0]
            if hasattr(sca_first, "conv"):
                sca_kernel_size = sca_first.conv.kernel_size[0]
            elif hasattr(sca_first, "kernel_size"):
                sca_kernel_size = sca_first.kernel_size[0] if isinstance(sca_first.kernel_size, tuple) else sca_first.kernel_size

        time_block_kernel = gpkffn.kernel_list

        first_freq_block = ts_block.freq_stage[0]
        freq_gpkffn = first_freq_block[1]
        freq_block_kernel = freq_gpkffn.kernel_list

        streaming_ts_block = cls(
            dense_channel=dense_channel,
            time_block_num=time_block_num,
            freq_block_num=freq_block_num,
            time_dw_kernel_size=time_dw_kernel_size,
            time_block_kernel=time_block_kernel,
            freq_block_kernel=freq_block_kernel,
            sca_kernel_size=sca_kernel_size,
        )

        for i, block in enumerate(ts_block.time_stage):
            cab_src = block[0]
            gpkffn_src = block[1]
            _transfer_cab_weights_to_stateful(
                cab_src, streaming_ts_block.time_cabs[i], axis="time"
            )
            _transfer_gpkffn_weights_to_stateful(
                gpkffn_src, streaming_ts_block.time_gpkffns[i], axis="time"
            )

        for i, block in enumerate(ts_block.freq_stage):
            cab_src = block[0]
            gpkffn_src = block[1]
            freq_module = streaming_ts_block.freq_stage[i]
            _transfer_cab_weights_to_freq(cab_src, freq_module[0], axis="freq")
            _transfer_gpkffn_weights_to_freq(gpkffn_src, freq_module[1], axis="freq")

        streaming_ts_block.beta_t.data = ts_block.beta_t.data.unsqueeze(-1)
        streaming_ts_block.beta_f.data = ts_block.beta_f.data.unsqueeze(-1)

        logger.info(
            f"Converted TSBlock to Streaming: "
            f"{time_block_num} time blocks, {freq_block_num} freq blocks"
        )

        return streaming_ts_block

    @staticmethod
    def convert_sequence_block(
        sequence_block: nn.Sequential,
    ) -> nn.ModuleList:
        """
        Convert a sequence of TSBlocks to streaming versions.

        Args:
            sequence_block: Sequential container of TSBlocks

        Returns:
            ModuleList of StreamingTSBlocks
        """
        streaming_blocks = nn.ModuleList()
        for i, ts_block in enumerate(sequence_block):
            streaming_block = StreamingTSBlock.from_backbone_tsblock(ts_block)
            streaming_blocks.append(streaming_block)
            logger.info(f"Converted TSBlock {i} to Streaming")

        return streaming_blocks


# =============================================================================
# Section 3 — Weight Transfer Helpers (private)
# =============================================================================


def _convert_layernorm1d_to_channel(
    ln1d: nn.Module,
) -> ChannelLayerNorm2d:
    """
    Convert LayerNorm1d to ChannelLayerNorm2d.

    ChannelLayerNorm2d normalizes over the channel dimension (dim=1) in 4D,
    which is the correct 4D equivalent of LayerNorm1d that normalizes
    over channels (dim=1) in 3D.

    Args:
        ln1d: Original LayerNorm1d layer

    Returns:
        ChannelLayerNorm2d with transferred weights
    """
    if hasattr(ln1d, "weight"):
        channels = ln1d.weight.shape[0]
    else:
        raise ValueError(f"Cannot determine channels from {type(ln1d)}")

    eps = getattr(ln1d, "eps", 1e-6)
    ch_ln = ChannelLayerNorm2d(channels=channels, eps=eps)

    # Transfer weights: [C] → [1, C, 1, 1]
    if hasattr(ln1d, "weight") and ln1d.weight is not None:
        ch_ln.weight.data = ln1d.weight.data.view(1, channels, 1, 1)
    if hasattr(ln1d, "bias") and ln1d.bias is not None:
        ch_ln.bias.data = ln1d.bias.data.view(1, channels, 1, 1)

    return ch_ln


def _transfer_cab_weights_to_stateful(
    cab_src: nn.Module,
    cab_dst: StreamingCAB,
    axis: str,
) -> None:
    """Transfer weights from original CAB to StreamingCAB."""
    if hasattr(cab_src, "norm"):
        cab_dst.norm = _convert_layernorm1d_to_channel(cab_src.norm)

    if hasattr(cab_src, "pwconv1"):
        cab_dst.pwconv1.weight.data = cab_src.pwconv1.weight.data.unsqueeze(-1)
        if cab_src.pwconv1.bias is not None:
            cab_dst.pwconv1.bias.data = cab_src.pwconv1.bias.data.clone()

    if hasattr(cab_src, "dwconv"):
        dwconv_src = cab_src.dwconv.conv if hasattr(cab_src.dwconv, "conv") else cab_src.dwconv
        dwconv_dst = cab_dst.dwconv.conv if hasattr(cab_dst.dwconv, "conv") else cab_dst.dwconv
        if axis == "time":
            dwconv_dst.weight.data = dwconv_src.weight.data.unsqueeze(-1)
        else:
            dwconv_dst.weight.data = dwconv_src.weight.data.unsqueeze(2)
        if dwconv_src.bias is not None:
            dwconv_dst.bias.data = dwconv_src.bias.data.clone()

    if hasattr(cab_src, "sca") and isinstance(cab_src.sca, nn.Sequential):
        sca = cab_src.sca
        sca_dw_src = sca[0]
        sca_pw_src = sca[1] if len(sca) > 1 else None

        if hasattr(cab_dst, "sca_dwconv") and cab_dst.sca_dwconv is not None:
            dw_src_conv = sca_dw_src.conv if hasattr(sca_dw_src, "conv") else sca_dw_src
            dw_dst_conv = cab_dst.sca_dwconv.conv
            if isinstance(dw_src_conv, nn.Conv1d):
                dw_dst_conv.weight.data = dw_src_conv.weight.data.unsqueeze(-1)
                if dw_src_conv.bias is not None:
                    dw_dst_conv.bias.data = dw_src_conv.bias.data.clone()

        if sca_pw_src is not None and isinstance(sca_pw_src, nn.Conv1d):
            cab_dst.sca_conv.weight.data = sca_pw_src.weight.data.unsqueeze(-1)
            if sca_pw_src.bias is not None:
                cab_dst.sca_conv.bias.data = sca_pw_src.bias.data.clone()

    if hasattr(cab_src, "pwconv2"):
        cab_dst.pwconv2.weight.data = cab_src.pwconv2.weight.data.unsqueeze(-1)
        if cab_src.pwconv2.bias is not None:
            cab_dst.pwconv2.bias.data = cab_src.pwconv2.bias.data.clone()

    if hasattr(cab_src, "beta"):
        cab_dst.beta.data = cab_src.beta.data.unsqueeze(-1)


def _transfer_gpkffn_weights_to_stateful(
    gpkffn_src: nn.Module,
    gpkffn_dst: StreamingGPKFFN,
    axis: str,
) -> None:
    """Transfer weights from original GPKFFN to StreamingGPKFFN."""
    if hasattr(gpkffn_src, "norm"):
        gpkffn_dst.norm = _convert_layernorm1d_to_channel(gpkffn_src.norm)

    if hasattr(gpkffn_src, "proj_first"):
        proj_conv = (
            gpkffn_src.proj_first[0]
            if isinstance(gpkffn_src.proj_first, nn.Sequential)
            else gpkffn_src.proj_first
        )
        gpkffn_dst.proj_first.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            gpkffn_dst.proj_first.bias.data = proj_conv.bias.data.clone()

    if hasattr(gpkffn_src, "proj_last"):
        proj_conv = (
            gpkffn_src.proj_last[0]
            if isinstance(gpkffn_src.proj_last, nn.Sequential)
            else gpkffn_src.proj_last
        )
        gpkffn_dst.proj_last.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            gpkffn_dst.proj_last.bias.data = proj_conv.bias.data.clone()

    if hasattr(gpkffn_src, "scale"):
        gpkffn_dst.scale.data = gpkffn_src.scale.data.unsqueeze(-1)

    kernel_list = gpkffn_src.kernel_list
    for k in kernel_list:
        attn_src = getattr(gpkffn_src, f"attn_{k}")
        attn_dw_dst = getattr(gpkffn_dst, f"attn_dw_{k}")
        attn_pw_dst = getattr(gpkffn_dst, f"attn_pw_{k}")

        if isinstance(attn_src, nn.Sequential):
            src_dw = attn_src[0].conv if hasattr(attn_src[0], "conv") else attn_src[0]
            dst_dw = attn_dw_dst.conv if hasattr(attn_dw_dst, "conv") else attn_dw_dst
            if axis == "time":
                dst_dw.weight.data = src_dw.weight.data.unsqueeze(-1)
            else:
                dst_dw.weight.data = src_dw.weight.data.unsqueeze(2)
            if src_dw.bias is not None:
                dst_dw.bias.data = src_dw.bias.data.clone()

            src_pw = attn_src[1]
            attn_pw_dst.weight.data = src_pw.weight.data.unsqueeze(-1)
            if src_pw.bias is not None:
                attn_pw_dst.bias.data = src_pw.bias.data.clone()

        conv_src = getattr(gpkffn_src, f"conv_{k}")
        conv_src_actual = conv_src.conv if hasattr(conv_src, "conv") else conv_src
        conv_dst = getattr(gpkffn_dst, f"conv_{k}")
        conv_dst_actual = conv_dst.conv if hasattr(conv_dst, "conv") else conv_dst

        if axis == "time":
            conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(-1)
        else:
            conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(2)
        if conv_src_actual.bias is not None:
            conv_dst_actual.bias.data = conv_src_actual.bias.data.clone()


def _transfer_cab_weights_to_freq(
    cab_src: nn.Module,
    cab_dst: FreqCAB,
    axis: str,
) -> None:
    """Transfer weights from original CAB to FreqCAB (non-stateful)."""
    if hasattr(cab_src, "norm"):
        cab_dst.norm = _convert_layernorm1d_to_channel(cab_src.norm)

    if hasattr(cab_src, "pwconv1"):
        cab_dst.pwconv1.weight.data = cab_src.pwconv1.weight.data.unsqueeze(-1)
        if cab_src.pwconv1.bias is not None:
            cab_dst.pwconv1.bias.data = cab_src.pwconv1.bias.data.clone()

    if hasattr(cab_src, "dwconv"):
        dwconv_src = cab_src.dwconv.conv if hasattr(cab_src.dwconv, "conv") else cab_src.dwconv
        if axis == "time":
            cab_dst.dwconv.weight.data = dwconv_src.weight.data.unsqueeze(-1)
        else:
            cab_dst.dwconv.weight.data = dwconv_src.weight.data.unsqueeze(2)
        if dwconv_src.bias is not None:
            cab_dst.dwconv.bias.data = dwconv_src.bias.data.clone()

    if hasattr(cab_src, "sca") and isinstance(cab_src.sca, nn.Sequential) and len(cab_src.sca) > 1:
        sca_conv = cab_src.sca[-1]
        if isinstance(sca_conv, nn.Conv1d):
            cab_dst.sca_conv.weight.data = sca_conv.weight.data.unsqueeze(-1)
            if sca_conv.bias is not None:
                cab_dst.sca_conv.bias.data = sca_conv.bias.data.clone()

    if hasattr(cab_src, "pwconv2"):
        cab_dst.pwconv2.weight.data = cab_src.pwconv2.weight.data.unsqueeze(-1)
        if cab_src.pwconv2.bias is not None:
            cab_dst.pwconv2.bias.data = cab_src.pwconv2.bias.data.clone()

    if hasattr(cab_src, "beta"):
        cab_dst.beta.data = cab_src.beta.data.unsqueeze(-1)


def _transfer_gpkffn_weights_to_freq(
    gpkffn_src: nn.Module,
    gpkffn_dst: FreqGPKFFN,
    axis: str,
) -> None:
    """Transfer weights from original GPKFFN to FreqGPKFFN (non-stateful)."""
    if hasattr(gpkffn_src, "norm"):
        gpkffn_dst.norm = _convert_layernorm1d_to_channel(gpkffn_src.norm)

    if hasattr(gpkffn_src, "proj_first"):
        proj_conv = (
            gpkffn_src.proj_first[0]
            if isinstance(gpkffn_src.proj_first, nn.Sequential)
            else gpkffn_src.proj_first
        )
        gpkffn_dst.proj_first.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            gpkffn_dst.proj_first.bias.data = proj_conv.bias.data.clone()

    if hasattr(gpkffn_src, "proj_last"):
        proj_conv = (
            gpkffn_src.proj_last[0]
            if isinstance(gpkffn_src.proj_last, nn.Sequential)
            else gpkffn_src.proj_last
        )
        gpkffn_dst.proj_last.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            gpkffn_dst.proj_last.bias.data = proj_conv.bias.data.clone()

    if hasattr(gpkffn_src, "scale"):
        gpkffn_dst.scale.data = gpkffn_src.scale.data.unsqueeze(-1)

    kernel_list = gpkffn_src.kernel_list
    for k in kernel_list:
        attn_src = getattr(gpkffn_src, f"attn_{k}")
        attn_dst = getattr(gpkffn_dst, f"attn_{k}")

        if isinstance(attn_src, nn.Sequential):
            for src_layer, dst_layer in zip(attn_src, attn_dst):
                src_conv = src_layer.conv if hasattr(src_layer, "conv") else src_layer
                dst_conv = dst_layer.conv if hasattr(dst_layer, "conv") else dst_layer
                if isinstance(src_conv, nn.Conv1d):
                    if axis == "time":
                        dst_conv.weight.data = src_conv.weight.data.unsqueeze(-1)
                    else:
                        dst_conv.weight.data = src_conv.weight.data.unsqueeze(2)
                    if src_conv.bias is not None:
                        dst_conv.bias.data = src_conv.bias.data.clone()

        conv_src = getattr(gpkffn_src, f"conv_{k}")
        conv_src_actual = conv_src.conv if hasattr(conv_src, "conv") else conv_src
        conv_dst = getattr(gpkffn_dst, f"conv_{k}")
        conv_dst_actual = conv_dst.conv if hasattr(conv_dst, "conv") else conv_dst

        if axis == "time":
            conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(-1)
        else:
            conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(2)
        if conv_src_actual.bias is not None:
            conv_dst_actual.bias.data = conv_src_actual.bias.data.clone()


__all__ = [
    "ChannelLayerNorm2d",
    "SimpleGate2d",
    "CausalConv2dTime",
    "FreqCAB",
    "FreqGPKFFN",
    "StreamingConv2d",
    "StreamingCAB",
    "StreamingGPKFFN",
    "StreamingTSBlock",
]
