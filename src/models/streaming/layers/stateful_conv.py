"""Stateful convolution layers for streaming inference.

Stateful versions of ``CausalConv1d``, ``AsymmetricConv2d`` and ``CausalConv2d``
that keep an activation buffer (``_state``) carrying the previous chunk's last
frames, using it as left-padding instead of zeros. This removes the
zero-padding discontinuity at chunk boundaries so chunk-by-chunk output equals
the full-sequence output (within float32 reordering tolerance), provided the
training model used the same asymmetric padding.

Key classes:
    - ``StatefulCausalConv1d``: stateful ``CausalConv1d`` (1D temporal data).
    - ``StatefulAsymmetricConv2d``: stateful ``AsymmetricConv2d`` (2D spectrogram).
    - ``StatefulCausalConv2d``: stateful ``CausalConv2d``.

State is saved from the **input** (input-level continuity), bounded to the
leading ``state_frames`` frames — supplied explicitly or via
``StateFramesContext`` — so lookahead frames in an extended input do not corrupt
the state. Right (future) time padding is always zeros; the streaming wrapper is
responsible for feeding an extended input whose lookahead frames land in those
slots.

Ported from LaCoSENet ``src/models/streaming/layers/stateful_conv.py``; adjusted
for BAFNet+ module paths (``src.models.backbone`` / ``src.models.streaming.context``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn  # 'fn' avoids clashing with the 'F' (frequency) variable

from src.models.streaming.context import (
    check_batch_size_change,
    check_streaming_allowed,
    get_state_frames_context,
    sync_state_device_dtype,
)

if TYPE_CHECKING:
    from src.models.backbone import AsymmetricConv2d, CausalConv1d, CausalConv2d

logger = logging.getLogger(__name__)


class StatefulLayerMixin:
    """Common interface for stateful streaming layers.

    Provides ``reset_state()``, ``set_streaming()`` and the ``state`` property.
    Subclasses must set ``self._state = None`` and ``self._streaming = False`` in
    their own ``__init__`` (after ``super().__init__()``) and must also inherit
    from ``nn.Module`` (this mixin reads ``self.training``).
    """

    _state: Optional[torch.Tensor]
    _streaming: bool
    training: bool  # provided by nn.Module (the mixin's host class)

    def reset_state(self) -> None:
        """Clear the state buffer (call before a new utterance)."""
        self._state = None

    def set_streaming(self, streaming: bool) -> None:
        """Enable/disable streaming mode with safety checks.

        Args:
            streaming: ``True`` to enable streaming mode.

        Raises:
            RuntimeError: If streaming is enabled while the module is training.
        """
        if streaming:
            check_streaming_allowed(self.training, type(self).__name__)
        if self._streaming and not streaming:
            self.reset_state()
        self._streaming = streaming

    @property
    def state(self) -> Optional[torch.Tensor]:
        """Current state buffer (``None`` before the first streamed chunk)."""
        return self._state


class StatefulCausalConv1d(StatefulLayerMixin, nn.Module):
    """Stateful ``CausalConv1d`` for streaming inference.

    Maintains a ``[B, C, padding_size]`` buffer with the previous chunk's last
    ``padding_size`` input frames, used as left-padding instead of zeros.

    Attributes:
        padding_size: Left-padding size (frames). Matches the original
            ``CausalConv1d`` which doubles its ``padding`` argument.
        conv: Underlying ``nn.Conv1d`` (padding=0).
        _state: Previous-activation buffer ``[B, C, padding_size]`` (or ``None``).
        _streaming: Whether streaming mode is enabled.

    Notes:
        - State is only maintained while ``streaming=True``.
        - State is reset when streaming mode is disabled.
        - The first chunk uses zero padding (no previous context).
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
        # Original CausalConv1d doubles the padding argument (all on the left).
        self.padding_size = padding * 2
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
        self._state: Optional[torch.Tensor] = None
        self._streaming = False

    @classmethod
    def from_causal_conv(cls, causal_conv: "CausalConv1d") -> "StatefulCausalConv1d":
        """Build a ``StatefulCausalConv1d`` from a ``CausalConv1d`` (copies weights).

        Uses a full ``__init__`` + ``load_state_dict`` (not reference sharing) so
        the result is independent of the source module and stays
        save/load- and ONNX-export-compatible.

        Args:
            causal_conv: Source ``CausalConv1d``. ``causal_conv.padding`` is
                already ``padding * 2`` (see ``backbone.py``), so the original
                argument is recovered as ``padding // 2``.

        Returns:
            A new ``StatefulCausalConv1d`` with copied weights.
        """
        orig_conv = causal_conv.conv
        instance = cls(
            in_channels=orig_conv.in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size[0],
            padding=causal_conv.padding // 2,
            stride=orig_conv.stride[0],
            dilation=orig_conv.dilation[0],
            groups=orig_conv.groups,
            bias=orig_conv.bias is not None,
        )
        instance.conv.load_state_dict(orig_conv.state_dict())
        return instance

    def forward(
        self,
        x: torch.Tensor,
        state_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass, optionally stateful.

        Args:
            x: Input tensor ``[B, C, T]``.
            state_frames: Number of leading frames allowed to update state. If
                ``None``, falls back to ``StateFramesContext``, then to ``T``.

        Returns:
            Output tensor ``[B, C_out, T]``.
        """
        if not self._streaming:
            # Non-streaming: original zero-padded causal behaviour.
            x = fn.pad(x, [self.padding_size, 0])
            return self.conv(x)

        B, C, T = x.shape
        device, dtype = x.device, x.dtype

        if check_batch_size_change(self._state, B, "StatefulCausalConv1d"):
            self.reset_state()
        self._state = sync_state_device_dtype(self._state, device, dtype)

        if self._state is None:
            left_pad = torch.zeros(B, C, self.padding_size, device=device, dtype=dtype)
        else:
            left_pad = self._state

        x_padded = torch.cat([left_pad, x], dim=2)

        # Update state from the INPUT (input-level continuity), bounded to the
        # leading state_frames frames so lookahead frames cannot corrupt it.
        if state_frames is None:
            state_frames = get_state_frames_context()
        effective_t = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_t]

        if effective_t >= self.padding_size:
            self._state = x_for_state[:, :, -self.padding_size :].clone().detach()
        else:
            keep = self.padding_size - effective_t
            if self._state is not None:
                old_part = self._state[:, :, -keep:]
            else:
                old_part = torch.zeros(B, C, keep, device=device, dtype=dtype)
            self._state = torch.cat([old_part, x_for_state], dim=2).clone().detach()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[StatefulCausalConv1d] streaming=%s state=%s input=%s state_frames=%s",
                self._streaming,
                "init" if self._state is not None else "None",
                list(x.shape),
                effective_t,
            )

        return self.conv(x_padded)


class StatefulAsymmetricConv2d(StatefulLayerMixin, nn.Module):
    """Stateful ``AsymmetricConv2d`` for streaming inference.

    Asymmetric padding on the time axis; symmetric (always-zero) padding on the
    frequency axis. State carries only the left (past) time padding — the right
    (future) time padding is zero-filled by default; in a buffered streaming
    wrapper the extended input fills those slots with real lookahead frames.

    Attributes:
        time_padding_left: Left (past) time padding size.
        time_padding_right: Right (future) time padding size.
        freq_padding: Symmetric frequency padding (one side).
        padding_ratio: ``(left_ratio, right_ratio)`` used to split time padding.
        conv: Underlying ``nn.Conv2d`` (padding=0).
        _state: Previous-activation buffer ``[B, C, time_padding_left, F_padded]``.
        _streaming: Whether streaming mode is enabled.
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
        # Reproduce AsymmetricConv2d's Python round() split, including the
        # actual_total != total re-derivation of the right padding.
        time_padding_total = padding[0] * 2
        freq_padding = padding[1]
        left_ratio, right_ratio = padding_ratio

        self.time_padding_left = round(time_padding_total * left_ratio)
        self.time_padding_right = round(time_padding_total * right_ratio)
        if self.time_padding_left + self.time_padding_right != time_padding_total:
            self.time_padding_right = time_padding_total - self.time_padding_left

        self.freq_padding = freq_padding
        self.padding_ratio = padding_ratio

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
        self._state: Optional[torch.Tensor] = None
        self._streaming = False

    @classmethod
    def from_asymmetric_conv(cls, asym_conv: "AsymmetricConv2d") -> "StatefulAsymmetricConv2d":
        """Build a ``StatefulAsymmetricConv2d`` from an ``AsymmetricConv2d`` (copies weights).

        Args:
            asym_conv: Source ``AsymmetricConv2d``. Its ``padding`` field is
                stored as ``(freq, freq, time_left, time_right)``; the original
                time padding argument is recovered as
                ``(time_left + time_right) // 2``.

        Returns:
            A new ``StatefulAsymmetricConv2d`` with copied weights.

        Note:
            ``stride`` and ``dilation`` are preserved as the original 2D tuples
            (both time and freq axes must match the source conv).
        """
        orig_conv = asym_conv.conv
        total_time_padding = asym_conv.time_padding_left + asym_conv.time_padding_right
        original_time_padding = total_time_padding // 2
        original_freq_padding = asym_conv.padding[0]  # freq_padding stored directly

        instance = cls(
            in_channels=orig_conv.in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            padding=(original_time_padding, original_freq_padding),
            padding_ratio=asym_conv.padding_ratio,
            stride=orig_conv.stride,
            dilation=orig_conv.dilation,
            groups=orig_conv.groups,
            bias=orig_conv.bias is not None,
        )
        instance.conv.load_state_dict(orig_conv.state_dict())
        return instance

    def forward(
        self,
        x: torch.Tensor,
        state_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass, optionally stateful.

        Args:
            x: Input tensor ``[B, C, T, F]`` (``F`` is the frequency dim).
            state_frames: Number of leading frames allowed to update state. If
                ``None``, falls back to ``StateFramesContext``, then to ``T``.

        Returns:
            Output tensor ``[B, C_out, T, F]``.

        Notes:
            - Frequency padding is always symmetric and zero.
            - State is saved AFTER frequency padding is applied (so it includes
              the freq-padded width).
            - When ``state_frames < T``, the lookahead tail is processed but does
              not enter the next chunk's state.
        """
        B, C, T, _ = x.shape
        device, dtype = x.device, x.dtype

        if not self._streaming:
            pad = (
                self.freq_padding,
                self.freq_padding,
                self.time_padding_left,
                self.time_padding_right,
            )
            x = fn.pad(x, pad)
            return self.conv(x)

        if check_batch_size_change(self._state, B, "StatefulAsymmetricConv2d"):
            self.reset_state()
        self._state = sync_state_device_dtype(self._state, device, dtype)

        # 1. Frequency padding (always symmetric, always zero — no time dependency).
        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        freq_padded = x.shape[3]

        # 2. Time padding: left = state (or zeros on first chunk), right = zeros.
        if self._state is None:
            left_pad = torch.zeros(B, C, self.time_padding_left, freq_padded, device=device, dtype=dtype)
        else:
            left_pad = self._state
        right_pad = torch.zeros(B, C, self.time_padding_right, freq_padded, device=device, dtype=dtype)
        x_padded = torch.cat([left_pad, x, right_pad], dim=2)

        # 3. Update state: last time_padding_left freq-padded frames of the input.
        if state_frames is None:
            state_frames = get_state_frames_context()
        effective_t = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_t, :]

        if effective_t >= self.time_padding_left:
            self._state = x_for_state[:, :, -self.time_padding_left :, :].clone().detach()
        else:
            keep = self.time_padding_left - effective_t
            if self._state is not None:
                old_part = self._state[:, :, -keep:, :]
            else:
                old_part = torch.zeros(B, C, keep, freq_padded, device=device, dtype=dtype)
            self._state = torch.cat([old_part, x_for_state], dim=2).clone().detach()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[StatefulAsymmetricConv2d] streaming=%s state=%s input=%s state_frames=%s",
                self._streaming,
                "init" if self._state is not None else "None",
                list(x.shape),
                effective_t,
            )

        return self.conv(x_padded)


class StatefulCausalConv2d(StatefulLayerMixin, nn.Module):
    """Stateful ``CausalConv2d`` for streaming inference.

    Like ``StatefulAsymmetricConv2d`` but fully causal on the time axis
    (``padding_ratio == (1.0, 0.0)``; no right padding).

    Attributes:
        time_padding: Left (past) time padding size.
        freq_padding: Symmetric frequency padding (one side).
        conv: Underlying ``nn.Conv2d`` (padding=0).
        _state: Previous-activation buffer ``[B, C, time_padding, F_padded]``.
        _streaming: Whether streaming mode is enabled.
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
        # Original CausalConv2d doubles the time padding (all on the left).
        self.time_padding = padding[0] * 2
        self.freq_padding = padding[1]
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
        self._state: Optional[torch.Tensor] = None
        self._streaming = False

    @classmethod
    def from_causal_conv2d(cls, causal_conv: "CausalConv2d") -> "StatefulCausalConv2d":
        """Build a ``StatefulCausalConv2d`` from a ``CausalConv2d`` (copies weights).

        Args:
            causal_conv: Source ``CausalConv2d``. Its ``padding`` field is stored
                as ``(freq, freq, time_total, 0)`` where ``time_total`` is
                already doubled; the original argument is ``time_total // 2``.

        Returns:
            A new ``StatefulCausalConv2d`` with copied weights.
        """
        orig_conv = causal_conv.conv
        time_padding_total = causal_conv.padding[2]
        freq_padding = causal_conv.padding[0]
        instance = cls(
            in_channels=orig_conv.in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            padding=(time_padding_total // 2, freq_padding),
            stride=orig_conv.stride,
            dilation=orig_conv.dilation,
            groups=orig_conv.groups,
            bias=orig_conv.bias is not None,
        )
        instance.conv.load_state_dict(orig_conv.state_dict())
        return instance

    def forward(
        self,
        x: torch.Tensor,
        state_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass, optionally stateful.

        Args:
            x: Input tensor ``[B, C, T, F]``.
            state_frames: Number of leading frames allowed to update state. If
                ``None``, falls back to ``StateFramesContext``, then to ``T``.

        Returns:
            Output tensor ``[B, C_out, T, F]``.
        """
        B, C, T, _ = x.shape
        device, dtype = x.device, x.dtype

        if not self._streaming:
            pad = (self.freq_padding, self.freq_padding, self.time_padding, 0)
            x = fn.pad(x, pad)
            return self.conv(x)

        if check_batch_size_change(self._state, B, "StatefulCausalConv2d"):
            self.reset_state()
        self._state = sync_state_device_dtype(self._state, device, dtype)

        x = fn.pad(x, (self.freq_padding, self.freq_padding, 0, 0))
        freq_padded = x.shape[3]

        if self._state is None:
            left_pad = torch.zeros(B, C, self.time_padding, freq_padded, device=device, dtype=dtype)
        else:
            left_pad = self._state
        x_padded = torch.cat([left_pad, x], dim=2)

        if state_frames is None:
            state_frames = get_state_frames_context()
        effective_t = state_frames if state_frames is not None else T
        x_for_state = x[:, :, :effective_t, :]

        if effective_t >= self.time_padding:
            self._state = x_for_state[:, :, -self.time_padding :, :].clone().detach()
        else:
            keep = self.time_padding - effective_t
            if self._state is not None:
                old_part = self._state[:, :, -keep:, :]
            else:
                old_part = torch.zeros(B, C, keep, freq_padded, device=device, dtype=dtype)
            self._state = torch.cat([old_part, x_for_state], dim=2).clone().detach()

        return self.conv(x_padded)


__all__ = [
    "StatefulLayerMixin",
    "StatefulCausalConv1d",
    "StatefulAsymmetricConv2d",
    "StatefulCausalConv2d",
]
