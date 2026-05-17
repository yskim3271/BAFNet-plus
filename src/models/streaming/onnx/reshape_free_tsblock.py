"""Reshape-free TSBlock variant + weight-preserving converter (cycle 11).

Drop-in replacement for :class:`src.models.backbone.TSBlock` that processes the
``[B, C, T, F]`` tensor without ever permuting/reshaping into the 3D
``[B*F, C, T]`` / ``[B*T, C, F]`` forms. The submodule attribute structure
mirrors the original TSBlock so the existing
:class:`~src.models.streaming.onnx.backbone_core.StateIterator` collects state
names in the same DFS order (the only difference: the time-stage stateful conv
type is :class:`~src.models.streaming.onnx.reshape_free.FunctionalStatefulConv2dTimeAxis`
instead of :class:`~src.models.streaming.onnx.functional_stateful.FunctionalStatefulConv1d`,
and its state tensor is 4D ``[B, C, padding, F]`` rather than 3D
``[B*F, C, padding]``).

State-name contract (per RF TSBlock, identical to default-path TSBlock):
    ``time_stage.{i}.0.dwconv`` — CAB depthwise (causal time axis, kernel K)
    ``time_stage.{i}.0.sca.0`` — SCA depthwise (causal time axis, kernel sca_K)
    ``time_stage.{i}.0.sca.1`` — SCA pointwise (stateless 1×1)
    ``time_stage.{i}.1.attn_{k}.0`` — GPKFFN attn depthwise (causal, kernel k)
    ``time_stage.{i}.1.attn_{k}.1`` — GPKFFN attn pointwise (stateless 1×1)
    ``time_stage.{i}.1.conv_{k}`` — GPKFFN conv (causal, kernel k)

The freq stage has no stateful convs (BAFNet+'s TSBlock hard-codes
``causal=False`` for the freq stage, with ``AdaptiveAvgPool1d(1)`` SCA), so the
state-name list is unchanged: 10 stateful convs per time-stage block × 2 blocks
= 20 stateful convs per RF TSBlock — same as the default-path TSBlock.

Numerical equivalence with the original TSBlock holds bit-identically (up to FP32
rounding ≤ ~1e-6) on a single block because:

* ``nn.Conv2d(kernel=(K, 1))`` on ``[B, C, T, F]`` with weight ``unsqueeze(-1)``
  reduces exactly to ``nn.Conv1d(kernel=K)`` applied per ``F`` column (the
  Conv2d kernel only spans the time axis; ``F`` indices are independent).
* ``nn.Conv2d(kernel=(1, K))`` on ``[B, C, T, F]`` with weight ``unsqueeze(2)``
  reduces exactly to ``nn.Conv1d(kernel=K)`` applied per ``T`` row.
* :class:`LayerNorm4dChannel` algebra equals ``LayerNorm1d`` algebra on the
  ``B*F`` (or ``B*T``) flattened tensor (channels are still on ``dim=1``).
* SCA freq-axis ``AdaptiveAvgPool1d(1)`` on ``[B*T, C, F]`` ≡ ``mean(dim=3,
  keepdim=True)`` on ``[B, C, T, F]``.

The converter :func:`convert_tsblock_to_reshape_free` reads weights from the
ORIGINAL (non-stateful) TSBlock submodules — the caller is expected to pass a
pre-conversion TSBlock (the same module ``ExportableBackboneCore.from_backbone``
captures lookahead from). All conv weights are reshaped via ``unsqueeze`` to
the target 4D layout without any value mutation, so the resulting RF TSBlock is
bit-identical to the original on any input.
"""

from __future__ import annotations

import logging
from typing import List, Optional, cast

import torch
import torch.nn as nn

from src.models.backbone import (
    CausalConv1d,
    ChannelAttentionBlock,
    GroupPrimeKernelFFN,
    LayerNorm1d,
    SimpleGate,
    TSBlock,
)
from src.models.streaming.onnx.reshape_free import (
    FunctionalStatefulConv2dTimeAxis,
    LayerNorm4dChannel,
    SimpleGate4d,
)

logger = logging.getLogger(__name__)


# ============================================================ reshape-free CAB / FFN


class ReshapeFreeChannelAttentionBlock(nn.Module):
    """4D analog of :class:`src.models.backbone.ChannelAttentionBlock`.

    Same submodule attribute names (``norm`` / ``pwconv1`` / ``dwconv`` / ``sg``
    / ``sca`` / ``pwconv2`` / optional ``beta`` / ``post_norm_layer``) so the
    state-name DFS walk by :class:`StateIterator` produces identical names to
    the default-path CAB. The ``axis`` argument controls which dimension the
    depthwise conv operates over:

    - ``axis='time'`` (causal): ``dwconv`` is a
      :class:`FunctionalStatefulConv2dTimeAxis`; ``sca`` is
      ``Sequential(FunctionalStatefulConv2dTimeAxis(depthwise), Conv2d(1, 1))``.
    - ``axis='freq'`` (non-causal, freq stage): ``dwconv`` is a stateless
      ``nn.Conv2d(kernel=(1, dw_kernel_size), padding=(0, dw_kernel_size//2))``;
      ``sca`` is ``Sequential(_FreqMeanPool, Conv2d(1, 1))`` where
      ``_FreqMeanPool`` reduces ``dim=3`` (== :class:`nn.AdaptiveAvgPool1d` on
      the original 3D freq-stage path).

    The ``causal`` parameter is intentionally not exposed: ``axis='time'`` ⇒ causal,
    ``axis='freq'`` ⇒ non-causal (matches the BAFNet+ convention enforced inside
    :class:`TSBlock` where ``causal_ts_block`` only flips the time-stage convs).

    Args:
        in_channels: Channel count (= ``dense_channel``).
        dw_kernel_size: Depthwise kernel size (3 in the deployed config).
        axis: ``'time'`` (causal, stateful) or ``'freq'`` (non-causal, stateless).
        sca_kernel_size: SCA depthwise kernel size for the time-axis variant
            (11 in the deployed config). Ignored for ``axis='freq'``.
        fold_residual_scale: If False (default), keeps the learnable
            ``beta [1, C, 1, 1]`` residual scale (same algebra as ``LayerNorm1d``
            CAB with the freshly-broadcast 4D shape).
        post_norm: If True, applies a trailing :class:`LayerNorm4dChannel`.
    """

    def __init__(
        self,
        in_channels: int,
        dw_kernel_size: int = 3,
        axis: str = "time",
        sca_kernel_size: int = 11,
        fold_residual_scale: bool = False,
        post_norm: bool = False,
    ) -> None:
        super().__init__()
        if axis not in ("time", "freq"):
            raise ValueError(f"axis must be 'time' or 'freq', got {axis!r}")
        self.in_channels = int(in_channels)
        self.axis = axis
        self.dw_kernel_size = int(dw_kernel_size)
        self.sca_kernel_size = int(sca_kernel_size)
        self.fold_residual_scale = bool(fold_residual_scale)
        self.post_norm = bool(post_norm)

        dw_channel = in_channels * 2

        self.norm = LayerNorm4dChannel(in_channels)
        # pwconv1: 1×1 conv on 4D (matches the BAFNet+ Conv1d(in_C, dw_C, 1)).
        self.pwconv1 = nn.Conv2d(in_channels, dw_channel, kernel_size=1, bias=True)

        if axis == "time":
            # Causal time-axis depthwise — get_padding(K) = K // 2 in the source
            # (CausalConv1d doubles it internally to a left-only pad of K-1 for
            # odd K). FunctionalStatefulConv2dTimeAxis follows the same
            # ``padding_size = padding * 2`` doubling.
            self.dwconv = FunctionalStatefulConv2dTimeAxis(
                in_channels=dw_channel,
                out_channels=dw_channel,
                kernel_size=dw_kernel_size,
                padding=dw_kernel_size // 2,
                groups=dw_channel,
                bias=True,
            )
        else:
            # Non-causal freq depthwise — symmetric F padding via padding=(0, K//2).
            self.dwconv = nn.Conv2d(
                dw_channel,
                dw_channel,
                kernel_size=(1, dw_kernel_size),
                padding=(0, dw_kernel_size // 2),
                groups=dw_channel,
                bias=True,
            )

        self.sg = SimpleGate4d()

        # SCA path. SimpleGate halves channels so SCA operates on dw_channel // 2.
        sca_channels = dw_channel // 2  # == in_channels
        if axis == "time":
            # BAFNet+ time-stage SCA: Sequential(CausalConv1d depthwise sca_K, Conv1d 1×1).
            self.sca = nn.Sequential(
                FunctionalStatefulConv2dTimeAxis(
                    in_channels=sca_channels,
                    out_channels=sca_channels,
                    kernel_size=sca_kernel_size,
                    padding=sca_kernel_size // 2,
                    groups=sca_channels,
                    bias=False,
                ),
                nn.Conv2d(sca_channels, sca_channels, kernel_size=1, bias=True),
            )
        else:
            # BAFNet+ freq-stage SCA: Sequential(AdaptiveAvgPool1d(1), Conv1d 1×1).
            # AdaptiveAvgPool1d(1) on [B*T, C, F] ≡ mean(dim=3, keepdim=True) on
            # [B, C, T, F]. The result [B, C, T, 1] broadcasts against the 4D x.
            self.sca = nn.Sequential(
                _FreqMeanPool(),
                nn.Conv2d(sca_channels, sca_channels, kernel_size=1, bias=True),
            )

        self.pwconv2 = nn.Conv2d(sca_channels, in_channels, kernel_size=1, bias=True)
        if not self.fold_residual_scale:
            self.register_parameter("beta", nn.Parameter(torch.zeros(1, in_channels, 1, 1)))
        if self.post_norm:
            self.post_norm_layer = LayerNorm4dChannel(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the reshape-free CAB.

        Note: when ``axis='time'`` the ``dwconv`` and ``sca.0`` modules are
        :class:`FunctionalStatefulConv2dTimeAxis` instances and require state I/O
        — calling ``self(x)`` directly will fail. The exportable core routes
        them through :class:`StateIterator`. This forward is provided as a
        development convenience for the ``axis='freq'`` variant (no state),
        and falls back to ``state=zero`` for the time variant (single-shot
        single-chunk usage, matching the original TSBlock's
        ``LayerNorm1d → Conv1d`` zero-init equivalence on a complete sequence).
        """
        skip = x
        x = self.norm(x)
        x = self.pwconv1(x)

        if isinstance(self.dwconv, FunctionalStatefulConv2dTimeAxis):
            _, _, _, f = x.shape
            zero_state = self.dwconv.init_state(batch_size=x.shape[0], freq_size=f, device=x.device, dtype=x.dtype)
            x, _ = self.dwconv(x, zero_state, state_frames=None)
        else:
            x = self.dwconv(x)

        x = self.sg(x)

        sca_out = x
        for layer in self.sca:
            if isinstance(layer, FunctionalStatefulConv2dTimeAxis):
                zero_state = layer.init_state(
                    batch_size=sca_out.shape[0], freq_size=sca_out.shape[3], device=sca_out.device, dtype=sca_out.dtype
                )
                sca_out, _ = layer(sca_out, zero_state, state_frames=None)
            else:
                sca_out = layer(sca_out)
        x = x * sca_out

        x = self.pwconv2(x)
        if self.fold_residual_scale:
            x = skip + x
        else:
            x = skip + x * self.beta
        if self.post_norm:
            x = self.post_norm_layer(x)
        return x


class ReshapeFreeGroupPrimeKernelFFN(nn.Module):
    """4D analog of :class:`src.models.backbone.GroupPrimeKernelFFN`.

    Same submodule attribute names — ``norm`` / ``proj_first`` (single 1×1
    Conv2d wrapped in a ``Sequential`` of length 1, mirroring the original
    BAFNet+ ``nn.Sequential(nn.Conv1d(C, mid_C, 1))``) / ``proj_last`` (same
    pattern) / optional ``scale [1, C, 1, 1]`` / ``post_norm_layer`` / per-
    kernel ``attn_{k}`` (``Sequential(depthwise, pointwise)``) and ``conv_{k}``
    (single depthwise). The depthwise / conv leaves are
    :class:`FunctionalStatefulConv2dTimeAxis` for ``axis='time'``, stateless
    ``nn.Conv2d(kernel=(1, k))`` for ``axis='freq'``.

    Args:
        in_channel: Channel count (= ``dense_channel``).
        kernel_list: Sorted-by-construction list of prime kernels (e.g.
            ``[3, 5, 7, 9]`` for time, ``[3, 11, 23, 31]`` for freq).
        axis: ``'time'`` or ``'freq'``.
        fold_residual_scale: If False (default), keeps the learnable
            ``scale [1, C, 1, 1]`` residual scale.
        post_norm: If True, applies a trailing :class:`LayerNorm4dChannel`.
    """

    def __init__(
        self,
        in_channel: int,
        kernel_list: List[int],
        axis: str = "time",
        fold_residual_scale: bool = False,
        post_norm: bool = False,
    ) -> None:
        super().__init__()
        if axis not in ("time", "freq"):
            raise ValueError(f"axis must be 'time' or 'freq', got {axis!r}")
        self.in_channel = int(in_channel)
        self.kernel_list = list(kernel_list)
        self.axis = axis
        self.fold_residual_scale = bool(fold_residual_scale)
        self.post_norm = bool(post_norm)

        self.expand_ratio = len(self.kernel_list)
        self.mid_channel = self.in_channel * self.expand_ratio

        # proj_first / proj_last: 1×1 wrapped in nn.Sequential (mirrors the
        # original BAFNet+ ``nn.Sequential(nn.Conv1d(in_C, mid_C, 1))``).
        self.proj_first = nn.Sequential(
            nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=1, bias=True),
        )
        self.proj_last = nn.Sequential(
            nn.Conv2d(self.mid_channel, self.in_channel, kernel_size=1, bias=True),
        )
        self.norm = LayerNorm4dChannel(self.in_channel)
        if not self.fold_residual_scale:
            self.register_parameter("scale", nn.Parameter(torch.zeros(1, self.in_channel, 1, 1)))
        if self.post_norm:
            self.post_norm_layer = LayerNorm4dChannel(self.in_channel)

        for ks in self.kernel_list:
            if axis == "time":
                attn_seq = nn.Sequential(
                    FunctionalStatefulConv2dTimeAxis(
                        in_channels=self.in_channel,
                        out_channels=self.in_channel,
                        kernel_size=ks,
                        padding=ks // 2,
                        groups=self.in_channel,
                        bias=True,
                    ),
                    nn.Conv2d(self.in_channel, self.in_channel, kernel_size=1, bias=True),
                )
                main_conv = FunctionalStatefulConv2dTimeAxis(
                    in_channels=self.in_channel,
                    out_channels=self.in_channel,
                    kernel_size=ks,
                    padding=ks // 2,
                    groups=self.in_channel,
                    bias=True,
                )
            else:
                attn_seq = nn.Sequential(
                    nn.Conv2d(
                        self.in_channel,
                        self.in_channel,
                        kernel_size=(1, ks),
                        padding=(0, ks // 2),
                        groups=self.in_channel,
                        bias=True,
                    ),
                    nn.Conv2d(self.in_channel, self.in_channel, kernel_size=1, bias=True),
                )
                main_conv = nn.Conv2d(
                    self.in_channel,
                    self.in_channel,
                    kernel_size=(1, ks),
                    padding=(0, ks // 2),
                    groups=self.in_channel,
                    bias=True,
                )
            setattr(self, f"attn_{ks}", attn_seq)
            setattr(self, f"conv_{ks}", main_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the reshape-free GPKFFN.

        Same single-shot-zero-state caveat as :meth:`ReshapeFreeChannelAttentionBlock.forward`
        for the ``axis='time'`` variant — production usage routes through the
        exportable core's :class:`StateIterator`.
        """
        shortcut = x
        x = self.norm(x)
        x = self.proj_first(x)

        chunks = list(torch.chunk(x, self.expand_ratio, dim=1))
        for i, ks in enumerate(self.kernel_list):
            attn_module = cast(nn.Sequential, getattr(self, f"attn_{ks}"))
            conv_module = getattr(self, f"conv_{ks}")
            attn_out = chunks[i]
            for layer in attn_module:
                if isinstance(layer, FunctionalStatefulConv2dTimeAxis):
                    zero_state = layer.init_state(
                        batch_size=attn_out.shape[0],
                        freq_size=attn_out.shape[3],
                        device=attn_out.device,
                        dtype=attn_out.dtype,
                    )
                    attn_out, _ = layer(attn_out, zero_state, state_frames=None)
                else:
                    attn_out = layer(attn_out)

            if isinstance(conv_module, FunctionalStatefulConv2dTimeAxis):
                zero_state = conv_module.init_state(
                    batch_size=chunks[i].shape[0],
                    freq_size=chunks[i].shape[3],
                    device=chunks[i].device,
                    dtype=chunks[i].dtype,
                )
                conv_out, _ = conv_module(chunks[i], zero_state, state_frames=None)
            else:
                conv_out = conv_module(chunks[i])

            chunks[i] = attn_out * conv_out

        x = torch.cat(chunks, dim=1)
        if self.fold_residual_scale:
            x = self.proj_last(x) + shortcut
        else:
            x = self.proj_last(x) * self.scale + shortcut
        if self.post_norm:
            x = self.post_norm_layer(x)
        return x


class _FreqMeanPool(nn.Module):
    """Reshape-free analog of :class:`nn.AdaptiveAvgPool1d` (output_size=1) over the freq axis.

    Reduces ``[B, C, T, F]`` to ``[B, C, T, 1]`` via ``mean(dim=3, keepdim=True)``.
    Pure 4D op, no internal reshape. Bit-identical to ``AdaptiveAvgPool1d(1)``
    applied to ``[B*T, C, F]`` (both compute the unweighted mean over ``F``).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=3, keepdim=True)


# ============================================================ reshape-free TSBlock


class ReshapeFreeTSBlock(nn.Module):
    """Drop-in 4D replacement for :class:`src.models.backbone.TSBlock`.

    Same constructor signature, same submodule attribute structure
    (``time_stage`` / ``freq_stage`` are ``nn.Sequential`` of
    ``nn.Sequential(CAB, GPKFFN)`` pairs; ``beta_t`` / ``beta_f`` are learnable
    ``[1, C, 1, 1]`` scalars) — the only difference is the layout of the
    intermediate tensors (``[B, C, T, F]`` end-to-end, no permute/reshape).

    Forward does **not** itself manage state — the exportable core's modified
    ``_forward_ts_block`` routes through :class:`StateIterator` for the
    stateful time-stage convs and calls each leaf directly otherwise. The
    in-module ``forward`` provided here is a single-shot convenience that
    zero-initialises every state on the fly (suitable for unit-parity tests
    against the original TSBlock).
    """

    def __init__(
        self,
        dense_channel: int = 64,
        time_block_num: int = 2,
        freq_block_num: int = 2,
        time_dw_kernel_size: int = 3,
        time_block_kernel: Optional[List[int]] = None,
        freq_block_kernel: Optional[List[int]] = None,
        sca_kernel_size: int = 11,
        fold_residual_scale: bool = False,
        post_norm: bool = False,
    ) -> None:
        super().__init__()
        if time_block_kernel is None:
            time_block_kernel = [3, 5, 7, 9]
        if freq_block_kernel is None:
            freq_block_kernel = [3, 11, 23, 31]
        self.dense_channel = int(dense_channel)
        self.time_block_num = int(time_block_num)
        self.freq_block_num = int(freq_block_num)
        self.time_dw_kernel_size = int(time_dw_kernel_size)
        self.time_block_kernel = list(time_block_kernel)
        self.freq_block_kernel = list(freq_block_kernel)
        self.sca_kernel_size = int(sca_kernel_size)
        self.fold_residual_scale = bool(fold_residual_scale)
        self.post_norm = bool(post_norm)

        time_stage: List[nn.Module] = []
        for _ in range(time_block_num):
            time_stage.append(
                nn.Sequential(
                    ReshapeFreeChannelAttentionBlock(
                        in_channels=dense_channel,
                        dw_kernel_size=time_dw_kernel_size,
                        axis="time",
                        sca_kernel_size=sca_kernel_size,
                        fold_residual_scale=fold_residual_scale,
                        post_norm=post_norm,
                    ),
                    ReshapeFreeGroupPrimeKernelFFN(
                        in_channel=dense_channel,
                        kernel_list=time_block_kernel,
                        axis="time",
                        fold_residual_scale=fold_residual_scale,
                        post_norm=post_norm,
                    ),
                )
            )
        freq_stage: List[nn.Module] = []
        for _ in range(freq_block_num):
            freq_stage.append(
                nn.Sequential(
                    ReshapeFreeChannelAttentionBlock(
                        in_channels=dense_channel,
                        dw_kernel_size=3,
                        axis="freq",
                        sca_kernel_size=sca_kernel_size,  # unused for freq
                        fold_residual_scale=fold_residual_scale,
                        post_norm=post_norm,
                    ),
                    ReshapeFreeGroupPrimeKernelFFN(
                        in_channel=dense_channel,
                        kernel_list=freq_block_kernel,
                        axis="freq",
                        fold_residual_scale=fold_residual_scale,
                        post_norm=post_norm,
                    ),
                )
            )
        self.time_stage = nn.Sequential(*time_stage)
        self.freq_stage = nn.Sequential(*freq_stage)

        self.register_parameter("beta_t", nn.Parameter(torch.zeros(1, dense_channel, 1, 1)))
        self.register_parameter("beta_f", nn.Parameter(torch.zeros(1, dense_channel, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-shot 4D forward — convenient for parity tests, NOT for streaming export.

        The exportable core handles state I/O explicitly; this forward
        zero-initialises every state and discards next-states. Bit-identical to
        :meth:`TSBlock.forward` on a complete sequence (the original also runs
        with implicit zero left-context).
        """
        residual = x
        x = self.time_stage(x) + residual * self.beta_t
        residual = x
        x = self.freq_stage(x) + residual * self.beta_f
        return x


# ============================================================ converter


def _copy_pwconv1d_to_2d(src_pw: nn.Conv1d, dst_pw: nn.Conv2d) -> None:
    """Copy a 1×1 ``nn.Conv1d`` weight tensor into a 1×1 ``nn.Conv2d`` (unsqueeze the
    trailing axis).

    The original ``Conv1d`` weight is ``[O, I/groups, 1]``; the target
    ``Conv2d`` weight is ``[O, I/groups, 1, 1]``. ``unsqueeze(-1)`` is exact
    (just adds a degenerate axis).
    """
    dst_pw.weight.data = src_pw.weight.data.unsqueeze(-1).clone()
    if src_pw.bias is not None and dst_pw.bias is not None:
        dst_pw.bias.data = src_pw.bias.data.clone()


def _copy_time_axis_dw_conv(src_conv: CausalConv1d, dst: FunctionalStatefulConv2dTimeAxis) -> None:
    """Copy a time-axis ``CausalConv1d`` weight into a 4D ``Conv2d(kernel=(K, 1))``.

    Original weight: ``[O, I/groups, K]``. Target weight (inside
    ``dst.conv``): ``[O, I/groups, K, 1]``. ``unsqueeze(-1)`` is exact.

    Raises:
        ValueError: If the source kernel size / groups do not match the target.
    """
    inner = src_conv.conv  # the nn.Conv1d inside CausalConv1d
    if int(inner.kernel_size[0]) != int(dst.kernel_size):
        raise ValueError(
            f"kernel size mismatch: src={int(inner.kernel_size[0])}, dst={int(dst.kernel_size)}"
        )
    dst.conv.weight.data = inner.weight.data.unsqueeze(-1).clone()
    if inner.bias is not None and dst.conv.bias is not None:
        dst.conv.bias.data = inner.bias.data.clone()


def _copy_freq_axis_dw_conv(src_conv: nn.Conv1d, dst_conv: nn.Conv2d) -> None:
    """Copy a freq-axis ``nn.Conv1d`` weight into a ``Conv2d(kernel=(1, K))``.

    Original weight: ``[O, I/groups, K]``. Target weight: ``[O, I/groups, 1, K]``.
    ``unsqueeze(2)`` is exact.
    """
    dst_conv.weight.data = src_conv.weight.data.unsqueeze(2).clone()
    if src_conv.bias is not None and dst_conv.bias is not None:
        dst_conv.bias.data = src_conv.bias.data.clone()


def _copy_layernorm1d(src_ln: LayerNorm1d, dst_ln: LayerNorm4dChannel) -> None:
    """Copy a :class:`LayerNorm1d` weight/bias into a :class:`LayerNorm4dChannel`.

    Both store ``[C]`` parameters — a bare ``.clone()`` is sufficient.
    """
    dst_ln.weight.data = src_ln.weight.data.clone()
    dst_ln.bias.data = src_ln.bias.data.clone()


def _convert_cab(
    src_cab: ChannelAttentionBlock,
    *,
    axis: str,
    sca_kernel_size: int,
    fold_residual_scale: bool,
    post_norm: bool,
) -> ReshapeFreeChannelAttentionBlock:
    """Build a reshape-free CAB from a trained :class:`ChannelAttentionBlock`."""
    in_channels = int(src_cab.norm.weight.shape[0])
    # Discover dw_kernel_size from src_cab.dwconv.
    src_dw = src_cab.dwconv
    if isinstance(src_dw, CausalConv1d):
        dw_kernel_size = int(src_dw.conv.kernel_size[0])
    elif isinstance(src_dw, nn.Conv1d):
        dw_kernel_size = int(src_dw.kernel_size[0])
    else:
        raise TypeError(f"unsupported CAB dwconv type: {type(src_dw).__name__}")

    dst = ReshapeFreeChannelAttentionBlock(
        in_channels=in_channels,
        dw_kernel_size=dw_kernel_size,
        axis=axis,
        sca_kernel_size=sca_kernel_size,
        fold_residual_scale=fold_residual_scale,
        post_norm=post_norm,
    )

    _copy_layernorm1d(src_cab.norm, dst.norm)
    _copy_pwconv1d_to_2d(src_cab.pwconv1, dst.pwconv1)

    if axis == "time":
        assert isinstance(src_dw, CausalConv1d)
        assert isinstance(dst.dwconv, FunctionalStatefulConv2dTimeAxis)
        _copy_time_axis_dw_conv(src_dw, dst.dwconv)
    else:
        assert isinstance(src_dw, nn.Conv1d)
        assert isinstance(dst.dwconv, nn.Conv2d)
        _copy_freq_axis_dw_conv(src_dw, dst.dwconv)

    # SCA path: time-stage SCA is Sequential(CausalConv1d depthwise, Conv1d 1×1);
    # freq-stage SCA is Sequential(AdaptiveAvgPool1d(1), Conv1d 1×1).
    src_sca = src_cab.sca
    if not isinstance(src_sca, nn.Sequential):
        raise TypeError(f"unsupported CAB sca type: {type(src_sca).__name__}")

    if axis == "time":
        sca_dw_src = src_sca[0]
        sca_pw_src = src_sca[1]
        if not isinstance(sca_dw_src, CausalConv1d):
            raise TypeError(f"time-stage CAB sca[0] expected CausalConv1d, got {type(sca_dw_src).__name__}")
        if not isinstance(sca_pw_src, nn.Conv1d):
            raise TypeError(f"time-stage CAB sca[1] expected nn.Conv1d, got {type(sca_pw_src).__name__}")
        sca_dw_dst = dst.sca[0]
        sca_pw_dst = dst.sca[1]
        assert isinstance(sca_dw_dst, FunctionalStatefulConv2dTimeAxis)
        assert isinstance(sca_pw_dst, nn.Conv2d)
        _copy_time_axis_dw_conv(sca_dw_src, sca_dw_dst)
        _copy_pwconv1d_to_2d(sca_pw_src, sca_pw_dst)
    else:
        # freq stage: AdaptiveAvgPool1d(1) carries no params; only the 1×1 Conv1d does.
        sca_pw_src = src_sca[1]
        if not isinstance(sca_pw_src, nn.Conv1d):
            raise TypeError(f"freq-stage CAB sca[1] expected nn.Conv1d, got {type(sca_pw_src).__name__}")
        sca_pw_dst = dst.sca[1]
        assert isinstance(sca_pw_dst, nn.Conv2d)
        _copy_pwconv1d_to_2d(sca_pw_src, sca_pw_dst)

    _copy_pwconv1d_to_2d(src_cab.pwconv2, dst.pwconv2)
    if not fold_residual_scale:
        dst.beta.data = src_cab.beta.data.unsqueeze(-1).clone()
    if post_norm:
        _copy_layernorm1d(src_cab.post_norm_layer, dst.post_norm_layer)
    return dst


def _convert_gpkffn(
    src_ffn: GroupPrimeKernelFFN,
    *,
    axis: str,
    fold_residual_scale: bool,
    post_norm: bool,
) -> ReshapeFreeGroupPrimeKernelFFN:
    """Build a reshape-free GPKFFN from a trained :class:`GroupPrimeKernelFFN`."""
    in_channel = int(src_ffn.in_channel)
    kernel_list = list(src_ffn.kernel_list)

    dst = ReshapeFreeGroupPrimeKernelFFN(
        in_channel=in_channel,
        kernel_list=kernel_list,
        axis=axis,
        fold_residual_scale=fold_residual_scale,
        post_norm=post_norm,
    )

    _copy_layernorm1d(src_ffn.norm, dst.norm)
    # proj_first / proj_last are nn.Sequential(nn.Conv1d(...)) — single-element sequentials.
    src_proj_first = src_ffn.proj_first[0]
    src_proj_last = src_ffn.proj_last[0]
    if not isinstance(src_proj_first, nn.Conv1d):
        raise TypeError(f"GPKFFN proj_first[0] expected nn.Conv1d, got {type(src_proj_first).__name__}")
    if not isinstance(src_proj_last, nn.Conv1d):
        raise TypeError(f"GPKFFN proj_last[0] expected nn.Conv1d, got {type(src_proj_last).__name__}")
    _copy_pwconv1d_to_2d(src_proj_first, dst.proj_first[0])
    _copy_pwconv1d_to_2d(src_proj_last, dst.proj_last[0])

    if not fold_residual_scale:
        dst.scale.data = src_ffn.scale.data.unsqueeze(-1).clone()
    if post_norm:
        _copy_layernorm1d(src_ffn.post_norm_layer, dst.post_norm_layer)

    for ks in kernel_list:
        src_attn = getattr(src_ffn, f"attn_{ks}")
        src_conv = getattr(src_ffn, f"conv_{ks}")
        dst_attn = getattr(dst, f"attn_{ks}")
        dst_conv = getattr(dst, f"conv_{ks}")

        # attn = Sequential(depthwise_conv, pointwise_Conv1d).
        if not isinstance(src_attn, nn.Sequential):
            raise TypeError(f"GPKFFN attn_{ks} expected nn.Sequential, got {type(src_attn).__name__}")
        src_attn_dw = src_attn[0]
        src_attn_pw = src_attn[1]
        if not isinstance(src_attn_pw, nn.Conv1d):
            raise TypeError(f"GPKFFN attn_{ks}[1] expected nn.Conv1d, got {type(src_attn_pw).__name__}")

        dst_attn_dw = dst_attn[0]
        dst_attn_pw = dst_attn[1]

        if axis == "time":
            if not isinstance(src_attn_dw, CausalConv1d):
                raise TypeError(f"time-stage GPKFFN attn_{ks}[0] expected CausalConv1d, got {type(src_attn_dw).__name__}")
            assert isinstance(dst_attn_dw, FunctionalStatefulConv2dTimeAxis)
            _copy_time_axis_dw_conv(src_attn_dw, dst_attn_dw)
        else:
            if not isinstance(src_attn_dw, nn.Conv1d):
                raise TypeError(f"freq-stage GPKFFN attn_{ks}[0] expected nn.Conv1d, got {type(src_attn_dw).__name__}")
            assert isinstance(dst_attn_dw, nn.Conv2d)
            _copy_freq_axis_dw_conv(src_attn_dw, dst_attn_dw)
        assert isinstance(dst_attn_pw, nn.Conv2d)
        _copy_pwconv1d_to_2d(src_attn_pw, dst_attn_pw)

        # conv_{k}: single depthwise.
        if axis == "time":
            if not isinstance(src_conv, CausalConv1d):
                raise TypeError(f"time-stage GPKFFN conv_{ks} expected CausalConv1d, got {type(src_conv).__name__}")
            assert isinstance(dst_conv, FunctionalStatefulConv2dTimeAxis)
            _copy_time_axis_dw_conv(src_conv, dst_conv)
        else:
            if not isinstance(src_conv, nn.Conv1d):
                raise TypeError(f"freq-stage GPKFFN conv_{ks} expected nn.Conv1d, got {type(src_conv).__name__}")
            assert isinstance(dst_conv, nn.Conv2d)
            _copy_freq_axis_dw_conv(src_conv, dst_conv)
    return dst


def convert_tsblock_to_reshape_free(tsblock: TSBlock) -> ReshapeFreeTSBlock:
    """Build a :class:`ReshapeFreeTSBlock` from a trained :class:`TSBlock` (weight copy).

    The source :class:`TSBlock` is **not** mutated. The destination has the same
    submodule structure (same attribute paths), the same trained weights
    (reshaped via ``unsqueeze`` to 4D — no value change), and the same
    ``fold_residual_scale`` / ``post_norm`` algebraic flags inferred from the
    presence of the corresponding submodule.

    Args:
        tsblock: Original :class:`TSBlock` (causal=True for the canonical
            BAFNet+ time-stage; the freq-stage submodules are always
            ``causal=False`` per :meth:`TSBlock.__init__`).

    Returns:
        A :class:`ReshapeFreeTSBlock` with bit-equivalent weights (subject to
        the 4D layout) and the same forward semantics.
    """
    if not isinstance(tsblock, TSBlock):
        raise TypeError(f"convert_tsblock_to_reshape_free expects a TSBlock, got {type(tsblock).__name__}")

    # Discover algebraic flags from the first time-stage CAB.
    first_time = tsblock.time_stage[0]
    first_time_cab = first_time[0]
    first_time_ffn = first_time[1]
    fold_residual_scale = not hasattr(first_time_cab, "beta")
    post_norm = hasattr(first_time_cab, "post_norm_layer")

    dense_channel = int(tsblock.dense_channel)
    time_block_num = len(tsblock.time_stage)
    freq_block_num = len(tsblock.freq_stage)

    # Discover kernel sizes.
    if isinstance(first_time_cab.dwconv, CausalConv1d):
        time_dw_kernel_size = int(first_time_cab.dwconv.conv.kernel_size[0])
    elif isinstance(first_time_cab.dwconv, nn.Conv1d):
        time_dw_kernel_size = int(first_time_cab.dwconv.kernel_size[0])
    else:
        raise TypeError(f"unsupported time-stage CAB dwconv type: {type(first_time_cab.dwconv).__name__}")
    time_block_kernel = list(first_time_ffn.kernel_list)
    freq_block_kernel = list(tsblock.freq_stage[0][1].kernel_list)

    # SCA kernel size from the time-stage CAB sca depthwise.
    sca_kernel_size = 11  # default
    src_sca = first_time_cab.sca
    if isinstance(src_sca, nn.Sequential) and isinstance(src_sca[0], CausalConv1d):
        sca_kernel_size = int(src_sca[0].conv.kernel_size[0])

    dst = ReshapeFreeTSBlock(
        dense_channel=dense_channel,
        time_block_num=time_block_num,
        freq_block_num=freq_block_num,
        time_dw_kernel_size=time_dw_kernel_size,
        time_block_kernel=time_block_kernel,
        freq_block_kernel=freq_block_kernel,
        sca_kernel_size=sca_kernel_size,
        fold_residual_scale=fold_residual_scale,
        post_norm=post_norm,
    )

    # Copy time-stage blocks.
    for i, src_block in enumerate(tsblock.time_stage):
        src_cab = src_block[0]
        src_ffn = src_block[1]
        if not isinstance(src_cab, ChannelAttentionBlock):
            raise TypeError(f"time_stage.{i}.0 expected ChannelAttentionBlock, got {type(src_cab).__name__}")
        if not isinstance(src_ffn, GroupPrimeKernelFFN):
            raise TypeError(f"time_stage.{i}.1 expected GroupPrimeKernelFFN, got {type(src_ffn).__name__}")
        dst_block = dst.time_stage[i]
        dst_block[0] = _convert_cab(
            src_cab,
            axis="time",
            sca_kernel_size=sca_kernel_size,
            fold_residual_scale=fold_residual_scale,
            post_norm=post_norm,
        )
        dst_block[1] = _convert_gpkffn(
            src_ffn,
            axis="time",
            fold_residual_scale=fold_residual_scale,
            post_norm=post_norm,
        )

    # Copy freq-stage blocks.
    for i, src_block in enumerate(tsblock.freq_stage):
        src_cab = src_block[0]
        src_ffn = src_block[1]
        if not isinstance(src_cab, ChannelAttentionBlock):
            raise TypeError(f"freq_stage.{i}.0 expected ChannelAttentionBlock, got {type(src_cab).__name__}")
        if not isinstance(src_ffn, GroupPrimeKernelFFN):
            raise TypeError(f"freq_stage.{i}.1 expected GroupPrimeKernelFFN, got {type(src_ffn).__name__}")
        dst_block = dst.freq_stage[i]
        dst_block[0] = _convert_cab(
            src_cab,
            axis="freq",
            sca_kernel_size=sca_kernel_size,  # unused for freq
            fold_residual_scale=fold_residual_scale,
            post_norm=post_norm,
        )
        dst_block[1] = _convert_gpkffn(
            src_ffn,
            axis="freq",
            fold_residual_scale=fold_residual_scale,
            post_norm=post_norm,
        )

    # beta_t / beta_f: (1, C, 1) → (1, C, 1, 1).
    dst.beta_t.data = tsblock.beta_t.data.unsqueeze(-1).clone()
    dst.beta_f.data = tsblock.beta_f.data.unsqueeze(-1).clone()

    return dst


def convert_sequence_block_to_reshape_free(sequence_block: nn.Sequential) -> nn.Sequential:
    """Build a parallel :class:`nn.Sequential` of :class:`ReshapeFreeTSBlock` from
    BAFNet+'s :class:`nn.Sequential` of trained :class:`TSBlock` instances.

    The result is a fresh ``nn.Sequential`` (same length, same per-block
    submodule attribute paths) — the source is unchanged.

    Args:
        sequence_block: The ``Backbone.sequence_block`` attribute (an
            ``nn.Sequential`` of N TSBlocks).

    Returns:
        An ``nn.Sequential`` of ``ReshapeFreeTSBlock`` instances with copied
        weights. Same length.
    """
    if not isinstance(sequence_block, nn.Sequential):
        raise TypeError(
            f"convert_sequence_block_to_reshape_free expects an nn.Sequential, "
            f"got {type(sequence_block).__name__}"
        )
    rf_blocks: List[nn.Module] = []
    for i, tsblock in enumerate(sequence_block):
        if not isinstance(tsblock, TSBlock):
            raise TypeError(f"sequence_block[{i}] expected TSBlock, got {type(tsblock).__name__}")
        rf_blocks.append(convert_tsblock_to_reshape_free(tsblock))
    return nn.Sequential(*rf_blocks)


__all__ = [
    "ReshapeFreeChannelAttentionBlock",
    "ReshapeFreeGroupPrimeKernelFFN",
    "ReshapeFreeTSBlock",
    "convert_tsblock_to_reshape_free",
    "convert_sequence_block_to_reshape_free",
]
