"""
Reshape-Free Model Converter.

This module provides utilities to convert existing Backbone models to
Reshape-Free versions that eliminate reshape operations for batch_size=1
optimized inference.

Key conversions:
    - LayerNorm1d → ChannelLayerNorm2d
    - TSBlock → ReshapeFreeTSBlock / StatefulReshapeFreeTSBlock (with weight transfer)

Usage:
    >>> from src.models.streaming.converters.reshape_free_converter import (
    ...     convert_ts_block_to_stateful_reshape_free,
    ... )
    >>> stateful_block = convert_ts_block_to_stateful_reshape_free(ts_block)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.streaming.layers.reshape_free import (
    ReshapeFreeCAB,
    ReshapeFreeGPKFFN,
    ReshapeFreeTSBlock,
)

logger = logging.getLogger(__name__)


def _convert_layernorm1d_to_channel(
    ln1d: nn.Module,
) -> "ChannelLayerNorm2d":
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
    from src.models.streaming.layers.reshape_free import ChannelLayerNorm2d

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


def _is_causal_conv(conv_module: nn.Module) -> bool:
    """Check if conv module is a CausalConv1d."""
    cls_name = conv_module.__class__.__name__
    if "Causal" in cls_name:
        return True
    if hasattr(conv_module, "conv") and hasattr(conv_module, "padding"):
        return conv_module.conv.padding[0] == 0
    return False


def convert_cab_to_reshape_free(
    cab: nn.Module,
    axis: str,
    causal: Optional[bool] = None,
) -> ReshapeFreeCAB:
    """
    Convert ChannelAttentionBlock to ReshapeFreeCAB.

    Args:
        cab: Original CAB module
        axis: Processing axis ('time' or 'freq')
        causal: If provided, use this value; otherwise auto-detect from dwconv

    Returns:
        ReshapeFreeCAB with transferred weights
    """
    channels = cab.norm.weight.shape[0] if hasattr(cab.norm, "weight") else 64

    if hasattr(cab.dwconv, "conv"):
        kernel_size = cab.dwconv.conv.kernel_size[0]
    elif hasattr(cab.dwconv, "kernel_size"):
        kernel_size = cab.dwconv.kernel_size[0]
    else:
        kernel_size = 3

    if causal is None:
        causal = _is_causal_conv(cab.dwconv) if axis == "time" else False

    rf_cab = ReshapeFreeCAB(channels=channels, kernel_size=kernel_size, axis=axis, causal=causal)

    # Transfer weights
    if hasattr(cab, "norm"):
        rf_cab.norm = _convert_layernorm1d_to_channel(cab.norm)

    if hasattr(cab, "pwconv1"):
        rf_cab.pwconv1.weight.data = cab.pwconv1.weight.data.unsqueeze(-1)
        if cab.pwconv1.bias is not None:
            rf_cab.pwconv1.bias.data = cab.pwconv1.bias.data.clone()

    if hasattr(cab, "dwconv"):
        dwconv_src = cab.dwconv.conv if hasattr(cab.dwconv, "conv") else cab.dwconv
        if axis == "time":
            rf_cab.dwconv.weight.data = dwconv_src.weight.data.unsqueeze(-1)
        else:
            rf_cab.dwconv.weight.data = dwconv_src.weight.data.unsqueeze(2)
        if dwconv_src.bias is not None:
            rf_cab.dwconv.bias.data = dwconv_src.bias.data.clone()

    if hasattr(cab, "sca") and isinstance(cab.sca, nn.Sequential) and len(cab.sca) > 1:
        sca_conv = cab.sca[-1]
        if isinstance(sca_conv, nn.Conv1d):
            rf_cab.sca_conv.weight.data = sca_conv.weight.data.unsqueeze(-1)
            if sca_conv.bias is not None:
                rf_cab.sca_conv.bias.data = sca_conv.bias.data.clone()

    if hasattr(cab, "pwconv2"):
        rf_cab.pwconv2.weight.data = cab.pwconv2.weight.data.unsqueeze(-1)
        if cab.pwconv2.bias is not None:
            rf_cab.pwconv2.bias.data = cab.pwconv2.bias.data.clone()

    if hasattr(cab, "beta"):
        rf_cab.beta.data = cab.beta.data.unsqueeze(-1)

    return rf_cab


def convert_gpkffn_to_reshape_free(
    gpkffn: nn.Module,
    axis: str,
    causal: Optional[bool] = None,
) -> ReshapeFreeGPKFFN:
    """
    Convert GroupPrimeKernelFFN to ReshapeFreeGPKFFN.

    Args:
        gpkffn: Original GPKFFN module
        axis: Processing axis ('time' or 'freq')
        causal: If provided, use this value; otherwise auto-detect

    Returns:
        ReshapeFreeGPKFFN with transferred weights
    """
    channels = gpkffn.in_channel
    kernel_list = gpkffn.kernel_list

    if causal is None and axis == "time":
        first_k = kernel_list[0]
        conv_src = getattr(gpkffn, f"conv_{first_k}", None)
        causal = _is_causal_conv(conv_src) if conv_src else False
    elif causal is None:
        causal = False

    rf_gpkffn = ReshapeFreeGPKFFN(
        channels=channels,
        kernel_list=kernel_list,
        axis=axis,
        causal=causal,
    )

    if hasattr(gpkffn, "norm"):
        rf_gpkffn.norm = _convert_layernorm1d_to_channel(gpkffn.norm)

    if hasattr(gpkffn, "proj_first"):
        proj_conv = gpkffn.proj_first[0] if isinstance(gpkffn.proj_first, nn.Sequential) else gpkffn.proj_first
        rf_gpkffn.proj_first.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            rf_gpkffn.proj_first.bias.data = proj_conv.bias.data.clone()

    if hasattr(gpkffn, "proj_last"):
        proj_conv = gpkffn.proj_last[0] if isinstance(gpkffn.proj_last, nn.Sequential) else gpkffn.proj_last
        rf_gpkffn.proj_last.weight.data = proj_conv.weight.data.unsqueeze(-1)
        if proj_conv.bias is not None:
            rf_gpkffn.proj_last.bias.data = proj_conv.bias.data.clone()

    if hasattr(gpkffn, "scale"):
        rf_gpkffn.scale.data = gpkffn.scale.data.unsqueeze(-1)

    for k in kernel_list:
        attn_src = getattr(gpkffn, f"attn_{k}")
        attn_dst = getattr(rf_gpkffn, f"attn_{k}")

        if isinstance(attn_src, nn.Sequential):
            for i, (src_layer, dst_layer) in enumerate(zip(attn_src, attn_dst)):
                src_conv = src_layer.conv if hasattr(src_layer, "conv") else src_layer
                dst_conv = dst_layer.conv if hasattr(dst_layer, "conv") else dst_layer
                if isinstance(src_conv, nn.Conv1d):
                    if axis == "time":
                        dst_conv.weight.data = src_conv.weight.data.unsqueeze(-1)
                    else:
                        dst_conv.weight.data = src_conv.weight.data.unsqueeze(2)
                    if src_conv.bias is not None:
                        dst_conv.bias.data = src_conv.bias.data.clone()

        conv_src = getattr(gpkffn, f"conv_{k}")
        conv_src_actual = conv_src.conv if hasattr(conv_src, "conv") else conv_src
        conv_dst = getattr(rf_gpkffn, f"conv_{k}")
        conv_dst_actual = conv_dst.conv if hasattr(conv_dst, "conv") else conv_dst

        if isinstance(conv_src_actual, nn.Conv1d):
            if axis == "time":
                conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(-1)
            else:
                conv_dst_actual.weight.data = conv_src_actual.weight.data.unsqueeze(2)
            if conv_src_actual.bias is not None:
                conv_dst_actual.bias.data = conv_src_actual.bias.data.clone()

    return rf_gpkffn


def convert_ts_block_to_reshape_free(
    ts_block: nn.Module,
) -> ReshapeFreeTSBlock:
    """
    Convert TSBlock to ReshapeFreeTSBlock.

    Args:
        ts_block: Original TSBlock module

    Returns:
        ReshapeFreeTSBlock with transferred weights
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

    time_block_kernel = gpkffn.kernel_list

    first_freq_block = ts_block.freq_stage[0]
    freq_gpkffn = first_freq_block[1]
    freq_block_kernel = freq_gpkffn.kernel_list

    causal = _is_causal_conv(cab.dwconv)
    logger.info(f"Detected causal mode: {causal}")

    rf_ts_block = ReshapeFreeTSBlock(
        dense_channel=dense_channel,
        time_block_num=time_block_num,
        freq_block_num=freq_block_num,
        time_dw_kernel_size=time_dw_kernel_size,
        time_block_kernel=time_block_kernel,
        freq_block_kernel=freq_block_kernel,
        causal=causal,
    )

    for i, block in enumerate(ts_block.time_stage):
        cab_src = block[0]
        gpkffn_src = block[1]
        rf_ts_block.time_stage[i][0] = convert_cab_to_reshape_free(cab_src, axis="time", causal=causal)
        rf_ts_block.time_stage[i][1] = convert_gpkffn_to_reshape_free(gpkffn_src, axis="time", causal=causal)

    for i, block in enumerate(ts_block.freq_stage):
        cab_src = block[0]
        gpkffn_src = block[1]
        rf_ts_block.freq_stage[i][0] = convert_cab_to_reshape_free(cab_src, axis="freq")
        rf_ts_block.freq_stage[i][1] = convert_gpkffn_to_reshape_free(gpkffn_src, axis="freq")

    rf_ts_block.beta_t.data = ts_block.beta_t.data.unsqueeze(-1)
    rf_ts_block.beta_f.data = ts_block.beta_f.data.unsqueeze(-1)

    logger.info(
        f"Converted TSBlock: {time_block_num} time blocks, {freq_block_num} freq blocks"
    )

    return rf_ts_block


def convert_ts_block_to_stateful_reshape_free(
    ts_block: nn.Module,
) -> "StatefulReshapeFreeTSBlock":
    """
    Convert TSBlock to StatefulReshapeFreeTSBlock for streaming inference.

    Args:
        ts_block: Original TSBlock module

    Returns:
        StatefulReshapeFreeTSBlock with transferred weights
    """
    from src.models.streaming.layers.reshape_free_stateful import (
        StatefulReshapeFreeTSBlock,
    )

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

    stateful_ts_block = StatefulReshapeFreeTSBlock(
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
            cab_src, stateful_ts_block.time_cabs[i], axis="time"
        )
        _transfer_gpkffn_weights_to_stateful(
            gpkffn_src, stateful_ts_block.time_gpkffns[i], axis="time"
        )

    for i, block in enumerate(ts_block.freq_stage):
        cab_src = block[0]
        gpkffn_src = block[1]
        freq_module = stateful_ts_block.freq_stage[i]
        _transfer_cab_weights_to_reshape_free(cab_src, freq_module[0], axis="freq")
        _transfer_gpkffn_weights_to_reshape_free(gpkffn_src, freq_module[1], axis="freq")

    stateful_ts_block.beta_t.data = ts_block.beta_t.data.unsqueeze(-1)
    stateful_ts_block.beta_f.data = ts_block.beta_f.data.unsqueeze(-1)

    logger.info(
        f"Converted TSBlock to Stateful Reshape-Free: "
        f"{time_block_num} time blocks, {freq_block_num} freq blocks"
    )

    return stateful_ts_block


def _transfer_cab_weights_to_stateful(
    cab_src: nn.Module,
    cab_dst: "StatefulReshapeFreeCAB",
    axis: str,
) -> None:
    """Transfer weights from original CAB to StatefulReshapeFreeCAB."""
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
    gpkffn_dst: "StatefulReshapeFreeGPKFFN",
    axis: str,
) -> None:
    """Transfer weights from original GPKFFN to StatefulReshapeFreeGPKFFN."""
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


def _transfer_cab_weights_to_reshape_free(
    cab_src: nn.Module,
    cab_dst: "ReshapeFreeCAB",
    axis: str,
) -> None:
    """Transfer weights from original CAB to ReshapeFreeCAB (non-stateful)."""
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


def _transfer_gpkffn_weights_to_reshape_free(
    gpkffn_src: nn.Module,
    gpkffn_dst: "ReshapeFreeGPKFFN",
    axis: str,
) -> None:
    """Transfer weights from original GPKFFN to ReshapeFreeGPKFFN (non-stateful)."""
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


def convert_sequence_block_to_stateful_reshape_free(
    sequence_block: nn.Sequential,
) -> nn.ModuleList:
    """
    Convert a sequence of TSBlocks to stateful reshape-free versions.

    Args:
        sequence_block: Sequential container of TSBlocks

    Returns:
        ModuleList of StatefulReshapeFreeTSBlocks
    """
    rf_blocks = nn.ModuleList()
    for i, ts_block in enumerate(sequence_block):
        rf_block = convert_ts_block_to_stateful_reshape_free(ts_block)
        rf_blocks.append(rf_block)
        logger.info(f"Converted TSBlock {i} to Stateful Reshape-Free")

    return rf_blocks


__all__ = [
    "convert_cab_to_reshape_free",
    "convert_gpkffn_to_reshape_free",
    "convert_ts_block_to_reshape_free",
    "convert_ts_block_to_stateful_reshape_free",
    "convert_sequence_block_to_stateful_reshape_free",
]
