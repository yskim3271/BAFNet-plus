"""Functional-stateful core helpers for BAFNet+ ONNX export.

Layer-by-layer state routing helpers that mirror the PyTorch streaming
forward path of a single backbone (DenseEncoder → SequenceBlock(TSBlocks)
→ MaskDecoder + PhaseDecoder). Together with :mod:`functional_stateful`
they replace the previous inline-math wrapper, eliminating the
``set_streaming_mode(False)`` zero-pad on encoder/decoder that caused
audio-domain divergence vs ``BAFNetPlusStreaming.process_samples``.

Mirrors ``LaCoSENet/src/models/onnx_export/stateful_core.py``'s
``_forward_*`` helpers but adapted for BAFNet+ usage where the backbone
is invoked twice (mapping + masking) by a single export wrapper.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.streaming.onnx.functional_stateful import (
    FunctionalStatefulCausalConv2d,
    FunctionalStatefulConv1d,
    FunctionalStatefulConv2d,
)

FUNCTIONAL_TYPES = (
    FunctionalStatefulConv1d,
    FunctionalStatefulConv2d,
    FunctionalStatefulCausalConv2d,
)


class StateIterator:
    """Tracks position in a flat state list while routing prev/next states.

    Usage:
        it = StateIterator(prev_states)
        out, ns = layer(x, it.take(), state_frames=N); it.push(ns)
    """

    def __init__(self, prev_states: List[Tensor]):
        self._prev = list(prev_states)
        self._next: List[Tensor] = []
        self._idx = 0

    def take(self) -> Tensor:
        s = self._prev[self._idx]
        self._idx += 1
        return s

    def push(self, ns: Tensor) -> None:
        self._next.append(ns)

    def step(self, layer: nn.Module, x: Tensor, state_frames: Optional[int]) -> Tensor:
        """Convenience: take state, call layer(x, state, state_frames), push next."""
        prev = self.take()
        y, ns = layer(x, prev, state_frames=state_frames)
        self.push(ns)
        return y

    @property
    def next_states(self) -> List[Tensor]:
        return list(self._next)

    @property
    def consumed(self) -> int:
        return self._idx


def collect_functional_modules(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    """List ``(name, mod)`` for every FunctionalStateful descendant of ``module``.

    Order is DFS module-tree order (matches ``named_modules``); we exclude
    nested duplicates so that, e.g., an SCA Sequential that contains one
    FunctionalConv1d is counted once via the inner conv.
    """
    out: List[Tuple[str, nn.Module]] = []
    seen = set()
    for name, m in module.named_modules():
        if isinstance(m, FUNCTIONAL_TYPES):
            if any(name.startswith(p + ".") for p in seen):
                continue
            out.append((name, m))
            seen.add(name)
    return out


# ---------------------------------------------------------------------------
# Backbone forward helpers (single backbone)
# ---------------------------------------------------------------------------


def _forward_cab(cab: nn.Module, x: Tensor, state_iter: StateIterator, state_frames: Optional[int]) -> Tensor:
    """ChannelAttentionBlock with state routing."""
    skip = x
    x = cab.norm(x)
    x = cab.pwconv1(x)

    if isinstance(cab.dwconv, FUNCTIONAL_TYPES):
        x = state_iter.step(cab.dwconv, x, state_frames)
    else:
        x = cab.dwconv(x)

    x = cab.sg(x)

    if isinstance(cab.sca, nn.Sequential):
        sca_out = x
        for layer in cab.sca:
            if isinstance(layer, FUNCTIONAL_TYPES):
                sca_out = state_iter.step(layer, sca_out, state_frames)
            else:
                sca_out = layer(sca_out)
        x = x * sca_out
    else:
        x = x * cab.sca(x)

    x = cab.pwconv2(x)
    x = skip + x * cab.beta
    return x


def _forward_gpkffn(gpkffn: nn.Module, x: Tensor, state_iter: StateIterator, state_frames: Optional[int]) -> Tensor:
    """GroupPrimeKernelFFN with state routing."""
    shortcut = x
    x = gpkffn.norm(x)
    x = gpkffn.proj_first(x)

    expand_ratio = gpkffn.expand_ratio
    kernel_list = gpkffn.kernel_list
    x_chunks = list(torch.chunk(x, expand_ratio, dim=1))
    for i in range(expand_ratio):
        ksize = kernel_list[i]
        attn_module = getattr(gpkffn, f"attn_{ksize}")
        conv_module = getattr(gpkffn, f"conv_{ksize}")

        # attn path (Sequential: conv_fn + 1x1 conv)
        attn_out = x_chunks[i]
        for layer in attn_module:
            if isinstance(layer, FUNCTIONAL_TYPES):
                attn_out = state_iter.step(layer, attn_out, state_frames)
            else:
                attn_out = layer(attn_out)

        # conv path (single conv_fn)
        if isinstance(conv_module, FUNCTIONAL_TYPES):
            conv_out = state_iter.step(conv_module, x_chunks[i], state_frames)
        else:
            conv_out = conv_module(x_chunks[i])

        x_chunks[i] = attn_out * conv_out

    x = torch.cat(x_chunks, dim=1)
    x = gpkffn.proj_last(x) * gpkffn.scale + shortcut
    return x


def _forward_stage(stage: nn.Module, x: Tensor, state_iter: StateIterator, state_frames: Optional[int]) -> Tensor:
    """time_stage or freq_stage (Sequential of (CAB, GPKFFN) blocks)."""
    for block in stage:
        for sub_block in block:
            if hasattr(sub_block, "sca"):
                x = _forward_cab(sub_block, x, state_iter, state_frames)
            elif hasattr(sub_block, "proj_first"):
                x = _forward_gpkffn(sub_block, x, state_iter, state_frames)
            else:
                x = sub_block(x)
    return x


def _forward_ts_block(
    ts_block: nn.Module, x: Tensor, state_iter: StateIterator, state_frames: Optional[int],
) -> Tensor:
    """Single TSBlock with state routing — matches TSBlock.forward layout."""
    C = ts_block.dense_channel
    B, _, T, F = x.shape

    # Time stage
    x = x.permute(0, 3, 1, 2).reshape(B * F, C, T)
    x = _forward_stage(ts_block.time_stage, x, state_iter, state_frames) + x * ts_block.beta_t

    # Freq stage
    x = x.view(B, F, C, T).permute(0, 3, 2, 1).reshape(B * T, C, F)
    x = _forward_stage(ts_block.freq_stage, x, state_iter, state_frames) + x * ts_block.beta_f

    # Back to [B, C, T, F]
    x = x.view(B, T, C, F).permute(0, 2, 1, 3)
    return x


def _forward_sequence(sequence: nn.Module, x: Tensor, state_iter: StateIterator, state_frames: Optional[int]) -> Tensor:
    """Iterate TSBlocks."""
    for ts_block in sequence:
        x = _forward_ts_block(ts_block, x, state_iter, state_frames)
    return x


def _forward_ds_ddb(ds_ddb: nn.Module, x: Tensor, state_iter: StateIterator, state_frames: Optional[int]) -> Tensor:
    """DenseDilatedBlock with state routing.

    Mirrors ``DenseDilatedBlock.forward`` exactly:
        skip = x
        for i: x = dense_block[i](skip); skip = cat([x, skip])
    """
    skip = x
    for i, dense_conv in enumerate(ds_ddb.dense_block):
        # dense_conv is Sequential([FunctionalConv2d-or-Asymmetric, 1x1 Conv2d, BN, PReLU])
        layer_input = skip
        for layer in dense_conv:
            if isinstance(layer, FUNCTIONAL_TYPES):
                layer_input = state_iter.step(layer, layer_input, state_frames)
            else:
                layer_input = layer(layer_input)
        x = layer_input
        skip = torch.cat([x, skip], dim=1)
    return x


def _forward_dense_encoder(
    encoder: nn.Module, x: Tensor, state_iter: StateIterator, state_frames: Optional[int],
) -> Tensor:
    """DenseEncoder with state routing — preserves stateless conv_1/conv_2."""
    x = encoder.dense_conv_1(x)
    x = _forward_ds_ddb(encoder.dense_block, x, state_iter, state_frames)
    x = encoder.dense_conv_2(x)
    return x


def _forward_mask_decoder_to_mask(
    decoder: nn.Module, x: Tensor, state_iter: StateIterator, state_frames: Optional[int],
) -> Tensor:
    """MaskDecoder with state routing; returns mask before final unsqueeze.

    Replicates ``MaskDecoder.forward`` step by step, with state routed
    through the ``dense_block`` only (mask_conv and lsigmoid are stateless).
    Returns ``[B, F, T]`` in the same layout as the original ``mask_full``
    intermediate used by the export wrapper (squeezed + transposed).
    """
    x = _forward_ds_ddb(decoder.dense_block, x, state_iter, state_frames)
    x = decoder.mask_conv(x)                              # [B, 1, T, F']
    x = x.squeeze(1).transpose(1, 2)                      # [B, F', T]
    x = decoder.lsigmoid(x)                               # [B, F', T]
    return x


def _forward_phase_decoder_to_complex(
    decoder: nn.Module, x: Tensor, state_iter: StateIterator, state_frames: Optional[int],
) -> Tuple[Tensor, Tensor]:
    """PhaseDecoder with state routing; returns (phase_real, phase_imag) BEFORE atan2.

    Replicates ``PhaseDecoder.forward`` *up to* the atan2 step — we skip
    the in-graph atan2 to preserve INT8 precision (host computes atan2).
    Output shapes are ``[B, F', T]`` matching the squeezed+transposed
    ``phase_real_full / phase_imag_full`` intermediates used by the old
    wrapper.
    """
    x = _forward_ds_ddb(decoder.dense_block, x, state_iter, state_frames)
    x = decoder.phase_conv(x)                                              # [B, C, T, F']
    x_r = decoder.phase_conv_r(x).squeeze(1).transpose(1, 2)               # [B, F', T]
    x_i = decoder.phase_conv_i(x).squeeze(1).transpose(1, 2)
    return x_r, x_i


# ---------------------------------------------------------------------------
# Single-backbone forward (mapping or masking)
# ---------------------------------------------------------------------------


def forward_backbone(
    backbone: nn.Module,
    mag: Tensor,
    pha: Tensor,
    state_iter: StateIterator,
    chunk_size: int,
    infer_type: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Full backbone forward with state routing.

    Returns:
        est_mag:      [B, F, chunk_size] — for masking: ``mag * mask``; for mapping: ``mask``.
        mask_cropped: [B, F, chunk_size] — raw mask output (returned mainly for masking branch).
        com_out:      [B, F, chunk_size, 2] — complex spec from atan2-bypass identity.
    """
    # Input stack — match ``_run_backbone`` line 397: [B, 2, T_ext, F]
    x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)

    # Encoder
    x = _forward_dense_encoder(backbone.dense_encoder, x, state_iter, state_frames=chunk_size)

    # TSBlocks
    x = _forward_sequence(backbone.sequence_block, x, state_iter, state_frames=chunk_size)

    # Mask + phase decoders (state routed through their dense_block)
    mask_full = _forward_mask_decoder_to_mask(backbone.mask_decoder, x, state_iter, state_frames=chunk_size)
    pr_full, pi_full = _forward_phase_decoder_to_complex(backbone.phase_decoder, x, state_iter, state_frames=chunk_size)

    # Crop to chunk_size and assemble complex out via atan2-bypass identity (same as old _run_backbone)
    mask_cropped = mask_full[:, :, :chunk_size]
    phase_real = pr_full[:, :, :chunk_size]
    phase_imag = pi_full[:, :, :chunk_size]

    if infer_type == "masking":
        est_mag = mag[:, :, :chunk_size] * mask_cropped
    else:
        est_mag = mask_cropped

    eps = 1e-8
    pr = phase_real + eps
    pi = phase_imag + eps
    norm = torch.sqrt(pr * pr + pi * pi)
    com_real = est_mag * pr / norm
    com_imag = est_mag * pi / norm
    com_out = torch.stack([com_real, com_imag], dim=-1)
    return est_mag, mask_cropped, com_out


def forward_single_backbone(
    backbone: nn.Module,
    mag: Tensor,
    pha: Tensor,
    state_iter: StateIterator,
    chunk_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Single-backbone forward returning raw (mask, phase_real, phase_imag) at full T.

    Output format matches the LaCoSENet ONNX export wrapper's old
    ``StatefulExportableNNCore.forward`` — no crop, no atan2-bypass
    normalization. Host computes ``atan2(imag, real)`` and applies
    ``mag * mask`` externally.

    Returns:
        mask:        [B, F, T_full] — raw mask (lsigmoid applied)
        phase_real:  [B, F, T_full] — raw phase real conv output
        phase_imag:  [B, F, T_full] — raw phase imag conv output
    """
    x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)
    x = _forward_dense_encoder(backbone.dense_encoder, x, state_iter, state_frames=chunk_size)
    x = _forward_sequence(backbone.sequence_block, x, state_iter, state_frames=chunk_size)
    mask = _forward_mask_decoder_to_mask(backbone.mask_decoder, x, state_iter, state_frames=chunk_size)
    phase_real, phase_imag = _forward_phase_decoder_to_complex(backbone.phase_decoder, x, state_iter, state_frames=chunk_size)
    return mask, phase_real, phase_imag


# ---------------------------------------------------------------------------
# State registry — collect FunctionalStateful modules in forward order with shapes
# ---------------------------------------------------------------------------


def _compute_state_shape(
    full_name: str,
    module: nn.Module,
    freq_bins: int,
    freq_size_encoded: int,
    export_time_frames: int,
) -> Tuple[int, ...]:
    """Determine the state tensor shape for ``module`` given its forward-order ``full_name``.

    The shape depends on the location in the network because TSBlock 1D
    convs operate after a reshape (B*F_enc or B*T) and 2D convs see
    different freq dimensions depending on whether they are in the
    encoder (pre-stride) or decoder/fusion (post-stride).
    """
    if isinstance(module, FunctionalStatefulConv1d):
        # 1D conv. batch depends on TSBlock stage reshape; otherwise 1.
        if ".time_stage." in full_name:
            batch = freq_size_encoded
        elif ".freq_stage." in full_name:
            batch = export_time_frames
        else:
            batch = 1
        return (batch, module.in_channels, module.padding_size)

    if isinstance(module, FunctionalStatefulConv2d):
        # 2D asymmetric (used in encoder/decoder dense_block, NOT alpha).
        # dense_encoder sees freq_bins; mask/phase_decoder sees freq_size_encoded.
        # Match both single-backbone ("dense_encoder.*") and dual-backbone
        # ("<branch>.dense_encoder.*") full_name shapes.
        is_encoder = (
            full_name == "dense_encoder"
            or full_name.startswith("dense_encoder.")
            or ".dense_encoder." in full_name
            or full_name.endswith(".dense_encoder")
        )
        f_in = freq_bins if is_encoder else freq_size_encoded
        f_padded = f_in + 2 * module.freq_padding
        return (1, module.in_channels, module.time_padding_left, f_padded)

    if isinstance(module, FunctionalStatefulCausalConv2d):
        # 2D causal (alpha_convblocks). Sees freq_bins (after alpha_feat assembly).
        f_in = freq_bins
        f_padded = f_in + 2 * module.freq_padding
        return (1, module.in_channels, module.time_padding, f_padded)

    raise TypeError(f"unknown FunctionalStateful: {type(module).__name__}")


_BACKBONE_SUBMODULES = ("dense_encoder", "sequence_block", "mask_decoder", "phase_decoder")


def build_generic_backbone_state_entries(
    backbones: List[Tuple[str, str, nn.Module]],
    freq_bins: int,
    freq_size_encoded: int,
    export_time_frames: int,
    fusion_modules: Optional[List[Tuple[str, str, nn.Module]]] = None,
) -> List[Tuple[str, Tuple[int, ...], Tuple[str, str]]]:
    """Unified state registry builder for any backbone count + optional fusion.

    Walks each backbone in canonical forward submodule order (``dense_encoder
    → sequence_block → mask_decoder → phase_decoder``) and appends fusion
    modules afterwards. Replaces the previous dual/single-specific helpers
    and supports triple+ backbone models by extending the ``backbones`` list.

    Args:
        backbones: ``[(tag, prefix, module), ...]`` per backbone.
            - ``prefix=""`` → no path prefix (LaCoSENet single)
            - ``prefix="mapping"`` → ``mapping.dense_encoder.*`` (BAFNet+ dual)
            - ``tag`` is the locator branch_tag stored in the entry tuple.
        fusion_modules: ``[(tag, prefix, module), ...]`` appended after all
            backbones. Used by BAFNet+ for ``calibration_encoder`` (tag
            ``calibration``) and ``alpha_convblocks`` (tag ``alpha``).

    Returns:
        Forward-ordered list of ``(state_name, shape, (tag, full_path))``.
    """
    entries: List[Tuple[str, Tuple[int, ...], Tuple[str, str]]] = []

    def _emit(prefix: str, branch: nn.Module, branch_tag: str):
        for sub_name, m in collect_functional_modules(branch):
            if prefix and sub_name:
                full = f"{prefix}.{sub_name}"
            else:
                full = prefix or sub_name
            shape = _compute_state_shape(full, m, freq_bins, freq_size_encoded, export_time_frames)
            name = f"state_{full.replace('.', '_')}"
            entries.append((name, shape, (branch_tag, full)))

    for tag, prefix, backbone in backbones:
        for sub_prefix in _BACKBONE_SUBMODULES:
            sub = backbone.get_submodule(sub_prefix)
            full_prefix = f"{prefix}.{sub_prefix}" if prefix else sub_prefix
            _emit(full_prefix, sub, tag)

    if fusion_modules:
        for tag, prefix, module in fusion_modules:
            _emit(prefix, module, tag)

    return entries


def build_functional_state_entries(
    bafnetplus: nn.Module,
    freq_bins: int,
    freq_size_encoded: int,
    export_time_frames: int,
) -> List[Tuple[str, Tuple[int, ...], Tuple[str, str]]]:
    """BAFNet+ dual-backbone state registry (mapping + masking + calibration + alpha).

    Thin wrapper around :func:`build_generic_backbone_state_entries`.
    """
    backbones = [
        ("mapping", "mapping", bafnetplus.mapping),
        ("masking", "masking", bafnetplus.masking),
    ]
    fusion: List[Tuple[str, str, nn.Module]] = []
    cal = getattr(bafnetplus, "calibration_encoder", None)
    if cal is not None:
        fusion.append(("calibration", "calibration_encoder", cal))
    alpha = getattr(bafnetplus, "alpha_convblocks", None)
    if alpha is not None:
        fusion.append(("alpha", "alpha_convblocks", alpha))
    return build_generic_backbone_state_entries(
        backbones, freq_bins, freq_size_encoded, export_time_frames,
        fusion_modules=fusion or None,
    )


def build_single_backbone_state_entries(
    backbone: nn.Module,
    freq_bins: int,
    freq_size_encoded: int,
    export_time_frames: int,
    branch_tag: str = "single",
) -> List[Tuple[str, Tuple[int, ...], Tuple[str, str]]]:
    """Single-backbone (LaCoSENet) state registry, no fusion, no path prefix.

    Thin wrapper around :func:`build_generic_backbone_state_entries`.
    """
    return build_generic_backbone_state_entries(
        [(branch_tag, "", backbone)],
        freq_bins, freq_size_encoded, export_time_frames,
        fusion_modules=None,
    )


class FunctionalStateRegistry:
    """Forward-ordered list of ``(name, shape, locator)`` entries.

    Lightweight replacement for ``BafnetPlusStateRegistry`` so the rest of
    the export pipeline can keep using ``state_registry.names``,
    ``.shapes``, ``len(...)``.
    """

    def __init__(self, entries: List[Tuple[str, Tuple[int, ...], Tuple[str, str]]]):
        self.entries = list(entries)
        self._lookup = {e[0]: e[1] for e in self.entries}

    @property
    def names(self) -> List[str]:
        return [e[0] for e in self.entries]

    @property
    def shapes(self) -> List[Tuple[int, ...]]:
        return [e[1] for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, name: str) -> Tuple[int, ...]:
        return self._lookup[name]

    def keys(self) -> List[str]:
        return self.names


__all__ = [
    "StateIterator",
    "collect_functional_modules",
    "forward_backbone",
    "forward_single_backbone",
    "build_generic_backbone_state_entries",
    "build_functional_state_entries",
    "build_single_backbone_state_entries",
    "FunctionalStateRegistry",
    "FUNCTIONAL_TYPES",
]
