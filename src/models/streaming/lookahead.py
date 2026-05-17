"""Algorithmic-lookahead calculator for a configured ``Backbone``.

The right (future) time padding of each ``AsymmetricConv2d`` is the only source
of algorithmic lookahead in the backbone (``CausalConv1d`` in TS-blocks is
left-only; the encoder/decoder ``nn.Conv2d`` / ``nn.ConvTranspose2d`` have
time-kernel 1). In the dense dilated block the convs form a dense chain — each
layer's input is the running ``skip`` concat of all prior outputs — so the
right-side reach **accumulates**: the block's total lookahead is the *sum* of
the per-layer ``time_padding_right`` values.

This module derives ``L_enc`` / ``L_dec`` straight from the instantiated
module fields (``time_padding_right``, which already encodes the
``padding_ratio`` + Python ``round()`` split, including the
``actual_total != total`` re-derivation in ``AsymmetricConv2d.__init__``).
It deliberately does **not** read any hand-entered latency labels.

For the 50 ms BAFNet+ backbone (``encoder_padding_ratio == decoder_padding_ratio
== (0.9, 0.1)``, dense depth 4, kernel 3) the per-DS_DDB right-pad frames are
``0 + 0 + 1 + 2`` so this returns ``L_enc == 3`` and ``L_dec == 3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch.nn as nn

from src.models.backbone import AsymmetricConv2d


def sum_right_padding(module: nn.Module) -> int:
    """Sum ``time_padding_right`` over every ``AsymmetricConv2d`` under ``module``.

    Args:
        module: Any module subtree (e.g. ``backbone.dense_encoder``).

    Returns:
        Total right (future) time-padding frames contributed by asymmetric convs.
    """
    return sum(m.time_padding_right for m in module.modules() if isinstance(m, AsymmetricConv2d))


def right_padding_breakdown(module: nn.Module) -> List[Tuple[str, int]]:
    """List ``(module_path, time_padding_right)`` for each ``AsymmetricConv2d``.

    Args:
        module: Any module subtree.

    Returns:
        ``(name, right_pad)`` pairs in ``named_modules`` order (diagnostic).
    """
    return [(name, m.time_padding_right) for name, m in module.named_modules() if isinstance(m, AsymmetricConv2d)]


@dataclass
class LookaheadInfo:
    """Resolved algorithmic lookahead for a backbone.

    Attributes:
        encoder_lookahead: ``L_enc`` — encoder right-padding frame sum.
        decoder_lookahead: ``L_dec`` — decoder right-padding frame sum.
        encoder_breakdown: Per-conv ``(path, right_pad)`` for the encoder.
        decoder_breakdown: Per-conv ``(path, right_pad)`` for the mask decoder
            (the phase decoder is asserted to match).
    """

    encoder_lookahead: int
    decoder_lookahead: int
    encoder_breakdown: List[Tuple[str, int]] = field(default_factory=list)
    decoder_breakdown: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def total_lookahead(self) -> int:
        """``L_enc + L_dec`` (used for the ``T_export = chunk + L_enc + L_dec`` geometry)."""
        return self.encoder_lookahead + self.decoder_lookahead


def compute_lookahead(backbone: nn.Module) -> LookaheadInfo:
    """Compute ``L_enc`` / ``L_dec`` for a configured ``Backbone``.

    Args:
        backbone: An instantiated ``Backbone`` (must expose ``dense_encoder``,
            ``mask_decoder`` and ``phase_decoder``).

    Returns:
        A :class:`LookaheadInfo` with the encoder/decoder lookahead frame counts.

    Raises:
        AttributeError: If ``backbone`` lacks the expected submodules.
        ValueError: If the mask and phase decoders disagree on lookahead (they
            share ``decoder_padding_ratio`` and depth, so this should not happen).
    """
    parts: dict = {}
    for attr in ("dense_encoder", "mask_decoder", "phase_decoder"):
        sub = getattr(backbone, attr, None)
        if not isinstance(sub, nn.Module):
            raise AttributeError(f"compute_lookahead expects a Backbone with a '{attr}' submodule")
        parts[attr] = sub

    l_enc = sum_right_padding(parts["dense_encoder"])
    l_dec_mask = sum_right_padding(parts["mask_decoder"])
    l_dec_phase = sum_right_padding(parts["phase_decoder"])
    if l_dec_mask != l_dec_phase:
        raise ValueError(
            "mask_decoder and phase_decoder disagree on lookahead "
            f"({l_dec_mask} vs {l_dec_phase}); they must share decoder_padding_ratio/depth"
        )

    return LookaheadInfo(
        encoder_lookahead=l_enc,
        decoder_lookahead=l_dec_mask,
        encoder_breakdown=right_padding_breakdown(parts["dense_encoder"]),
        decoder_breakdown=right_padding_breakdown(parts["mask_decoder"]),
    )


__all__ = [
    "LookaheadInfo",
    "compute_lookahead",
    "sum_right_padding",
    "right_padding_breakdown",
]
