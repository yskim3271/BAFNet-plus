"""Path E-4 surgery — merge parallel ``phase_conv_r`` + ``phase_conv_i``.

Background
----------
On Hexagon V79 + QNN SDK 2.42 + ORT 1.24.2 at W8A16, the QNN HTP op registry
has ``q::prelu.opt`` (the 3-input optimized PReLU kernel variant) **disabled**;
only the reference ``q::prelu`` (2-input) is enabled. The QNN compiler picks
``.opt`` when the PReLU output is consumed by multiple downstream ops that
benefit from TCM-resident stride-4 staging — the rejection JSON
(`Android_projects/results/parity/htp_compile_unified_v79.json`, schema
`s23-phase-c-preflight-htp-compile-v1`) shows the root rejection at
``/mapping_core/phase_conv/phase_conv.2/PRelu_3`` whose downstream fanout is
exactly two: ``phase_conv_r`` and ``phase_conv_i`` (both 1×1 Conv2d, dense
channel → 1).

This module collapses those two parallel 1×1 Convs into a single 1×1 Conv with
``out_channels=2`` followed by a channel-axis split, so the PReLU's output is
consumed by exactly one downstream op. The arithmetic is identical (concatenating
the weight rows of the two original Convs produces the same per-row outputs as
the originals), so host parity is numerically preserved at FP32 ULP.

The surgery is **opt-in** and operates on an :class:`ExportableBackboneCore`
instance only — the canonical :class:`~src.models.backbone.PhaseDecoder` in
``backbone.py`` is untouched, so training / evaluation / checkpoint-load paths
are unaffected.

Usage::

    core = ExportableBackboneCore.from_backbone(backbone, phase_output_mode='complex')
    merge_phase_conv_ri_inplace(core)
    # ... then call the existing export driver

Cycle context: S25 (cycle-4 mini-fix A2 candidate), 2026-05-16.
"""

from __future__ import annotations

import logging
import types
from typing import Tuple, Union, cast

import torch
from torch import Tensor, nn

from .backbone_core import ExportableBackboneCore, StateIterator

logger = logging.getLogger(__name__)


def merge_phase_conv_ri_inplace(core: ExportableBackboneCore) -> None:
    """Merge ``phase_conv_r`` + ``phase_conv_i`` into a single ``phase_conv_ri``.

    Builds a new ``nn.Conv2d`` whose weight is the row-concatenation of
    ``core.phase_decoder.phase_conv_r.weight`` and
    ``core.phase_decoder.phase_conv_i.weight`` (and same for bias if present),
    attaches it as ``core.phase_decoder.phase_conv_ri``, and monkey-patches
    ``core._forward_phase_decoder`` to route through the merged Conv + a
    channel-axis slice. The original ``phase_conv_r`` / ``phase_conv_i`` modules
    are left in place but become dead code from ``core``'s forward perspective;
    ``torch.onnx.export`` traces only the live forward path and so the exported
    ONNX will not contain them.

    Args:
        core: The :class:`ExportableBackboneCore` to modify. Must expose a
            ``phase_decoder`` submodule that holds two 1×1 :class:`nn.Conv2d`
            attributes named ``phase_conv_r`` and ``phase_conv_i`` with
            identical configuration.

    Raises:
        ValueError: If ``core.phase_decoder`` is missing the expected
            submodules, if the submodules' shapes / kernels / stride / padding
            / dilation / groups / bias-presence disagree, or if surgery has
            already been applied (``phase_conv_ri`` exists).

    Side effects (in-place on ``core``):
        - ``core.phase_decoder.phase_conv_ri`` is added (``nn.Conv2d`` with
          ``out_channels = phase_conv_r.out_channels + phase_conv_i.out_channels``).
        - ``core._forward_phase_decoder`` is rebound to a merged variant on
          ``core`` only (the class-level method is unchanged for other
          instances).
    """
    pd = core.phase_decoder
    if not (hasattr(pd, "phase_conv_r") and hasattr(pd, "phase_conv_i")):
        raise ValueError(
            "core.phase_decoder must expose 'phase_conv_r' and 'phase_conv_i'; "
            f"got attrs {sorted(pd._modules.keys())}"
        )
    if hasattr(pd, "phase_conv_ri"):
        raise ValueError(
            "core.phase_decoder.phase_conv_ri already exists — surgery has been applied"
        )

    r = cast(nn.Conv2d, pd.phase_conv_r)
    i = cast(nn.Conv2d, pd.phase_conv_i)
    if not (isinstance(r, nn.Conv2d) and isinstance(i, nn.Conv2d)):
        raise ValueError(
            f"phase_conv_r / phase_conv_i must be nn.Conv2d; "
            f"got {type(r).__name__} / {type(i).__name__}"
        )
    if r.in_channels != i.in_channels:
        raise ValueError(
            f"phase_conv_r/i in_channels mismatch: {r.in_channels} vs {i.in_channels}"
        )
    if r.kernel_size != i.kernel_size:
        raise ValueError(
            f"phase_conv_r/i kernel_size mismatch: {r.kernel_size} vs {i.kernel_size}"
        )
    if r.kernel_size != (1, 1):
        raise ValueError(
            f"E-4 surgery assumes 1×1 phase_conv_r/i; got kernel_size={r.kernel_size}"
        )
    if r.stride != i.stride or r.padding != i.padding or r.dilation != i.dilation or r.groups != i.groups:
        raise ValueError(
            "phase_conv_r/i config mismatch: "
            f"r=(stride={r.stride}, padding={r.padding}, dilation={r.dilation}, groups={r.groups}) vs "
            f"i=(stride={i.stride}, padding={i.padding}, dilation={i.dilation}, groups={i.groups})"
        )
    has_bias_r = r.bias is not None
    has_bias_i = i.bias is not None
    if has_bias_r != has_bias_i:
        raise ValueError(
            f"phase_conv_r/i bias-presence mismatch: r={has_bias_r}, i={has_bias_i}"
        )

    r_out = r.out_channels
    i_out = i.out_channels
    merged_out = r_out + i_out

    merged = nn.Conv2d(
        in_channels=r.in_channels,
        out_channels=merged_out,
        kernel_size=(1, 1),
        stride=r.stride,
        padding=r.padding,
        dilation=r.dilation,
        groups=r.groups,
        bias=has_bias_r,
    )
    merged.to(device=r.weight.device, dtype=r.weight.dtype)
    with torch.no_grad():
        merged.weight.copy_(torch.cat([r.weight.data, i.weight.data], dim=0))
        if has_bias_r:
            merged.bias.copy_(torch.cat([r.bias.data, i.bias.data], dim=0))

    pd.phase_conv_ri = merged

    def _forward_phase_decoder_merged(
        self: ExportableBackboneCore,
        x: Tensor,
        state_iter: StateIterator,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """E-4 merged variant of :meth:`_forward_phase_decoder`.

        Routes the post-``phase_conv`` activation through a single
        ``phase_conv_ri`` (2-output 1×1 Conv) and slices the result along the
        channel axis — same arithmetic as the original two parallel 1×1 Convs,
        but with PReLU fanout reduced from 2 to 1 so the QNN HTP compiler does
        not pattern-match into ``q::prelu.opt``.
        """
        decoder = self.phase_decoder
        x = self._forward_ds_ddb(x, decoder.dense_block, state_iter)
        x = decoder.phase_conv(x)
        x_combined = decoder.phase_conv_ri(x)
        x_r = x_combined[:, :r_out, :, :]
        x_i = x_combined[:, r_out:, :, :]
        if self.phase_output_mode == "complex":
            return x_r, x_i
        return torch.atan2(x_i + 1e-8, x_r + 1e-8)

    core._forward_phase_decoder = types.MethodType(  # type: ignore[method-assign]
        _forward_phase_decoder_merged, core
    )

    logger.info(
        "merge_phase_conv_ri_inplace: phase_conv_r(out=%d) + phase_conv_i(out=%d) "
        "→ phase_conv_ri(out=%d); _forward_phase_decoder rebound on instance.",
        r_out, i_out, merged_out,
    )


__all__ = ["merge_phase_conv_ri_inplace"]
