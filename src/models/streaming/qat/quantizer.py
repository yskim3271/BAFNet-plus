"""Custom PT2E :class:`Quantizer` matching the D1 PTQ scheme.

Scheme (cycle 17, mirrors the D1 INT8 QDQ trunk
``bafnetplus_50ms_int8_qdq_trunk_t2.onnx``):

- Conv1d / Conv2d / Linear weight → **per-channel QInt8** (axis=0, output channel)
- Activation (input + output of quantizable ops) → **per-tensor asymmetric QUInt16**
  (matches the PTQ recipe ``activation_type="QUInt16"`` in
  :func:`src.models.streaming.onnx.export.quantize_bafnetplus_qdq`)
- Bias: FP32 (the INT8 conv accumulates in INT32 and dequants; explicit fake-quant
  on bias is unnecessary and matches the PTQ pipeline behavior)

Skip patterns (mirror :func:`_auto_precision_sensitive_nodes`):

- ``phase_conv_r/`` / ``phase_conv_i/`` / ``phase_conv/phase_conv.`` Conv ops
  in the mapping_core / masking_core trunk decoders (cycle-12 patch — Hexagon
  V79 rejects them at HTP graph-prepare time when INT8'd; FP32 fallback is
  the required behavior).
- All ops whose FX node path lies under the head-graph cluster
  (``alpha_convblocks/``, ``alpha_out/``, ``calibration_encoder/``,
  ``common_gain_head/``, ``relative_gain_head/``). QAT scope is **trunk only**;
  head stays FP32 (matches the D1 split-graph deployable contract).

The XNNPACKQuantizer in `torch/ao/quantization/quantizer/xnnpack_quantizer.py`
served as a structural reference; the scheme + skip rules differ enough that
a direct subclass would have produced more drag than reimplementation.

Notes on torch.ao deprecation
------------------------------
PyTorch 2.10 removes ``torch.ao.quantization``; the PT2E pipeline migrates to
``torchao.quantization.pt2e``. We're pinned to torch 2.9.1+cu128 for cycle 17;
when PyTorch is upgraded the imports will need to flip — the API surface is the
same.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import torch
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)


# ----------------------------------------------------------------------- specs

def _make_activation_spec() -> QuantizationSpec:
    """Per-tensor asymmetric QUInt16 activation (matches D1 PTQ recipe)."""
    return QuantizationSpec(
        dtype=torch.uint16,
        quant_min=0,
        quant_max=65535,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            dtype=torch.uint16,
            quant_min=0,
            quant_max=65535,
            qscheme=torch.per_tensor_affine,
            eps=2 ** -12,
        ),
    )


def _make_weight_spec(ch_axis: int = 0) -> QuantizationSpec:
    """Per-channel symmetric QInt8 weight along ``ch_axis`` (default: out-channel)."""
    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
        is_dynamic=False,
        observer_or_fake_quant_ctr=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=ch_axis,
            eps=2 ** -12,
        ),
    )


# ---------------------------------------------------------------- target ops

# Conv-family aten targets we quantize. ``convolution.default`` is the generic
# form that PT2E often produces; ``conv1d.default`` / ``conv2d.default`` are the
# specialized targets. ``linear.default`` is for Linear modules in the trunk
# (FC after AdaptiveAvgPool, etc.).
_CONV_TARGETS: Tuple = (
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
)
_LINEAR_TARGETS: Tuple = (
    torch.ops.aten.linear.default,
)


# -------------------------------------------------------------- skip patterns

# Default skip patterns: trunk phase-decoder cluster + head-graph cluster.
# Matched against the FX node ``name`` (PT2E uses dotted module-path style).
DEFAULT_SKIP_NAME_SUBSTRINGS: Tuple[str, ...] = (
    # Trunk phase-decoder cluster (cycle-12 patch — V79 HTP-incompatible if INT8'd)
    "phase_conv_r",
    "phase_conv_i",
    "phase_conv_phase_conv",  # matches mapping/masking sub-block style
    # Head-graph cluster (QAT scope = trunk only; head stays FP32)
    "alpha_convblocks",
    "alpha_out",
    "calibration_encoder",
    "common_gain_head",
    "relative_gain_head",
)


def _node_should_skip(node, skip_substrings: Tuple[str, ...]) -> bool:
    """True if any skip substring appears in the node's identifying strings.

    PT2E ``torch.export`` produces flat names (``conv2d`` / ``conv2d_1``) that
    don't directly carry the source ``nn.Module`` hierarchy. The hierarchy IS
    available through (a) the weight/bias argument names (FX inlines them as
    ``alpha_out_weight``, ``regular_weight``, ...) and (b) ``nn_module_stack``
    meta, which records the module FQN of every wrapper the call traversed.
    We OR-match the skip substrings against all three sources.
    """
    candidates: List[str] = [getattr(node, "name", "") or ""]
    for arg in getattr(node, "args", ()):
        name = getattr(arg, "name", None)
        if isinstance(name, str):
            candidates.append(name)
    stack = node.meta.get("nn_module_stack", {}) if hasattr(node, "meta") else {}
    for entry in stack.values():
        if isinstance(entry, (tuple, list)) and entry:
            candidates.append(str(entry[0]))
    blob = " ".join(candidates).lower()
    return any(s.lower() in blob for s in skip_substrings)


# -------------------------------------------------------------- the Quantizer

class QNNQuantizer(Quantizer):
    """PT2E Quantizer matching the D1 PTQ scheme for the BAFNet+ trunk.

    Args:
        skip_phase_conv: If True (default), exclude the phase-decoder cluster
            (``phase_conv_r/i`` and ``phase_conv/phase_conv.{0,1,2}``) — these
            ops are V79-HTP-incompatible at INT8.
        skip_head: If True (default), exclude all head-graph ops (alpha,
            calibration, gain heads). QAT scope is trunk only.
        extra_skip_substrings: Additional substrings to skip (caller can add
            project-specific patterns without subclassing).

    Inserted observers:
        - Activations: per-tensor asymmetric QUInt16, MovingAverageMinMax.
        - Weights: per-channel symmetric QInt8 (axis=0), MovingAveragePerChannelMinMax.
        - Bias: not annotated (FP32 → INT32 accumulator → dequant, matches PTQ).
    """

    def __init__(
        self,
        *,
        skip_phase_conv: bool = True,
        skip_head: bool = True,
        extra_skip_substrings: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        self.skip_phase_conv = skip_phase_conv
        self.skip_head = skip_head

        skips: List[str] = []
        if skip_phase_conv:
            skips.extend(s for s in DEFAULT_SKIP_NAME_SUBSTRINGS if "phase_conv" in s)
        if skip_head:
            skips.extend(
                s
                for s in DEFAULT_SKIP_NAME_SUBSTRINGS
                if "phase_conv" not in s
            )
        if extra_skip_substrings:
            skips.extend(extra_skip_substrings)
        self.skip_substrings: Tuple[str, ...] = tuple(skips)

        self.act_spec = _make_activation_spec()
        self.weight_spec = _make_weight_spec(ch_axis=0)

        # Populated during annotate() — debugging surface for the warm-start step.
        self.annotated_node_names: List[str] = []
        self.skipped_node_names: List[str] = []

    # ----------- Quantizer protocol --------------------------------------

    def annotate(self, model):
        """Walk the GraphModule and annotate quantizable Conv/Linear nodes."""
        self.annotated_node_names = []
        self.skipped_node_names = []

        seen_input_acts: Set = set()

        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            is_conv = target in _CONV_TARGETS
            is_linear = target in _LINEAR_TARGETS
            if not (is_conv or is_linear):
                continue

            if _node_should_skip(node, self.skip_substrings):
                self.skipped_node_names.append(node.name)
                continue

            inp_node = node.args[0]
            wt_node = node.args[1]
            input_qspec_map = {wt_node: self.weight_spec}

            # Only insert the input activation observer the first time we see
            # this tensor as a quantizable-op input. If two annotated convs
            # share an input, both pointing at the same activation observer is
            # fine — PT2E deduplicates internally, but skipping the second
            # insertion keeps the annotation map cleaner.
            if id(inp_node) not in seen_input_acts:
                input_qspec_map[inp_node] = self.act_spec
                seen_input_acts.add(id(inp_node))
            else:
                input_qspec_map[inp_node] = self.act_spec  # PT2E handles sharing

            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=self.act_spec,
                _annotated=True,
            )
            self.annotated_node_names.append(node.name)

        return model

    def validate(self, model) -> None:  # noqa: D401 — PT2E protocol
        """No-op validation; the annotate pass produces a self-consistent map."""
        return None

    # ----------- introspection helpers (used by warm-start + tests) -------

    def summary(self) -> dict:
        return {
            "scheme": "w8a16_qnn",
            "weight": "per_channel_symmetric_int8_axis0",
            "activation": "per_tensor_affine_uint16",
            "skip_phase_conv": self.skip_phase_conv,
            "skip_head": self.skip_head,
            "skip_substrings": list(self.skip_substrings),
            "num_annotated": len(self.annotated_node_names),
            "num_skipped": len(self.skipped_node_names),
        }
