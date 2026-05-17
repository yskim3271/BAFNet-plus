"""Path T2 surgery — replace ``PRelu`` with ``Max(0,x) + slope * Min(0,x)``.

Background
----------
On Hexagon V79 + QNN SDK 2.42 at QUInt16 activation precision, QNN HTP's op
registry has ``q::prelu.opt`` (the 3-input optimized PReLU kernel) DISABLED;
only the reference ``q::PRelu`` (2-input) is enabled, and the compiler's
pattern matcher picks ``.opt`` whenever it can (cycles 1, 2 mini-measurement,
4 verified this on three distinct graphs). Cycle 5 / T1 sidestepped the
problem by down-converting the PReLU's activation input to QUInt8 (where the
``.opt`` variant is registered) — that compiles cleanly and runs at p50
63.02 ms.

T2 is an alternative escape valve: remove the ``PRelu`` op from the graph
entirely by rewriting it as its algebraic decomposition

    PReLU(x, slope) = Max(x, 0)  +  slope * Min(x, 0)

so the QNN HTP compiler never gets the chance to pattern-match into
``q::prelu.opt``. The result is a graph that uses only ``Max``, ``Min``,
``Mul``, ``Add`` for the phase-decoder activation, all of which are
unambiguously registered at QUInt16 on V79 (they're plentiful elsewhere in
the trunk). This surgery is numerically equivalent to the original PReLU at
FP32 (bit-identical for any reasonable broadcasting of the per-channel
slope), so it should preserve host parity precisely.

Compared to T1:
- T2 changes the topology of the graph (PReLU removed, 4 new ops per
  target node); T1 changes only the quantization metadata.
- T2 has no Q-DQ rewrap around the activation (rest of trunk's QUInt16
  type stays end-to-end); T1 has one Q16↔Q8 convert pair per branch.
- T2 fan-in on ``Mul`` is the per-channel slope tensor (was PReLU's
  ``input[1]``); the same slope initializer is reused so no quantization
  scale shift is introduced.

The surgery is **opt-in** and operates on an ONNX ``ModelProto`` directly —
no PT-level changes; export the FP32 trunk as-is, then run this surgery on
the exported ONNX before INT8 quantization.

Usage::

    from src.models.streaming.onnx.prelu_decompose import decompose_phase_prelu_inplace
    m = onnx.load("results/onnx/bafnetplus_50ms_fp32_trunk.onnx")
    decompose_phase_prelu_inplace(m, target_node_names=(
        "/phase_conv/phase_conv.2/PRelu",
        "/mapping_core/phase_conv/phase_conv.2/PRelu",
    ))
    onnx.save(m, "results/onnx/bafnetplus_50ms_fp32_trunk_t2.onnx")

Cycle context: S25 cycle-5 T2 (web-research path #2), 2026-05-16.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

logger = logging.getLogger(__name__)


_DEFAULT_TARGETS: Sequence[str] = (
    "/phase_conv/phase_conv.2/PRelu",
    "/mapping_core/phase_conv/phase_conv.2/PRelu",
)


def _find_zero_constant_or_make(
    model: onnx.ModelProto,
    *,
    dtype: int = TensorProto.FLOAT,
) -> str:
    """Return the name of a scalar 0.0 initializer (creating one if absent).

    Reusing a single zero constant across both PReLU rewrites keeps the
    graph initializer count tight (vs one zero-per-PReLU).
    """
    target_name = "prelu_decomp_zero"
    for ini in model.graph.initializer:
        if ini.name == target_name:
            return target_name
    zero_arr = np.zeros((), dtype=np.float32)
    zero_initializer = numpy_helper.from_array(zero_arr, name=target_name)
    model.graph.initializer.append(zero_initializer)
    logger.info("decompose_phase_prelu: added scalar-zero initializer %r", target_name)
    return target_name


def decompose_phase_prelu_inplace(
    model: onnx.ModelProto,
    *,
    target_node_names: Iterable[str] = _DEFAULT_TARGETS,
) -> List[str]:
    """Rewrite each named ``PRelu`` node as ``Max(0,x) + slope * Min(0,x)``.

    For each target ``PRelu`` node ``P`` with inputs ``[data, slope]`` and
    output ``y``, this function:

    1. Removes ``P`` from ``model.graph.node``.
    2. Inserts four new nodes at the same position:
       - ``Max``  with inputs ``[data, zero]``, output ``y_max``.
       - ``Min``  with inputs ``[data, zero]``, output ``y_min``.
       - ``Mul``  with inputs ``[y_min, slope]``, output ``y_neg``.
       - ``Add``  with inputs ``[y_max, y_neg]``, output ``y`` (same name
         as the original PReLU output, so downstream consumers don't need
         rewiring).
    3. Reuses ``slope`` exactly as-is (no copy / no re-initializer), so the
       per-channel learned values stay bit-identical.

    A scalar ``0.0`` initializer is added once (shared across all rewrites).

    Args:
        model: The ONNX ModelProto to modify in-place.
        target_node_names: Names of the PReLU nodes to decompose. Default:
            the two phase-decoder PReLUs (one per branch).

    Returns:
        The list of node names that were actually rewritten (in graph order).

    Raises:
        ValueError: If any target node is not found, is not a ``PRelu``,
            or has the wrong number of inputs/outputs.
    """
    targets = list(target_node_names)
    if not targets:
        raise ValueError("target_node_names is empty")

    nodes_by_name = {n.name: (i, n) for i, n in enumerate(model.graph.node) if n.name}
    missing = [t for t in targets if t not in nodes_by_name]
    if missing:
        raise ValueError(f"PRelu target nodes not found in graph: {missing}")
    for t in targets:
        n = nodes_by_name[t][1]
        if n.op_type != "PRelu":
            raise ValueError(f"Target node {t!r} has op_type={n.op_type!r}, expected 'PRelu'")
        if len(n.input) != 2 or len(n.output) != 1:
            raise ValueError(
                f"PRelu node {t!r} has unexpected I/O: inputs={list(n.input)}, outputs={list(n.output)}"
            )

    # Lazily create a single shared scalar-zero initializer (broadcasts trivially).
    zero_name = _find_zero_constant_or_make(model)

    # Collect rewrites first, then mutate the graph in one pass so node-index
    # positions stay stable across iterations.
    rewrites = []  # (idx, name, [new nodes])
    for t in targets:
        idx, prelu = nodes_by_name[t]
        data_in, slope_in = prelu.input[0], prelu.input[1]
        out = prelu.output[0]
        base = t.rstrip("/").replace("/", "_")  # safe-ish derived-name stem

        y_max = f"{out}__prelu_decomp_max"
        y_min = f"{out}__prelu_decomp_min"
        y_neg = f"{out}__prelu_decomp_neg"

        max_node = helper.make_node(
            "Max",
            inputs=[data_in, zero_name],
            outputs=[y_max],
            name=f"{t}_decomp_Max",
        )
        min_node = helper.make_node(
            "Min",
            inputs=[data_in, zero_name],
            outputs=[y_min],
            name=f"{t}_decomp_Min",
        )
        mul_node = helper.make_node(
            "Mul",
            inputs=[y_min, slope_in],
            outputs=[y_neg],
            name=f"{t}_decomp_Mul",
        )
        add_node = helper.make_node(
            "Add",
            inputs=[y_max, y_neg],
            outputs=[out],
            name=f"{t}_decomp_Add",
        )
        rewrites.append((idx, t, [max_node, min_node, mul_node, add_node]))

    # Apply rewrites in reverse-idx order so removing a node doesn't shift
    # the indices of pending rewrites.
    rewrites.sort(key=lambda r: -r[0])
    rewritten_names: List[str] = []
    for idx, name, new_nodes in rewrites:
        del model.graph.node[idx]
        for offset, new in enumerate(new_nodes):
            model.graph.node.insert(idx + offset, new)
        rewritten_names.append(name)
        logger.info("decompose_phase_prelu: rewrote %r as Max+Min+Mul+Add (4 new nodes)", name)

    # Re-run shape inference so downstream tools (quantizer, ORT) see the
    # new intermediate tensor shapes.
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        del model.graph.value_info[:]
        model.graph.value_info.extend(inferred.graph.value_info)
    except Exception as e:  # noqa: BLE001
        logger.warning("decompose_phase_prelu: shape inference re-run failed (%s); proceeding", e)

    rewritten_names.reverse()  # graph order
    return rewritten_names


__all__ = ["decompose_phase_prelu_inplace"]
