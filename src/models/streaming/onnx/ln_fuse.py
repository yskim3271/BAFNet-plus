"""LN cluster fusion surgery — replace a 9-op LN primitive cluster with ``Transpose → LayerNormalization(axis=-1) → Transpose``.

Background
----------
BAFNet+'s :class:`LayerNorm1d` (and RF :class:`LayerNorm4dChannel`) export to a
9-primitive cluster in ONNX:

    ReduceMean(axes=[1]) → Sub → Pow → ReduceMean(axes=[1]) → Add(eps) → Sqrt
        → Div → Mul(gamma) → Add(beta)

This normalises **only over the channel axis** (``axis=1``) of
``[B, C, T, F] = [1, 64, 14, 100]``, broadcasting per ``(B, T, F)`` location.

In the INT8 QDQ trunk, every primitive output is wrapped Q→DQ before the next
consumer, blocking ORT's ``LayerNormalizationFusion`` (cycle 7 falsification).
On V79 the cluster's runtime cost is ~8.7 % of total cycles (~2.8 ms wall;
cycle 14 profile re-measurement).

ONNX semantic finding (Cycle 15 Path α probe, 2026-05-17)
---------------------------------------------------------
**ONNX 17 ``LayerNormalization(axis=k)`` normalises over** ``[k:]`` — that is,
**ALL axes from k onward**, not just dimension k. For ``[1, 64, 14, 100]``
with ``axis=1`` the op would normalise over the 89 600-element ``(C, T, F)``
volume and expect scale/bias of shape ``[64, 14, 100]`` — NOT the
``[C]``-only normalisation BAFNet+ wants. (Empirically verified: ORT raises
``InvalidArgument: Got scale size of 64 and bias size of 64`` when given
``axis=1`` with ``[64]`` scale/bias on ``[1, 64, 14, 100]`` input.)

The "Option A" hypothesis from the cycle 14 plan — emit a bare
``LayerNormalization(axis=1)`` and let V79 sort it out — therefore **does not
exist** in ONNX semantics. There is no way to express BAFNet+'s
channel-only LN with a single LayerNormalization op while keeping the
``[B, C, T, F]`` layout.

The only viable surgery is **Option B (Transpose pair)**:

    [B, C, T, F]
      → Transpose perm=(0, 2, 3, 1)  → [B, T, F, C]
      → LayerNormalization(axis=-1, gamma[C], beta[C])  → [B, T, F, C]
      → Transpose perm=(0, 3, 1, 2)  → [B, C, T, F]

The two Transposes operate on ``[1, 14, 100, 64]`` = 89 600 elements each
(~90 KB at INT8 / ~358 KB at FP32). Cycle 14's per-Transpose cost estimate
from cycle 8 data was ~0.23 ms non-profile per Transpose; the V79 sandwich
will measure the actual cost.

This module implements the **Cycle 15 Path α 1-cluster probe**: surgery that
fuses exactly one ``/<branch>/norm/`` LN cluster (the first one, no suffix)
into a ``Transpose → LayerNormalization(axis=-1) → Transpose`` triplet.
Gamma / beta values are dequantised from their stored QInt initializers and
emitted as raw FP32 initializers (shape ``[C]``). The other 63 clusters
remain unchanged. If the V79 sandwich shows Δ p50 ≤ −0.5 ms vs the intra-
anchor for a SINGLE cluster, extrapolation suggests Cycle 16 full 64-cluster
surgery is worthwhile.

The surgery operates on the **post-PTQ INT8 trunk** (in-place on the
``ModelProto``) — no PT-level changes, no re-quantization. The 9 primitive
nodes + 8 intermediate Q nodes + 8 intermediate DQ nodes are removed; the
cluster-output ``QuantizeLinear`` is rewired to consume the post-LN
``Transpose`` output. The cluster-input ``DequantizeLinear`` is preserved
(feeds the pre-LN ``Transpose``).

Per-cluster delta:

- −9 primitive ops (ReduceMean × 2, Sub, Pow, Add, Sqrt, Div, Mul, Add)
- −8 intermediate Q nodes (all primitive-output Quantize except Add_1's)
- −8 intermediate DQ nodes (each intermediate Q's downstream DQ)
- +1 LayerNormalization node
- +2 Transpose nodes (pre + post)
- +2 FP32 initializers (gamma_fp32 [C], beta_fp32 [C])

Net node count: −25 + 3 = **−22 ops per cluster**.

Numerical equivalence at FP32 (modulo libm precision in LN op's internal
algorithm). The replaced cluster's INT8 quantisation noise is **removed** —
the new LN runs naked FP32 between QDQ boundaries — so the drift vs the
unfused INT8 trunk is bounded by one cluster's worth of quant noise.

Usage::

    from src.models.streaming.onnx.ln_fuse import fuse_first_branch_norm_ln_inplace
    m = onnx.load("results/onnx/bafnetplus_50ms_int8_qdq_trunk_t2.onnx")
    info = fuse_first_branch_norm_ln_inplace(m, branch="mapping_core")
    onnx.save(m, "results/onnx/bafnetplus_50ms_int8_qdq_trunk_t2_1ln_a.onnx")

Cycle context: Phase C pre-flight cycle 15 Path α, 2026-05-17.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

logger = logging.getLogger(__name__)


_PRIMITIVE_SUFFIXES = (
    "ReduceMean",  # rm_first (axes=[1], mean)
    "Sub",         # x - mean
    "Pow",         # (x - mean) ** 2
    "ReduceMean_1",  # rm_second (axes=[1], var)
    "Add",         # var + eps
    "Sqrt",        # sqrt(var + eps)
    "Div",         # (x - mean) / std
    "Mul",         # gamma * normalized
    "Add_1",       # beta + scaled
)


def _dequantize_initializer(
    dq_output_name: str,
    by_output_map: Dict[str, onnx.NodeProto],
    initializers: Dict[str, onnx.TensorProto],
) -> np.ndarray:
    """Trace ``dq_output_name`` back to the quantized initializer + scale + zp and return FP32 values.

    Supports per-tensor (scalar scale/zp) and per-channel quantization
    (1-D scale/zp with a corresponding ``axis`` attribute on the DQ node).
    """
    dq_node = by_output_map[dq_output_name]
    if dq_node.op_type != "DequantizeLinear":
        raise ValueError(f"Expected DequantizeLinear producing {dq_output_name!r}, got {dq_node.op_type}")
    if len(dq_node.input) < 2:
        raise ValueError(f"DequantizeLinear {dq_node.name!r} has too few inputs: {list(dq_node.input)}")
    q_init_name = dq_node.input[0]
    scale_name = dq_node.input[1]
    zp_name = dq_node.input[2] if len(dq_node.input) >= 3 else None
    if q_init_name not in initializers or scale_name not in initializers:
        raise ValueError(
            f"DequantizeLinear {dq_node.name!r} inputs not all in initializers: "
            f"q_init={q_init_name} scale={scale_name} zp={zp_name}"
        )
    q_arr = numpy_helper.to_array(initializers[q_init_name])
    scale = numpy_helper.to_array(initializers[scale_name]).astype(np.float32)
    if zp_name and zp_name in initializers:
        zp = numpy_helper.to_array(initializers[zp_name])
    else:
        zp = np.zeros((), dtype=q_arr.dtype)

    axis = 1
    for attr in dq_node.attribute:
        if attr.name == "axis":
            axis = attr.i

    if scale.size == 1:
        fp32 = (q_arr.astype(np.float32) - float(zp.item() if zp.size == 1 else 0)) * float(scale.item())
    else:
        shape = [1] * q_arr.ndim
        shape[axis] = -1
        scale_b = scale.reshape(shape)
        zp_b = zp.astype(np.float32).reshape(shape) if zp.size > 1 else np.zeros_like(scale_b)
        fp32 = (q_arr.astype(np.float32) - zp_b) * scale_b
    return fp32.astype(np.float32)


def _find_consumer_qlinear(
    output_name: str,
    nodes: List[onnx.NodeProto],
) -> Optional[onnx.NodeProto]:
    """Return the single ``QuantizeLinear`` consumer of ``output_name``, or None / raise on ambiguity."""
    consumers = [n for n in nodes if output_name in n.input]
    q_consumers = [n for n in consumers if n.op_type == "QuantizeLinear"]
    if len(q_consumers) == 0:
        return None
    if len(q_consumers) > 1:
        raise ValueError(
            f"Expected exactly 1 QuantizeLinear consumer of {output_name!r}, got {len(q_consumers)}: "
            f"{[c.name for c in q_consumers]}"
        )
    return q_consumers[0]


def fuse_first_branch_norm_ln_inplace(
    model: onnx.ModelProto,
    *,
    branch: str = "mapping_core",
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """Fuse the first ``/<branch>/norm/`` LN cluster into a single ``LayerNormalization(axis=1)``.

    Args:
        model: The ONNX ModelProto to modify in-place.
        branch: One of ``"mapping_core"``, ``"masking_core"``, or ``""``
            (the bare ``/norm/`` cluster at the masking-core branch root, if any).
            The function targets the first cluster with primitive names
            ``/{branch}/norm/ReduceMean`` etc. (no trailing ``_N`` index).
        epsilon: LN epsilon. Default 1e-6 matches BAFNet+'s :class:`LayerNorm1d`.

    Returns:
        A dict with surgery audit info: ``branch``, ``deleted_node_names`` (set),
        ``deleted_node_count``, ``added_node_name`` (the new LN op),
        ``gamma_init_name``, ``beta_init_name``, ``gamma_shape``, ``beta_shape``,
        ``cluster_input_tensor``, ``cluster_output_q_node``.

    Raises:
        ValueError: If the targeted cluster's 9 primitives aren't all found,
            or if the cluster output's Q wrapper is missing / ambiguous.
    """
    if not branch:
        prefix = "/norm"
    else:
        prefix = f"/{branch}/norm"

    primitive_names = [f"{prefix}/{suffix}" for suffix in _PRIMITIVE_SUFFIXES]

    by_name = {n.name: n for n in model.graph.node if n.name}
    missing = [nm for nm in primitive_names if nm not in by_name]
    if missing:
        raise ValueError(
            f"LN cluster surgery: primitive nodes not found for branch={branch!r}: {missing}"
        )
    primitives = [by_name[nm] for nm in primitive_names]
    rm_first, sub_node, pow_node, rm_second, add_eps, sqrt_node, div_node, mul_gamma, add_beta = primitives

    cluster_input_name = rm_first.input[0]
    cluster_output_name = add_beta.output[0]

    # Locate the existing Q wrapper at cluster output — its input we'll rewire.
    nodes_list = list(model.graph.node)
    cluster_output_q = _find_consumer_qlinear(cluster_output_name, nodes_list)
    if cluster_output_q is None:
        raise ValueError(
            f"LN cluster surgery: no QuantizeLinear found downstream of {cluster_output_name!r} "
            f"(Add_1 output) — cannot rewire cluster boundary."
        )

    # Build initializer + producer maps for gamma/beta extraction.
    inits = {init.name: init for init in model.graph.initializer}
    by_output = {o: n for n in model.graph.node for o in n.output}

    # Mul's input[1] = gamma DQ output; Add_1's input[1] = beta DQ output.
    gamma_fp32 = _dequantize_initializer(mul_gamma.input[1], by_output, inits)
    beta_fp32 = _dequantize_initializer(add_beta.input[1], by_output, inits)

    if gamma_fp32.size != beta_fp32.size:
        raise ValueError(
            f"gamma size {gamma_fp32.size} != beta size {beta_fp32.size}"
        )
    channels = int(gamma_fp32.size)

    # LayerNormalization(axis=1) expects scale/bias of shape matching the normalized dim — here [C].
    gamma_1d = gamma_fp32.reshape(-1)
    beta_1d = beta_fp32.reshape(-1)

    gamma_init_name = f"ln_fuse_gamma_{branch or 'root'}_norm_0"
    beta_init_name = f"ln_fuse_beta_{branch or 'root'}_norm_0"
    gamma_init = numpy_helper.from_array(gamma_1d, name=gamma_init_name)
    beta_init = numpy_helper.from_array(beta_1d, name=beta_init_name)
    model.graph.initializer.extend([gamma_init, beta_init])

    # Build the Transpose-LN-Transpose triplet.
    #
    # Pre-Transpose: [B, C, T, F] → [B, T, F, C]   perm=(0, 2, 3, 1)
    # LayerNormalization(axis=-1) over [B, T, F, C]: normalise over the last axis (C).
    #   scale/bias shape [C] match this last-axis-only semantic.
    # Post-Transpose: [B, T, F, C] → [B, C, T, F]   perm=(0, 3, 1, 2)
    #
    # Mathematically equivalent to BAFNet+ LayerNorm1d on the channel axis of
    # [B, C, T, F] because ONNX LN(axis=-1) on [B, T, F, C] computes per-
    # (B, T, F)-location normalisation over C with [C]-shaped scale/bias.
    pre_t_name = f"{prefix}/ln_fuse_pre_transpose"
    pre_t_out = f"{prefix}/ln_fuse_pre_t_output"
    pre_t_node = helper.make_node(
        "Transpose",
        inputs=[cluster_input_name],
        outputs=[pre_t_out],
        name=pre_t_name,
        perm=[0, 2, 3, 1],
    )

    ln_node_name = f"{prefix}/LayerNormalizationFused"
    ln_output_name = f"{prefix}/ln_fuse_output"
    ln_node = helper.make_node(
        "LayerNormalization",
        inputs=[pre_t_out, gamma_init_name, beta_init_name],
        outputs=[ln_output_name],
        name=ln_node_name,
        axis=-1,
        epsilon=epsilon,
    )

    post_t_name = f"{prefix}/ln_fuse_post_transpose"
    post_t_out = f"{prefix}/ln_fuse_post_t_output"
    post_t_node = helper.make_node(
        "Transpose",
        inputs=[ln_output_name],
        outputs=[post_t_out],
        name=post_t_name,
        perm=[0, 3, 1, 2],
    )

    # Insert the triplet at the position of rm_first.
    rm_first_idx = None
    for i, n in enumerate(model.graph.node):
        if n is rm_first:
            rm_first_idx = i
            break
    if rm_first_idx is None:
        raise ValueError("LN cluster surgery: could not locate rm_first index in graph (internal error)")
    model.graph.node.insert(rm_first_idx, post_t_node)
    model.graph.node.insert(rm_first_idx, ln_node)
    model.graph.node.insert(rm_first_idx, pre_t_node)

    # Rewire cluster_output_q's input slot to post-Transpose output.
    rewired = False
    for i, inp in enumerate(cluster_output_q.input):
        if inp == cluster_output_name:
            cluster_output_q.input[i] = post_t_out
            rewired = True
            break
    if not rewired:
        raise ValueError(
            f"LN cluster surgery: could not rewire cluster_output_q={cluster_output_q.name!r} "
            f"to use {post_t_out!r}"
        )

    # Identify intermediate Q and DQ nodes downstream of primitive outputs (except the
    # cluster_output_q which we just rewired).
    primitive_outputs = set()
    for prim in primitives:
        primitive_outputs.update(prim.output)

    nodes_to_delete: List[str] = [p.name for p in primitives]
    cluster_output_q_id = id(cluster_output_q)

    # Walk Q→DQ pairs whose Q consumes a primitive output (excluding the cluster_output_q).
    for n in list(model.graph.node):
        if n.op_type != "QuantizeLinear":
            continue
        if id(n) == cluster_output_q_id:
            continue
        # Does this Q consume any primitive output?
        consumed_prim_outputs = [inp for inp in n.input if inp in primitive_outputs]
        if not consumed_prim_outputs:
            continue
        nodes_to_delete.append(n.name)
        # Find the DQ that consumes this Q's output (typically exactly one DQ per Q).
        q_out_names = list(n.output)
        for n2 in list(model.graph.node):
            if n2.op_type != "DequantizeLinear":
                continue
            if any(out_name in n2.input for out_name in q_out_names):
                nodes_to_delete.append(n2.name)

    nodes_to_delete_set = set(nodes_to_delete)

    # Delete by reverse index so positions stay stable.
    indices_to_delete = sorted(
        [i for i, n in enumerate(model.graph.node) if n.name in nodes_to_delete_set],
        reverse=True,
    )
    for i in indices_to_delete:
        del model.graph.node[i]

    # Re-run shape inference so downstream consumers see the new LN output's shape.
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False)
        del model.graph.value_info[:]
        model.graph.value_info.extend(inferred.graph.value_info)
    except Exception as e:  # noqa: BLE001
        logger.warning("ln_fuse: shape inference re-run failed (%s); proceeding", e)

    audit = {
        "branch": branch,
        "primitive_names": list(primitive_names),
        "deleted_node_names": sorted(nodes_to_delete_set),
        "deleted_node_count": len(nodes_to_delete_set),
        "added_node_names": [pre_t_name, ln_node_name, post_t_name],
        "added_initializer_names": [gamma_init_name, beta_init_name],
        "ln_node_name": ln_node_name,
        "pre_transpose_node_name": pre_t_name,
        "post_transpose_node_name": post_t_name,
        "gamma_init_name": gamma_init_name,
        "beta_init_name": beta_init_name,
        "gamma_shape": list(gamma_1d.shape),
        "beta_shape": list(beta_1d.shape),
        "channels": channels,
        "epsilon": epsilon,
        "axis_internal": -1,
        "logical_normalize_axis": 1,
        "pre_transpose_perm": [0, 2, 3, 1],
        "post_transpose_perm": [0, 3, 1, 2],
        "cluster_input_tensor": cluster_input_name,
        "cluster_output_q_node": cluster_output_q.name,
        "pre_transpose_output_tensor": pre_t_out,
        "ln_output_tensor": ln_output_name,
        "post_transpose_output_tensor": post_t_out,
    }
    logger.info(
        "ln_fuse: fused %s cluster — deleted %d nodes (9 primitives + %d Q/DQ wrappers), "
        "added Transpose → LayerNormalization(axis=-1, eps=%s) → Transpose + 2 FP32 initializers, "
        "gamma/beta shape=[C=%d]",
        prefix, len(nodes_to_delete_set), len(nodes_to_delete_set) - 9, epsilon, channels,
    )
    return audit


__all__ = ["fuse_first_branch_norm_ln_inplace"]
