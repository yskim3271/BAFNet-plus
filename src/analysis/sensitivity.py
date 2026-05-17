"""Per-node QDQ sensitivity analysis for INT8 BAFNet+ trunk graphs (S23 Phase A1).

SeQTO-style per-node sensitivity diagnostic — reimplementation of the two
core metrics from Louloudakis & Rajan, "Selective Quantization Tuning for
ONNX Models" (ICSE 2026, arXiv:2507.12196 v2) adapted to the BAFNet+
audio-regression W8A16 pipeline. Drives Phase A2/B scope decisions by
ranking which graph nodes contribute most quantization noise.

Two metrics (verbatim formula from the paper, simplified isolation proxy):

  * ``qdq_err`` — per-tensor mean L2 norm of (fp32 - int8) across calibration
    samples. SeQTO's qdq_err is technically the per-node isolation-quantization
    error (only that node quantized); we use the cheaper fully-quantized
    proxy since N+1 isolation runs are prohibitive for a 951-node graph.
  * ``xmodel_err`` — per-tensor mean relative L2 error
    ``||fp - int||₂ / max(||fp||₂, eps)``. SeQTO's "in-context" error.
  * ``combined_metric`` — ``0.5 * norm(qdq_err) + 0.5 * norm(xmodel_err)``
    after per-metric min-max normalization across all paired nodes.

Pairing strategy: NODE NAME, not output tensor name. ORT's QDQ pass
renames some output tensors (e.g. Sigmoid: ``Sigmoid_output_0`` →
``Sigmoid``) but preserves node names. Each FP32 node is paired with the
INT8 node of identical name + op_type. Nodes fused away during INT8
conversion (many Mul/Add) are skipped and counted in
``unpaired_skipped``.

CLI::

    python -m src.analysis.sensitivity \\
        --fp32-onnx results/onnx/bafnetplus_50ms_fp32_trunk.onnx \\
        --int8-onnx results/onnx/bafnetplus_50ms_int8_qdq_trunk.onnx \\
        --calibration-dir /tmp/bafnet_calib_taps_v3 \\
        --output-json results/analysis/sensitivity_s21_baseline.json \\
        --top-k 30

Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "s23-bafnetplus-sensitivity-v1"
METHOD_REFERENCE = "Louloudakis & Rajan, ICSE 2026 (arXiv:2507.12196 v2)"
DEFAULT_OP_TYPES: Tuple[str, ...] = (
    "Conv",
    "ConvTranspose",
    "MatMul",
    "Mul",
    "Add",
    "Sigmoid",
)
RELATIVE_NORM_EPS = 1e-6
NORM_SCOPE_EXCLUDE = "/norm"  # LN-internal stable distributions


@dataclass(frozen=True)
class _AnalysisNode:
    node_name: str
    op_type: str
    fp32_tensor: str
    int8_tensor: str


# ----------------------------------------------------------------- helpers
def _md5_of(path: Path) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _node_index(onnx_path: Path) -> Dict[str, Tuple[str, str]]:
    """Return dict node_name -> (op_type, output_tensor_name [first output])."""
    import onnx  # noqa: PLC0415

    model = onnx.load(str(onnx_path))
    idx: Dict[str, Tuple[str, str]] = {}
    for n in model.graph.node:
        if not n.output:
            continue
        idx[n.name] = (n.op_type, n.output[0])
    return idx


def _intermediate_outputs_to_expose(
    onnx_path: Any,
    op_types: Optional[Sequence[str]] = None,
) -> List[str]:
    """Walk ONNX graph; return output-tensor names for nodes whose intermediate
    activations should be exposed for sensitivity analysis.

    Default ``op_types``: :data:`DEFAULT_OP_TYPES`
    (``Conv``, ``ConvTranspose``, ``MatMul``, ``Mul``, ``Add``, ``Sigmoid``).
    Excludes nodes whose name contains ``/norm`` (LN-internal stable
    distributions — these quantize cleanly and would saturate the ranking
    with noise-floor entries). This is the inverse of the
    :func:`_auto_precision_sensitive_nodes` logic in
    :mod:`src.models.streaming.onnx.export`: that function INCLUDES
    top-level Pow/Sqrt (sensitive); we EXCLUDE /norm/-scoped ops (stable).
    """
    import onnx  # noqa: PLC0415

    model = onnx.load(str(onnx_path))
    types = tuple(op_types) if op_types is not None else DEFAULT_OP_TYPES
    out: List[str] = []
    for n in model.graph.node:
        if n.op_type not in types:
            continue
        if NORM_SCOPE_EXCLUDE in n.name:
            continue
        if not n.output:
            continue
        out.append(n.output[0])
    return out


def _expose_intermediate_outputs(
    src_onnx: Any,
    exposed_tensors: Sequence[str],
    *,
    output_path: Optional[Path] = None,
) -> Tuple[Path, List[str]]:
    """Add each tensor in ``exposed_tensors`` as a graph output of ``src_onnx``.

    Idempotent: tensors already present in ``model.graph.output`` are skipped
    (not re-added). Tensors not produced by any node and not in any
    existing value_info are also skipped (cannot be exposed).

    Args:
        src_onnx: Source ONNX path.
        exposed_tensors: Tensor names to add as outputs.
        output_path: Output ONNX path. If ``None``, a tempfile in the
            system temp dir is used.

    Returns:
        ``(output_path, actually_added_names)``. ``actually_added_names`` is
        the subset of ``exposed_tensors`` that were newly added (so callers
        can distinguish duplicates / unproducible tensors).
    """
    import onnx  # noqa: PLC0415
    from onnx import TensorProto  # noqa: PLC0415

    model = onnx.load(str(src_onnx))
    existing = {o.name for o in model.graph.output}
    value_info = {vi.name: vi for vi in model.graph.value_info}
    producers: Dict[str, Any] = {}
    for n in model.graph.node:
        for o in n.output:
            producers[o] = n

    added: List[str] = []
    for name in exposed_tensors:
        if name in existing:
            continue
        if name not in producers and name not in value_info:
            logger.debug("skip exposing %s: no producer / value_info found", name)
            continue
        if name in value_info:
            vi = onnx.ValueInfoProto()
            vi.CopyFrom(value_info[name])
        else:
            # Fallback: float tensor with unknown shape. ORT will infer.
            vi = onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        model.graph.output.append(vi)
        existing.add(name)
        added.append(name)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix="sensitivity_exposed_", suffix=".onnx", delete=False
        )
        tmp.close()
        output_path = Path(tmp.name)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path, added


def _pair_nodes_for_analysis(
    fp32_path: Path,
    int8_path: Path,
    op_types: Optional[Sequence[str]] = None,
) -> Tuple[List[_AnalysisNode], List[str]]:
    """Build the list of ``(node_name, fp32_tensor, int8_tensor)`` pairs for
    analysis. Pairs by NODE NAME (robust to ORT's tensor renames).

    Returns ``(analysis_nodes, unpaired_fp32_node_names)``. ``unpaired`` is
    sorted; typically these are Mul/Add nodes fused away during INT8
    conversion (ConvMulFusion etc.).
    """
    import onnx  # noqa: PLC0415

    types = tuple(op_types) if op_types is not None else DEFAULT_OP_TYPES
    fp_model = onnx.load(str(fp32_path))
    int_model = onnx.load(str(int8_path))
    int_nodes = {n.name: (n.op_type, n.output[0]) for n in int_model.graph.node if n.output}

    paired: List[_AnalysisNode] = []
    unpaired: List[str] = []
    for n in fp_model.graph.node:
        if n.op_type not in types or NORM_SCOPE_EXCLUDE in n.name or not n.output:
            continue
        match = int_nodes.get(n.name)
        if match is None or match[0] != n.op_type:
            unpaired.append(n.name)
            continue
        paired.append(
            _AnalysisNode(
                node_name=n.name,
                op_type=n.op_type,
                fp32_tensor=n.output[0],
                int8_tensor=match[1],
            )
        )
    return paired, sorted(unpaired)


# -------------------------------------------------------------------- metrics
def compute_qdq_err(
    act_fp32: Dict[str, Sequence[np.ndarray]],
    act_int8: Dict[str, Sequence[np.ndarray]],
) -> Dict[str, float]:
    """Per-tensor mean L2 norm of ``(fp32 - int8)`` across calibration samples.

    Both inputs are ``dict[tensor_name → list of per-sample arrays]``.
    Returns ``dict[tensor_name → float]`` (mean across samples). Tensors
    must appear in BOTH input dicts with EQUAL list length.
    """
    out: Dict[str, float] = {}
    for name in act_fp32:
        if name not in act_int8:
            raise KeyError(f"tensor {name!r} missing from act_int8")
        fp_list = act_fp32[name]
        int_list = act_int8[name]
        if len(fp_list) != len(int_list):
            raise ValueError(
                f"sample count mismatch for {name!r}: fp32={len(fp_list)} int8={len(int_list)}"
            )
        if not fp_list:
            out[name] = 0.0
            continue
        s = 0.0
        for fp, it in zip(fp_list, int_list):
            diff = (np.asarray(fp, dtype=np.float64) - np.asarray(it, dtype=np.float64)).ravel()
            s += float(np.sqrt(np.dot(diff, diff)))
        out[name] = s / len(fp_list)
    return out


def compute_xmodel_err(
    act_fp32: Dict[str, Sequence[np.ndarray]],
    act_int8: Dict[str, Sequence[np.ndarray]],
    eps: float = RELATIVE_NORM_EPS,
) -> Dict[str, float]:
    """Per-tensor mean relative L2 error ``||fp - int||₂ / max(||fp||₂, eps)``.

    Same input contract as :func:`compute_qdq_err`. The ``eps`` floor
    prevents blow-up on tensors with near-zero FP32 norm (e.g.,
    LN-internal numerics on cold-start state).
    """
    out: Dict[str, float] = {}
    for name in act_fp32:
        if name not in act_int8:
            raise KeyError(f"tensor {name!r} missing from act_int8")
        fp_list = act_fp32[name]
        int_list = act_int8[name]
        if len(fp_list) != len(int_list):
            raise ValueError(
                f"sample count mismatch for {name!r}: fp32={len(fp_list)} int8={len(int_list)}"
            )
        if not fp_list:
            out[name] = 0.0
            continue
        s = 0.0
        for fp, it in zip(fp_list, int_list):
            fp_arr = np.asarray(fp, dtype=np.float64).ravel()
            it_arr = np.asarray(it, dtype=np.float64).ravel()
            diff = fp_arr - it_arr
            diff_l2 = float(np.sqrt(np.dot(diff, diff)))
            fp_l2 = float(np.sqrt(np.dot(fp_arr, fp_arr)))
            s += diff_l2 / max(fp_l2, eps)
        out[name] = s / len(fp_list)
    return out


def _minmax_normalize(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    values = list(d.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def combine_metrics(
    qdq: Dict[str, float],
    xmodel: Dict[str, float],
) -> Dict[str, float]:
    """Min-max normalize each metric across all nodes, then
    ``combined = 0.5 * norm_qdq + 0.5 * norm_xmodel``. SeQTO error_metric
    formula (verbatim, §IV of the paper)."""
    keys = sorted(set(qdq) & set(xmodel))
    if not keys:
        return {}
    norm_qdq = _minmax_normalize({k: qdq[k] for k in keys})
    norm_xmodel = _minmax_normalize({k: xmodel[k] for k in keys})
    return {k: 0.5 * norm_qdq[k] + 0.5 * norm_xmodel[k] for k in keys}


def by_module_rollup(
    per_node_metrics: Dict[str, float],
    prefix_depth: int = 2,
) -> Dict[str, float]:
    """Aggregate per-node metric values by the top-N components of node names.

    ``per_node_metrics`` keys are full node names (e.g.
    ``/mapping_core/sequence_block.0/time_stage/proj_first/Conv``). With
    ``prefix_depth=2`` the rollup buckets at ``/mapping_core/sequence_block.0``;
    with ``prefix_depth=3`` at ``/mapping_core/sequence_block.0/time_stage``.

    Plain sum aggregation (no param-count weighting) — favors modules with
    more nodes, which is the intent for "where is the noise mass".
    """
    rollup: Dict[str, float] = {}
    for node_name, value in per_node_metrics.items():
        parts = [p for p in node_name.split("/") if p]
        scope_parts = parts[:prefix_depth] if len(parts) >= prefix_depth else parts
        scope = "/" + "/".join(scope_parts) if scope_parts else node_name
        rollup[scope] = rollup.get(scope, 0.0) + value
    return rollup


# ------------------------------------------------ ORT streaming capture
def _run_streaming_capture(
    fp32_with_exposed: Path,
    int8_with_exposed: Path,
    calibration_dir: Path,
    paired: Sequence[_AnalysisNode],
    *,
    max_samples: Optional[int] = None,
    eps: float = RELATIVE_NORM_EPS,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """Run both ORT sessions over the calibration corpus, computing streaming
    qdq_err + xmodel_err per node (keyed by ``node_name``).

    Streaming aggregation — O(num_nodes) memory regardless of sample count.
    Returns ``(qdq_err, xmodel_err, n_samples_used)``.
    """
    import onnxruntime as ort  # noqa: PLC0415

    from src.models.streaming.onnx.export import (  # noqa: PLC0415 — runtime composition
        _BAFNetPlusCalibrationDataReader,
    )

    fp_sess = ort.InferenceSession(
        str(fp32_with_exposed), providers=["CPUExecutionProvider"]
    )
    int_sess = ort.InferenceSession(
        str(int8_with_exposed), providers=["CPUExecutionProvider"]
    )

    fp_outputs = {o.name for o in fp_sess.get_outputs()}
    int_outputs = {o.name for o in int_sess.get_outputs()}
    runnable = [
        p for p in paired
        if p.fp32_tensor in fp_outputs and p.int8_tensor in int_outputs
    ]
    dropped = len(paired) - len(runnable)
    if dropped:
        logger.warning(
            "%d/%d paired nodes had exposed tensors missing from one session — dropped",
            dropped, len(paired),
        )

    reader = _BAFNetPlusCalibrationDataReader(
        calibration_dir, fp32_with_exposed, max_samples=max_samples
    )

    fp_fetch = [p.fp32_tensor for p in runnable]
    int_fetch = [p.int8_tensor for p in runnable]
    sum_qdq = {p.node_name: 0.0 for p in runnable}
    sum_xmodel = {p.node_name: 0.0 for p in runnable}
    n = 0
    while True:
        feed = reader.get_next()
        if feed is None:
            break
        fp_outs = fp_sess.run(fp_fetch, feed)
        int_outs = int_sess.run(int_fetch, feed)
        for p, fp_arr, int_arr in zip(runnable, fp_outs, int_outs):
            fp_flat = np.asarray(fp_arr, dtype=np.float64).ravel()
            int_flat = np.asarray(int_arr, dtype=np.float64).ravel()
            diff = fp_flat - int_flat
            diff_l2 = float(np.sqrt(np.dot(diff, diff)))
            fp_l2 = float(np.sqrt(np.dot(fp_flat, fp_flat)))
            sum_qdq[p.node_name] += diff_l2
            sum_xmodel[p.node_name] += diff_l2 / max(fp_l2, eps)
        n += 1
        if n % 50 == 0:
            logger.info("  processed %d samples", n)
    if n == 0:
        zero = {p.node_name: 0.0 for p in runnable}
        return zero, dict(zero), 0
    qdq = {k: v / n for k, v in sum_qdq.items()}
    xmodel = {k: v / n for k, v in sum_xmodel.items()}
    return qdq, xmodel, n


# ----------------------------------------------------------------- driver
def run_sensitivity_analysis(
    fp32_onnx: Any,
    int8_onnx: Any,
    calibration_dir: Any,
    output_json: Any,
    *,
    top_k: int = 30,
    max_samples: Optional[int] = None,
    op_types: Optional[Sequence[str]] = None,
    prefix_depth: int = 3,
    eps: float = RELATIVE_NORM_EPS,
) -> Dict[str, Any]:
    """End-to-end Phase A1 sensitivity driver.

    1. Pair nodes between FP32 and INT8 graphs by node name (+ op_type),
       filtered to ``op_types`` and excluding ``/norm/``-scoped nodes.
    2. Expose paired-node output tensors as graph outputs in temp ONNX
       copies of each graph.
    3. Stream the calibration corpus through both ORT sessions; accumulate
       qdq_err + xmodel_err per node.
    4. Compute combined_metric (0.5/0.5 normalized blend).
    5. Rank, roll up by module scope, and write the structured JSON.

    Returns the result dict (also written to ``output_json``).
    """
    fp32_onnx = Path(fp32_onnx)
    int8_onnx = Path(int8_onnx)
    calibration_dir = Path(calibration_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    paired, unpaired = _pair_nodes_for_analysis(fp32_onnx, int8_onnx, op_types=op_types)
    logger.info(
        "paired %d nodes for analysis (skipped %d unpaired)",
        len(paired), len(unpaired),
    )
    if not paired:
        raise RuntimeError(
            f"no nodes paired between {fp32_onnx} and {int8_onnx} for op_types="
            f"{op_types or DEFAULT_OP_TYPES}; check graphs match expected topology."
        )

    fp32_exposed_path = output_json.with_name(output_json.stem + ".fp32_exposed.onnx")
    int8_exposed_path = output_json.with_name(output_json.stem + ".int8_exposed.onnx")
    try:
        _expose_intermediate_outputs(
            fp32_onnx, [p.fp32_tensor for p in paired], output_path=fp32_exposed_path
        )
        _expose_intermediate_outputs(
            int8_onnx, [p.int8_tensor for p in paired], output_path=int8_exposed_path
        )

        qdq, xmodel, n_samples = _run_streaming_capture(
            fp32_exposed_path, int8_exposed_path, calibration_dir, paired,
            max_samples=max_samples, eps=eps,
        )
    finally:
        # Tmp exposed ONNX files are large (~12 MB each); clean up.
        for p in (fp32_exposed_path, int8_exposed_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    combined = combine_metrics(qdq, xmodel)
    rollup = by_module_rollup(
        {p.node_name: combined.get(p.node_name, 0.0) for p in paired},
        prefix_depth=prefix_depth,
    )

    # Build ranked nodes (top_k by combined_metric; ties broken by qdq_err).
    by_node_name = {p.node_name: p for p in paired}
    ranked_pairs: List[Tuple[str, float]] = sorted(
        ((nm, combined[nm]) for nm in combined),
        key=lambda kv: (-kv[1], -qdq.get(kv[0], 0.0), kv[0]),
    )
    top_ranked = ranked_pairs[: max(0, int(top_k))]
    ranked_nodes: List[Dict[str, Any]] = []
    for rank, (node_name, score) in enumerate(top_ranked, start=1):
        meta = by_node_name[node_name]
        ranked_nodes.append({
            "rank": rank,
            "node_name": node_name,
            "op_type": meta.op_type,
            "tensor_name": meta.fp32_tensor,
            "qdq_err": qdq[node_name],
            "xmodel_err": xmodel[node_name],
            "combined_metric": score,
        })

    op_counts: Dict[str, int] = {}
    for p in paired:
        op_counts[p.op_type] = op_counts.get(p.op_type, 0) + 1

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "produced_by": "src.analysis.sensitivity.run_sensitivity_analysis",
        "method_reference": METHOD_REFERENCE,
        "fp32_onnx_path": str(fp32_onnx.resolve()),
        "fp32_onnx_md5": _md5_of(fp32_onnx),
        "int8_onnx_path": str(int8_onnx.resolve()),
        "int8_onnx_md5": _md5_of(int8_onnx),
        "calibration_dir": str(calibration_dir.resolve()),
        "calibration_samples_used": n_samples,
        "op_types_analyzed": list(op_types) if op_types is not None else list(DEFAULT_OP_TYPES),
        "op_type_distribution": op_counts,
        "n_nodes_analyzed": len(paired),
        "n_unpaired_skipped": len(unpaired),
        "module_rollup_prefix_depth": prefix_depth,
        "relative_norm_eps": eps,
        "elapsed_seconds": round(time.time() - t0, 2),
        "ranked_nodes": ranked_nodes,
        "by_module_rollup": rollup,
    }

    output_json.write_text(json.dumps(result, indent=2, sort_keys=False))
    logger.info(
        "wrote %s (n_nodes=%d, top_k=%d, samples=%d, elapsed=%.1fs)",
        output_json, len(paired), len(ranked_nodes), n_samples, result["elapsed_seconds"],
    )
    return result


# ----------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.analysis.sensitivity",
        description="SeQTO-style per-node QDQ sensitivity analysis (S23 Phase A1).",
    )
    p.add_argument("--fp32-onnx", required=True, type=Path, help="FP32 reference ONNX (trunk).")
    p.add_argument("--int8-onnx", required=True, type=Path, help="INT8 QDQ ONNX (trunk).")
    p.add_argument("--calibration-dir", required=True, type=Path, help="Directory of calib_*.npz.")
    p.add_argument("--output-json", required=True, type=Path, help="Output JSON path.")
    p.add_argument("--top-k", type=int, default=30, help="Number of top-ranked nodes to report.")
    p.add_argument("--max-samples", type=int, default=None, help="Cap on calibration samples (default: all).")
    p.add_argument(
        "--op-types", nargs="+", default=None,
        help=f"Op-types to analyze (default: {' '.join(DEFAULT_OP_TYPES)}).",
    )
    p.add_argument("--prefix-depth", type=int, default=3, help="by_module rollup path depth (default 3).")
    p.add_argument("--eps", type=float, default=RELATIVE_NORM_EPS, help="xmodel_err denominator floor.")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_sensitivity_analysis(
        fp32_onnx=args.fp32_onnx,
        int8_onnx=args.int8_onnx,
        calibration_dir=args.calibration_dir,
        output_json=args.output_json,
        top_k=args.top_k,
        max_samples=args.max_samples,
        op_types=args.op_types,
        prefix_depth=args.prefix_depth,
        eps=args.eps,
    )
    print(
        f"sensitivity analysis: {result['n_nodes_analyzed']} nodes, "
        f"{result['calibration_samples_used']} samples, "
        f"top combined_metric={result['ranked_nodes'][0]['combined_metric']:.4f} "
        f"({result['ranked_nodes'][0]['node_name']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
