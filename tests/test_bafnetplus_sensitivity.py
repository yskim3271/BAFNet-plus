"""Tests for the S23 Phase A1 per-node QDQ sensitivity module.

Four structural tests use synthetic ONNX graphs / numpy fixtures and are
ORT-free. The fifth (``test_smoke_run_sensitivity_analysis_real_trunk``) is
gated on the S21 baseline assets + the ``/tmp/bafnet_calib_taps_v3``
calibration corpus and exercises the full CLI end-to-end with
``--max-samples 10`` for speed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from src.analysis.sensitivity import (
    DEFAULT_OP_TYPES,
    NORM_SCOPE_EXCLUDE,
    RELATIVE_NORM_EPS,
    SCHEMA_VERSION,
    _expose_intermediate_outputs,
    _intermediate_outputs_to_expose,
    by_module_rollup,
    combine_metrics,
    compute_qdq_err,
    compute_xmodel_err,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_FP32_TRUNK = _REPO_ROOT / "results/onnx/bafnetplus_50ms_fp32_trunk.onnx"
_REAL_INT8_TRUNK = _REPO_ROOT / "results/onnx/bafnetplus_50ms_int8_qdq_trunk.onnx"
_REAL_CALIB_DIR = Path("/tmp/bafnet_calib_taps_v3")


def _real_assets_available() -> bool:
    return (
        _REAL_FP32_TRUNK.exists()
        and _REAL_INT8_TRUNK.exists()
        and _REAL_CALIB_DIR.is_dir()
        and any(_REAL_CALIB_DIR.glob("calib_*.npz"))
    )


def _make_mixed_optype_graph(tmp_path: Path) -> Path:
    """Tiny ONNX with one node per scenario the op-type filter must handle.

    Nodes:
        * ``/encoder/conv0/Conv``         — Conv outside /norm/  (INCLUDED)
        * ``/encoder/sig0/Sigmoid``       — Sigmoid outside /norm/ (INCLUDED)
        * ``/encoder/add0/Add``           — Add outside /norm/   (INCLUDED)
        * ``/encoder/norm/add_norm/Add``  — Add inside /norm/    (EXCLUDED — /norm/)
        * ``/encoder/pow0/Pow``           — Pow                 (EXCLUDED — not in default op_types)

    The graph is structurally minimal: each op consumes the primary input
    or a single Constant; outputs are unused after the next op so we don't
    need to thread them through downstream consumers.
    """
    fp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])

    w_conv = helper.make_tensor(
        name="w_conv",
        data_type=TensorProto.FLOAT,
        dims=[4, 4, 1],
        vals=np.eye(4, dtype=np.float32).reshape(4, 4, 1).flatten().tolist(),
    )
    c_one = helper.make_tensor(
        name="c_one", data_type=TensorProto.FLOAT, dims=[1], vals=[1.0]
    )
    c_two = helper.make_tensor(
        name="c_two", data_type=TensorProto.FLOAT, dims=[1], vals=[2.0]
    )

    n_conv = helper.make_node(
        "Conv", inputs=["x", "w_conv"], outputs=["conv0_out"],
        name="/encoder/conv0/Conv", kernel_shape=[1],
    )
    n_sig = helper.make_node(
        "Sigmoid", inputs=["conv0_out"], outputs=["sig0_out"],
        name="/encoder/sig0/Sigmoid",
    )
    n_add = helper.make_node(
        "Add", inputs=["sig0_out", "c_one"], outputs=["add0_out"],
        name="/encoder/add0/Add",
    )
    n_norm_add = helper.make_node(
        "Add", inputs=["add0_out", "c_one"], outputs=["norm_add_out"],
        name="/encoder/norm/add_norm/Add",
    )
    n_pow = helper.make_node(
        "Pow", inputs=["norm_add_out", "c_two"], outputs=["pow0_out"],
        name="/encoder/pow0/Pow",
    )

    out = helper.make_tensor_value_info("pow0_out", TensorProto.FLOAT, [1, 4, 8])

    graph = helper.make_graph(
        nodes=[n_conv, n_sig, n_add, n_norm_add, n_pow],
        name="mixed_optype",
        inputs=[fp],
        outputs=[out],
        initializer=[w_conv, c_one, c_two],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "mixed_optype.onnx"
    onnx.save(model, str(path))
    return path


# ============================================================================
# (1) _intermediate_outputs_to_expose default op_types + /norm/ exclusion
# ============================================================================
def test_intermediate_outputs_to_expose_default_optypes(tmp_path: Path) -> None:
    """Default op_types include Conv/Sigmoid/Add; /norm/-scoped Add is excluded;
    Pow is excluded by absence from DEFAULT_OP_TYPES."""
    onnx_path = _make_mixed_optype_graph(tmp_path)
    exposed = _intermediate_outputs_to_expose(onnx_path)
    assert "conv0_out" in exposed
    assert "sig0_out" in exposed
    assert "add0_out" in exposed
    assert "norm_add_out" not in exposed, "must exclude /norm/-scoped nodes"
    assert "pow0_out" not in exposed, "Pow not in DEFAULT_OP_TYPES"
    assert len(exposed) == 3

    # Explicit op_types override still respects /norm/ exclusion.
    exposed_with_pow = _intermediate_outputs_to_expose(
        onnx_path, op_types=("Pow", "Add", "Conv"),
    )
    assert "pow0_out" in exposed_with_pow
    assert "add0_out" in exposed_with_pow
    assert "conv0_out" in exposed_with_pow
    assert "norm_add_out" not in exposed_with_pow

    # NORM_SCOPE_EXCLUDE constant is documented + used.
    assert NORM_SCOPE_EXCLUDE == "/norm"
    # Default op_types matches the SeQTO-derived selection.
    assert set(DEFAULT_OP_TYPES) == {
        "Conv", "ConvTranspose", "MatMul", "Mul", "Add", "Sigmoid",
    }


# ============================================================================
# (2) _expose_intermediate_outputs idempotency
# ============================================================================
def test_expose_intermediate_outputs_idempotent(tmp_path: Path) -> None:
    """Tensors already in graph.output are not re-added; calling twice is a no-op."""
    onnx_path = _make_mixed_optype_graph(tmp_path)
    # Source graph has 1 output (pow0_out). Expose 3 new + 1 already-present.
    exposed1, added1 = _expose_intermediate_outputs(
        onnx_path, ["conv0_out", "sig0_out", "add0_out", "pow0_out"],
        output_path=tmp_path / "exposed1.onnx",
    )
    model1 = onnx.load(str(exposed1))
    out_names1 = [o.name for o in model1.graph.output]
    assert sorted(out_names1) == sorted(["pow0_out", "conv0_out", "sig0_out", "add0_out"])
    assert sorted(added1) == sorted(["conv0_out", "sig0_out", "add0_out"])
    assert "pow0_out" not in added1, "already-existing graph output must not be re-added"

    # Second pass on the already-exposed graph — nothing new.
    exposed2, added2 = _expose_intermediate_outputs(
        exposed1, ["conv0_out", "sig0_out", "add0_out"],
        output_path=tmp_path / "exposed2.onnx",
    )
    model2 = onnx.load(str(exposed2))
    out_names2 = [o.name for o in model2.graph.output]
    assert sorted(out_names2) == sorted(out_names1), "second pass changed graph outputs"
    assert added2 == []

    # Unproducible tensor name is silently skipped (not in producers / value_info).
    exposed3, added3 = _expose_intermediate_outputs(
        onnx_path, ["does_not_exist_anywhere"],
        output_path=tmp_path / "exposed3.onnx",
    )
    assert added3 == []
    model3 = onnx.load(str(exposed3))
    assert "does_not_exist_anywhere" not in {o.name for o in model3.graph.output}


# ============================================================================
# (3) compute_qdq_err on synthetic activations — formula correctness
# ============================================================================
def test_compute_qdq_err_synthetic_2layer_conv() -> None:
    """Hand-computed expected values against the streaming-mean L2 norm formula.

    Two synthetic tensors, two samples each. Verified independently:
    * Sample 0 ``diff`` = [0.5, 0.5, 0.5, 0.5] → ||diff||₂ = 1.0
    * Sample 1 ``diff`` = [0, 1, 0, 1]         → ||diff||₂ = sqrt(2) ≈ 1.41421356
    * mean = (1.0 + 1.41421356) / 2 ≈ 1.20710678
    """
    fp = {
        "conv1_out": [
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([2.0, 3.0, 2.0, 3.0], dtype=np.float32),
        ],
        "conv2_out": [
            np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ],
    }
    int_ = {
        "conv1_out": [
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),  # diff = 0.5 each
            np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),  # diff = [0,1,0,1]
        ],
        "conv2_out": [
            np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),  # diff = 0 → L2 = 0
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # diff = 0 → L2 = 0
        ],
    }
    qdq = compute_qdq_err(fp, int_)
    assert qdq["conv1_out"] == pytest.approx((1.0 + np.sqrt(2.0)) / 2.0, rel=1e-7)
    assert qdq["conv2_out"] == pytest.approx(0.0, abs=1e-12)

    # xmodel_err on the same data:
    # conv1, sample 0: ||diff||=1.0, ||fp||=2.0 → 0.5
    # conv1, sample 1: ||diff||=√2,  ||fp||=√(4+9+4+9)=√26 → √2/√26
    # mean = (0.5 + √2/√26) / 2
    xmodel = compute_xmodel_err(fp, int_)
    expected = (0.5 + np.sqrt(2.0) / np.sqrt(26.0)) / 2.0
    assert xmodel["conv1_out"] == pytest.approx(expected, rel=1e-7)
    assert xmodel["conv2_out"] == pytest.approx(0.0, abs=1e-12)

    # combine_metrics min-max normalizes each independently.
    combined = combine_metrics(qdq, xmodel)
    # Only 2 keys → norm collapses to {top: 1.0, bottom: 0.0} for both metrics
    # → combined = {top: 1.0, bottom: 0.0}.
    assert combined["conv1_out"] == pytest.approx(1.0)
    assert combined["conv2_out"] == pytest.approx(0.0)

    # by_module_rollup with prefix_depth=2 buckets both into the same scope
    # if names share the top-2 components. Here tensor names are not
    # path-shaped, so each gets its own scope.
    rollup = by_module_rollup({"a/b/c": 0.3, "a/b/d": 0.7, "x/y": 1.0}, prefix_depth=2)
    assert rollup["/a/b"] == pytest.approx(1.0)  # 0.3 + 0.7
    assert rollup["/x/y"] == pytest.approx(1.0)


# ============================================================================
# (4) xmodel_err eps floor — handles near-zero FP32 norm
# ============================================================================
def test_xmodel_err_handles_zero_norm_fp32() -> None:
    """A zero-norm FP32 tensor must not divide-by-zero — eps floor is engaged."""
    fp = {"zero": [np.zeros(8, dtype=np.float32)]}
    int_ = {"zero": [np.array([1e-3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)]}
    out = compute_xmodel_err(fp, int_, eps=RELATIVE_NORM_EPS)
    # diff_l2 = 1e-3; denom = max(0, eps) = eps = 1e-6
    # xmodel_err = 1e-3 / 1e-6 = 1000.0 (finite)
    assert np.isfinite(out["zero"]), "zero-norm fp32 must yield finite xmodel_err"
    assert out["zero"] == pytest.approx(1e-3 / RELATIVE_NORM_EPS, rel=1e-7)

    # Very small but non-zero FP32 norm — eps still binds if ||fp|| < eps.
    fp2 = {"tiny": [np.array([1e-9, 0, 0, 0], dtype=np.float32)]}
    int2 = {"tiny": [np.array([1e-3, 0, 0, 0], dtype=np.float32)]}
    out2 = compute_xmodel_err(fp2, int2, eps=RELATIVE_NORM_EPS)
    diff_l2 = np.sqrt((1e-3 - 1e-9) ** 2)
    expected = diff_l2 / RELATIVE_NORM_EPS
    assert out2["tiny"] == pytest.approx(expected, rel=1e-4)

    # Per-tensor sample-count mismatch → ValueError.
    with pytest.raises(ValueError, match="sample count mismatch"):
        compute_qdq_err({"a": [np.zeros(2)]}, {"a": [np.zeros(2), np.zeros(2)]})


# ============================================================================
# (5) Smoke: full CLI on the S21 baseline trunk with max_samples=10
# ============================================================================
@pytest.mark.skipif(
    not _real_assets_available(),
    reason="S21 baseline trunk + /tmp/bafnet_calib_taps_v3 not available",
)
def test_smoke_run_sensitivity_analysis_real_trunk(tmp_path: Path) -> None:
    """CLI end-to-end on the real S21 baseline trunk (max_samples=10 for speed)."""
    import json

    output_json = tmp_path / "sensitivity_smoke.json"
    cmd = [
        sys.executable, "-m", "src.analysis.sensitivity",
        "--fp32-onnx", str(_REAL_FP32_TRUNK),
        "--int8-onnx", str(_REAL_INT8_TRUNK),
        "--calibration-dir", str(_REAL_CALIB_DIR),
        "--output-json", str(output_json),
        "--max-samples", "10",
        "--top-k", "20",
    ]
    proc = subprocess.run(
        cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"sensitivity CLI failed (exit {proc.returncode}):\nSTDOUT:\n{proc.stdout}"
        f"\nSTDERR:\n{proc.stderr}"
    )
    assert output_json.exists()

    result = json.loads(output_json.read_text())
    # Structural checks: schema + required keys.
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["produced_by"] == "src.analysis.sensitivity.run_sensitivity_analysis"
    assert result["calibration_samples_used"] == 10
    assert result["n_nodes_analyzed"] > 100, "trunk has hundreds of analyzable nodes"
    assert len(result["ranked_nodes"]) == 20
    assert result["ranked_nodes"][0]["rank"] == 1
    # Top combined_metric is bounded [0, 1] (min-max blend); it equals 1.0 only
    # when the same node tops BOTH metrics after normalization, which is not
    # guaranteed. We assert the valid range + descending order instead.
    top_combined = result["ranked_nodes"][0]["combined_metric"]
    assert 0.0 < top_combined <= 1.0, f"top combined out of range: {top_combined}"
    combined_values = [e["combined_metric"] for e in result["ranked_nodes"]]
    assert combined_values == sorted(combined_values, reverse=True), "ranked not descending"
    assert result["ranked_nodes"][-1]["rank"] == 20

    # Each ranked entry has the schema's required keys.
    for entry in result["ranked_nodes"]:
        for k in ("rank", "node_name", "op_type", "tensor_name", "qdq_err", "xmodel_err", "combined_metric"):
            assert k in entry, f"missing key {k!r} in ranked entry"
        assert np.isfinite(entry["qdq_err"])
        assert np.isfinite(entry["xmodel_err"])
        assert 0.0 <= entry["combined_metric"] <= 1.0

    # by_module_rollup contains at least one BAFNet+ scope key.
    rollup_keys = list(result["by_module_rollup"])
    assert any("mapping_core" in k or "masking_core" in k for k in rollup_keys), (
        f"expected BAFNet+ scope in rollup; got {rollup_keys[:5]}"
    )

    # Temp exposed ONNX files should have been cleaned up.
    assert not (output_json.parent / "sensitivity_smoke.fp32_exposed.onnx").exists()
    assert not (output_json.parent / "sensitivity_smoke.int8_exposed.onnx").exists()
