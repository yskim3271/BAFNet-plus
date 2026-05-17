"""Tests for ``src.models.streaming.onnx.prelu_decompose``.

Three properties to cover:

1. **Numerical equivalence** at FP32: the decomposed graph produces
   bit-identical (within libm precision) outputs vs the original PReLU
   for arbitrary FP32 inputs and the learned per-channel slope.
2. **Graph rewrite correctness**: target PRelu nodes are gone; expected
   Max/Min/Mul/Add quartets are present; downstream consumer edges are
   preserved (same output tensor name as before surgery).
3. **Error paths**: empty target list, missing node, wrong op_type → raise.
"""

from __future__ import annotations

from typing import List

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from src.models.streaming.onnx.prelu_decompose import decompose_phase_prelu_inplace


def _build_minimal_prelu_graph(num_channels: int = 4) -> onnx.ModelProto:
    """Build a 2-node graph: input -> PRelu -> output, per-channel slope."""
    rng = np.random.default_rng(0)
    slope_np = (rng.uniform(0.1, 0.3, size=(num_channels,)).astype(np.float32)).reshape(
        1, num_channels, 1, 1
    )
    slope_initializer = numpy_helper.from_array(slope_np, name="slope")

    input_tensor = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [1, num_channels, 8, 12]
    )
    output_tensor = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, num_channels, 8, 12]
    )

    prelu_node = helper.make_node(
        "PRelu",
        inputs=["x", "slope"],
        outputs=["y"],
        name="/phase_conv/phase_conv.2/PRelu",
    )

    graph = helper.make_graph(
        nodes=[prelu_node],
        name="test_prelu_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[slope_initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def _run_ort(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(["y"], {"x": x})[0]


def test_numerical_equivalence_random_input():
    """Decomposed graph matches PRelu output to <= 1e-6 max-abs delta."""
    model = _build_minimal_prelu_graph(num_channels=4)
    rng = np.random.default_rng(42)
    x = rng.uniform(-3.0, 3.0, size=(1, 4, 8, 12)).astype(np.float32)
    ref = _run_ort(model, x)

    decompose_phase_prelu_inplace(model, target_node_names=("/phase_conv/phase_conv.2/PRelu",))
    onnx.checker.check_model(model)
    got = _run_ort(model, x)

    diff = float(np.abs(ref - got).max())
    assert diff <= 1e-6, f"max-abs delta {diff} exceeds 1e-6"


def test_numerical_equivalence_edge_cases():
    """Decomposition is exact on the PReLU's sign discontinuity (x == 0)."""
    model = _build_minimal_prelu_graph(num_channels=4)
    # Mix: deliberate zeros, tiny negatives, tiny positives, large values.
    edge_vals = np.array(
        [-1e-9, 0.0, 1e-9, -1.0, 1.0, -100.0, 100.0, -1e-30, 1e-30, 0.5, -0.5, 0.0],
        dtype=np.float32,
    )  # shape (12,) — matches the model's last-dim
    x = np.broadcast_to(edge_vals.reshape(1, 1, 1, 12), (1, 4, 8, 12)).astype(np.float32).copy()
    ref = _run_ort(model, x)

    decompose_phase_prelu_inplace(model, target_node_names=("/phase_conv/phase_conv.2/PRelu",))
    got = _run_ort(model, x)

    diff = float(np.abs(ref - got).max())
    assert diff == 0.0 or diff <= 1e-7, f"edge-case max-abs delta {diff} too large"


def test_graph_rewrite_drops_prelu_and_inserts_quartet():
    """After surgery: zero PRelu nodes; four new Max/Min/Mul/Add nodes per target."""
    model = _build_minimal_prelu_graph(num_channels=4)
    target = "/phase_conv/phase_conv.2/PRelu"
    rewritten = decompose_phase_prelu_inplace(model, target_node_names=(target,))
    assert rewritten == [target]

    op_types = [n.op_type for n in model.graph.node]
    assert "PRelu" not in op_types
    assert op_types.count("Max") == 1
    assert op_types.count("Min") == 1
    assert op_types.count("Mul") == 1
    assert op_types.count("Add") == 1
    # Output edge name preserved.
    add_nodes = [n for n in model.graph.node if n.op_type == "Add"]
    assert add_nodes[0].output[0] == "y"


def test_graph_rewrite_reuses_slope_initializer():
    """Slope initializer is not duplicated; the Mul node references the same name."""
    model = _build_minimal_prelu_graph(num_channels=4)
    decompose_phase_prelu_inplace(model, target_node_names=("/phase_conv/phase_conv.2/PRelu",))

    initializer_names = [ini.name for ini in model.graph.initializer]
    # Original slope plus exactly one new scalar-zero initializer.
    assert "slope" in initializer_names
    assert "prelu_decomp_zero" in initializer_names
    # No second "slope" copy.
    assert initializer_names.count("slope") == 1

    mul_node = next(n for n in model.graph.node if n.op_type == "Mul")
    assert "slope" in mul_node.input, f"Mul.input={list(mul_node.input)}"


def test_zero_initializer_shared_across_multiple_targets():
    """When decomposing two PReLUs, exactly one shared scalar-zero initializer."""
    # Build a graph with two PReLU nodes feeding two outputs.
    rng = np.random.default_rng(1)
    slope_a = numpy_helper.from_array(
        rng.uniform(0.1, 0.3, size=(1, 4, 1, 1)).astype(np.float32), name="slope_a"
    )
    slope_b = numpy_helper.from_array(
        rng.uniform(0.1, 0.3, size=(1, 4, 1, 1)).astype(np.float32), name="slope_b"
    )
    x_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 4])
    y_a = helper.make_tensor_value_info("ya", TensorProto.FLOAT, [1, 4, 4, 4])
    y_b = helper.make_tensor_value_info("yb", TensorProto.FLOAT, [1, 4, 4, 4])
    pa = helper.make_node("PRelu", ["x", "slope_a"], ["ya"], name="/branch_a/PRelu")
    pb = helper.make_node("PRelu", ["x", "slope_b"], ["yb"], name="/branch_b/PRelu")
    graph = helper.make_graph(
        nodes=[pa, pb],
        name="two_prelu",
        inputs=[x_in],
        outputs=[y_a, y_b],
        initializer=[slope_a, slope_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)

    decompose_phase_prelu_inplace(
        model, target_node_names=("/branch_a/PRelu", "/branch_b/PRelu")
    )
    onnx.checker.check_model(model)

    zero_inis = [ini for ini in model.graph.initializer if ini.name == "prelu_decomp_zero"]
    assert len(zero_inis) == 1, f"expected exactly 1 shared zero initializer, found {len(zero_inis)}"


def test_empty_target_list_raises():
    model = _build_minimal_prelu_graph()
    with pytest.raises(ValueError, match="target_node_names is empty"):
        decompose_phase_prelu_inplace(model, target_node_names=())


def test_missing_target_raises():
    model = _build_minimal_prelu_graph()
    with pytest.raises(ValueError, match="not found"):
        decompose_phase_prelu_inplace(model, target_node_names=("/does/not/exist",))


def test_wrong_op_type_raises():
    """A non-PRelu node named the same way should be rejected with a clear error."""
    model = _build_minimal_prelu_graph()
    # Force the existing PRelu node to a different op_type to simulate a name collision.
    model.graph.node[0].op_type = "Relu"
    with pytest.raises(ValueError, match="op_type='Relu'"):
        decompose_phase_prelu_inplace(model, target_node_names=("/phase_conv/phase_conv.2/PRelu",))
