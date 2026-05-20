"""Cycle 18b-16 Candidate (c) SmoothQuant — algebraic identity insert at the
two mapping TSBlock-3 freq_stage GPKFFN proj_first Conv nodes.

Plan C category (c): SmoothQuant-style channel-wise scaling (Xiao et al. 2023,
ICML). Inserts a per-input-channel ``Mul(1/s)`` node between LayerNorm output
and Conv 1x1 input, then folds ``s`` into the Conv weight (``W_new = W * s``,
broadcast along input-channel dim). This is FP32 BYTE-IDENTICAL to the
production graph (algebraic identity for Conv 1x1):

  Y[o, t] = Σ_j (X[j,t] / s[j]) · (W[o, j, 0] · s[j])
         = Σ_j X[j,t] · W[o, j, 0]   ✓

``s`` is sourced from
``cycle18b16_proj_first_activation_profile_idx0.json`` (Phase 1) with the
canonical SmoothQuant α = 0.5 formula:

  s[j] = max(|X[j]|)^α / max(|W[j]|)^α

Targets ONLY:

  * ``/mapping_core/proj_first/proj_first.0_14/Conv`` (GPKFFN_0, norm_29)
  * ``/mapping_core/proj_first/proj_first.0_15/Conv`` (GPKFFN_1, norm_31)

Production D3 ONNX (md5 ``980aa363…``) is NOT modified — output is written to
a separate file ``bafnetplus_50ms_fp32_complex_cycle18b16_smoothq.onnx`` with
sidecar schema ``s20-bafnetplus-functional-fp32-complex-cycle18b16-smoothq-v1``.

Usage::

    python scripts/export_bafnetplus_cycle18b16_candidate_c_smoothquant.py \\
        --src results/onnx/bafnetplus_50ms_fp32_complex.onnx \\
        --dst results/onnx/bafnetplus_50ms_fp32_complex_cycle18b16_smoothq.onnx \\
        --profile-json ../Android_projects/results/parity/cycle18b16_proj_first_activation_profile_idx0.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


TARGETS = {
    "g0": {
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_14/Conv",
        "weight_name": "mapping_core.sequence_block.3.freq_stage.0.1.proj_first.0.weight",
        "s_key": "g0_inv_s_64ch",  # inverse-s vector from profile (= 1/s)
    },
    "g1": {
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_15/Conv",
        "weight_name": "mapping_core.sequence_block.3.freq_stage.1.1.proj_first.0.weight",
        "s_key": "g1_inv_s_64ch",
    },
}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_smoothquant_to_conv(
    model: onnx.ModelProto,
    conv_node_name: str,
    weight_name: str,
    inv_s: np.ndarray,
) -> None:
    """Insert per-channel ``Mul(inv_s)`` before Conv input and fold ``s = 1/inv_s`` into Conv weight."""
    # Find target Conv node.
    target_node = None
    for n in model.graph.node:
        if n.name == conv_node_name:
            target_node = n
            break
    if target_node is None:
        raise ValueError(f"Conv node {conv_node_name} not found")
    orig_input = target_node.input[0]  # LayerNorm output
    scaled_input = orig_input + "_cycle18b16_smq"

    # Find target weight initializer.
    weight_init = None
    weight_idx = None
    for i, t in enumerate(model.graph.initializer):
        if t.name == weight_name:
            weight_init = t
            weight_idx = i
            break
    if weight_init is None:
        raise ValueError(f"Weight initializer {weight_name} not found")
    W = numpy_helper.to_array(weight_init).astype(np.float32)  # [256, 64, 1]
    assert W.ndim == 3, f"unexpected weight shape {W.shape}"
    assert W.shape[1] == inv_s.shape[0], f"inv_s shape {inv_s.shape} mismatches W input dim {W.shape}"

    # Fold s = 1/inv_s into W: W_new[:, j, :] = W[:, j, :] * s[j]
    s = 1.0 / np.maximum(inv_s, 1e-12)
    W_new = W * s.reshape(1, -1, 1)  # broadcast [1, 64, 1] over [256, 64, 1]
    new_weight_init = numpy_helper.from_array(W_new.astype(np.float32), name=weight_name)
    model.graph.initializer.remove(weight_init)
    model.graph.initializer.insert(weight_idx, new_weight_init)

    # Create inv_s initializer broadcast tensor: shape [1, 64, 1] for [B, C, T] Conv input.
    inv_s_tensor = inv_s.astype(np.float32).reshape(1, -1, 1)
    inv_s_init = numpy_helper.from_array(
        inv_s_tensor,
        name=f"{conv_node_name}_inv_s_cycle18b16",
    )
    model.graph.initializer.append(inv_s_init)

    # Create Mul node: scaled_input = orig_input * inv_s
    mul_node = helper.make_node(
        "Mul",
        inputs=[orig_input, inv_s_init.name],
        outputs=[scaled_input],
        name=f"{conv_node_name}_smoothquant_Mul_inv_s",
    )
    insert_idx = list(model.graph.node).index(target_node)
    model.graph.node.insert(insert_idx, mul_node)

    # Rewire Conv input to scaled_input.
    target_node.input[0] = scaled_input


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx"))
    parser.add_argument("--dst", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle18b16_smoothq.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
    parser.add_argument("--profile-json", type=Path, required=True)
    args = parser.parse_args()

    src_md5 = md5sum(args.src)
    expected = "980aa36308eba63232bb253c36249f38"
    assert src_md5 == expected, f"production D3 ONNX md5 drift: {src_md5}"
    print(f"[cycle18b16 smq] src md5: {src_md5} (production D3 OK)")

    profile = json.loads(args.profile_json.read_text())
    sv = profile["smoothquant_s_full_vectors"]
    inv_s_g0 = np.array(sv["g0_inv_s_64ch"], dtype=np.float32)
    inv_s_g1 = np.array(sv["g1_inv_s_64ch"], dtype=np.float32)
    print(f"[cycle18b16 smq] g0 inv_s range: [{inv_s_g0.min():.4f}, {inv_s_g0.max():.4f}] p50={np.percentile(inv_s_g0, 50):.4f}")
    print(f"[cycle18b16 smq] g1 inv_s range: [{inv_s_g1.min():.4f}, {inv_s_g1.max():.4f}] p50={np.percentile(inv_s_g1, 50):.4f}")

    model = onnx.load(str(args.src))
    apply_smoothquant_to_conv(model, TARGETS["g0"]["conv_node_name"], TARGETS["g0"]["weight_name"], inv_s_g0)
    apply_smoothquant_to_conv(model, TARGETS["g1"]["conv_node_name"], TARGETS["g1"]["weight_name"], inv_s_g1)

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    dst_md5 = md5sum(args.dst)
    print(f"[cycle18b16 smq] wrote {args.dst.name} md5={dst_md5}")

    src_sidecar = json.loads(args.sidecar_src.read_text())
    side = copy.deepcopy(src_sidecar)
    side["schema_version"] = "s20-bafnetplus-functional-fp32-complex-cycle18b16-smoothq-v1"
    side["produced_by"] = "BAFNetPlus.scripts.export_bafnetplus_cycle18b16_candidate_c_smoothquant"
    side["cycle18b16"] = {
        "candidate": "c_smoothquant",
        "smoothquant_alpha": 0.5,
        "targets": [
            {"conv_node_name": v["conv_node_name"], "weight_name": v["weight_name"]} for v in TARGETS.values()
        ],
        "inv_s_g0_summary": {"min": float(inv_s_g0.min()), "max": float(inv_s_g0.max()),
                              "p50": float(np.percentile(inv_s_g0, 50))},
        "inv_s_g1_summary": {"min": float(inv_s_g1.min()), "max": float(inv_s_g1.max()),
                              "p50": float(np.percentile(inv_s_g1, 50))},
        "rationale": "Per-input-channel SmoothQuant algebraic identity Y = (X/s)·(W·s) = X·W; "
                     "Mul(inv_s) inserted before Conv input, s folded into Conv weight. "
                     "FP32 BYTE-IDENTICAL to production; reduces per-channel input-side magnitudes "
                     "feeding HTP-FP16 Conv accumulate.",
        "source_production_md5": src_md5,
        "phase1_profile_json": str(args.profile_json),
    }
    side["onnx_file_md5"] = dst_md5
    dst_sidecar = args.dst.with_suffix(".onnx.json")
    dst_sidecar.write_text(json.dumps(side, indent=2))
    print(f"[cycle18b16 smq] wrote {dst_sidecar.name} md5={md5sum(dst_sidecar)}")

    assert md5sum(args.src) == expected, "production source mutated!"
    print("[cycle18b16 smq] G3 OK — algebraic identity insert + weight fold, candidate (c) SmoothQuant emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
