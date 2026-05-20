"""Cycle 18b-17 Candidate D1 — LayerNorm affine (gamma, beta) absorption into proj_first Conv.

Plan C extension category: pure algebraic identity at FP32, removes intermediate
affine activation from the LN -> Conv 1x1 sequence so the HTP-FP16 cascade has
one less FP16 tensor in its chain.

Identity:

  Original
    n[c]      = (x[c] - mean) / sqrt(var + eps)
    z[c]      = gamma[c] * n[c] + beta[c]
    y[o]      = sum_c W[o, c, 0] * z[c] + b[o]
                = sum_c W[o, c, 0] * (gamma[c] * n[c] + beta[c]) + b[o]

  Re-written
    W'[o, c, 0] = W[o, c, 0] * gamma[c]    (broadcast multiply)
    b'[o]       = b[o] + sum_c W[o, c, 0] * beta[c]
    y[o]        = sum_c W'[o, c, 0] * n[c] + b'[o]

In FP32 real arithmetic the two forms are byte-identical; the new graph
removes the LN gamma Mul + beta Add nodes from the active path (they become
dead nodes — kept in the file but disconnected from any downstream consumer).

Targets ONLY the 2 mapping TSBlock-3 freq_stage proj_first Conv nodes
identified by cycle 18b-15 Pre_cpu:

  * ``/mapping_core/proj_first/proj_first.0_14/Conv``  (GPKFFN_0, norm_29)
  * ``/mapping_core/proj_first/proj_first.0_15/Conv``  (GPKFFN_1, norm_31)

Production D3 ONNX (md5 ``980aa363...``) is NOT modified; output is written to
``bafnetplus_50ms_fp32_complex_cycle18b17_d1_lnabsorb.onnx`` with sidecar
schema ``s20-bafnetplus-functional-fp32-complex-cycle18b17-d1-lnabsorb-v1``.

Usage::

    python BAFNetPlus/scripts/export_bafnetplus_cycle18b17_candidate_d1_lnabsorb.py \\
        --src BAFNetPlus/results/onnx/bafnetplus_50ms_fp32_complex.onnx \\
        --dst BAFNetPlus/results/onnx/bafnetplus_50ms_fp32_complex_cycle18b17_d1_lnabsorb.onnx
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


EXPECTED_SRC_MD5 = "980aa36308eba63232bb253c36249f38"

TARGETS = (
    {
        "key": "g0",
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_14/Conv",
        "ln_div_output": "/mapping_core/norm_29/Div_output_0",
        "gamma_init": "mapping_core.sequence_block.3.freq_stage.0.1.norm.weight",
        "beta_init":  "mapping_core.sequence_block.3.freq_stage.0.1.norm.bias",
        "weight_init": "mapping_core.sequence_block.3.freq_stage.0.1.proj_first.0.weight",
        "bias_init":   "mapping_core.sequence_block.3.freq_stage.0.1.proj_first.0.bias",
    },
    {
        "key": "g1",
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_15/Conv",
        "ln_div_output": "/mapping_core/norm_31/Div_output_0",
        "gamma_init": "mapping_core.sequence_block.3.freq_stage.1.1.norm.weight",
        "beta_init":  "mapping_core.sequence_block.3.freq_stage.1.1.norm.bias",
        "weight_init": "mapping_core.sequence_block.3.freq_stage.1.1.proj_first.0.weight",
        "bias_init":   "mapping_core.sequence_block.3.freq_stage.1.1.proj_first.0.bias",
    },
)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def absorb_one(model: onnx.ModelProto, target: dict) -> dict:
    """Apply LN -> Conv affine absorption for a single target group."""
    inits = {t.name: (i, t) for i, t in enumerate(model.graph.initializer)}
    if target["gamma_init"] not in inits:
        raise ValueError(f"gamma init {target['gamma_init']} not found")
    if target["beta_init"] not in inits:
        raise ValueError(f"beta init {target['beta_init']} not found")
    if target["weight_init"] not in inits:
        raise ValueError(f"weight init {target['weight_init']} not found")
    if target["bias_init"] not in inits:
        raise ValueError(f"bias init {target['bias_init']} not found")

    gamma = numpy_helper.to_array(inits[target["gamma_init"]][1]).astype(np.float32)
    beta = numpy_helper.to_array(inits[target["beta_init"]][1]).astype(np.float32)
    W = numpy_helper.to_array(inits[target["weight_init"]][1]).astype(np.float32)
    b = numpy_helper.to_array(inits[target["bias_init"]][1]).astype(np.float32)

    assert gamma.shape == (W.shape[1],), f"gamma {gamma.shape} vs W {W.shape}"
    assert beta.shape == (W.shape[1],), f"beta {beta.shape} vs W {W.shape}"
    assert W.ndim == 3 and W.shape[2] == 1, f"unexpected W shape {W.shape}"
    assert b.shape == (W.shape[0],), f"b {b.shape} vs W {W.shape}"

    # W'[o, c, 0] = W[o, c, 0] * gamma[c]
    W_new = W * gamma.reshape(1, -1, 1)
    # b'[o] = b[o] + sum_c W[o, c, 0] * beta[c]
    b_new = b + (W[:, :, 0] @ beta)

    # Replace the proj_first weight & bias initializers (in-place: same name, new array).
    w_idx, _ = inits[target["weight_init"]]
    new_w_init = numpy_helper.from_array(W_new.astype(np.float32), name=target["weight_init"])
    model.graph.initializer.remove(inits[target["weight_init"]][1])
    model.graph.initializer.insert(w_idx, new_w_init)
    b_idx, _ = inits[target["bias_init"]]
    new_b_init = numpy_helper.from_array(b_new.astype(np.float32), name=target["bias_init"])
    model.graph.initializer.remove(inits[target["bias_init"]][1])
    model.graph.initializer.insert(b_idx, new_b_init)

    # Rewire Conv input[0] from LN/Add_1_output_0 to LN/Div_output_0 (= normalized n).
    target_node = None
    for n in model.graph.node:
        if n.name == target["conv_node_name"]:
            target_node = n
            break
    if target_node is None:
        raise ValueError(f"Conv node {target['conv_node_name']} not found")
    target_node.input[0] = target["ln_div_output"]

    return {
        "key": target["key"],
        "conv_node_name": target["conv_node_name"],
        "new_conv_input": target["ln_div_output"],
        "W_orig_max_abs": float(np.abs(W).max()),
        "W_new_max_abs": float(np.abs(W_new).max()),
        "b_orig_max_abs": float(np.abs(b).max()),
        "b_new_max_abs": float(np.abs(b_new).max()),
        "gamma_max_abs": float(np.abs(gamma).max()),
        "beta_max_abs": float(np.abs(beta).max()),
        "W_scaling_factor_top5_ch": {int(c): float(gamma[c]) for c in np.argsort(-np.abs(gamma))[:5]},
        "b_shift_max_abs": float(np.abs(b_new - b).max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx"))
    parser.add_argument("--dst", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle18b17_d1_lnabsorb.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
    args = parser.parse_args()

    src_md5 = md5sum(args.src)
    assert src_md5 == EXPECTED_SRC_MD5, f"production D3 md5 drift: {src_md5}"
    print(f"[d1] src md5: {src_md5} (production D3 OK)")

    model = onnx.load(str(args.src))
    per_target = [absorb_one(model, t) for t in TARGETS]

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    dst_md5 = md5sum(args.dst)
    print(f"[d1] wrote {args.dst.name} md5={dst_md5}")
    for pt in per_target:
        print(f"[d1] {pt['key']}: W max_abs {pt['W_orig_max_abs']:.3f} -> {pt['W_new_max_abs']:.3f}, "
              f"b max_abs {pt['b_orig_max_abs']:.3f} -> {pt['b_new_max_abs']:.3f}, "
              f"gamma_max={pt['gamma_max_abs']:.3f}, beta_max={pt['beta_max_abs']:.3f}, "
              f"b_shift_max={pt['b_shift_max_abs']:.3f}")

    src_sidecar = json.loads(args.sidecar_src.read_text())
    side = copy.deepcopy(src_sidecar)
    side["schema_version"] = "s20-bafnetplus-functional-fp32-complex-cycle18b17-d1-lnabsorb-v1"
    side["produced_by"] = "BAFNetPlus.scripts.export_bafnetplus_cycle18b17_candidate_d1_lnabsorb"
    side["cycle18b17"] = {
        "candidate": "d1_lnabsorb",
        "rationale": (
            "Absorb LayerNorm affine (gamma, beta) into proj_first Conv weight and bias. "
            "Algebraic identity in real arithmetic: W'[o,c] = W[o,c] * gamma[c]; "
            "b'[o] = b[o] + sum_c W[o,c] * beta[c]. Conv input rewired from LN Add_1_output "
            "(post-affine) to LN Div_output (just normalized n). Removes the intermediate "
            "affine FP16 tensor from the LN -> Conv cascade entirely (the gamma Mul and beta "
            "Add nodes become dead nodes in the graph). This intervention modifies the Conv "
            "weight at export time -- ORT cannot fold it back to the original since the "
            "weight values themselves are different."
        ),
        "targets": per_target,
        "source_production_md5": src_md5,
    }
    side["onnx_file_md5"] = dst_md5
    dst_sidecar = args.dst.with_suffix(".onnx.json")
    dst_sidecar.write_text(json.dumps(side, indent=2))
    print(f"[d1] wrote {dst_sidecar.name} md5={md5sum(dst_sidecar)}")

    assert md5sum(args.src) == EXPECTED_SRC_MD5, "production source mutated!"
    print("[d1] G1 OK -- production D3 byte-identical, candidate D1 lnabsorb emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
