"""Cycle 18b-16 Candidate (b+d) LayerNorm rewrite — targeted ε retarget for
norm_29 (GPKFFN_0) and norm_31 (GPKFFN_1) only.

Plan C categories (b) + (d):

* (d) **ε retarget**: per cycle 18b-11 lesson (`fp16_safe_epsilon` 1e-3/1e-6
  for ALL norms caused host PESQ regression via semantic shift). This cycle
  applies the ε bump ONLY to the two TSBlock 3 freq_stage GPKFFN proj_first
  LayerNorms (norm_29 + norm_31). Production-default ε = 1e-6 is replaced
  with ε_target (default 1e-2). The original shared eps initializer
  (``/mapping_core/norm/Constant_2_output_0`` = 1e-6) is unchanged so all
  other LayerNorms keep production behaviour.

* (b) **Decomposition**: optional ``Pow(2)`` → ``Mul(x, x)`` rewrite for the
  Sub→Pow→ReduceMean path. ``Pow(2)`` is implemented as ``Exp(2·Log(x))``
  on some FP16 hardware and can lose precision; ``Mul(x, x)`` is a direct
  IEEE FMA that is more stable on HTP. Off by default (``--rewrite-pow=false``)
  since (d) alone is the primary intervention.

The intervention is **NOT** an algebraic identity — it changes the LayerNorm
output by ``ε`` worth of bias. Per cycle 18b-11: at ``ε=1e-3`` the host est_mag
shift was 9.49e-2 (over 5e-2 envelope). At ``ε=1e-2`` the bias is larger;
device-only HTP behaviour should still be sane since FP16 captures ε ≈ 1e-2
exactly. Phase 5 device PESQ measurement is required to validate idx 0
recovery vs idx 3 regression risk.

Production D3 ONNX (md5 ``980aa363…``) is NOT modified — output is written
to ``bafnetplus_50ms_fp32_complex_cycle18b16_lnorm.onnx`` with sidecar schema
``s20-bafnetplus-functional-fp32-complex-cycle18b16-lnorm-v1``.

Usage::

    python scripts/export_bafnetplus_cycle18b16_candidate_bd_lnorm.py \\
        --src results/onnx/bafnetplus_50ms_fp32_complex.onnx \\
        --dst results/onnx/bafnetplus_50ms_fp32_complex_cycle18b16_lnorm.onnx \\
        --epsilon-target 1e-2
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


TARGET_NORM_ADD_NODES = (
    "/mapping_core/norm_29/Add",
    "/mapping_core/norm_31/Add",
)
TARGET_NORM_POW_NODES = (
    "/mapping_core/norm_29/Pow",
    "/mapping_core/norm_31/Pow",
)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def retarget_norm_epsilon(model: onnx.ModelProto, add_node_name: str, eps_target: float) -> None:
    """Replace the shared eps initializer with a dedicated one for ``add_node_name``."""
    add_node = None
    for n in model.graph.node:
        if n.name == add_node_name:
            add_node = n
            break
    if add_node is None:
        raise ValueError(f"Add node {add_node_name} not found")
    assert len(add_node.input) == 2, f"unexpected Add inputs {list(add_node.input)}"

    # The second input should be the shared eps initializer (e.g. /mapping_core/norm/Constant_2_output_0).
    eps_orig_name = add_node.input[1]
    new_eps_name = add_node_name.replace("/Add", "/Add_eps_cycle18b16")
    eps_init = numpy_helper.from_array(
        np.array(eps_target, dtype=np.float32),
        name=new_eps_name,
    )
    model.graph.initializer.append(eps_init)
    add_node.input[1] = new_eps_name


def rewrite_pow2_as_mul(model: onnx.ModelProto, pow_node_name: str) -> None:
    """Replace ``y = Pow(x, 2)`` with ``y = Mul(x, x)`` (more FP16-stable)."""
    pow_node = None
    pow_idx = None
    for i, n in enumerate(model.graph.node):
        if n.name == pow_node_name:
            pow_node = n
            pow_idx = i
            break
    if pow_node is None:
        raise ValueError(f"Pow node {pow_node_name} not found")
    assert pow_node.op_type == "Pow"
    x_in = pow_node.input[0]
    pow_exp = pow_node.input[1]
    # Verify the exponent is 2.0
    for t in model.graph.initializer:
        if t.name == pow_exp:
            val = numpy_helper.to_array(t)
            assert np.allclose(val, 2.0), f"Pow exponent != 2: {val}"
            break
    pow_out = pow_node.output[0]
    mul_node = helper.make_node(
        "Mul",
        inputs=[x_in, x_in],
        outputs=[pow_out],
        name=pow_node_name.replace("/Pow", "/Mul_x2_cycle18b16"),
    )
    model.graph.node.remove(pow_node)
    model.graph.node.insert(pow_idx, mul_node)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx"))
    parser.add_argument("--dst", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle18b16_lnorm.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
    parser.add_argument("--epsilon-target", type=float, default=1e-2)
    parser.add_argument("--rewrite-pow", action="store_true", default=False,
                        help="Also replace Pow(2) with Mul(x, x) for the two target norm Pow nodes.")
    args = parser.parse_args()

    src_md5 = md5sum(args.src)
    expected = "980aa36308eba63232bb253c36249f38"
    assert src_md5 == expected, f"production D3 ONNX md5 drift: {src_md5}"
    print(f"[cycle18b16 lnorm] src md5: {src_md5} (production D3 OK)")
    print(f"[cycle18b16 lnorm] target Add nodes: {TARGET_NORM_ADD_NODES}")
    print(f"[cycle18b16 lnorm] epsilon target = {args.epsilon_target}")
    print(f"[cycle18b16 lnorm] rewrite Pow(2) -> Mul(x, x): {args.rewrite_pow}")

    model = onnx.load(str(args.src))
    for n in TARGET_NORM_ADD_NODES:
        retarget_norm_epsilon(model, n, args.epsilon_target)
    if args.rewrite_pow:
        for n in TARGET_NORM_POW_NODES:
            rewrite_pow2_as_mul(model, n)

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    dst_md5 = md5sum(args.dst)
    print(f"[cycle18b16 lnorm] wrote {args.dst.name} md5={dst_md5}")

    src_sidecar = json.loads(args.sidecar_src.read_text())
    side = copy.deepcopy(src_sidecar)
    side["schema_version"] = "s20-bafnetplus-functional-fp32-complex-cycle18b16-lnorm-v1"
    side["produced_by"] = "BAFNetPlus.scripts.export_bafnetplus_cycle18b16_candidate_bd_lnorm"
    side["cycle18b16"] = {
        "candidate": "bd_lnorm",
        "epsilon_target": args.epsilon_target,
        "rewrite_pow_to_mul": args.rewrite_pow,
        "target_norm_add_nodes": list(TARGET_NORM_ADD_NODES),
        "target_norm_pow_nodes": list(TARGET_NORM_POW_NODES) if args.rewrite_pow else [],
        "rationale": "Targeted LayerNorm epsilon retarget (1e-6 -> 1e-2) for ONLY norm_29 and "
                     "norm_31 (mapping TSBlock 3 freq_stage GPKFFN proj_first norms). Other "
                     "LayerNorm nodes keep production epsilon (1e-6) so cycle 18b-11's global "
                     "host-PESQ regression is avoided. Optional Pow(2)->Mul(x,x) for FP16 "
                     "stability in Sub->squared->ReduceMean->Add path.",
        "source_production_md5": src_md5,
    }
    side["onnx_file_md5"] = dst_md5
    dst_sidecar = args.dst.with_suffix(".onnx.json")
    dst_sidecar.write_text(json.dumps(side, indent=2))
    print(f"[cycle18b16 lnorm] wrote {dst_sidecar.name} md5={md5sum(dst_sidecar)}")

    assert md5sum(args.src) == expected, "production source mutated!"
    print("[cycle18b16 lnorm] G4 OK — candidate (b+d) LayerNorm rewrite emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
