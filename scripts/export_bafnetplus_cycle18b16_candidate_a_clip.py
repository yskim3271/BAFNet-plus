"""Cycle 18b-16 Candidate (a) Clip — insert Clip(-T, +T) after the two mapping
TSBlock-3 freq_stage GPKFFN proj_first Conv outputs.

Plan C category (a): hard activation clipping. Mathematically NOT identity
(values outside [-T, +T] are clamped) but for T chosen above the host-FP32
observed Conv output max-abs (Phase 1 profile), the clip is a no-op in FP32
(host-parity BYTE-IDENTICAL) and a safety guard only at HTP-FP16 runtime where
quantization noise can push a value above the host range.

Targets ONLY the 2 nodes identified by cycle 18b-15 Pre_cpu (mapping branch
TSBlock 3 freq_stage GPKFFNs):

  * ``/mapping_core/proj_first/proj_first.0_14/Conv`` (GPKFFN_0, norm_29)
  * ``/mapping_core/proj_first/proj_first.0_15/Conv`` (GPKFFN_1, norm_31)

Each Conv output is rewired via a new ``Clip`` node:

  Conv_output_0  →  Clip_node  →  Conv_output_0_clipped
  (and every downstream consumer is re-pointed to the clipped tensor.)

The production D3 ONNX (md5 ``980aa363…``) is NOT modified — output is written
to a separate file ``bafnetplus_50ms_fp32_complex_cycle18b16_clip.onnx`` with
sidecar schema ``s20-bafnetplus-functional-fp32-complex-cycle18b16-clip-v1``.

Usage::

    python scripts/export_bafnetplus_cycle18b16_candidate_a_clip.py \\
        --src results/onnx/bafnetplus_50ms_fp32_complex.onnx \\
        --dst results/onnx/bafnetplus_50ms_fp32_complex_cycle18b16_clip.onnx \\
        --clip-threshold 256.0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


TARGET_CONV_NODE_NAMES = (
    "/mapping_core/proj_first/proj_first.0_14/Conv",
    "/mapping_core/proj_first/proj_first.0_15/Conv",
)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def insert_clip_after_conv(model: onnx.ModelProto, conv_node_name: str, threshold: float) -> None:
    """Insert a Clip node consuming ``conv_node_name``'s output; rewire downstream."""
    target_node = None
    for n in model.graph.node:
        if n.name == conv_node_name:
            target_node = n
            break
    if target_node is None:
        raise ValueError(f"Conv node {conv_node_name} not found")
    orig_out = target_node.output[0]
    clipped_out = orig_out + "_clipped"

    # Create min/max initializer scalars (FP32).
    min_init = numpy_helper.from_array(
        np.array(-threshold, dtype=np.float32),
        name=f"{orig_out}_clip_min",
    )
    max_init = numpy_helper.from_array(
        np.array(threshold, dtype=np.float32),
        name=f"{orig_out}_clip_max",
    )
    model.graph.initializer.extend([min_init, max_init])

    clip_node = helper.make_node(
        "Clip",
        inputs=[orig_out, min_init.name, max_init.name],
        outputs=[clipped_out],
        name=f"{conv_node_name}_Clip_cycle18b16",
    )
    # Insert clip node immediately after the Conv in graph.node order.
    insert_idx = list(model.graph.node).index(target_node) + 1
    model.graph.node.insert(insert_idx, clip_node)

    # Rewire every downstream consumer that reads orig_out (except the new Clip itself).
    for n in model.graph.node:
        if n.name == clip_node.name:
            continue
        for i, inp in enumerate(n.input):
            if inp == orig_out:
                n.input[i] = clipped_out

    # If orig_out is in graph.output, redirect to clipped_out too (defensive — unlikely here).
    for o in model.graph.output:
        if o.name == orig_out:
            o.name = clipped_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx"))
    parser.add_argument("--dst", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle18b16_clip.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
    parser.add_argument("--clip-threshold", type=float, default=256.0)
    args = parser.parse_args()

    src_md5 = md5sum(args.src)
    expected_src_md5 = "980aa36308eba63232bb253c36249f38"
    assert src_md5 == expected_src_md5, f"production D3 ONNX md5 drift: {src_md5}"
    print(f"[cycle18b16 clip] src md5: {src_md5} (production D3 OK)")
    print(f"[cycle18b16 clip] target Conv nodes: {TARGET_CONV_NODE_NAMES}")
    print(f"[cycle18b16 clip] clip threshold T = {args.clip_threshold}")

    model = onnx.load(str(args.src))
    for name in TARGET_CONV_NODE_NAMES:
        insert_clip_after_conv(model, name, args.clip_threshold)

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    dst_md5 = md5sum(args.dst)
    print(f"[cycle18b16 clip] wrote {args.dst.name} md5={dst_md5}")

    # Sidecar JSON: derived from production sidecar with schema override + clip metadata.
    src_sidecar = json.loads(args.sidecar_src.read_text())
    side = copy.deepcopy(src_sidecar)
    side["schema_version"] = "s20-bafnetplus-functional-fp32-complex-cycle18b16-clip-v1"
    side["produced_by"] = "BAFNetPlus.scripts.export_bafnetplus_cycle18b16_candidate_a_clip"
    side["cycle18b16"] = {
        "candidate": "a_clip",
        "target_conv_node_names": list(TARGET_CONV_NODE_NAMES),
        "clip_threshold": args.clip_threshold,
        "rationale": "Hard activation clip after the two mapping TSBlock-3 freq_stage GPKFFN "
                     "proj_first Conv outputs (cycle 18b-15 Pre_cpu target). T chosen above "
                     "host-FP32 observed Conv output max-abs so the clip is FP32 no-op; only "
                     "fires at HTP-FP16 quantization-noise excursions.",
        "source_production_md5": src_md5,
    }
    side["onnx_file_md5"] = dst_md5
    dst_sidecar = args.dst.with_suffix(".onnx.json")
    dst_sidecar.write_text(json.dumps(side, indent=2))
    print(f"[cycle18b16 clip] wrote {dst_sidecar.name} md5={md5sum(dst_sidecar)}")

    # Sanity: production source NOT mutated.
    assert md5sum(args.src) == expected_src_md5, "production source mutated!"
    print("[cycle18b16 clip] G2 OK — production D3 ONNX byte-identical, candidate (a) Clip emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
