"""Cycle 18b-18 step 1 (A1c) — per-tensor single-node Clip variant of cycle 18b-17 A1.

cycle 18b-17 A1 (per-channel Max + Min, 2 nodes per target) achieved idx 0 PESQ
recovery (4.5542) but at 66.24 ms latency = 1.32x over 50 ms budget. The
latency overhead is dominated by QNN HTP partition boundary fragmentation
(adding any node breaks the production fused micro-kernel — confirmed by
cycle 18b-16/17 cluster at ~66-70 ms regardless of node count).

A1c hypothesis: ONNX `Clip` is a *single* op that QNN HTP may recognise as a
post-Conv activation function and fuse back into the Conv kernel template,
recovering the production ~30 ms latency.

Constraint: ONNX Clip min/max are scalars (rank-0 or 1-element tensors).
Per-channel calibration is not directly representable. As a compromise A1c
uses per-target per-tensor T = 1.10 x max(per_channel_calib[c]):

  * g0:  T = 1.10 x 12.638 = 13.902
  * g1:  T = 1.10 x 13.072 = 14.379

This is 18x tighter than cycle 18b-16 clip (T=256) and proportional to the
per-channel worst case in cycle 18b-17 A1. The expectation:
  * PESQ recovery: likely (T_g is per-channel max with 10% margin — same as
    cycle 18b-17 A1 except channels with lower max get over-protected).
  * Latency: target ~50 ms if QNN HTP fuses Clip into Conv post-activation.

If PESQ regresses below 4.508 → cycle 18b-18 step 2 = per-channel-scaled Clip
trick (`Mul(diag) -> Clip(per-tensor) -> Mul(1/diag)` to recover per-channel
effect with single Clip).

Targets ONLY the 2 mapping TSBlock-3 freq_stage proj_first Conv nodes:

  * ``/mapping_core/proj_first/proj_first.0_14/Conv`` (g0)
  * ``/mapping_core/proj_first/proj_first.0_15/Conv`` (g1)

Production D3 ONNX (md5 ``980aa363...``) NOT modified; output is
``bafnetplus_50ms_fp32_complex_cycle18b18_a1c_clip_pertensor.onnx`` with
sidecar schema ``s20-bafnetplus-functional-fp32-complex-cycle18b18-a1c-clip-pertensor-v1``.

Usage::

    python BAFNetPlus/scripts/export_bafnetplus_cycle18b18_candidate_a1c_clip_pertensor.py \\
        --calib-json Android_projects/results/parity/cycle18b17_proj_first_per_channel_calib_idx0.json
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


EXPECTED_SRC_MD5 = "980aa36308eba63232bb253c36249f38"
DEFAULT_MARGIN = 1.10

TARGETS = (
    {
        "key": "g0",
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_14/Conv",
        "calib_key": "g0_conv_out_per_channel_max_abs",
    },
    {
        "key": "g1",
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_15/Conv",
        "calib_key": "g1_conv_out_per_channel_max_abs",
    },
)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def insert_pertensor_clip(model: onnx.ModelProto, target: dict, calib_per_ch: np.ndarray, margin: float) -> dict:
    target_node = None
    target_idx = None
    for i, n in enumerate(model.graph.node):
        if n.name == target["conv_node_name"]:
            target_node = n
            target_idx = i
            break
    if target_node is None:
        raise ValueError(f"Conv node not found: {target['conv_node_name']}")

    orig_output = target_node.output[0]
    final_output = orig_output + "_pt_clip_cycle18b18"

    T = float(margin * float(calib_per_ch.max()))

    prefix = target["conv_node_name"]
    # ONNX Clip min/max are scalar (rank-0) initializers.
    min_init = numpy_helper.from_array(np.array(-T, dtype=np.float32),
                                       name=f"{prefix}_clip_min_cycle18b18")
    max_init = numpy_helper.from_array(np.array(T, dtype=np.float32),
                                       name=f"{prefix}_clip_max_cycle18b18")
    model.graph.initializer.extend([min_init, max_init])

    clip_node = helper.make_node(
        "Clip",
        inputs=[orig_output, min_init.name, max_init.name],
        outputs=[final_output],
        name=f"{prefix}_Clip_cycle18b18",
    )
    model.graph.node.insert(target_idx + 1, clip_node)

    # Rewire every downstream consumer of orig_output to use final_output,
    # except the Clip node we just inserted.
    new_node_names = {clip_node.name}
    for n in model.graph.node:
        if n.name in new_node_names:
            continue
        for j, inp in enumerate(n.input):
            if inp == orig_output:
                n.input[j] = final_output

    for o in model.graph.output:
        if o.name == orig_output:
            o.name = final_output

    return {
        "key": target["key"],
        "conv_node_name": target["conv_node_name"],
        "T_per_tensor": T,
        "margin": float(margin),
        "calib_per_channel_summary": {
            "min": float(calib_per_ch.min()),
            "p50": float(np.percentile(calib_per_ch, 50)),
            "p90": float(np.percentile(calib_per_ch, 90)),
            "max": float(calib_per_ch.max()),
        },
        "T_vs_cycle18b16_per_tensor_T_256_ratio_tighter": float(256.0 / T),
        "T_vs_cycle18b17_a1_max_hi_per_channel_ratio": float(T / (margin * float(calib_per_ch.max()))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx"))
    parser.add_argument("--dst", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle18b18_a1c_clip_pertensor.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
    parser.add_argument("--calib-json", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    args = parser.parse_args()

    src_md5 = md5sum(args.src)
    assert src_md5 == EXPECTED_SRC_MD5, f"production D3 md5 drift: {src_md5}"
    calib_md5 = md5sum(args.calib_json)
    print(f"[a1c] src md5: {src_md5} (production D3 OK)")
    print(f"[a1c] calib JSON: {args.calib_json.name} md5={calib_md5}")

    calib = json.loads(args.calib_json.read_text())
    calib_per_target = {
        "g0": np.array(calib["g0_conv_out_per_channel_max_abs"], dtype=np.float32),
        "g1": np.array(calib["g1_conv_out_per_channel_max_abs"], dtype=np.float32),
    }

    model = onnx.load(str(args.src))
    per_target = []
    for t in TARGETS:
        per_target.append(insert_pertensor_clip(model, t, calib_per_target[t["key"]], args.margin))
        print(f"[a1c] {t['key']}: T={per_target[-1]['T_per_tensor']:.4f} "
              f"(calib max {per_target[-1]['calib_per_channel_summary']['max']:.4f}, "
              f"margin {args.margin})")

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    dst_md5 = md5sum(args.dst)
    print(f"[a1c] wrote {args.dst.name} md5={dst_md5}")

    src_sidecar = json.loads(args.sidecar_src.read_text())
    side = copy.deepcopy(src_sidecar)
    side["schema_version"] = "s20-bafnetplus-functional-fp32-complex-cycle18b18-a1c-clip-pertensor-v1"
    side["produced_by"] = "BAFNetPlus.scripts.export_bafnetplus_cycle18b18_candidate_a1c_clip_pertensor"
    side["cycle18b18"] = {
        "candidate": "a1c_clip_pertensor",
        "step": "step1_latency_opt_single_node_clip",
        "predecessor": "cycle18b17 A1 per-channel Max+Min (66.24 ms / idx0 PESQ 4.5542)",
        "rationale": (
            "Replace cycle 18b-17 A1's per-channel (Max + Min) 2-node clip with a single "
            "ONNX Clip node per target Conv, hoping QNN HTP will fuse Clip into the Conv "
            "kernel template as a post-activation function (cycle 18b-16/17 cluster ~66-70 "
            "ms regardless of node count suggests partition boundary is the dominant overhead). "
            "Per-tensor T = margin * max(per-channel-calib) — over-protects channels with "
            "smaller per-channel max but uses a single scalar threshold per target, which is "
            "what ONNX Clip accepts (min/max are scalar)."
        ),
        "margin": float(args.margin),
        "calib_json": str(args.calib_json),
        "calib_json_md5": calib_md5,
        "targets": per_target,
        "source_production_md5": src_md5,
    }
    side["onnx_file_md5"] = dst_md5
    dst_sidecar = args.dst.with_suffix(".onnx.json")
    dst_sidecar.write_text(json.dumps(side, indent=2))
    print(f"[a1c] wrote {dst_sidecar.name} md5={md5sum(dst_sidecar)}")

    assert md5sum(args.src) == EXPECTED_SRC_MD5, "production source mutated!"
    print("[a1c] OK -- production D3 byte-identical, candidate A1c per-tensor Clip emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
