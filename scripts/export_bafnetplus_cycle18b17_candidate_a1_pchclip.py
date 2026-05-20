"""Cycle 18b-17 Candidate A1 — per-channel symmetric Clip after proj_first Conv outputs.

Plan C extension category (a): post-Conv asymmetric clipping with per-output-
channel thresholds calibrated to 1.10x the host-FP32 observed max-abs.

Per-channel hi[c] = 1.10 * max_abs_calib_g{0,1}[c] (from cycle 18b-17 Phase 0
per-channel calibration on TAPS idx 0). F4=(a) symmetric: lo[c] = -hi[c].
T_c is much tighter than cycle 18b-16's per-tensor T=256 (host max ~13, so
cycle 18b-16 T was ~20x margin; cycle 18b-17 A1 hi[c] is exactly 1.10x channel
max -> on host this is BYTE-IDENTICAL because every host value is within
bounds, but on HTP-FP16 any out-of-host-range outlier in channel c gets
clamped tightly).

Insert structure (per Conv target):

  y       = Conv(z, W, b)            # unchanged
  tmp     = Max(y, lo_per_ch[1,256,1])
  y_clip  = Min(tmp, hi_per_ch[1,256,1])

The original Conv output name is rewired to ``y_clip`` and every downstream
consumer that referenced the Conv output now reads ``y_clip``.

Targets ONLY the 2 mapping TSBlock-3 freq_stage proj_first Conv nodes:

  * ``/mapping_core/proj_first/proj_first.0_14/Conv`` (g0, calibrated to g0_conv_out max-abs)
  * ``/mapping_core/proj_first/proj_first.0_15/Conv`` (g1, calibrated to g1_conv_out max-abs)

Production D3 ONNX (md5 ``980aa363...``) is NOT modified; output is written to
``bafnetplus_50ms_fp32_complex_cycle18b17_a1_pchclip.onnx`` with sidecar
schema ``s20-bafnetplus-functional-fp32-complex-cycle18b17-a1-pchclip-v1``.

Usage::

    python BAFNetPlus/scripts/export_bafnetplus_cycle18b17_candidate_a1_pchclip.py \\
        --src BAFNetPlus/results/onnx/bafnetplus_50ms_fp32_complex.onnx \\
        --dst BAFNetPlus/results/onnx/bafnetplus_50ms_fp32_complex_cycle18b17_a1_pchclip.onnx \\
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

DEFAULT_MARGIN = 1.10


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def insert_per_channel_clip(model: onnx.ModelProto, target: dict, calib: np.ndarray, margin: float) -> dict:
    target_node = None
    target_idx = None
    for i, n in enumerate(model.graph.node):
        if n.name == target["conv_node_name"]:
            target_node = n
            target_idx = i
            break
    if target_node is None:
        raise ValueError(f"Conv node {target['conv_node_name']} not found")

    orig_output = target_node.output[0]
    clip_intermediate = orig_output + "_pchclip_max_cycle18b17"
    final_output = orig_output + "_pchclip_cycle18b17"

    assert calib.shape == (256,), f"calib shape {calib.shape} expected (256,)"
    hi = (margin * calib).astype(np.float32).reshape(1, 256, 1)
    lo = (-hi).astype(np.float32)

    prefix = target["conv_node_name"]
    lo_init = numpy_helper.from_array(lo, name=f"{prefix}_pchclip_lo_cycle18b17")
    hi_init = numpy_helper.from_array(hi, name=f"{prefix}_pchclip_hi_cycle18b17")
    model.graph.initializer.extend([lo_init, hi_init])

    max_node = helper.make_node(
        "Max",
        inputs=[orig_output, lo_init.name],
        outputs=[clip_intermediate],
        name=f"{prefix}_pchclip_Max_cycle18b17",
    )
    min_node = helper.make_node(
        "Min",
        inputs=[clip_intermediate, hi_init.name],
        outputs=[final_output],
        name=f"{prefix}_pchclip_Min_cycle18b17",
    )
    # Insert immediately after target Conv (preserve topological order).
    model.graph.node.insert(target_idx + 1, max_node)
    model.graph.node.insert(target_idx + 2, min_node)

    # Rewire every downstream consumer of the original Conv output to use the clipped output,
    # EXCEPT the new Max node we just inserted.
    new_node_names = {max_node.name}
    for n in model.graph.node:
        if n.name in new_node_names:
            continue
        for j, inp in enumerate(n.input):
            if inp == orig_output:
                n.input[j] = final_output

    # If the original Conv output is in graph.output, redirect.
    for o in model.graph.output:
        if o.name == orig_output:
            o.name = final_output

    return {
        "key": target["key"],
        "conv_node_name": target["conv_node_name"],
        "calib_max_abs_per_channel_summary": {
            "min": float(calib.min()),
            "p50": float(np.percentile(calib, 50)),
            "p90": float(np.percentile(calib, 90)),
            "max": float(calib.max()),
        },
        "margin": float(margin),
        "hi_per_channel_summary": {
            "min": float(hi.min()),
            "p50": float(np.percentile(hi, 50)),
            "p90": float(np.percentile(hi, 90)),
            "max": float(hi.max()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx"))
    parser.add_argument("--dst", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle18b17_a1_pchclip.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
    parser.add_argument("--calib-json", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    args = parser.parse_args()

    src_md5 = md5sum(args.src)
    assert src_md5 == EXPECTED_SRC_MD5, f"production D3 md5 drift: {src_md5}"
    print(f"[a1] src md5: {src_md5} (production D3 OK)")
    calib_md5 = md5sum(args.calib_json)
    print(f"[a1] calib JSON: {args.calib_json.name} md5={calib_md5}")

    calib_data = json.loads(args.calib_json.read_text())
    calib_per_target = {
        "g0": np.array(calib_data["g0_conv_out_per_channel_max_abs"], dtype=np.float32),
        "g1": np.array(calib_data["g1_conv_out_per_channel_max_abs"], dtype=np.float32),
    }
    for k, v in calib_per_target.items():
        print(f"[a1] calib {k}: min={v.min():.3f} p50={np.percentile(v,50):.3f} p90={np.percentile(v,90):.3f} max={v.max():.3f}")

    model = onnx.load(str(args.src))
    per_target = []
    for t in TARGETS:
        calib_vec = calib_per_target[t["key"]]
        per_target.append(insert_per_channel_clip(model, t, calib_vec, args.margin))

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    dst_md5 = md5sum(args.dst)
    print(f"[a1] wrote {args.dst.name} md5={dst_md5}")

    src_sidecar = json.loads(args.sidecar_src.read_text())
    side = copy.deepcopy(src_sidecar)
    side["schema_version"] = "s20-bafnetplus-functional-fp32-complex-cycle18b17-a1-pchclip-v1"
    side["produced_by"] = "BAFNetPlus.scripts.export_bafnetplus_cycle18b17_candidate_a1_pchclip"
    side["cycle18b17"] = {
        "candidate": "a1_pchclip",
        "rationale": (
            "Post-Conv per-output-channel symmetric Clip via Max(y, lo_per_ch) then "
            "Min(tmp, hi_per_ch). hi[c] = margin * max_abs_calib[c] from cycle 18b-17 "
            "Phase 0 per-channel calibration (idx 0, 213 steps). lo[c] = -hi[c]. "
            "On host (FP32) every value is within bounds so the clip is byte-identical "
            "to the original Conv output. On HTP-FP16, any out-of-host-range outlier in "
            "channel c gets clamped tightly (vs cycle 18b-16's per-tensor T=256 which "
            "was 20x looser). The Max/Min nodes cannot fold back into Conv since "
            "Min/Max are not part of any Conv-fusion pattern in ORT."
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
    print(f"[a1] wrote {dst_sidecar.name} md5={md5sum(dst_sidecar)}")

    assert md5sum(args.src) == EXPECTED_SRC_MD5, "production source mutated!"
    print("[a1] G3 OK -- production D3 byte-identical, candidate A1 per-channel Clip emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
