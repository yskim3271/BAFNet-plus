"""Cycle 20 A1 (clean) — per-channel symmetric Clip on the CLEANER production D3.

This module is kept as a small helper library for the c09 5-tier export path:
``export_bafnetplus_c09_5tier_htp_fp16.py`` reuses ``TARGETS``,
``insert_per_channel_clip``, and ``md5sum``. The legacy one-shot CLI remains
available when its archived source ONNX, sidecar, and calibration JSON are
supplied explicitly; current deployment reproduction is driven by the c09
script and the bound D3 assets.
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


EXPECTED_SRC_MD5 = "86109fe20a821316e0e267a4fe56cb4e"

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
    clip_intermediate = orig_output + "_pchclip_max_cycle20"
    final_output = orig_output + "_pchclip_cycle20"

    assert calib.shape == (256,), f"calib shape {calib.shape} expected (256,)"
    hi = (margin * calib).astype(np.float32).reshape(1, 256, 1, 1)
    lo = (-hi).astype(np.float32)

    prefix = target["conv_node_name"]
    lo_init = numpy_helper.from_array(lo, name=f"{prefix}_pchclip_lo_cycle20")
    hi_init = numpy_helper.from_array(hi, name=f"{prefix}_pchclip_hi_cycle20")
    model.graph.initializer.extend([lo_init, hi_init])

    max_node = helper.make_node(
        "Max",
        inputs=[orig_output, lo_init.name],
        outputs=[clip_intermediate],
        name=f"{prefix}_pchclip_Max_cycle20",
    )
    min_node = helper.make_node(
        "Min",
        inputs=[clip_intermediate, hi_init.name],
        outputs=[final_output],
        name=f"{prefix}_pchclip_Min_cycle20",
    )
    model.graph.node.insert(target_idx + 1, max_node)
    model.graph.node.insert(target_idx + 2, min_node)

    for n in model.graph.node:
        if n.name == max_node.name:
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
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle20_a1_pchclip_clean.onnx"),
    )
    parser.add_argument("--sidecar-src", type=Path, required=True)
    parser.add_argument("--calib-json", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    args = parser.parse_args()

    src_md5 = md5sum(args.src)
    assert src_md5 == EXPECTED_SRC_MD5, f"cleaner production D3 md5 drift: {src_md5}"
    print(f"[a1-clean] src md5: {src_md5} (cleaner production D3 OK)")
    calib_md5 = md5sum(args.calib_json)
    print(f"[a1-clean] calib JSON: {args.calib_json.name} md5={calib_md5}")

    calib_data = json.loads(args.calib_json.read_text())
    calib_per_target = {
        "g0": np.array(calib_data["g0_conv_out_per_channel_max_abs"], dtype=np.float32),
        "g1": np.array(calib_data["g1_conv_out_per_channel_max_abs"], dtype=np.float32),
    }

    model = onnx.load(str(args.src))
    per_target = []
    for t in TARGETS:
        per_target.append(insert_per_channel_clip(model, t, calib_per_target[t["key"]], args.margin))

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    dst_md5 = md5sum(args.dst)
    print(f"[a1-clean] wrote {args.dst.name} md5={dst_md5}")

    src_sidecar = json.loads(args.sidecar_src.read_text())
    side = copy.deepcopy(src_sidecar)
    side["produced_by"] = "BAFNetPlus.scripts.export_bafnetplus_cycle20_a1_pchclip_clean"
    side["cycle20"] = {
        "schema_variant_label": "s20-bafnetplus-functional-fp32-complex-cycle20-a1-pchclip-clean-v1",
        "candidate": "a1_pchclip_clean",
        "margin": float(args.margin),
        "calib_json": str(args.calib_json),
        "calib_json_md5": calib_md5,
        "targets": per_target,
        "source_production_md5": src_md5,
    }
    side["onnx_file_md5"] = dst_md5
    side["onnx_file"] = args.dst.name
    dst_sidecar = args.dst.with_suffix(".onnx.json")
    dst_sidecar.write_text(json.dumps(side, indent=2))
    print(f"[a1-clean] wrote {dst_sidecar.name} md5={md5sum(dst_sidecar)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
