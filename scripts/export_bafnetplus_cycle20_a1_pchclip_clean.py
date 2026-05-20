"""Cycle 20 A1 (clean) — per-channel symmetric Clip on the CLEANER production D3.

Cycle 20 redo of cycle 18b-17 Candidate A1, applied to the cycle-20-rebuilt
production D3 anchor (md5 ``86109fe2...``, 6122 nodes, 12,220,945 bytes) that
replaced the stale 2026-05-14 anchor (md5 ``980aa363...``, 7460 nodes,
12,391,036 bytes). Cycle 19 forensic confirmed the stale anchor was a WIP
export with 1338 dead nodes that inflated device latency to ~66 ms; the
cleaner re-export restores ~33 ms idx 0. Cycle 18b-17 A1's per-channel Clip
adds 4 nodes (2x Max + 2x Min) for which cycle 18b-3 partition forensic
established 0 latency penalty on the HTP delegate. Expected outcome on
cleaner D3: device idx 0 latency 33-35 ms with host PESQ 4.5542 preserved
(byte-identical host behavior since every host value is within the 1.10x
margin).

Per-channel hi[c] = 1.10 * max_abs_calib_g{0,1}[c] (from cycle 18b-17 Phase 0
per-channel calibration on TAPS idx 0, host PyTorch trajectory invariant to
D3 dead-code-elimination). Symmetric lo[c] = -hi[c].

Insert structure (per Conv target):

  y       = Conv(z, W, b)            # unchanged (Conv2d on cleaner D3)
  tmp     = Max(y, lo_per_ch[1,256,1,1])
  y_clip  = Min(tmp, hi_per_ch[1,256,1,1])

Cleaner D3 uses Conv2d (kernel ``[1,1]``, weight ``[256,64,1,1]``, 4D output
``[B,256,F,T]``) — different from the stale anchor's Conv1d (kernel ``[1]``,
weight ``[256,64,1]``, 3D output ``[B,256,T]``) used by cycle 18b-17. The
per-channel scaler is therefore 4D ``[1,256,1,1]`` so the broadcast matches
along the C dim of the cleaner 4D Conv2d output.

Targets ONLY the 2 mapping TSBlock-3 freq_stage proj_first Conv nodes:

  * ``/mapping_core/proj_first/proj_first.0_14/Conv`` (g0)
  * ``/mapping_core/proj_first/proj_first.0_15/Conv`` (g1)

Cleaner production D3 (md5 ``86109fe2...``) is NOT modified; output is
written to ``bafnetplus_50ms_fp32_complex_cycle20_a1_pchclip_clean.onnx``
with the base sidecar schema ``s20-bafnetplus-functional-fp32-complex``
preserved (so ``BAFNetPlusOrtStreaming.from_onnx`` accepts it) — the cycle-
specific variant metadata is recorded in a top-level ``cycle20`` block.

Usage::

    python BAFNetPlus/scripts/export_bafnetplus_cycle20_a1_pchclip_clean.py \\
        --src BAFNetPlus/results/onnx/bafnetplus_50ms_fp32_complex.onnx \\
        --dst BAFNetPlus/results/onnx/bafnetplus_50ms_fp32_complex_cycle20_a1_pchclip_clean.onnx \\
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
    # Cleaner D3 Conv2d output is 4D [B,256,F,T] — reshape per-channel scaler to 4D.
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
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex_cycle20_a1_pchclip_clean.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
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
    for k, v in calib_per_target.items():
        print(f"[a1-clean] calib {k}: min={v.min():.3f} p50={np.percentile(v,50):.3f} p90={np.percentile(v,90):.3f} max={v.max():.3f}")

    model = onnx.load(str(args.src))
    per_target = []
    for t in TARGETS:
        calib_vec = calib_per_target[t["key"]]
        per_target.append(insert_per_channel_clip(model, t, calib_vec, args.margin))

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))
    dst_md5 = md5sum(args.dst)
    print(f"[a1-clean] wrote {args.dst.name} md5={dst_md5}")

    src_sidecar = json.loads(args.sidecar_src.read_text())
    side = copy.deepcopy(src_sidecar)
    # Preserve base schema so BAFNetPlusOrtStreaming.from_onnx accepts the sidecar
    # (the whitelist in ort_wrapper does not accept variant suffixes). Variant metadata
    # lives in the cycle20 block below.
    side["produced_by"] = "BAFNetPlus.scripts.export_bafnetplus_cycle20_a1_pchclip_clean"
    side["cycle20"] = {
        "schema_variant_label": "s20-bafnetplus-functional-fp32-complex-cycle20-a1-pchclip-clean-v1",
        "candidate": "a1_pchclip_clean",
        "redo_of": "cycle18b17_a1_pchclip",
        "rationale": (
            "Cycle 20 redo of cycle 18b-17 A1 on the cycle-20-rebuilt cleaner "
            "production D3 (md5 86109fe2..., 6122 nodes). Cycle 19 forensic "
            "confirmed the stale 2026-05-14 anchor (md5 980aa363..., 7460 nodes "
            "= 1338 dead nodes) was the root cause of the ~66 ms device latency; "
            "cleaner re-export restores ~33 ms idx 0. A1's 4 Max/Min nodes "
            "carry 0 HTP latency penalty (cycle 18b-3 partition forensic), so "
            "predicted device idx 0 ~33-35 ms with host PESQ 4.5542 preserved."
        ),
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

    assert md5sum(args.src) == EXPECTED_SRC_MD5, "production source mutated!"
    print("[a1-clean] G3 OK -- cleaner production D3 byte-identical, cycle 20 A1 per-channel Clip emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
