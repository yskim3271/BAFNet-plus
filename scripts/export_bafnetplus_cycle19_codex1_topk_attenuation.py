"""Cycle 19 Phase 2 — Codex Idea 1: 0-node top-K row attenuation on production D3.

For BOTH targets g0 and g1, the top-K outlier output channels (from cycle 18b-17
per-channel calib JSON) get their proj_first Conv weight rows + biases scaled
by ``r`` in a SINGLE jointly-modified ONNX (vs Phase 0.2 which produces one
ONNX per single-channel perturbation). The intent is to suppress the FP16
overflow risk that hot channels carry on QNN-HTP without breaking PESQ.

  W'[o, :, :] = r * W[o, :, :]     for o in top_k_channels[target]
  b'[o]       = r * b[o]

Variants (defaults; override via --top-k / --r):
  Wv1 : g0 top-5 + g1 top-5, r=0.65
  Wv2 : g0 top-5 + g1 top-5, r=0.50
  Wv3 : g0 top-10 + g1 top-10, r=0.65 (lambda-aware -- ranking from
        cycle19_lambda_calibration_idx0.json, low-lambda channels preferred)

Output ONNX schema_version stays canonical
``s20-bafnetplus-functional-fp32-complex`` so BAFNetPlusOrtStreaming.from_onnx
accepts it for host eval. Cycle 19 provenance lives in the sidecar
``cycle19_phase2`` sub-block.

Usage (from BAFNetPlus/ as cwd)::

    # Wv1 -- top-5 max-abs, r=0.65 (most conservative):
    python scripts/export_bafnetplus_cycle19_codex1_topk_attenuation.py \\
        --calib-json ../Android_projects/results/parity/cycle18b17_proj_first_per_channel_calib_idx0.json \\
        --variant-name Wv1 --top-k 5 --r 0.65 \\
        --dst results/onnx/bafnetplus_50ms_fp32_complex_cycle19_codex1_topk_attenuation_Wv1.onnx

    # Wv2 -- top-5, r=0.50 (most aggressive):
    python ... --variant-name Wv2 --top-k 5 --r 0.50 \\
        --dst ...Wv2.onnx

    # Wv3 -- top-10 lambda-aware (requires Phase 0.2 lambda JSON):
    python ... --variant-name Wv3 --top-k 10 --r 0.65 \\
        --lambda-json ../Android_projects/results/parity/cycle19_lambda_calibration_idx0.json \\
        --dst ...Wv3.onnx
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
from onnx import numpy_helper


EXPECTED_SRC_MD5 = "980aa36308eba63232bb253c36249f38"
VARIANT_SIDECAR_SCHEMA = "s20-bafnetplus-functional-fp32-complex"
PRODUCED_BY = "BAFNetPlus.scripts.export_bafnetplus_cycle19_codex1_topk_attenuation"

TARGETS = (
    {
        "key": "g0",
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_14/Conv",
        "weight_init_name": "mapping_core.sequence_block.3.freq_stage.0.1.proj_first.0.weight",
        "bias_init_name":   "mapping_core.sequence_block.3.freq_stage.0.1.proj_first.0.bias",
        "calib_key": "g0_conv_out_per_channel_max_abs",
    },
    {
        "key": "g1",
        "conv_node_name": "/mapping_core/proj_first/proj_first.0_15/Conv",
        "weight_init_name": "mapping_core.sequence_block.3.freq_stage.1.1.proj_first.0.weight",
        "bias_init_name":   "mapping_core.sequence_block.3.freq_stage.1.1.proj_first.0.bias",
        "calib_key": "g1_conv_out_per_channel_max_abs",
    },
)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_topk_channels(calib_data: dict,
                       lambda_data: Optional[dict],
                       target_key: str,
                       calib_key: str,
                       top_k: int) -> List[int]:
    """Channel selection -- if lambda_data is provided, prefer low-lambda
    (low perceptual sensitivity) channels among the high-magnitude ones."""
    mag = np.array(calib_data[calib_key], dtype=np.float32)
    if lambda_data is None:
        return [int(c) for c in np.argsort(mag)[::-1][:top_k]]

    per_target = lambda_data["per_target"].get(target_key)
    if per_target is None:
        raise ValueError(f"lambda JSON missing per_target[{target_key}]")
    # variants list: each entry has channel + lambda.
    lam_by_ch = {entry["channel"]: float(entry["lambda"]) for entry in per_target["variants"]}
    cand_pool = [int(c) for c in np.argsort(mag)[::-1][: top_k * 2]]
    cand_with_lambda = []
    for c in cand_pool:
        if c in lam_by_ch:
            cand_with_lambda.append((c, lam_by_ch[c]))
    if len(cand_with_lambda) < top_k:
        # fall back to magnitude-only if lambda data is sparse
        return [int(c) for c in np.argsort(mag)[::-1][:top_k]]
    cand_with_lambda.sort(key=lambda x: x[1])  # lowest lambda first
    return [c for c, _ in cand_with_lambda[:top_k]]


def attenuate_initializer_rows(model: onnx.ModelProto,
                               target: dict,
                               channels: List[int],
                               r: float) -> dict:
    init_by_name = {i.name: i for i in model.graph.initializer}
    w_init = init_by_name[target["weight_init_name"]]
    b_init = init_by_name[target["bias_init_name"]]

    W = numpy_helper.to_array(w_init).copy()
    B = numpy_helper.to_array(b_init).copy()
    if W.shape != (256, 64, 1):
        raise ValueError(f"W shape {W.shape} unexpected for {target['key']}")
    if B.shape != (256,):
        raise ValueError(f"B shape {B.shape} unexpected for {target['key']}")

    for ch in channels:
        W[ch, :, :] = (r * W[ch, :, :]).astype(np.float32)
        B[ch] = np.float32(r * B[ch])

    w_init.CopyFrom(numpy_helper.from_array(W, name=w_init.name))
    b_init.CopyFrom(numpy_helper.from_array(B, name=b_init.name))

    return {
        "target": target["key"],
        "conv_node_name": target["conv_node_name"],
        "weight_init": target["weight_init_name"],
        "bias_init": target["bias_init_name"],
        "channels": [int(c) for c in channels],
        "r": float(r),
        "math": "W'[o,:,:] = r * W[o,:,:]; b'[o] = r * b[o]",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bn-root", type=Path, default=Path.cwd())
    parser.add_argument("--src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
    parser.add_argument("--calib-json", type=Path, required=True)
    parser.add_argument("--lambda-json", type=Path,
                        help="cycle19_lambda_calibration_idx0.json (Wv3 only)")
    parser.add_argument("--variant-name", type=str, required=True,
                        help="Wv1 / Wv2 / Wv3 etc. -- used in sidecar provenance only")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--dst", type=Path, required=True,
                        help="Output ONNX path (variant-specific)")
    args = parser.parse_args()

    bn_root = args.bn_root.resolve()
    src_path = (bn_root / args.src).resolve()
    src_md5 = md5sum(src_path)
    if src_md5 != EXPECTED_SRC_MD5:
        raise SystemExit(f"production D3 md5 drift: {src_md5}")
    print(f"[codex1] src D3 md5 OK: {src_md5}")

    sidecar_src_path = (bn_root / args.sidecar_src).resolve()
    src_sidecar = json.loads(sidecar_src_path.read_text())

    calib_path = (bn_root / args.calib_json).resolve() if not args.calib_json.is_absolute() else args.calib_json
    calib_md5 = md5sum(calib_path)
    calib_data = json.loads(calib_path.read_text())
    print(f"[codex1] calib JSON md5: {calib_md5}")

    lambda_md5 = None
    lambda_data: Optional[dict] = None
    if args.lambda_json is not None:
        lp = (bn_root / args.lambda_json).resolve() if not args.lambda_json.is_absolute() else args.lambda_json
        lambda_md5 = md5sum(lp)
        lambda_data = json.loads(lp.read_text())
        print(f"[codex1] lambda JSON md5: {lambda_md5}")

    model = onnx.load(str(src_path))
    per_target_meta = []
    for t in TARGETS:
        channels = pick_topk_channels(calib_data, lambda_data, t["key"], t["calib_key"], args.top_k)
        meta = attenuate_initializer_rows(model, t, channels, args.r)
        meta["max_abs_per_channel"] = [
            float(calib_data[t["calib_key"]][c]) for c in channels
        ]
        if lambda_data is not None:
            lam_lookup = {e["channel"]: e["lambda"] for e in lambda_data["per_target"][t["key"]]["variants"]}
            meta["lambda_per_channel"] = [float(lam_lookup.get(c, float("nan"))) for c in channels]
        per_target_meta.append(meta)
        print(f"[codex1] {args.variant_name} {t['key']}: channels={channels} r={args.r}")

    onnx.checker.check_model(model)
    dst_path = (bn_root / args.dst).resolve() if not args.dst.is_absolute() else args.dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_path))
    dst_md5 = md5sum(dst_path)
    print(f"[codex1] wrote {dst_path.name} md5={dst_md5}")

    side = copy.deepcopy(src_sidecar)
    side["schema_version"] = VARIANT_SIDECAR_SCHEMA
    side["produced_by"] = PRODUCED_BY
    side["cycle19_phase2"] = {
        "variant_name": args.variant_name,
        "top_k": int(args.top_k),
        "r": float(args.r),
        "calib_json": str(args.calib_json),
        "calib_json_md5": calib_md5,
        "lambda_json": str(args.lambda_json) if args.lambda_json else None,
        "lambda_json_md5": lambda_md5,
        "per_target": per_target_meta,
        "source_production_md5": src_md5,
    }
    side["onnx_file_md5"] = dst_md5
    dst_sidecar = dst_path.with_suffix(".onnx.json")
    dst_sidecar.write_text(json.dumps(side, indent=2))
    print(f"[codex1] wrote {dst_sidecar.name} md5={md5sum(dst_sidecar)}")

    assert md5sum(src_path) == EXPECTED_SRC_MD5, "production source mutated!"
    print(f"[codex1] OK {args.variant_name} -- production D3 byte-identical, variant emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
