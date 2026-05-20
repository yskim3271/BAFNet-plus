"""Cycle 19 Phase 0.2 — host lambda_o perceptual sensitivity calibration.

For each (target in {g0, g1}, channel in top-K from cycle 18b-17 per-channel
calib JSON), emit a variant ONNX where the production D3 proj_first Conv
weight row and bias for that single channel are scaled by ``r`` (default 0.5):

  W'[o, :, :] = r * W[o, :, :]      (o = channel index)
  b'[o]       = r * b[o]

Then run host eval_streaming on TAPS idx 0 for the production D3 baseline and
each variant, capture ``pesq_ort_streaming``, and define the per-channel
perceptual sensitivity:

  lambda[target, ch] = baseline_pesq_ort_streaming - variant_pesq_ort_streaming

Large lambda = a single-channel attenuation hurts host PESQ a lot = that
channel is perceptually load-bearing. Phase 2 / Phase 3 use the per-channel
lambda ranking to pick which outliers to attenuate jointly.

Anchor production D3 (md5 ``980aa363...``) is NOT modified; the 2 * top-K
variants are emitted under --dst-dir and the aggregate JSON (schema
``cycle19-lambda-calib-v1``) lands at --output-json.

Mean activation mu_o per channel (Phase 4 bias-recenter input) is NOT
collected here -- if Phase 2 + Phase 3 both fail (PESQ < 4.40), a separate
mu_o capture pass (cycle 18b-17 11-segment chain template) is needed before
Phase 4 export.

Usage (from BAFNetPlus/ as cwd)::

    python scripts/calib_proj_first_host_lambda_idx0.py \\
        --calib-json ../Android_projects/results/parity/cycle18b17_proj_first_per_channel_calib_idx0.json \\
        --output-json ../Android_projects/results/parity/cycle19_lambda_calibration_idx0.json

Restartable: existing variant ONNXs and eval JSONs are reused on rerun. Use
--skip-export / --skip-eval / --skip-aggregate to isolate stages.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
from onnx import numpy_helper


EXPECTED_SRC_MD5 = "980aa36308eba63232bb253c36249f38"
SCHEMA = "cycle19-lambda-calib-v1"
PRODUCED_BY = "BAFNetPlus.scripts.calib_proj_first_host_lambda_idx0"
# Keep canonical schema (s20-fp32-complex) so BAFNetPlusOrtStreaming.from_onnx accepts
# the variant for host eval -- cycle19 provenance lives in the sub-block.
VARIANT_SIDECAR_SCHEMA = "s20-bafnetplus-functional-fp32-complex"

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


def variant_stem(target_key: str, channel: int, r: float) -> str:
    return f"bafnetplus_50ms_fp32_complex_cycle19_lambda_{target_key}_ch{channel:03d}_r{r:.2f}"


def export_variant(src_model: onnx.ModelProto,
                   target: dict,
                   channel: int,
                   r: float,
                   src_sidecar: dict,
                   src_md5: str,
                   dst_onnx: Path,
                   dst_sidecar: Path) -> dict:
    model = copy.deepcopy(src_model)
    init_by_name = {i.name: i for i in model.graph.initializer}
    w_init = init_by_name[target["weight_init_name"]]
    b_init = init_by_name[target["bias_init_name"]]

    W = numpy_helper.to_array(w_init).copy()
    B = numpy_helper.to_array(b_init).copy()
    if W.shape != (256, 64, 1):
        raise ValueError(f"W shape {W.shape} unexpected for {target['key']}")
    if B.shape != (256,):
        raise ValueError(f"B shape {B.shape} unexpected for {target['key']}")
    W[channel, :, :] = (r * W[channel, :, :]).astype(np.float32)
    B[channel] = np.float32(r * B[channel])

    w_init.CopyFrom(numpy_helper.from_array(W, name=w_init.name))
    b_init.CopyFrom(numpy_helper.from_array(B, name=b_init.name))

    onnx.checker.check_model(model)
    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst_onnx))
    dst_md5 = md5sum(dst_onnx)

    side = copy.deepcopy(src_sidecar)
    side["schema_version"] = VARIANT_SIDECAR_SCHEMA
    side["produced_by"] = PRODUCED_BY
    side["cycle19_phase0_lambda"] = {
        "target": target["key"],
        "conv_node": target["conv_node_name"],
        "weight_init": target["weight_init_name"],
        "bias_init": target["bias_init_name"],
        "channel": int(channel),
        "r": float(r),
        "math": "W'[o,:,:] = r * W[o,:,:]; b'[o] = r * b[o] (o = channel)",
        "source_production_md5": src_md5,
    }
    side["onnx_file_md5"] = dst_md5
    dst_sidecar.write_text(json.dumps(side, indent=2))
    return {
        "channel": int(channel),
        "variant_onnx_path": str(dst_onnx),
        "variant_onnx_md5": dst_md5,
        "sidecar_md5": md5sum(dst_sidecar),
    }


def run_eval(onnx_path: Path, eval_json: Path, unified_ckpt_dir: str,
             chkpt_file: str, taps_idx: int, bn_root: Path) -> dict:
    if eval_json.exists():
        try:
            existing = json.loads(eval_json.read_text())
            for u in existing.get("per_utterance", []):
                if u.get("taps_idx") == taps_idx and isinstance(u.get("pesq_ort_streaming"), (int, float)):
                    return existing
        except Exception:
            pass
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.analysis.eval_streaming",
        "--preset", "custom",
        "--unified-ckpt-dir", unified_ckpt_dir,
        "--chkpt-file", chkpt_file,
        "--onnx-artifact", str(onnx_path),
        "--taps-indices", str(taps_idx),
        "--pesq-tol", "5e-3",
        "--output-json", str(eval_json),
        "--summary-only",
    ]
    print(f"[eval] {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(bn_root), check=False)
    if res.returncode != 0 and not eval_json.exists():
        raise RuntimeError(f"eval_streaming failed: rc={res.returncode} ({onnx_path})")
    return json.loads(eval_json.read_text())


def extract_pesq(report: dict, taps_idx: int) -> Dict[str, float]:
    for u in report.get("per_utterance", []):
        if u.get("taps_idx") == taps_idx:
            return {
                "pesq_ref_pt":         float(u["pesq_ref_pt"]),
                "pesq_streaming_pt":   float(u["pesq_streaming_pt"]),
                "pesq_ort_streaming":  float(u["pesq_ort_streaming"]),
                "delta_pesq_ref_vs_ort": float(u["delta_pesq_ref_vs_ort"]),
                "delta_pesq_pt_vs_ort": float(u["delta_pesq_pt_vs_ort"]),
            }
    raise KeyError(f"taps_idx={taps_idx} not in eval report at {report.get('produced_by', '?')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bn-root", type=Path, default=Path.cwd(),
                        help="BAFNetPlus/ root (default cwd). All paths resolved from here.")
    parser.add_argument("--src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx"))
    parser.add_argument("--sidecar-src", type=Path,
                        default=Path("results/onnx/bafnetplus_50ms_fp32_complex.onnx.json"))
    parser.add_argument("--calib-json", type=Path, required=True,
                        help="cycle18b17_proj_first_per_channel_calib_idx0.json (256-element per-channel max-abs)")
    parser.add_argument("--dst-dir", type=Path,
                        default=Path("results/onnx/cycle19_phase0_lambda"))
    parser.add_argument("--eval-dir", type=Path,
                        default=Path("results/eval_streaming/cycle19_phase0_lambda"))
    parser.add_argument("--output-json", type=Path, required=True,
                        help="Aggregate lambda calibration JSON (schema cycle19-lambda-calib-v1)")
    parser.add_argument("--unified-ckpt-dir", default="results/experiments/bafnetplus_50ms")
    parser.add_argument("--chkpt-file", default="best.th")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--taps-idx", type=int, default=0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    bn_root = args.bn_root.resolve()

    src_path = (bn_root / args.src).resolve()
    src_md5 = md5sum(src_path)
    if src_md5 != EXPECTED_SRC_MD5:
        raise SystemExit(f"production D3 md5 drift: {src_md5} (expected {EXPECTED_SRC_MD5})")
    sidecar_src_path = (bn_root / args.sidecar_src).resolve()
    src_sidecar = json.loads(sidecar_src_path.read_text())

    calib_path = (bn_root / args.calib_json).resolve() if not args.calib_json.is_absolute() else args.calib_json
    calib_md5 = md5sum(calib_path)
    calib_data = json.loads(calib_path.read_text())
    print(f"[phase0.2] D3 src md5 OK: {src_md5}")
    print(f"[phase0.2] calib JSON md5: {calib_md5}")

    per_target_topk: Dict[str, List[dict]] = {}
    for t in TARGETS:
        vec = np.array(calib_data[t["calib_key"]], dtype=np.float32)
        order = np.argsort(vec)[::-1][:args.top_k]
        per_target_topk[t["key"]] = [
            {"channel": int(ch), "max_abs": float(vec[ch])} for ch in order
        ]
        ranking_str = ", ".join(f"ch{x['channel']:3d}({x['max_abs']:.2f})" for x in per_target_topk[t["key"]])
        print(f"[phase0.2] {t['key']} top-{args.top_k}: {ranking_str}")

    src_model = onnx.load(str(src_path))

    dst_dir = (bn_root / args.dst_dir).resolve()
    eval_dir = (bn_root / args.eval_dir).resolve()

    variants: List[dict] = []
    for t in TARGETS:
        for entry in per_target_topk[t["key"]]:
            stem = variant_stem(t["key"], entry["channel"], args.r)
            dst_onnx = dst_dir / f"{stem}.onnx"
            dst_sidecar = dst_dir / f"{stem}.onnx.json"
            if not args.skip_export and not dst_onnx.exists():
                meta = export_variant(src_model, t, entry["channel"], args.r,
                                      src_sidecar, src_md5, dst_onnx, dst_sidecar)
                print(f"[phase0.2] export {t['key']}/ch{entry['channel']:03d} -> {dst_onnx.name} md5={meta['variant_onnx_md5']}")
            else:
                meta = {
                    "channel": int(entry["channel"]),
                    "variant_onnx_path": str(dst_onnx),
                    "variant_onnx_md5": md5sum(dst_onnx) if dst_onnx.exists() else None,
                    "sidecar_md5": md5sum(dst_sidecar) if dst_sidecar.exists() else None,
                }
            meta.update({"target": t["key"], "max_abs": entry["max_abs"], "r": args.r})
            variants.append(meta)

    if not args.skip_eval:
        baseline_eval = eval_dir / "eval_baseline_d3_idx0.json"
        run_eval(src_path, baseline_eval, args.unified_ckpt_dir, args.chkpt_file, args.taps_idx, bn_root)
        for v in variants:
            stem = variant_stem(v["target"], v["channel"], v["r"])
            eval_json = eval_dir / f"eval_{stem}.json"
            run_eval(Path(v["variant_onnx_path"]), eval_json, args.unified_ckpt_dir,
                     args.chkpt_file, args.taps_idx, bn_root)

    if args.skip_aggregate:
        return 0

    baseline_eval = eval_dir / "eval_baseline_d3_idx0.json"
    base_metrics = extract_pesq(json.loads(baseline_eval.read_text()), args.taps_idx)
    base_pesq = base_metrics["pesq_ort_streaming"]
    print(f"[phase0.2] baseline pesq_ort_streaming = {base_pesq:.4f}")

    per_target_report: Dict[str, dict] = {}
    for t in TARGETS:
        tk = t["key"]
        tk_variants = [v for v in variants if v["target"] == tk]
        results = []
        for v in tk_variants:
            stem = variant_stem(tk, v["channel"], v["r"])
            eval_json = eval_dir / f"eval_{stem}.json"
            metrics = extract_pesq(json.loads(eval_json.read_text()), args.taps_idx)
            lam = base_pesq - metrics["pesq_ort_streaming"]
            results.append({
                "channel": v["channel"],
                "max_abs": v["max_abs"],
                "r": v["r"],
                "variant_onnx_md5": v["variant_onnx_md5"],
                **metrics,
                "lambda": lam,
            })
        ranking = sorted(results, key=lambda x: -x["lambda"])
        per_target_report[tk] = {
            "top_k_by_max_abs": per_target_topk[tk],
            "variants": results,
            "ranking_by_sensitivity": [{"channel": r["channel"], "lambda": r["lambda"]} for r in ranking],
        }

    report = {
        "schema_version": SCHEMA,
        "produced_by": PRODUCED_BY,
        "inputs": {
            "src": str(args.src),
            "src_md5": src_md5,
            "sidecar_src": str(args.sidecar_src),
            "calib_json": str(args.calib_json),
            "calib_json_md5": calib_md5,
            "unified_ckpt_dir": args.unified_ckpt_dir,
            "chkpt_file": args.chkpt_file,
            "top_k": args.top_k,
            "r": args.r,
            "taps_idx": args.taps_idx,
        },
        "baseline": {
            "onnx_md5": src_md5,
            **base_metrics,
        },
        "lambda_definition": "lambda = baseline_pesq_ort_streaming - variant_pesq_ort_streaming",
        "per_target": per_target_report,
        "note_mu_o": (
            "Per-channel mean activation (mu_o) NOT collected in Phase 0.2 -- "
            "Phase 4 (Codex Idea 4 bias recenter) is conditional on Phase 2+3 both "
            "failing. On Phase 4 entry, run a separate mu_o capture pass via the "
            "cycle 18b-17 11-segment chain (calib_proj_first_per_channel_idx0.py "
            "template extended with mean aggregation)."
        ),
    }
    out_path = (bn_root / args.output_json).resolve() if not args.output_json.is_absolute() else args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[phase0.2] wrote {out_path} md5={md5sum(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
