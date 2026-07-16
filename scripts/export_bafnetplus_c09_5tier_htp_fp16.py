#!/usr/bin/env python3
"""c09 Phase 1 — 5-tier HTP-FP16 streaming-ONNX export.

Builds the cross-platform model-family deployment artifacts for review cycle
c09: one streaming ONNX (FP32 graph) + HTP-FP16 sidecar per latency tier, by
applying the cycle 20-22 D3 deployment pipeline to each per-tier-trained
checkpoint.

Five-step pipeline per tier (12.5 / 25 / 75 / 100 ms; the 50 ms tier reuses the
deployed D3 and is verified, not re-exported):

  1. base export      export_bafnetplus_to_onnx_from_checkpoint(
                       results/experiments/bafnetplus_<T>ms, chunk_size,
                       phase_output_mode="complex") -> schema
                       s20-bafnetplus-functional-fp32-complex
  2. A1 clip          per-channel symmetric Clip on the 2 mapping TSBlock-3
                       freq_stage proj_first Conv nodes -- reuses
                       insert_per_channel_clip from the cycle 20 script and derives
                       the 50 ms D3 thresholds from the bound D3 ONNX
  3. onnxsim          graph simplification with --skip-fuse-bn
  4. namespace        regen_streaming_config.py (Android side) rebuilds the
     bridge            Kotlin StreamingConfig sidecar from the export sidecar
  5. HTP-FP16         supported_backends -> ["qnn_htp","cpu","nnapi"] +
     sidecar           qnn_config block, mirroring the deployed D3 sidecar

The same FP32 graph serves both Mobile CPU (CPU EP) and Mobile DSP (QNN EP +
HTP-FP16 sidecar). The 50 ms D3 per-channel Clip thresholds are reused for all
tiers (c09 plan option A -- per-tier calibration is intentionally not run).

The cycle 20 A1 script (export_bafnetplus_cycle20_a1_pchclip_clean.py) hardcodes
the 50 ms base-export md5 and emits a 50 ms-D3-specific sidecar block, so it
cannot be invoked directly on the other tiers. This driver reuses its
tier-generic clip core (insert_per_channel_clip) and wraps the full pipeline,
leaving that frozen forensic script untouched.

Usage::

    python -m scripts.export_bafnetplus_c09_5tier_htp_fp16             # 4 new tiers
    python -m scripts.export_bafnetplus_c09_5tier_htp_fp16 --tiers 12  # one tier
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnx
from onnx import numpy_helper

# A top-level `scripts` package installed in site-packages shadows this repo's
# namespace `scripts/` dir, so `python -m scripts.<name>` / `from scripts...`
# do not resolve here. Run this file directly (`python scripts/<file>.py`) and
# bootstrap the repo root + scripts dir onto sys.path for direct imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]  # BAFNetPlus/
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_bafnetplus_cycle20_a1_pchclip_clean import (  # noqa: E402
    TARGETS,
    insert_per_channel_clip,
    md5sum,
)
from src.models.streaming.onnx.export import (  # noqa: E402
    export_bafnetplus_to_onnx_from_checkpoint,
)

REPO_ROOT = _REPO_ROOT
ANDROID_ROOT = REPO_ROOT.parent / "Android_projects"
OUT_ROOT = REPO_ROOT / "results" / "onnx" / "c09_5tier_streaming"

BASE_SCHEMA = "s20-bafnetplus-functional-fp32-complex"
A1_MARGIN = 1.10
REGEN_SCRIPT = ANDROID_ROOT / "scripts" / "parity" / "regen_streaming_config.py"

# Latency tier table -- meta.md "Latency tier 컨벤션 (LOCKED)".
# rho is baked into each per-tier checkpoint; L_f = encoder_lookahead +
# decoder_lookahead; chunk_size = L_f + 2.
TIERS: Dict[str, Dict[str, Any]] = {
    "12": dict(t_alg_ms=12.5, rho=0.975, l_f=0, chunk=2),
    "25": dict(t_alg_ms=25.0, rho=0.95, l_f=2, chunk=4),
    "50": dict(t_alg_ms=50.0, rho=0.9, l_f=6, chunk=8),  # deployed D3 -- reused
    "75": dict(t_alg_ms=75.0, rho=0.84, l_f=10, chunk=12),
    "100": dict(t_alg_ms=100.0, rho=0.78, l_f=14, chunk=16),
}
NEW_TIERS = ["12", "25", "75", "100"]  # 50 ms reuses the deployed D3

# 50 ms deployed D3 anchors (cycle 30, recording-app assets) -- verified, not rebuilt.
# The sidecar hash reflects the 2026-05-29 metadata-only target_soc correction
# from SM8550 to the actual Galaxy S25 Ultra target, SM8750.
D3_ONNX = ANDROID_ROOT / "bafnet-recording-app" / "src" / "main" / "assets" / "bafnetplus_50ms_fp32.onnx"
D3_ONNX_MD5 = "cac4cf69c904b9fb5397bff328532f6c"
D3_SIDECAR_MD5 = "7027e77de03652930c0edfef9f101313"

# HTP-FP16 sidecar fields -- mirror the deployed D3 sidecar
# (bafnet-recording-app/.../bafnetplus_50ms_fp32.onnx.json, md5 7027e77d...).
HTP_FP16_BACKENDS = ["qnn_htp", "cpu", "nnapi"]
HTP_FP16_QNN_CONFIG = {
    "target_soc": "SM8750",
    "htp_performance_mode": "burst",
    "context_cache_enabled": True,
    "vtcm_mb": 8,
    "enable_htp_fp16_precision": True,
    "htp_graph_finalization_optimization_mode": 0,
    "enable_htp_shared_memory_allocator": True,
    "qnn_context_priority": "high",
    "profiling_level": "off",
}

NUM_NON_STATE_OUTPUTS = 3  # complex graph: est_mag, phase_real, phase_imag
EXPECTED_NUM_STATES = 190


def _load_a1_calib_from_d3() -> Dict[str, np.ndarray]:
    """Recover A1 calibration vectors from the bound D3 ONNX Clip thresholds."""
    model = onnx.load(str(D3_ONNX))
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    calib: Dict[str, np.ndarray] = {}
    for target in TARGETS:
        hi_name = f"{target['conv_node_name']}_pchclip_hi_cycle20"
        hi = initializers.get(hi_name)
        if hi is None:
            raise KeyError(f"D3 clip initializer not found: {hi_name}")
        vec = (hi.reshape(-1).astype(np.float32) / np.float32(A1_MARGIN)).astype(np.float32)
        if vec.shape != (256,):
            raise ValueError(f"{hi_name} recovered shape {vec.shape}, expected (256,)")
        calib[target["key"]] = vec
    return calib


def _step1_base_export(tier: str, work: Path) -> Dict[str, Any]:
    """Step 1 -- export the per-tier checkpoint to a base streaming ONNX."""
    chunk = TIERS[tier]["chunk"]
    base_onnx = work / "step1_base.onnx"
    res = export_bafnetplus_to_onnx_from_checkpoint(
        REPO_ROOT / "results" / "experiments" / f"bafnetplus_{tier}ms",
        base_onnx,
        chunk_size=chunk,
        phase_output_mode="complex",
        verbose=True,
    )
    meta = res.metadata
    geo = meta["geometry"]
    if meta["schema_version"] != BASE_SCHEMA:
        raise AssertionError(f"[{tier}ms] schema {meta['schema_version']!r} != {BASE_SCHEMA!r}")
    if geo["chunk_size"] != chunk:
        raise AssertionError(f"[{tier}ms] chunk_size {geo['chunk_size']} != {chunk}")
    l_f = int(geo["encoder_lookahead"]) + int(geo["decoder_lookahead"])
    if l_f != TIERS[tier]["l_f"]:
        raise AssertionError(
            f"[{tier}ms] L_f {l_f} (enc {geo['encoder_lookahead']} + dec {geo['decoder_lookahead']}) "
            f"!= LOCKED {TIERS[tier]['l_f']}"
        )
    if int(meta["core"]["num_states"]) != EXPECTED_NUM_STATES:
        raise AssertionError(f"[{tier}ms] num_states {meta['core']['num_states']} != {EXPECTED_NUM_STATES}")
    print(
        f"  [1/5] base export: chunk={chunk} L_f={l_f} T_export={geo['T_export']} "
        f"md5={md5sum(base_onnx)}"
    )
    return {"base_onnx": base_onnx, "base_sidecar": base_onnx.with_suffix(".onnx.json")}


def _step2_a1_clip(tier: str, base_onnx: Path, work: Path) -> Path:
    """Step 2 -- insert the per-channel A1 Clip (50 ms D3 thresholds reused)."""
    calib = _load_a1_calib_from_d3()
    model = onnx.load(str(base_onnx))
    for t in TARGETS:
        vec = calib[t["key"]]
        insert_per_channel_clip(model, t, vec, A1_MARGIN)
    onnx.checker.check_model(model)
    clipped = work / "step2_clipped.onnx"
    onnx.save(model, str(clipped))
    print(f"  [2/5] A1 per-channel clip (margin {A1_MARGIN}): md5={md5sum(clipped)}")
    return clipped


def _step3_onnxsim(tier: str, clipped: Path, work: Path) -> Path:
    """Step 3 -- onnxsim graph simplification (--skip-fuse-bn)."""
    simplified = work / "step3_simplified.onnx"
    subprocess.run(
        [sys.executable, "-m", "onnxsim", str(clipped), str(simplified), "--skip-fuse-bn"],
        check=True,
        cwd=REPO_ROOT,
    )
    print(f"  [3/5] onnxsim --skip-fuse-bn: md5={md5sum(simplified)}")
    return simplified


def _step3b_export_sidecar(
    tier: str, base_sidecar: Path, simplified: Path, final_onnx_name: str, work: Path
) -> Path:
    """Carry the export-schema sidecar forward to the simplified ONNX (regen input).

    The A1 clip and onnxsim change only internal graph nodes -- geometry, STFT,
    core, and state I/O are unchanged -- so the base export sidecar is reused
    with the ONNX file reference updated and a c09 provenance block added.
    """
    side = json.loads(base_sidecar.read_text())
    side["onnx_file"] = final_onnx_name
    side["onnx_size_bytes"] = simplified.stat().st_size
    side["c09"] = {
        "pipeline": "c09 Phase 1 -- 5-tier HTP-FP16 export",
        "tier_t_alg_ms": TIERS[tier]["t_alg_ms"],
        "rho_left": TIERS[tier]["rho"],
        "steps": "base export -> A1 per-channel clip -> onnxsim --skip-fuse-bn -> regen -> HTP-FP16 sidecar",
        "a1_clip": {
            "margin": A1_MARGIN,
            "calib_source": "50 ms bound D3 ONNX Clip threshold initializers",
            "calib_source_onnx": str(D3_ONNX.relative_to(REPO_ROOT.parent)),
            "calib_source_onnx_md5": D3_ONNX_MD5,
            "calib_recovery": f"hi_threshold / margin({A1_MARGIN})",
            "targets": [t["conv_node_name"] for t in TARGETS],
        },
        "produced_by": "scripts.export_bafnetplus_c09_5tier_htp_fp16",
    }
    out = work / "step3_simplified.onnx.json"
    out.write_text(json.dumps(side, indent=2))
    return out


def _step4_regen(tier: str, export_sidecar: Path, work: Path) -> Path:
    """Step 4 -- namespace bridge: regen_streaming_config.py -> Kotlin sidecar."""
    kotlin = work / "step4_kotlin_sidecar.json"
    subprocess.run(
        [
            sys.executable,
            str(REGEN_SCRIPT),
            "--sidecar",
            str(export_sidecar.resolve()),
            "--output",
            str(kotlin.resolve()),
        ],
        check=True,
    )
    print(f"  [4/5] regen namespace bridge: {kotlin.name}")
    return kotlin


def _step5_htp_fp16(tier: str, kotlin: Path, final_sidecar: Path) -> None:
    """Step 5 -- HTP-FP16 sidecar: swap backends + inject qnn_config block."""
    kc = json.loads(kotlin.read_text())
    kc["model_info"]["supported_backends"] = list(HTP_FP16_BACKENDS)
    # Insert qnn_config immediately after streaming_config to match the
    # deployed D3 sidecar key order.
    rebuilt: Dict[str, Any] = {}
    for key, val in kc.items():
        rebuilt[key] = val
        if key == "streaming_config":
            rebuilt["qnn_config"] = dict(HTP_FP16_QNN_CONFIG)
    final_sidecar.write_text(json.dumps(rebuilt, indent=2) + "\n")
    print(f"  [5/5] HTP-FP16 sidecar: backends={HTP_FP16_BACKENDS} md5={md5sum(final_sidecar)}")


def _ort_smoke(final_onnx: Path) -> Dict[str, Any]:
    """Load the final ONNX in ORT CPU EP, run one zero-input chunk, check finiteness."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(final_onnx), providers=["CPUExecutionProvider"])
    feeds: Dict[str, np.ndarray] = {}
    for idx, inp in enumerate(sess.get_inputs()):
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        # First 4 inputs are bcs_mag/bcs_pha/acs_mag/acs_pha; feed a small
        # positive value so acs_mag stays > 0. States (rest) stay zero.
        fill = 0.1 if idx < 4 else 0.0
        feeds[inp.name] = np.full(shape, fill, dtype=np.float32)
    outs = sess.run(None, feeds)
    finite = all(bool(np.all(np.isfinite(o))) for o in outs[:NUM_NON_STATE_OUTPUTS])
    return {
        "num_inputs": len(sess.get_inputs()),
        "num_outputs": len(outs),
        "non_state_finite": finite,
        "non_state_shapes": [list(o.shape) for o in outs[:NUM_NON_STATE_OUTPUTS]],
    }


def run_tier(tier: str) -> Dict[str, Any]:
    """Run the full 5-step export pipeline for one new tier."""
    print(f"\n=== tier {TIERS[tier]['t_alg_ms']} ms (bafnetplus_{tier}ms, chunk={TIERS[tier]['chunk']}) ===")
    tier_dir = OUT_ROOT / f"{tier}ms"
    work = tier_dir / "_intermediate"
    work.mkdir(parents=True, exist_ok=True)
    final_onnx = tier_dir / f"bafnetplus_{tier}ms_fp32.onnx"
    final_sidecar = tier_dir / f"bafnetplus_{tier}ms_fp32.onnx.json"

    s1 = _step1_base_export(tier, work)
    clipped = _step2_a1_clip(tier, s1["base_onnx"], work)
    simplified = _step3_onnxsim(tier, clipped, work)
    export_sidecar = _step3b_export_sidecar(tier, s1["base_sidecar"], simplified, final_onnx.name, work)
    kotlin = _step4_regen(tier, export_sidecar, work)

    final_onnx.write_bytes(simplified.read_bytes())
    _step5_htp_fp16(tier, kotlin, final_sidecar)

    smoke = _ort_smoke(final_onnx)
    if not smoke["non_state_finite"]:
        raise AssertionError(f"[{tier}ms] ORT smoke: non-state outputs not finite")
    expected_outs = NUM_NON_STATE_OUTPUTS + EXPECTED_NUM_STATES
    if smoke["num_outputs"] != expected_outs:
        raise AssertionError(f"[{tier}ms] ORT smoke: {smoke['num_outputs']} outputs != {expected_outs}")
    print(
        f"  smoke OK: {smoke['num_inputs']} inputs / {smoke['num_outputs']} outputs, "
        f"non-state finite, shapes {smoke['non_state_shapes']}"
    )
    return {
        "tier": tier,
        "t_alg_ms": TIERS[tier]["t_alg_ms"],
        "chunk": TIERS[tier]["chunk"],
        "rho": TIERS[tier]["rho"],
        "onnx": final_onnx,
        "onnx_md5": md5sum(final_onnx),
        "onnx_bytes": final_onnx.stat().st_size,
        "sidecar_md5": md5sum(final_sidecar),
        "smoke": smoke,
    }


def populate_50ms() -> Dict[str, Any]:
    """Verify the deployed 50 ms D3 anchors and copy them into the 5-tier set."""
    print("\n=== tier 50.0 ms (deployed D3 -- verify + copy, not re-exported) ===")
    onnx_md5 = md5sum(D3_ONNX)
    sidecar_src = D3_ONNX.with_suffix(".onnx.json")
    sidecar_md5 = md5sum(sidecar_src)
    if onnx_md5 != D3_ONNX_MD5:
        raise AssertionError(f"50 ms D3 ONNX md5 drift: {onnx_md5} != {D3_ONNX_MD5}")
    if sidecar_md5 != D3_SIDECAR_MD5:
        raise AssertionError(f"50 ms D3 sidecar md5 drift: {sidecar_md5} != {D3_SIDECAR_MD5}")
    tier_dir = OUT_ROOT / "50ms"
    tier_dir.mkdir(parents=True, exist_ok=True)
    dst_onnx = tier_dir / "bafnetplus_50ms_fp32.onnx"
    dst_sidecar = tier_dir / "bafnetplus_50ms_fp32.onnx.json"
    dst_onnx.write_bytes(D3_ONNX.read_bytes())
    dst_sidecar.write_bytes(sidecar_src.read_bytes())
    print(f"  verified: onnx md5={onnx_md5} sidecar md5={sidecar_md5} -> copied to {tier_dir}")
    return {
        "tier": "50",
        "t_alg_ms": 50.0,
        "chunk": 8,
        "rho": 0.9,
        "onnx": dst_onnx,
        "onnx_md5": onnx_md5,
        "onnx_bytes": dst_onnx.stat().st_size,
        "sidecar_md5": sidecar_md5,
        "smoke": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=NEW_TIERS,
        choices=NEW_TIERS,
        help="New tiers to export (default: all 4). The 50 ms tier is always verified + copied.",
    )
    args = parser.parse_args()

    if not REGEN_SCRIPT.is_file():
        parser.error(f"regen_streaming_config.py not found: {REGEN_SCRIPT}")
    if not D3_ONNX.is_file():
        parser.error(f"deployed 50 ms D3 ONNX not found: {D3_ONNX}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = [run_tier(t) for t in args.tiers]
    results.append(populate_50ms())

    results.sort(key=lambda r: r["t_alg_ms"])
    print("\n=== c09 5-tier HTP-FP16 export summary ===")
    print(f"{'tier':>8} {'chunk':>6} {'rho':>6} {'onnx bytes':>12}  {'onnx md5':>34}  {'sidecar md5':>34}")
    for r in results:
        print(
            f"{r['t_alg_ms']:>6} ms {r['chunk']:>6} {r['rho']:>6} {r['onnx_bytes']:>12}  "
            f"{r['onnx_md5']:>34}  {r['sidecar_md5']:>34}"
        )
    chunks = [r["chunk"] for r in results]
    if chunks != sorted(chunks) or len(set(chunks)) != len(chunks):
        raise AssertionError(f"chunk sizes not strictly monotonic across tiers: {chunks}")
    print(f"\nchunk monotonic across {len(results)} tiers: {chunks}  OK")
    print(f"output: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
