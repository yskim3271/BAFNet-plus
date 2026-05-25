#!/usr/bin/env python3
"""c09 ORT version probe -- did the LaCoSENet team's "1.22 -> 1.24 = +46% on
CPU" lever (Android benchmark milestone) transfer to the Linux Xeon path?

Benches the same exported .onnx + sidecar that bench_local_ort_cpu.py uses,
with whichever onnxruntime is in the active env. No torch dependency -- the
exported ONNX files must already be on disk under results/onnx/c09_ort_cpu/.

Run identically in two envs:
  fullcomplex (ORT 1.23.2 -- the c09 baseline)
  sibling venv (ORT 1.24.2 / 1.26 -- the upgrade probes)

Then diff. The 2026-05-22 speedup probe never touched the ORT version axis;
LaCoSENet's milestone says it is the single biggest CPU lever.

Two passes:
  (1) clean timing sweep at tiers 12 / 50 / 100 ms, canonical seq intra=8;
  (2) one profiled pass at tier 50 -- to check whether newer ORT's CPU EP
      fuses the decomposed LayerNorm pattern (ReduceMean/Pow/Sqrt/Div) into
      LayerNormalization, which would shrink the 18% arith bucket and the
      ~13 ms tier-flat executor gap (fewer executed nodes).

Usage (BAFNetPlus repo root, pinned to one socket):
    taskset -c 12-23 env CUDA_VISIBLE_DEVICES= /tmp/c09_ort_newer_env/bin/python \\
        scripts/probe_ort_version.py
"""
from __future__ import annotations

import collections
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
ONNX_DIR = REPO_ROOT / "results" / "onnx" / "c09_ort_cpu"


def main() -> int:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import numpy as np
    import onnxruntime as ort
    # profile_ort_cpu's module-level imports are stdlib only; importing here does
    # not pull in torch (the torch-using export call lives inside its main()).
    from profile_ort_cpu import (TIERS, PROFILE_DIR, coarse, make_session,
                                 make_step, parse_profile)

    aff = sorted(os.sched_getaffinity(0))
    print(f"### onnxruntime {ort.__version__}  numpy {np.__version__}  python "
          f"{sys.version.split()[0]}")
    print(f"### providers={ort.get_available_providers()}  "
          f"CPU affinity={len(aff)} cpus {aff[0]}..{aff[-1]}")

    # ---- (1) clean timing at 3 tiers, canonical seq intra=8 ---------------
    print(f"\n## Clean timing  (seq intra_op=8, graph_opt=all, warmup=150, measure=400)")
    print(f"  {'tier':6s}  {'mean':>10s}  {'p50':>9s}  {'p99':>9s}  {'RTF':>7s}")
    summary: dict = {}
    for tier_int in (12, 50, 100):
        tag, _, _, t_chunk = TIERS[tier_int]
        onnx_path = ONNX_DIR / tag / f"bafnetplus_{tier_int}ms_fp32.onnx"
        if not onnx_path.exists():
            print(f"  {tag:6s}  SKIP (missing {onnx_path})")
            continue
        with open(f"{onnx_path}.json") as f:
            sidecar = json.load(f)
        sess = make_session(ort, onnx_path, threads=8, inter_op=1,
                            exec_mode="sequential", graph_opt="all", profile_prefix=None)
        step = make_step(np, sess, sidecar)
        for _ in range(150):
            step()
        dts = []
        for _ in range(400):
            t0 = time.perf_counter_ns()
            step()
            dts.append(time.perf_counter_ns() - t0)
        a = np.asarray(dts, dtype=np.float64) / 1.0e6
        mean = float(a.mean())
        summary[tier_int] = mean
        print(f"  {tag:6s}  {mean:9.3f}ms  {np.percentile(a, 50):8.3f}  "
              f"{np.percentile(a, 99):8.3f}  {mean / t_chunk:7.3f}",
              flush=True)
        del sess
        gc.collect()

    # cross-tier linear fit (replicates profile_ort_cpu.fit_line)
    if len(summary) >= 2:
        xs = [TIERS[t][2] for t in sorted(summary)]
        ys = [summary[t] for t in sorted(summary)]
        s, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
        print(f"  fit  : {b:.3f} ms fixed + {s:.3f} ms/frame  (chunks={xs})")

    # ---- (2) profiled pass at tier 50: did LayerNorm fuse? ---------------
    print(f"\n## Profiled pass  (tier 50, prof_warmup=30, prof_measure=30)")
    tier_int = 50
    tag, _, _, t_chunk = TIERS[tier_int]
    onnx_path = ONNX_DIR / tag / f"bafnetplus_{tier_int}ms_fp32.onnx"
    if not onnx_path.exists():
        print(f"  SKIP (missing {onnx_path})")
        return 0
    with open(f"{onnx_path}.json") as f:
        sidecar = json.load(f)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(PROFILE_DIR / f"ortver_{ort.__version__.replace('.', '_')}_{tag}")
    sess = make_session(ort, onnx_path, threads=8, inter_op=1,
                        exec_mode="sequential", graph_opt="all", profile_prefix=prefix)
    step = make_step(np, sess, sidecar)
    for _ in range(60):
        step()
    prof_path = sess.end_profiling()
    del sess
    gc.collect()
    p = parse_profile(prof_path, n_warmup=30)
    print(f"  executed nodes  : {p['n_exec_nodes']}")
    print(f"  model_run mean  : {p['run_mean_ms']:7.3f} ms  (profiler-inflated)")
    print(f"  kernel_total    : {p['kernel_total_ms']:7.3f} ms")
    print(f"  executor gap    : {p['gap_ms']:7.3f} ms  "
          f"({100 * p['gap_ms'] / p['run_mean_ms']:5.1f}% of run)")
    cb: dict = collections.defaultdict(float)
    for op, ms in p["op_ms"].items():
        cb[coarse(op)] += ms
    kt = p["kernel_total_ms"]
    print(f"  buckets         : conv {cb['conv']:.2f}ms ({100*cb['conv']/kt:.1f}%) | "
          f"plumbing {cb['plumbing']:.2f}ms ({100*cb['plumbing']/kt:.1f}%) | "
          f"arith {cb['arith']:.2f}ms ({100*cb['arith']/kt:.1f}%)")
    print(f"  per-op (top 16 of {len(p['op_ms'])}):")
    print(f"     {'op':22s} {'count/chunk':>11s} {'ms/chunk':>10s} {'% kernel':>9s}")
    for op, ms in sorted(p["op_ms"].items(), key=lambda kv: -kv[1])[:16]:
        print(f"     {op:22s} {p['op_cnt'][op]:11.0f} {ms:10.3f} {100*ms/kt:8.1f}%")
    # explicit fused-op detection -- a positive hit means newer ORT *did* fuse
    fused_hits = []
    for op in ("LayerNormalization", "SkipLayerNormalization", "FusedConv",
               "FusedMatMul", "EmbedLayerNormalization", "InstanceNormalization"):
        if op in p["op_ms"]:
            fused_hits.append(f"{op}(n={p['op_cnt'][op]:.0f}, {p['op_ms'][op]:.2f}ms)")
    print(f"  fused-op hits   : {fused_hits if fused_hits else '(none -- LayerNorm pattern NOT fused)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
