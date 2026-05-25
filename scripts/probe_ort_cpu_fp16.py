#!/usr/bin/env python3
"""c09 investigation probe -- Local CPU ORT FP16 precision lever.

The 2026-05-22 precision sweep tested INT8 (dynamic + static QDQ) and found
both negative (dynamic = ConvInteger CPU kernel absent / static = 2.4x
slower from QDQ boundary overhead). The 2026-05-25 4-axis triangulation
cemented the FP32 floor across SessionOptions / ORT version / cross-platform
axes -- but never touched FP16, the one untested precision.

Sapphire Rapids (Xeon Silver 4510) has native FP16 SIMD via AVX-512-FP16
(``avx512_fp16`` cpuinfo flag), so in principle the MLAS path could exploit
it. In practice ORT 1.23.2's CPU EP FP16 kernel coverage on x86 is sparse
(most ops cast back to FP32 internally), and the 190-state streaming graph
inserts ~380 Cast nodes at the FP32/FP16 boundary if I/O stays FP32. Two
variants bracket the realistic vs best-case:

  fp16_keepio  -- ``convert_float_to_float16(keep_io_types=True)`` -- all 194
                  inputs / 193 outputs stay FP32, Cast nodes at boundary.
                  Deployment-realistic (surrounding pipeline is FP32) but the
                  ~380 Cast nodes are the cost of admission.
  fp16_fullio  -- ``convert_float_to_float16(keep_io_types=False)`` -- I/O is
                  FP16, no boundary Cast. Best-case for "pure FP16 compute"
                  isolation. The bench feeds FP16 zero tensors. Less deployment
                  -realistic (host pipeline would need FP16 staging).

For each variant: clean timing (N=400 warmup=150) + one profiled pass
(executed-node count, op histogram, Cast count). Compares against FP32
baseline (the unchanged 50ms tier ONNX bench_local_ort_cpu.py uses). Pin
externally with taskset. NOT a c09 deliverable script.

Usage (from the BAFNetPlus repo root, pinned to one NUMA node):
    taskset -c 12-23 env CUDA_VISIBLE_DEVICES= python scripts/probe_ort_cpu_fp16.py --tier 50
"""
from __future__ import annotations

import argparse
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
FP16_DIR = Path("/tmp/c09_ort_cpu_fp16")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", type=int, nargs="+", choices=[12, 25, 50, 75, 100], default=[50])
    p.add_argument("--threads", type=int, default=8, help="intra_op (calibrated optimum)")
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--measure", type=int, default=400)
    p.add_argument("--prof-warmup", type=int, default=30)
    p.add_argument("--prof-measure", type=int, default=30)
    return p.parse_args()


def convert_to_fp16(src: Path, dst: Path, keep_io: bool) -> tuple[int, int]:
    """Convert FP32 ONNX to FP16. Returns (orig_node_count, fp16_node_count)."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    model = onnx.load(str(src))
    orig_nodes = len(model.graph.node)
    fp16_model = convert_float_to_float16(model, keep_io_types=keep_io)
    fp16_nodes = len(fp16_model.graph.node)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(fp16_model, str(dst))
    return orig_nodes, fp16_nodes


def make_step(np, sess, sidecar, *, fp_dtype):
    io, geom = sidecar["io"], sidecar["geometry"]
    state_names = list(io["state_names"])
    state_shapes = [tuple(s) for s in io["state_shapes"]]
    n_nso = int(io["num_non_state_outputs"])
    f_bins, t_export = int(geom["freq_size"]), int(geom["T_export"])
    z = np.zeros((1, f_bins, t_export), dtype=fp_dtype)
    spec = {k: z for k in ("bcs_mag", "bcs_pha", "acs_mag", "acs_pha")}
    states = [np.zeros(s, dtype=fp_dtype) for s in state_shapes]

    def step() -> None:
        feed = dict(spec)
        for nm, arr in zip(state_names, states):
            feed[nm] = arr
        out = sess.run(None, feed)
        for i, s in enumerate(out[n_nso:]):
            states[i] = s
    return step


def make_session(ort, onnx_path: Path, *, threads: int, profile_prefix: str | None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix is not None:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


def bench_one(np, ort, onnx_path: Path, sidecar, *, threads: int, warmup: int, measure: int,
              fp_dtype) -> dict:
    sess = make_session(ort, onnx_path, threads=threads, profile_prefix=None)
    step = make_step(np, sess, sidecar, fp_dtype=fp_dtype)
    for _ in range(warmup):
        step()
    dts = []
    for _ in range(measure):
        t0 = time.perf_counter_ns()
        step()
        dts.append(time.perf_counter_ns() - t0)
    del sess
    gc.collect()
    a = np.asarray(dts, dtype=np.float64) / 1.0e6
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99))}


def profile_one(np, ort, onnx_path: Path, sidecar, tier_int: int, label: str, *, threads: int,
                fp_dtype, prof_warmup: int, prof_measure: int) -> dict:
    FP16_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(FP16_DIR / f"prof_{label}_{tier_int}ms")
    sess = make_session(ort, onnx_path, threads=threads, profile_prefix=prefix)
    step = make_step(np, sess, sidecar, fp_dtype=fp_dtype)
    for _ in range(prof_warmup + prof_measure):
        step()
    prof_path = sess.end_profiling()
    del sess
    gc.collect()
    # reuse the parser from profile_ort_cpu (no torch dep)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from profile_ort_cpu import parse_profile
    return parse_profile(prof_path, n_warmup=prof_warmup)


def main() -> int:
    args = parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    import onnxruntime as ort

    aff = sorted(os.sched_getaffinity(0))
    print(f"### onnxruntime {ort.__version__}  python {sys.version.split()[0]}")
    print(f"### CPU affinity: {len(aff)} cpus {aff[0]}..{aff[-1]}  intra_op={args.threads}")

    for tier_int in args.tier:
        t_chunk = {12: 12.5, 25: 25.0, 50: 50.0, 75: 75.0, 100: 100.0}[tier_int]
        tag = f"{tier_int}ms"
        fp32_path = ONNX_DIR / tag / f"bafnetplus_{tier_int}ms_fp32.onnx"
        with open(f"{fp32_path}.json") as f:
            sidecar = json.load(f)
        size_fp32 = fp32_path.stat().st_size / 1.0e6

        # ---- build FP16 variants
        keepio_path = FP16_DIR / f"bafnetplus_{tier_int}ms_fp16_keepio.onnx"
        fullio_path = FP16_DIR / f"bafnetplus_{tier_int}ms_fp16_fullio.onnx"
        print(f"\n### tier {tag}  (chunk={int(t_chunk/6.25)} frames, T_chunk={t_chunk} ms, FP32 base {size_fp32:.1f} MB)")
        print(f"  converting FP32 -> FP16 ...", flush=True)
        n_orig, n_keepio = convert_to_fp16(fp32_path, keepio_path, keep_io=True)
        _, n_fullio = convert_to_fp16(fp32_path, fullio_path, keep_io=False)
        size_keepio = keepio_path.stat().st_size / 1.0e6
        size_fullio = fullio_path.stat().st_size / 1.0e6
        print(f"  FP32 nodes={n_orig}  keepio FP16 nodes={n_keepio} ({size_keepio:.1f} MB)  "
              f"fullio FP16 nodes={n_fullio} ({size_fullio:.1f} MB)")

        # ---- bench all three (FP32 baseline + 2 FP16 variants)
        print(f"\n  clean timing (N={args.measure}, warmup={args.warmup}):")
        print(f"    {'variant':14s} {'mean':>9s} {'p50':>9s} {'p99':>9s} {'RTF':>7s} {'vs FP32':>9s}")
        results = {}
        for label, path, fp_dtype in [
            ("fp32", fp32_path, np.float32),
            ("fp16_keepio", keepio_path, np.float32),
            ("fp16_fullio", fullio_path, np.float16),
        ]:
            r = bench_one(np, ort, path, sidecar, threads=args.threads,
                          warmup=args.warmup, measure=args.measure, fp_dtype=fp_dtype)
            results[label] = r
            ratio_str = ""
            if label != "fp32" and "fp32" in results:
                ratio = results["fp32"]["mean"] / r["mean"]
                ratio_str = f"{ratio:.3f}x"
            print(f"    {label:14s} {r['mean']:8.3f}ms {r['p50']:8.3f} {r['p99']:8.3f} "
                  f"{r['mean']/t_chunk:7.3f} {ratio_str:>9s}", flush=True)

        # ---- profile each to see op-level dispatch
        print(f"\n  profiled per-op (N={args.prof_measure} after {args.prof_warmup} warmup):")
        for label, path, fp_dtype in [
            ("fp32", fp32_path, np.float32),
            ("fp16_keepio", keepio_path, np.float32),
            ("fp16_fullio", fullio_path, np.float16),
        ]:
            p = profile_one(np, ort, path, sidecar, tier_int, label, threads=args.threads,
                            fp_dtype=fp_dtype, prof_warmup=args.prof_warmup,
                            prof_measure=args.prof_measure)
            # top ops + Cast count
            cast_ms = p["op_ms"].get("Cast", 0.0)
            cast_cnt = p["op_cnt"].get("Cast", 0)
            conv_ms = p["op_ms"].get("Conv", 0.0)
            print(f"    [{label:14s}] executed={p['n_exec_nodes']}  model_run={p['run_mean_ms']:.2f}ms  "
                  f"kernel={p['kernel_total_ms']:.2f}ms  gap={p['gap_ms']:.2f}ms")
            print(f"      Cast: count={cast_cnt:5.0f}/chunk  ms={cast_ms:.3f}  "
                  f"({100*cast_ms/p['kernel_total_ms']:.1f}% kernel)")
            print(f"      Conv: count={p['op_cnt'].get('Conv',0):5.0f}/chunk  ms={conv_ms:.3f}  "
                  f"({100*conv_ms/p['kernel_total_ms']:.1f}% kernel)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
