#!/usr/bin/env python3
"""c09 investigation probe -- per-node profiling of the Local CPU ORT path.

Re-opens the 2026-05-22 "FP32 ORT CPU EP is the software optimisation floor"
conclusion. That probe tested precision / graph-simplify / alt-runtime levers
(INT8, onnxsim, OpenVINO -- all failed) but never profiled *where* the
~56 ms/chunk actually goes. This script turns on ORT's per-node profiler
(SessionOptions.enable_profiling) on the functional streaming graph and breaks
one chunk-step down into:

  * per-op-type kernel time -- Conv (real compute) vs Slice/Concat (190-state
    streaming plumbing) vs ReduceMean/Pow/Sqrt/Div (decomposed normalisation)
    vs Transpose/Reshape (layout) vs ...;
  * per-branch kernel time -- mapping_core vs masking_core, the two
    independent backbones (relevant to an inter-op-parallel lever);
  * the executor gap = model_run wall time - sum(kernel_time): the ORT
    sequential-executor per-node dispatch + tensor allocation + thread
    fork/join not attributed to any kernel.

Run it across tiers (chunk = 2 / 8 / 16 frames) to separate the tier-flat
fixed component from the per-frame component.

Two passes per tier: a clean timing pass (profiling off -- honest magnitude /
fixed-vs-scaling fit) and a profiled pass (profiling on, absolute numbers
inflated -- only the *breakdown proportions* are read off it). One timed
iteration = one ORT session.run over the streaming graph (4 spectrogram
inputs + 190 state inputs -> 3 + 190 outputs) + state feedback, the same
chunk-step as bench_local_ort_cpu.py. NOT a c09 deliverable script.

Usage (from the BAFNetPlus repo root, pinned to one NUMA node):
    taskset -c 12-23 env CUDA_VISIBLE_DEVICES= python scripts/profile_ort_cpu.py --tier 12 50 100
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
DEFAULT_ONNX = REPO_ROOT / "results" / "onnx" / "c09_ort_cpu"
DEFAULT_EXPERIMENTS = REPO_ROOT / "results" / "experiments"
PROFILE_DIR = Path("/tmp/c09_ort_profile")

# tier_int -> (tag, T_alg_ms, chunk_size_frames, T_chunk_ms) -- c09 LOCKED tier table
TIERS = {
    12:  ("12ms",  12.5,  2,  12.5),
    25:  ("25ms",  25.0,  4,  25.0),
    50:  ("50ms",  50.0,  8,  50.0),
    75:  ("75ms",  75.0, 12,  75.0),
    100: ("100ms", 100.0, 16, 100.0),
}

CONV = {"Conv", "ConvTranspose"}
PLUMBING = {"Slice", "Concat", "Pad", "Reshape", "Transpose", "Shape", "Gather",
            "Unsqueeze", "Squeeze", "ConstantOfShape", "Cast", "Constant",
            "Where", "Expand", "Equal", "Greater", "Less", "If"}


def coarse(op: str) -> str:
    if op in CONV:
        return "conv"
    if op in PLUMBING:
        return "plumbing"
    return "arith"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", type=int, nargs="+", choices=[12, 25, 50, 75, 100], default=[12, 50, 100])
    p.add_argument("--threads", type=int, default=8, help="intra_op_num_threads")
    p.add_argument("--inter-op", type=int, default=1, help="inter_op_num_threads")
    p.add_argument("--exec-mode", choices=["sequential", "parallel"], default="sequential")
    p.add_argument("--graph-opt", choices=["all", "extended", "basic", "disabled"], default="all")
    p.add_argument("--warmup", type=int, default=200, help="clean-pass warmup")
    p.add_argument("--clean-measure", type=int, default=1000, help="clean (no-profile) timed steps")
    p.add_argument("--prof-warmup", type=int, default=30, help="profiled-pass warmup (keeps JSON small)")
    p.add_argument("--prof-measure", type=int, default=30, help="profiled steps (kept after warmup)")
    p.add_argument("--chkpt-file", default="best.th")
    return p.parse_args()


def ensure_onnx(export_fn, tier_int: int, chunk_size: int, chkpt_file: str) -> Path:
    tag = TIERS[tier_int][0]
    onnx_path = DEFAULT_ONNX / tag / f"bafnetplus_{tier_int}ms_fp32.onnx"
    if onnx_path.exists() and Path(f"{onnx_path}.json").exists():
        return onnx_path
    ckpt_dir = DEFAULT_EXPERIMENTS / f"bafnetplus_{tier_int}ms"
    if not (ckpt_dir / ".hydra" / "config.yaml").exists():
        raise FileNotFoundError(f"checkpoint dir missing: {ckpt_dir}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  exporting FP32 streaming ONNX: {ckpt_dir.name} -> {onnx_path}", flush=True)
    export_fn(str(ckpt_dir), str(onnx_path), chunk_size=chunk_size, chkpt_file=chkpt_file, verbose=False)
    return onnx_path


def make_session(ort, onnx_path: Path, *, threads: int, inter_op: int, exec_mode: str,
                 graph_opt: str, profile_prefix: str | None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = inter_op
    so.execution_mode = (ort.ExecutionMode.ORT_PARALLEL if exec_mode == "parallel"
                         else ort.ExecutionMode.ORT_SEQUENTIAL)
    so.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[graph_opt]
    if profile_prefix is not None:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


def make_step(np, sess, sidecar):
    io, geom = sidecar["io"], sidecar["geometry"]
    state_names = list(io["state_names"])
    state_shapes = [tuple(s) for s in io["state_shapes"]]
    n_nso = int(io["num_non_state_outputs"])
    f_bins, t_export = int(geom["freq_size"]), int(geom["T_export"])
    z = np.zeros((1, f_bins, t_export), dtype=np.float32)
    spec = {k: z for k in ("bcs_mag", "bcs_pha", "acs_mag", "acs_pha")}
    states = [np.zeros(s, dtype=np.float32) for s in state_shapes]

    def step() -> None:
        feed = dict(spec)
        for nm, arr in zip(state_names, states):
            feed[nm] = arr
        out = sess.run(None, feed)
        for i, s in enumerate(out[n_nso:]):
            states[i] = s
    return step


def time_clean(np, ort, onnx_path, sidecar, args):
    sess = make_session(ort, onnx_path, threads=args.threads, inter_op=args.inter_op,
                        exec_mode=args.exec_mode, graph_opt=args.graph_opt, profile_prefix=None)
    step = make_step(np, sess, sidecar)
    for _ in range(args.warmup):
        step()
    dts = []
    for _ in range(args.clean_measure):
        t0 = time.perf_counter_ns()
        step()
        dts.append(time.perf_counter_ns() - t0)
    del sess
    gc.collect()
    return np.asarray(dts, dtype=np.float64) / 1.0e6


def profile_tier(np, ort, onnx_path, sidecar, tier_int, args):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(PROFILE_DIR / f"ort_{tier_int}ms_{args.exec_mode}_i{args.threads}x{args.inter_op}")
    sess = make_session(ort, onnx_path, threads=args.threads, inter_op=args.inter_op,
                        exec_mode=args.exec_mode, graph_opt=args.graph_opt, profile_prefix=prefix)
    step = make_step(np, sess, sidecar)
    for _ in range(args.prof_warmup + args.prof_measure):
        step()
    prof_path = sess.end_profiling()
    del sess
    gc.collect()
    return parse_profile(prof_path, args.prof_warmup)


def parse_profile(json_path: str, n_warmup: int) -> dict:
    """ORT profiling JSON -> per-op / per-branch / per-node chunk-step breakdown.

    model_run events are bucketed by ts; the first ``n_warmup`` runs are dropped
    so steady-state only. Per-node ``_kernel_time`` events inside the kept ts
    window are aggregated by op type and by mapping/masking branch.
    """
    with open(json_path) as f:
        events = json.load(f)
    runs = sorted((e for e in events if e.get("cat") == "Session" and e.get("name") == "model_run"),
                  key=lambda e: e["ts"])
    if len(runs) <= n_warmup:
        raise RuntimeError(f"{json_path}: only {len(runs)} model_run events, need > {n_warmup}")
    kept = runs[n_warmup:]
    t0 = kept[0]["ts"]
    t1 = kept[-1]["ts"] + kept[-1]["dur"]
    n = len(kept)
    run_mean_ms = sum(r["dur"] for r in kept) / 1000.0 / n

    op_us: dict = collections.defaultdict(float)
    op_cnt: dict = collections.defaultdict(int)
    branch_us: dict = collections.defaultdict(float)
    node_us: dict = collections.defaultdict(float)
    node_op: dict = {}
    for e in events:
        if e.get("cat") != "Node":
            continue
        name = e.get("name", "")
        if not name.endswith("_kernel_time"):
            continue
        ts = e["ts"]
        if ts < t0 or ts > t1:
            continue
        dur = e["dur"]
        op = e.get("args", {}).get("op_name", "?")
        node = name[: -len("_kernel_time")]
        op_us[op] += dur
        op_cnt[op] += 1
        node_us[node] += dur
        node_op[node] = op
        low = node.lower()
        if "mapping_core" in low:
            br = "mapping"
        elif "alpha" in low:
            br = "fusion"
        else:
            br = "masking+glue"
        branch_us[br] += dur

    kernel_total_ms = sum(op_us.values()) / 1000.0 / n
    return {
        "n": n,
        "run_mean_ms": run_mean_ms,
        "kernel_total_ms": kernel_total_ms,
        "gap_ms": run_mean_ms - kernel_total_ms,
        "op_ms": {op: op_us[op] / 1000.0 / n for op in op_us},
        "op_cnt": {op: op_cnt[op] / n for op in op_cnt},
        "branch_ms": {b: branch_us[b] / 1000.0 / n for b in branch_us},
        "node_ms": {nd: node_us[nd] / 1000.0 / n for nd in node_us},
        "node_op": node_op,
        "n_exec_nodes": len(node_us),
    }


def print_tier(tier_int: int, clean, prof: dict) -> None:
    tag, t_alg, chunk, t_chunk = TIERS[tier_int]
    print(f"\n{'=' * 72}")
    print(f"TIER {tag}  (chunk={chunk} frames, T_chunk={t_chunk} ms)")
    print(f"{'=' * 72}")
    print(f"  clean   : mean={clean.mean():7.3f}ms  p50={clean_p(clean,50):7.3f}  "
          f"p99={clean_p(clean,99):7.3f}  RTF={clean.mean()/t_chunk:.3f}  (n={len(clean)})")
    print(f"  profiled: model_run mean={prof['run_mean_ms']:7.3f}ms  (n={prof['n']}, "
          f"profiler-inflated -- read proportions only)")
    kt = prof["kernel_total_ms"]
    gap = prof["gap_ms"]
    rm = prof["run_mean_ms"]
    print(f"  executed nodes: {prof['n_exec_nodes']}")
    print(f"  Sum(kernel_time) = {kt:7.3f}ms  ({100*kt/rm:5.1f}%)   "
          f"executor gap = {gap:7.3f}ms  ({100*gap/rm:5.1f}%)")

    # coarse buckets
    cb: dict = collections.defaultdict(float)
    for op, ms in prof["op_ms"].items():
        cb[coarse(op)] += ms
    print(f"  coarse buckets (of kernel_time {kt:.3f}ms):")
    for b in ("conv", "plumbing", "arith"):
        v = cb.get(b, 0.0)
        print(f"     {b:10s} {v:7.3f}ms  {100*v/kt:5.1f}%")

    # per-op detail
    print(f"  per-op kernel time (top 16 of {len(prof['op_ms'])}):")
    print(f"     {'op':18s} {'exec#':>7s} {'ms/chunk':>10s} {'% kernel':>9s}")
    for op, ms in sorted(prof["op_ms"].items(), key=lambda kv: -kv[1])[:16]:
        print(f"     {op:18s} {prof['op_cnt'][op]:7.0f} {ms:10.3f} {100*ms/kt:8.1f}%")

    # branch split
    print(f"  branch split (kernel_time):")
    for b in ("mapping", "masking+glue", "fusion"):
        v = prof["branch_ms"].get(b, 0.0)
        print(f"     {b:14s} {v:7.3f}ms  {100*v/kt:5.1f}%")

    # top individual nodes
    print(f"  top 10 individual nodes (ms/chunk):")
    for nd, ms in sorted(prof["node_ms"].items(), key=lambda kv: -kv[1])[:10]:
        print(f"     {ms:7.3f}ms  [{prof['node_op'].get(nd,'?'):14s}] {nd}")


def clean_p(arr, q):
    import numpy as np
    return float(np.percentile(arr, q))


def fit_line(xs, ys):
    """Least-squares y = slope*x + intercept; returns (slope, intercept)."""
    import numpy as np
    if len(xs) < 2:
        return float("nan"), float("nan")
    s, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return float(s), float(b)


def print_cross_tier(summary: dict) -> None:
    tiers = sorted(summary)
    if len(tiers) < 2:
        return
    chunks = [TIERS[t][2] for t in tiers]
    print(f"\n{'=' * 72}")
    print("CROSS-TIER  fixed (intercept) vs per-frame (slope), least-squares")
    print(f"{'=' * 72}")
    print(f"  tiers={[TIERS[t][0] for t in tiers]}  chunk-frames={chunks}")

    def report(label, ys):
        s, b = fit_line(chunks, ys)
        vals = "  ".join(f"{y:7.3f}" for y in ys)
        print(f"  {label:22s} [{vals}]  ->  {b:7.3f}ms fixed + {s:6.3f}ms/frame")

    report("clean mean", [summary[t]["clean_mean"] for t in tiers])
    report("profiled model_run", [summary[t]["prof"]["run_mean_ms"] for t in tiers])
    report("  kernel_total", [summary[t]["prof"]["kernel_total_ms"] for t in tiers])
    report("  executor gap", [summary[t]["prof"]["gap_ms"] for t in tiers])
    for b in ("conv", "plumbing", "arith"):
        ys = []
        for t in tiers:
            cb: dict = collections.defaultdict(float)
            for op, ms in summary[t]["prof"]["op_ms"].items():
                cb[coarse(op)] += ms
            ys.append(cb.get(b, 0.0))
        report(f"  bucket:{b}", ys)


def main() -> int:
    args = parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    import onnxruntime as ort
    from src.models.streaming.onnx.export import export_bafnetplus_to_onnx_from_checkpoint

    aff = sorted(os.sched_getaffinity(0))
    print("=" * 72)
    print(f"c09 Local CPU profiling probe | onnxruntime {ort.__version__}")
    print(f"  CPU affinity: {len(aff)} cpus {aff[0]}..{aff[-1]}")
    print(f"  intra_op={args.threads} inter_op={args.inter_op} exec={args.exec_mode} "
          f"graph_opt={args.graph_opt}")
    print(f"  tiers={args.tier} warmup={args.warmup} clean={args.clean_measure} "
          f"prof={args.prof_measure}")
    print("=" * 72)

    summary: dict = {}
    for tier_int in args.tier:
        _, _, chunk, _ = TIERS[tier_int]
        onnx_path = ensure_onnx(export_bafnetplus_to_onnx_from_checkpoint, tier_int, chunk, args.chkpt_file)
        with open(f"{onnx_path}.json") as f:
            sidecar = json.load(f)
        t_start = time.time()
        clean = time_clean(np, ort, onnx_path, sidecar, args)
        prof = profile_tier(np, ort, onnx_path, sidecar, tier_int, args)
        print_tier(tier_int, clean, prof)
        print(f"  ({time.time() - t_start:.1f}s)")
        summary[tier_int] = {"clean_mean": float(clean.mean()), "prof": prof}

    print_cross_tier(summary)
    print(f"\n{'=' * 72}\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
