#!/usr/bin/env python3
"""c09 investigation probe -- Local CPU ORT threading / execution-mode levers.

Follows the profile_ort_cpu.py diagnosis: the FP32 ORT CPU EP chunk-step is
~25% Conv, ~15% MLAS layout reorders (ReorderInput/Output), ~34% streaming
Slice/Concat plumbing, and carries a tier-flat ~13 ms ORT sequential-executor
dispatch gap over ~3000 nodes. Two levers the 2026-05-22 speedup probe never
tested:

  intra-op floor : intra_op_num_threads incl. 1 / 2. An overhead-bound
                   tiny-tensor graph can be FASTER with fewer threads -- less
                   per-node fork/join. 2026-05-22 only swept 4 / 8 / 12.
  inter-op       : ExecutionMode.ORT_PARALLEL + inter_op>1. mapping_core and
                   the second backbone are data-independent until the alpha
                   fusion, so a parallel executor may overlap them.

Clean timing (no profiler), one (intra,inter,exec) config per row, all rows
in one process for a fair comparison. Pin externally with taskset; hide the
GPU with CUDA_VISIBLE_DEVICES=. NOT a c09 deliverable script.

Usage (from the BAFNetPlus repo root):
    taskset -c 12-23 env CUDA_VISIBLE_DEVICES= python scripts/probe_ort_cpu_levers.py --tier 50
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

# (kind, intra_op, inter_op, exec_mode)
CONFIGS = [
    ("seq", 1, 1, "sequential"),
    ("seq", 2, 1, "sequential"),
    ("seq", 4, 1, "sequential"),
    ("seq", 6, 1, "sequential"),
    ("seq", 8, 1, "sequential"),   # 2026-05-22 calibrated optimum -- reference row
    ("seq", 12, 1, "sequential"),
    ("par", 2, 2, "parallel"),
    ("par", 3, 2, "parallel"),
    ("par", 4, 2, "parallel"),
    ("par", 6, 2, "parallel"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", type=int, nargs="+", choices=[12, 25, 50, 75, 100], default=[50])
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--measure", type=int, default=400)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for pth in (str(REPO_ROOT), str(SCRIPTS_DIR)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    import numpy as np
    import onnxruntime as ort
    from src.models.streaming.onnx.export import export_bafnetplus_to_onnx_from_checkpoint
    from profile_ort_cpu import TIERS, ensure_onnx, make_session, make_step

    print("=" * 72)
    print(f"c09 Local CPU lever probe | onnxruntime {ort.__version__} | warmup={args.warmup} "
          f"measure={args.measure}")
    print("=" * 72)

    for tier_int in args.tier:
        tag, t_alg, chunk, t_chunk = TIERS[tier_int]
        onnx_path = ensure_onnx(export_bafnetplus_to_onnx_from_checkpoint, tier_int, chunk, "best.th")
        with open(f"{onnx_path}.json") as f:
            sidecar = json.load(f)
        print(f"\n### tier {tag}  chunk={chunk} frames  T_chunk={t_chunk} ms  "
              f"(realtime needs mean < {t_chunk} ms)")
        print(f"  {'config':24s} {'mean':>10s} {'p50':>9s} {'p99':>9s} {'RTF':>8s}")
        base = None
        for kind, intra, inter, mode in CONFIGS:
            sess = make_session(ort, onnx_path, threads=intra, inter_op=inter,
                                exec_mode=mode, graph_opt="all", profile_prefix=None)
            step = make_step(np, sess, sidecar)
            for _ in range(args.warmup):
                step()
            dts = []
            for _ in range(args.measure):
                t0 = time.perf_counter_ns()
                step()
                dts.append(time.perf_counter_ns() - t0)
            a = np.asarray(dts, dtype=np.float64) / 1.0e6
            mean = float(a.mean())
            if intra == 8 and mode == "sequential":
                base = mean
            label = f"{mode[:3]} intra={intra} inter={inter}"
            delta = f"  ({mean / base:.2f}x base)" if base and base != mean else ""
            print(f"  {label:24s} {mean:9.3f}ms {np.percentile(a, 50):8.3f} "
                  f"{np.percentile(a, 99):8.3f} {mean / t_chunk:8.3f}{delta}", flush=True)
            del sess
            gc.collect()
    print("\n" + "=" * 72 + "\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
