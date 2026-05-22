#!/usr/bin/env python3
"""c09 Phase 1 - Local CPU per-chunk inference latency, ONNX Runtime CPU EP.

Local CPU platform of the c09 cross-platform RTF matrix. Decided 2026-05-22:
Local CPU is measured on ONNX Runtime CPU EP (not the PyTorch eager wrapper) -
an investigation found ORT graph fusion makes it ~2x faster than PyTorch eager
and is the deployment-realistic CPU stack. Local GPU stays PyTorch
(bench_local_pytorch.py); Mobile DSP is ORT QNN EP (c09 isolated sweep). The
5-tier latency model family, FP32.

Per-chunk step
--------------
One timed iteration = one ORT ``session.run`` over the functional streaming
graph (4 spectrogram inputs + 190 state inputs -> 3 outputs + 190 next-states)
plus the state feedback. This is the ORT analogue of the PyTorch wrapper's
per-chunk step and of the Mobile DSP harness's ``session.run``, so the three
platforms are timed at the same granularity. The debug-only state-integrity
assertion in ``BAFNetPlusOrtStreaming._run_ort`` is intentionally excluded from
the timed region (it is not part of deployment inference).

Each tier's FP32 streaming ONNX is exported fresh from its per-tier checkpoint
(``results/experiments/bafnetplus_<T>ms``, per-tier-trained model family) via
``export_bafnetplus_to_onnx_from_checkpoint``.

Threading: ``SessionOptions.intra_op_num_threads`` (default 8 -- the calibrated
latency optimum; a thread sweep found 8 beats 12 for this small streaming
graph). Pin to one NUMA node externally with ``taskset``.

Output: ``<out-dir>/local_cpu_<tier>_run<R>.csv``  columns ``iter,dt_ns``
(no compile dimension - ORT has no torch.compile). Schema matches the Mobile
DSP isolated CSVs so ``aggregate_c09_cross_platform.py`` reuses the same loader.

Usage (from the BAFNetPlus repo root, pinned to one NUMA node):
    taskset -c 12-23 env CUDA_VISIBLE_DEVICES= python scripts/bench_local_ort_cpu.py --threads 8
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]            # BAFNetPlus/
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_EXPERIMENTS = REPO_ROOT / "results" / "experiments"
DEFAULT_ONNX = REPO_ROOT / "results" / "onnx" / "c09_ort_cpu"
DEFAULT_OUT = WORKSPACE_ROOT / "Paper_TASLP" / "tables" / "raw" / "c09_cross_platform"

# Locked 5-tier convention -- c09 meta.md "Latency tier 컨벤션 (LOCKED)".
# (tier_tag, T_alg_ms, rho, L_f, chunk_size_frames, T_chunk_ms)
TIERS = [
    ("12ms",   12.5, 0.975,  0,  2,  12.5),
    ("25ms",   25.0, 0.95,   2,  4,  25.0),
    ("50ms",   50.0, 0.90,   6,  8,  50.0),
    ("75ms",   75.0, 0.84,  10, 12,  75.0),
    ("100ms", 100.0, 0.78,  14, 16, 100.0),
]
TIER_INT_TO_TAG = {12: "12ms", 25: "25ms", 50: "50ms", 75: "75ms", 100: "100ms"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    p.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--tier", type=int, nargs="+", choices=[12, 25, 50, 75, 100],
                   default=[12, 25, 50, 75, 100])
    p.add_argument("--threads", type=int, default=8, help="ORT intra-op threads (calibrated optimum)")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--measure", type=int, default=10000)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=90.0)
    p.add_argument("--chkpt-file", default="best.th")
    return p.parse_args()


def ensure_onnx(export_fn, ckpt_dir: Path, onnx_path: Path, chunk_size: int, chkpt_file: str) -> None:
    """Export the tier's FP32 streaming ONNX if it is not already on disk."""
    if onnx_path.exists() and Path(f"{onnx_path}.json").exists():
        return
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  exporting FP32 streaming ONNX: {ckpt_dir.name} -> {onnx_path}", flush=True)
    export_fn(str(ckpt_dir), str(onnx_path), chunk_size=chunk_size, chkpt_file=chkpt_file, verbose=False)


def run_one(*, ort, np, BAFNetPlusOrtStreaming, onnx_path: Path, sidecar_path: Path,
            threads: int, warmup: int, measure: int) -> List[int]:
    """Build a fresh ORT session + wrapper, warm up, time ``measure`` chunk-steps."""
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    # Wrapper parses + validates the sidecar IO contract against the session.
    w = BAFNetPlusOrtStreaming(session=session, sidecar=sidecar, onnx_path=onnx_path, sidecar_path=sidecar_path)

    f_bins, t_export = w.freq_size, w.t_export
    n_non_state = w.num_non_state_outputs
    state_names = list(w.state_names)
    # Zero-input deterministic chunk (4 spectrogram inputs) + zero initial states.
    zero_spec = np.zeros((1, f_bins, t_export), dtype=np.float32)
    spec_inputs = {k: zero_spec for k in ("bcs_mag", "bcs_pha", "acs_mag", "acs_pha")}
    states = [np.zeros(tuple(shp), dtype=np.float32) for shp in w.state_shapes]

    def step() -> None:
        feed = dict(spec_inputs)
        for name, arr in zip(state_names, states):
            feed[name] = arr
        outputs = session.run(None, feed)
        next_states = outputs[n_non_state:]
        for i, s in enumerate(next_states):
            states[i] = s

    for _ in range(warmup):
        step()
    dts: List[int] = []
    for _ in range(measure):
        t0 = time.perf_counter_ns()
        step()
        dts.append(time.perf_counter_ns() - t0)

    del session, w
    gc.collect()
    return dts


def write_csv(path: Path, dts: List[int]) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["iter", "dt_ns"])
        for i, dt in enumerate(dts):
            wr.writerow([i, dt])


def main() -> int:
    args = parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import numpy as np
    import onnxruntime as ort
    from src.models.streaming.onnx.export import export_bafnetplus_to_onnx_from_checkpoint
    from src.models.streaming.onnx.ort_wrapper import BAFNetPlusOrtStreaming

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aff = sorted(os.sched_getaffinity(0))

    print("=" * 72)
    print(f"c09 Phase 1 Local CPU benchmark | onnxruntime {ort.__version__} | CPUExecutionProvider")
    print(f"  CPU affinity: {len(aff)} cpus {aff[0]}..{aff[-1]} | intra_op_threads={args.threads}")
    print(f"  tiers={args.tier} warmup={args.warmup} measure={args.measure} "
          f"runs={args.runs} cooldown={args.cooldown}s")
    print(f"  onnx-dir: {args.onnx_dir}")
    print(f"  out-dir : {args.out_dir}")
    print("=" * 72)

    n_total = len(args.tier) * args.runs
    done = 0
    for tier_int in args.tier:
        tag = TIER_INT_TO_TAG[tier_int]
        meta = next(t for t in TIERS if t[0] == tag)
        _, t_alg, rho, l_f, chunk_size, t_chunk = meta
        ckpt_dir = args.experiments_dir / f"bafnetplus_{tier_int}ms"
        if not (ckpt_dir / ".hydra" / "config.yaml").exists():
            print(f"ERROR: checkpoint dir missing: {ckpt_dir}", file=sys.stderr)
            return 3
        onnx_path = args.onnx_dir / f"{tag}" / f"bafnetplus_{tier_int}ms_fp32.onnx"
        ensure_onnx(export_bafnetplus_to_onnx_from_checkpoint, ckpt_dir, onnx_path, chunk_size, args.chkpt_file)
        sidecar_path = Path(f"{onnx_path}.json")

        for run in range(1, args.runs + 1):
            t_start = time.time()
            dts = run_one(
                ort=ort, np=np, BAFNetPlusOrtStreaming=BAFNetPlusOrtStreaming,
                onnx_path=onnx_path, sidecar_path=sidecar_path,
                threads=args.threads, warmup=args.warmup, measure=args.measure,
            )
            out_path = args.out_dir / f"local_cpu_{tag}_run{run}.csv"
            write_csv(out_path, dts)

            arr = np.asarray(dts, dtype=np.float64) / 1.0e6
            mean_ms = float(np.mean(arr))
            done += 1
            print(f"[{done:>2}/{n_total}] cpu {tag:>5} run{run} | "
                  f"mean={mean_ms:8.3f}ms p50={float(np.percentile(arr, 50)):8.3f} "
                  f"p99={float(np.percentile(arr, 99)):8.3f} | RTF={mean_ms / t_chunk:.4f} | "
                  f"{time.time() - t_start:6.1f}s | {out_path.name}", flush=True)

            if args.cooldown > 0 and done != n_total:
                time.sleep(args.cooldown)

    print("=" * 72)
    print(f"DONE: {done} CSV written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
