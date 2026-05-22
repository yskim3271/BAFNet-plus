#!/usr/bin/env python3
"""c09 Phase 1 - Local GPU per-chunk streaming latency benchmark (CUDA-graph path).

Companion to ``bench_local_pytorch.py`` (eager PyTorch) and
``bench_local_ort_cpu.py`` (ORT CPU EP). This script measures the Local GPU
per-chunk streaming inference latency of the 5-tier BAFNet+ latency model
family with the **CUDA-graph** deployment path:
:class:`~src.models.streaming.cuda_graph_streaming.BAFNetPlusCudaGraphStreaming`
captures one steady-state chunk-step (forward + 190-state copy-back) of the
functional ``ExportableBAFNetPlusCore`` into a ``torch.cuda.CUDAGraph`` and
replays it per chunk.

Motivation: the eager PyTorch sweep (``bench_local_pytorch.py``) is
launch-overhead-bound -- per-chunk latency is pinned at ~28 ms regardless of
tier (hundreds of tiny conv kernels dispatched through the Python eager loop on
a near-idle RTX 5090), so the 12.5 / 25 ms tiers miss real time. CUDA-graph
replay collapses those hundreds of launches into one, removing the CPU-side
launch overhead. This sweep produces the realistic Local GPU column of the c09
cross-platform RTF matrix.

Per-chunk step
--------------
One streaming chunk-step = one ``BAFNetPlusCudaGraphStreaming.step()`` = copy
the (deterministic zero-input) chunk into the static GPU buffers, then one
``graph.replay()``. This matches the Mobile DSP harness granularity (one ORT
``session.run`` per chunk-step) and the eager harness (one ``step()``).

Each latency tier is a SEPARATELY trained model -- loaded from its own
``results/experiments/bafnetplus_<T>ms`` checkpoint (per-tier-trained model
family). The functional core is built in ``phase_output_mode='complex'`` (the
deployed s20 graph CP10 exports to ONNX).

Tier convention (LOCKED -- c09 meta.md):
    T_alg   rho     L_f   chunk_size   T_chunk
    12.5    0.975   0     2            12.5 ms
    25      0.95    2     4            25   ms
    50      0.90    6     8            50   ms   (deployment default, D3)
    75      0.84    10    12           75   ms
    100     0.78    14    16           100  ms

Output
------
One raw per-chunk CSV per (tier, run):
    <out-dir>/local_gpu_<tier>_cudagraph_run<R>.csv     columns: iter,dt_ns
The ``dt_ns`` schema matches the eager / Mobile DSP CSVs, so
``aggregate_c09_cross_platform.py`` reuses the same loader.

Timing: ``torch.cuda.Event`` + ``synchronize()``.

Usage (run from the BAFNetPlus repo root)
-----------------------------------------
    # Full sweep (5 tier x 3 run = 15 CSV)
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_local_cuda_graph.py

    # Smoke test (fast)
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_local_cuda_graph.py \\
        --warmup 20 --measure 100 --runs 1 --cooldown 0
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]            # BAFNetPlus/
WORKSPACE_ROOT = REPO_ROOT.parent                          # BAFNet-plus/
DEFAULT_EXPERIMENTS = REPO_ROOT / "results" / "experiments"
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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--tier", type=int, nargs="+", choices=[12, 25, 50, 75, 100],
        default=[12, 25, 50, 75, 100], help="latency tiers to sweep (checkpoint suffix)",
    )
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--measure", type=int, default=10000)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=90.0, help="seconds between cold runs")
    p.add_argument("--chkpt-file", default="best.th")
    return p.parse_args()


def run_one(
    *,
    torch,
    BAFNetPlusCudaGraphStreaming,
    complex_to_mag_pha,
    ckpt_dir: Path,
    chkpt_file: str,
    chunk_size: int,
    warmup: int,
    measure: int,
) -> List[int]:
    """Build a CUDA-graph wrapper, warm up, and time ``measure`` per-chunk steps."""
    wrapper = BAFNetPlusCudaGraphStreaming.from_checkpoint(
        str(ckpt_dir), chunk_size, chkpt_file=chkpt_file, device="cuda", verbose=False,
    )
    if wrapper.chunk_size != chunk_size:
        raise RuntimeError(f"chunk_size mismatch: wrapper={wrapper.chunk_size} expected={chunk_size}")

    dev = torch.device("cuda")
    # Deterministic zero-input chunk (spectrogram domain) -> mag/pha via the same
    # complex_to_mag_pha boundary the eager / Mobile DSP harnesses use.
    zero_com = torch.zeros(1, wrapper.freq_size, wrapper.t_export, 2, device=dev)
    mag, pha = complex_to_mag_pha(zero_com, stack_dim=-1)
    bcs_mag, bcs_pha = mag.clone(), pha.clone()
    acs_mag, acs_pha = mag.clone(), pha.clone()

    wrapper.reset_state()
    out = None
    for _ in range(warmup):
        out = wrapper.step(bcs_mag, bcs_pha, acs_mag, acs_pha)
    torch.cuda.synchronize()
    if out is not None and not bool(torch.isfinite(out[0]).all()):
        print(f"    WARNING: non-finite est_mag after warmup (tier chunk_size={chunk_size})")

    dts: List[int] = []
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    for _ in range(measure):
        ev_s.record()
        wrapper.step(bcs_mag, bcs_pha, acs_mag, acs_pha)
        ev_e.record()
        torch.cuda.synchronize()
        dts.append(int(round(ev_s.elapsed_time(ev_e) * 1.0e6)))   # ms -> ns

    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    return dts


def write_csv(path: Path, dts: List[int]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "dt_ns"])
        for i, dt in enumerate(dts):
            w.writerow([i, dt])


def main() -> int:
    args = parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import numpy as np
    import torch
    from src.models.streaming.cuda_graph_streaming import BAFNetPlusCudaGraphStreaming
    from src.stft import complex_to_mag_pha

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"c09 Phase 1 Local GPU CUDA-graph benchmark | torch {torch.__version__}")
    print(f"  GPU: {torch.cuda.get_device_name(0)} | "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print(f"  tiers={args.tier} warmup={args.warmup} measure={args.measure} "
          f"runs={args.runs} cooldown={args.cooldown}s")
    print(f"  out-dir: {args.out_dir}")
    print("=" * 72)

    n_total_runs = len(args.tier) * args.runs
    done = 0

    for tier_int in args.tier:
        tag = TIER_INT_TO_TAG[tier_int]
        meta = next(t for t in TIERS if t[0] == tag)
        _, t_alg, rho, l_f, chunk_size, t_chunk = meta
        ckpt_dir = args.experiments_dir / f"bafnetplus_{tier_int}ms"
        if not (ckpt_dir / ".hydra" / "config.yaml").exists():
            print(f"ERROR: checkpoint dir missing: {ckpt_dir}", file=sys.stderr)
            return 3

        for run in range(1, args.runs + 1):
            t_start = time.time()
            dts = run_one(
                torch=torch,
                BAFNetPlusCudaGraphStreaming=BAFNetPlusCudaGraphStreaming,
                complex_to_mag_pha=complex_to_mag_pha,
                ckpt_dir=ckpt_dir,
                chkpt_file=args.chkpt_file,
                chunk_size=chunk_size,
                warmup=args.warmup,
                measure=args.measure,
            )
            out_path = args.out_dir / f"local_gpu_{tag}_cudagraph_run{run}.csv"
            write_csv(out_path, dts)

            arr = np.asarray(dts, dtype=np.float64) / 1.0e6   # ns -> ms
            mean_ms = float(np.mean(arr))
            p50 = float(np.percentile(arr, 50))
            p99 = float(np.percentile(arr, 99))
            done += 1
            print(f"[{done:>2}/{n_total_runs}] gpu {tag:>5} cudagraph run{run} | "
                  f"mean={mean_ms:8.4f}ms p50={p50:8.4f} p99={p99:8.4f} | "
                  f"RTF={mean_ms / t_chunk:.4f} | {time.time() - t_start:5.1f}s | "
                  f"{out_path.name}")

            is_last = done == n_total_runs
            if args.cooldown > 0 and not is_last:
                time.sleep(args.cooldown)

    print("=" * 72)
    print(f"DONE: {done} CSV written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
