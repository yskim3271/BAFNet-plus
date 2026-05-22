#!/usr/bin/env python3
"""c09 Phase 1 - Local CPU/GPU per-chunk streaming inference latency benchmark.

Measures the per-chunk inference latency of the BAFNet+ PyTorch streaming
wrapper (``BAFNetPlusStreaming``) for the 5-tier latency model family on the
local server (RTX 5090 GPU / Xeon Silver 4510 CPU), FP32, ``torch.compile`` ON
and OFF. Feeds the c09 cross-platform RTF matrix (new manuscript section
"Cross-Platform Model Family Deployment", ``tab:cross_platform``).

Per-chunk step
--------------
``BAFNetPlusStreaming.process_spectrogram`` is a *whole-utterance* driver (it
resets state then loops the time axis in chunk_size steps). For a per-chunk
latency measurement we time **one streaming chunk-step** -- exactly the body of
that driver's inner loop: push the per-call input mag onto the alignment FIFO,
run both branches' ``process_spectrogram_buffered``, then the BAFNet+ fusion
algebra (``_run_fusion``) on the matured pair. Streaming state persists across
iterations (no per-step reset); the warm-up steps fill the look-ahead pipeline
and reach steady state. This mirrors the Mobile DSP harness (one ORT
``session.run`` per chunk-step), so the three platforms are timed at the same
granularity.

Each latency tier is a SEPARATELY trained model -- loaded from its own
``results/experiments/bafnetplus_<T>ms`` checkpoint via
``BAFNetPlusStreaming.from_checkpoint`` (per-tier-trained model family; rho is
baked into the checkpoint, not a runtime override).

Tier convention (LOCKED -- c09 meta.md):
    T_alg   rho     L_f   chunk_size   T_chunk
    12.5    0.975   0     2            12.5 ms
    25      0.95    2     4            25   ms
    50      0.90    6     8            50   ms   (deployment default, D3)
    75      0.84    10    12           75   ms
    100     0.78    14    16           100  ms

Output
------
One raw per-chunk CSV per (device, tier, compile, run):
    <out-dir>/local_<dev>_<tier>_<compile>_run<R>.csv     columns: iter,dt_ns
    dev      in {gpu, cpu}
    tier     in {12ms,25ms,50ms,75ms,100ms}
    compile  in {eager, compile}
    R        in {1,2,3}
The ``dt_ns`` schema matches the Mobile DSP isolated CSVs, so
``aggregate_c09_cross_platform.py`` reuses the same loader.

Timing: GPU = ``torch.cuda.Event`` + ``synchronize()``; CPU = ``perf_counter_ns()``.

Usage (run from the BAFNetPlus repo root)
-----------------------------------------
    # Local GPU -- full sweep (5 tier x 2 compile x 3 run = 30 CSV)
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_local_pytorch.py --device cuda

    # Local CPU -- 1-socket NUMA pin (node0 = 12 physical cores 0-11)
    taskset -c 0-11 python scripts/bench_local_pytorch.py --device cpu --threads 12

    # Smoke test (fast)
    python scripts/bench_local_pytorch.py --device cpu --tier 50 --compile both \\
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
from typing import List, Optional

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
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--tier", type=int, nargs="+", choices=[12, 25, 50, 75, 100],
        default=[12, 25, 50, 75, 100], help="latency tiers to sweep (checkpoint suffix)",
    )
    p.add_argument("--compile", choices=["off", "on", "both"], default="both")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--measure", type=int, default=10000)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=90.0, help="seconds between cold runs")
    p.add_argument("--threads", type=int, default=12, help="CPU intra-op threads (device=cpu)")
    p.add_argument("--chkpt-file", default="best.th")
    return p.parse_args()


def run_one(
    *,
    torch,
    BAFNetPlusStreaming,
    complex_to_mag_pha,
    mag_pha_to_complex,
    ckpt_dir: Path,
    chkpt_file: str,
    chunk_size: int,
    device: str,
    use_compile: bool,
    warmup: int,
    measure: int,
) -> List[int]:
    """Build a wrapper, warm up, and time ``measure`` per-chunk steps. Returns dt_ns list."""
    wrapper = BAFNetPlusStreaming.from_checkpoint(
        str(ckpt_dir), chkpt_file=chkpt_file, chunk_size=chunk_size,
        device=device, verbose=False,
    )
    wrapper.eval()
    if wrapper.chunk_size != chunk_size:
        raise RuntimeError(f"chunk_size mismatch: wrapper={wrapper.chunk_size} expected={chunk_size}")

    f_bins = wrapper.n_fft // 2 + 1
    t_in = chunk_size + wrapper.encoder_lookahead          # frames per chunk-step input slice
    valid = chunk_size                                     # FIFO valid-frame trim (= min(chunk_size, t_in))
    dev = torch.device(device)
    zero_chunk = torch.zeros(1, f_bins, t_in, 2, device=dev)

    bcs = wrapper.bcs_streaming
    acs = wrapper.acs_streaming
    fifo = wrapper._input_mag_fifo
    run_fusion = wrapper._run_fusion

    def step():
        """One streaming chunk-step = body of process_spectrogram's inner loop."""
        bcs_mag, _ = complex_to_mag_pha(zero_chunk, stack_dim=-1)
        acs_mag, _ = complex_to_mag_pha(zero_chunk, stack_dim=-1)
        fifo.append((bcs_mag[:, :, :valid].clone(), acs_mag[:, :, :valid].clone()))
        bcs_result = bcs.process_spectrogram_buffered(zero_chunk)
        acs_result = acs.process_spectrogram_buffered(zero_chunk)
        if bcs_result is None or acs_result is None:
            return None
        bcs_mag_input, acs_mag_input = fifo.popleft()
        bcs_est_mag, bcs_est_pha, bcs_est_com = bcs_result
        acs_est_mag, acs_est_pha, acs_est_com = acs_result
        est_mag, est_pha = run_fusion(
            bcs_est_mag, bcs_est_pha, bcs_est_com,
            acs_est_mag, acs_est_pha, acs_est_com,
            bcs_mag_input, acs_mag_input,
        )
        return mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)

    # torch.compile: default backend (NOT mode="reduce-overhead" -- its CUDA
    # graphs overwrite the streaming wrapper's persistent _state tensors on
    # replay, which raises at runtime). dynamic=False specialises each of the
    # wrapper's fixed conv shapes; main() raises the Dynamo recompile limit so
    # every shape compiles and caches instead of falling back to eager.
    fn = torch.compile(step, dynamic=False) if use_compile else step
    is_cuda = device == "cuda"

    wrapper.reset_state()
    none_count = 0
    dts: List[int] = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if is_cuda:
            torch.cuda.synchronize()
        if is_cuda:
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            for _ in range(measure):
                ev_s.record()
                r = fn()
                ev_e.record()
                torch.cuda.synchronize()
                dts.append(int(round(ev_s.elapsed_time(ev_e) * 1.0e6)))   # ms -> ns
                if r is None:
                    none_count += 1
        else:
            for _ in range(measure):
                t0 = time.perf_counter_ns()
                r = fn()
                dts.append(time.perf_counter_ns() - t0)
                if r is None:
                    none_count += 1

    if none_count:
        print(f"    WARNING: {none_count}/{measure} measured steps did not mature "
              f"(warmup={warmup} too short for this tier?)")

    del wrapper, fn, step
    gc.collect()
    if is_cuda:
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

    if args.device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import numpy as np
    import torch
    from src.models.streaming.bafnetplus_streaming import BAFNetPlusStreaming
    from src.stft import complex_to_mag_pha, mag_pha_to_complex

    torch.manual_seed(0)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # inter-op pool already initialised this process

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda but torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    compiles = {"off": [False], "on": [True], "both": [False, True]}[args.compile]
    devtag = "gpu" if args.device == "cuda" else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if True in compiles:
        # The streaming wrapper routes many distinct conv shapes through the same
        # forward code; the default recompile limit (8) thrashes and falls back
        # to eager. The shape set is fixed and finite, so a generous limit lets
        # all of them compile once and stay cached.
        import torch._dynamo
        torch._dynamo.config.recompile_limit = 256
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 256

    print("=" * 72)
    print(f"c09 Phase 1 local benchmark | torch {torch.__version__} | device={args.device}")
    if args.device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} | "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    else:
        aff = sorted(os.sched_getaffinity(0))
        print(f"  CPU affinity: {len(aff)} cpus {aff[0]}..{aff[-1]} | "
              f"intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()} | "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"  tiers={args.tier} compile={compiles} warmup={args.warmup} "
          f"measure={args.measure} runs={args.runs} cooldown={args.cooldown}s")
    print(f"  out-dir: {args.out_dir}")
    print("=" * 72)

    cells = [(t, c) for t in args.tier for c in compiles]
    n_total_runs = len(cells) * args.runs
    done = 0

    for tier_int, use_compile in cells:
        tag = TIER_INT_TO_TAG[tier_int]
        meta = next(t for t in TIERS if t[0] == tag)
        _, t_alg, rho, l_f, chunk_size, t_chunk = meta
        ctag = "compile" if use_compile else "eager"
        ckpt_dir = args.experiments_dir / f"bafnetplus_{tier_int}ms"
        if not (ckpt_dir / ".hydra" / "config.yaml").exists():
            print(f"ERROR: checkpoint dir missing: {ckpt_dir}", file=sys.stderr)
            return 3

        for run in range(1, args.runs + 1):
            t_start = time.time()
            dts = run_one(
                torch=torch,
                BAFNetPlusStreaming=BAFNetPlusStreaming,
                complex_to_mag_pha=complex_to_mag_pha,
                mag_pha_to_complex=mag_pha_to_complex,
                ckpt_dir=ckpt_dir,
                chkpt_file=args.chkpt_file,
                chunk_size=chunk_size,
                device=args.device,
                use_compile=use_compile,
                warmup=args.warmup,
                measure=args.measure,
            )
            out_path = args.out_dir / f"local_{devtag}_{tag}_{ctag}_run{run}.csv"
            write_csv(out_path, dts)

            arr = np.asarray(dts, dtype=np.float64) / 1.0e6   # ns -> ms
            mean_ms = float(np.mean(arr))
            p50 = float(np.percentile(arr, 50))
            p99 = float(np.percentile(arr, 99))
            done += 1
            print(f"[{done:>2}/{n_total_runs}] {devtag} {tag:>5} {ctag:<7} run{run} | "
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
