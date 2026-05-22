#!/usr/bin/env python3
"""c09 investigation probe -- ONNX Runtime CPU EP vs PyTorch eager, per-chunk latency.

Throwaway investigation for the "ONNX-Local" framework question: does running
the BAFNet+ streaming model under ONNX Runtime CPU EP beat the PyTorch eager
streaming wrapper on the local Xeon? One (backend, thread-count) config per
invocation -- pin externally with taskset, hide the GPU with
CUDA_VISIBLE_DEVICES="". NOT a c09 deliverable script.

  pt  -- BAFNetPlusStreaming (S6 PT wrapper); one chunk-step = the body of
         process_spectrogram's inner loop (same as bench_local_pytorch.py).
  ort -- BAFNetPlusOrtStreaming (S9 ORT wrapper); exports a fresh FP32 streaming
         ONNX from the per-tier checkpoint, runs CPU EP; one chunk-step =
         _run_ort (one session.run + state feedback).

Usage (from the BAFNetPlus repo root):
    taskset -c 12-23 env CUDA_VISIBLE_DEVICES= python scripts/probe_ort_cpu.py \\
        --backend ort --tier 50 --threads 8 --warmup 20 --measure 150
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
ORT_DIR = Path("/tmp/c09_ort_probe")
# tier -> (chunk_size_frames, T_chunk_ms)
TIER = {12: (2, 12.5), 25: (4, 25.0), 50: (8, 50.0), 75: (12, 75.0), 100: (16, 100.0)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["pt", "ort"], required=True)
    p.add_argument("--tier", type=int, default=50, choices=[12, 25, 50, 75, 100])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=150)
    return p.parse_args()


def bench_pt(tier: int, chunk: int, threads: int, warmup: int, measure: int) -> List[int]:
    import torch
    from src.models.streaming.bafnetplus_streaming import BAFNetPlusStreaming
    from src.stft import complex_to_mag_pha, mag_pha_to_complex

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    ckpt = REPO_ROOT / "results" / "experiments" / f"bafnetplus_{tier}ms"
    w = BAFNetPlusStreaming.from_checkpoint(str(ckpt), chunk_size=chunk, device="cpu", verbose=False)
    w.eval()
    f_bins = w.n_fft // 2 + 1
    t_in = chunk + w.encoder_lookahead
    zero = torch.zeros(1, f_bins, t_in, 2)
    bcs, acs, fifo, run_fusion = w.bcs_streaming, w.acs_streaming, w._input_mag_fifo, w._run_fusion

    def step():
        bm, _ = complex_to_mag_pha(zero, stack_dim=-1)
        am, _ = complex_to_mag_pha(zero, stack_dim=-1)
        fifo.append((bm[:, :, :chunk].clone(), am[:, :, :chunk].clone()))
        br = bcs.process_spectrogram_buffered(zero)
        ar = acs.process_spectrogram_buffered(zero)
        if br is None or ar is None:
            return None
        bmi, ami = fifo.popleft()
        est_mag, est_pha = run_fusion(*br, *ar, bmi, ami)
        return mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)

    w.reset_state()
    dts: List[int] = []
    with torch.inference_mode():
        for _ in range(warmup):
            step()
        for _ in range(measure):
            t0 = time.perf_counter_ns()
            step()
            dts.append(time.perf_counter_ns() - t0)
    return dts


def bench_ort(tier: int, chunk: int, threads: int, warmup: int, measure: int) -> List[int]:
    import torch
    import onnxruntime as ort
    from src.models.streaming.onnx.export import export_bafnetplus_to_onnx_from_checkpoint
    from src.models.streaming.onnx.ort_wrapper import BAFNetPlusOrtStreaming

    ORT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ORT_DIR / f"bafnetplus_{tier}ms_fp32.onnx"
    sidecar_path = Path(f"{onnx_path}.json")
    if not (onnx_path.exists() and sidecar_path.exists()):
        ckpt = REPO_ROOT / "results" / "experiments" / f"bafnetplus_{tier}ms"
        print(f"  [ort] exporting fresh FP32 streaming ONNX from {ckpt.name} ...", flush=True)
        export_bafnetplus_to_onnx_from_checkpoint(str(ckpt), str(onnx_path), chunk_size=chunk, verbose=False)

    sidecar = json.load(open(sidecar_path))
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    w = BAFNetPlusOrtStreaming(session=session, sidecar=sidecar, onnx_path=onnx_path, sidecar_path=sidecar_path)

    zmag = torch.zeros(1, w.freq_size, w.t_export)
    zpha = torch.zeros(1, w.freq_size, w.t_export)
    w.reset_state()
    dts: List[int] = []
    for _ in range(warmup):
        w._run_ort(zmag, zpha, zmag, zpha)
    for _ in range(measure):
        t0 = time.perf_counter_ns()
        w._run_ort(zmag, zpha, zmag, zpha)
        dts.append(time.perf_counter_ns() - t0)
    return dts


def main() -> int:
    args = parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import numpy as np

    chunk, t_chunk_ms = TIER[args.tier]
    fn = bench_pt if args.backend == "pt" else bench_ort
    dts = fn(args.tier, chunk, args.threads, args.warmup, args.measure)

    a = np.asarray(dts, dtype=np.float64) / 1.0e6  # ns -> ms
    print(
        f"[{args.backend:3s} tier{args.tier:>3}ms threads={args.threads:>2}] "
        f"mean={a.mean():8.3f}ms  p50={np.percentile(a, 50):8.3f}  "
        f"p99={np.percentile(a, 99):8.3f}  RTF={a.mean() / t_chunk_ms:6.3f}  (n={len(a)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
