#!/usr/bin/env python3
"""c09 POC -- CUDA-graph capture of ExportableBAFNetPlusCore (Local GPU overhead probe).

The c09 Local GPU sweep found per-chunk eager-PyTorch streaming latency pinned
at ~28 ms regardless of tier -- launch-overhead-bound (hundreds of tiny conv
kernels per chunk on a near-idle RTX 5090), not compute-bound. The only lever
that removes per-launch overhead is CUDA-graph capture; torch.compile
reduce-overhead crashes because the eager streaming wrapper reassigns its
persistent ``_state`` tensors every step (stateful_conv.py: ``self._state =
...clone()``).

This probe tests the fix path: the functional ``ExportableBAFNetPlusCore``
(S8 -- explicit 190-state I/O, branch-free, the graph CP10 exports to ONNX) is
captured into a ``torch.cuda.CUDAGraph`` and replayed. It reports eager-core vs
graph-replay per-chunk latency for the 50 ms tier.

Gate: graph-replay < 12.5 ms  =>  the smallest tier's chunk budget is met, and
since replay latency is ~flat across tiers (kernel count ~constant), all 5
tiers (12.5..100 ms) would be realtime under a CUDA-graph deployment.

NOT a clean measurement -- intended to run even while training contends the
GPU; capture is a separate-process operation (no interference) and the gate
only needs (a) does capture succeed, (b) is replay in ms-land. The clean
5-tier x 3-run sweep is a separate step on an idle GPU.

Usage (from the BAFNetPlus repo root):
    CUDA_VISIBLE_DEVICES=0 python scripts/probe_cuda_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "results" / "experiments" / "bafnetplus_50ms"
CHUNK_SIZE = 8          # 50 ms tier (c09 LOCKED tier convention)
T_CHUNK_MS = 50.0
GATE_MS = 12.5          # smallest-tier budget; replay < this => all tiers realtime
WARMUP = 50
MEASURE = 500
TIERS_MS = [12.5, 25.0, 50.0, 75.0, 100.0]


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch
    from src.models.streaming.onnx.bafnetplus_core import ExportableBAFNetPlusCore
    from src.models.streaming.onnx.export import load_bafnetplus_from_checkpoint
    from src.stft import complex_to_mag_pha

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 2
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    print("=" * 72)
    print(f"c09 CUDA-graph POC | torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"  checkpoint: {CKPT_DIR}")

    # --- build the functional exportable core (50 ms tier) on the GPU
    bafnet, _info = load_bafnetplus_from_checkpoint(CKPT_DIR, "best.th", device="cpu", verbose=True)
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet)
    core = core.to(dev).eval()
    core.set_state_frames_for_update(CHUNK_SIZE)

    total_lookahead = core.total_lookahead
    t_export = CHUNK_SIZE + total_lookahead
    freq_size = core.n_fft // 2 + 1
    n_states = core.num_states
    print(f"  core: num_states={n_states}  total_lookahead={total_lookahead}  "
          f"t_export={t_export}  freq_size={freq_size}")

    # --- static I/O buffers (fixed addresses -- required for graph capture).
    # Zero complex chunk -> complex_to_mag_pha, matching the isolated-measurement
    # convention (mag carries the sqrt(...+eps) floor => strictly positive).
    zero_com = torch.zeros(1, freq_size, t_export, 2, device=dev)
    mag, pha = complex_to_mag_pha(zero_com, stack_dim=-1)
    bcs_mag, bcs_pha = mag.clone(), pha.clone()
    acs_mag, acs_pha = mag.clone(), pha.clone()
    static_states = core.init_states(
        batch_size=1, freq_size=freq_size, time_frames=t_export, device=dev
    )
    if len(static_states) != n_states:
        print(f"ERROR: init_states gave {len(static_states)} != num_states {n_states}", file=sys.stderr)
        return 2

    def call():
        return core(bcs_mag, bcs_pha, acs_mag, acs_pha, *static_states)

    def timed(fn, n):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            ev_s.record()
            fn()
            ev_e.record()
            torch.cuda.synchronize()
            ts.append(ev_s.elapsed_time(ev_e))   # ms
        ts.sort()
        return sum(ts) / len(ts), ts[len(ts) // 2], ts[min(int(len(ts) * 0.99), len(ts) - 1)]

    # --- (1) eager core forward (the "before")
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    eager_mean, eager_p50, eager_p99 = timed(call, MEASURE)
    print(f"\n[eager core  ] mean={eager_mean:8.3f}  p50={eager_p50:8.3f}  p99={eager_p99:8.3f} ms"
          f"  | RTF(50ms)={eager_mean / T_CHUNK_MS:.3f}")

    # --- (2) capture the steady-state chunk-step (forward + 190-state copy-back)
    print("\n[capture     ] side-stream warmup, then torch.cuda.CUDAGraph capture ...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(11):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            outs = core(bcs_mag, bcs_pha, acs_mag, acs_pha, *static_states)
            # Fold the 190-state copy-back INTO the captured graph: one replay
            # does forward + state advance, with zero per-chunk Python / launch
            # cost. forward reads static_states first, copy-back writes them
            # last -> correct streaming propagation across replays.
            for buf, nxt in zip(static_states, outs[3:]):
                buf.copy_(nxt)
    except Exception as exc:   # capture failure => names the remaining blocker
        print(f"[capture     ] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    print(f"[capture     ] OK -- captured forward + {n_states}-state copy-back")

    # --- (3) replay
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    replay_mean, replay_p50, replay_p99 = timed(graph.replay, MEASURE)
    finite = bool(torch.isfinite(outs[0]).all())
    print(f"[graph replay] mean={replay_mean:8.3f}  p50={replay_p50:8.3f}  p99={replay_p99:8.3f} ms"
          f"  | RTF(50ms)={replay_mean / T_CHUNK_MS:.3f}  | est_mag finite={finite}")

    # --- verdict
    print("\n" + "=" * 72)
    speedup = eager_mean / replay_mean if replay_mean > 0 else float("nan")
    print(f"  eager core   : {eager_mean:8.3f} ms")
    print(f"  graph replay : {replay_mean:8.3f} ms   ({speedup:.1f}x faster)")
    gate = replay_mean < GATE_MS
    print(f"  GATE (replay mean < {GATE_MS} ms): {'PASS' if gate else 'FAIL'}")
    print(f"  projected RTF if replay~const across tiers (replay {replay_mean:.2f} ms):")
    for tms in TIERS_MS:
        rtf = replay_mean / tms
        print(f"    {tms:6.1f} ms tier: RTF {rtf:5.3f}  {'realtime' if rtf < 1.0 else 'OVER'}")
    print("  NOTE: GPU is training-contended -- replay is an UPPER bound;")
    print("        clean 5-tier x 3-run numbers come from the idle re-measurement.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
