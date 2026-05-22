#!/usr/bin/env python3
"""c09 investigation probe -- Local CPU speedup options under ORT CPU EP.

Throwaway investigation (user request 2026-05-22): the plain FP32 ORT CPU EP
path is ~58 ms / RTF 1.17 at the 50 ms tier -- this probe measures speedup
variants of the BAFNet+ streaming graph to find headroom:

  raw      - FP32 streaming ONNX, base export (export_bafnetplus_to_onnx_from_checkpoint)
  onnxsim  - raw + onnxsim graph simplification (FP32, no precision change)
  int8dyn  - raw + ORT dynamic INT8 weight quantization (quantize_dynamic;
             AMX-INT8 / VNNI accelerated on this Sapphire Rapids Xeon)

One timed iteration = one ORT session.run over the functional streaming graph
+ 190-state feedback (same chunk-step as bench_local_ort_cpu.py). Pin
externally with taskset; hide the GPU with CUDA_VISIBLE_DEVICES=. NOT a c09
deliverable script.

Usage (from the BAFNetPlus repo root):
    taskset -c 12-23 env CUDA_VISIBLE_DEVICES= python scripts/probe_cpu_speedup.py --tier 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/tmp/c09_cpu_speedup")
TIER = {12: (2, 12.5), 25: (4, 25.0), 50: (8, 50.0), 75: (12, 75.0), 100: (16, 100.0)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", type=int, default=50, choices=[12, 25, 50, 75, 100])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=150)
    p.add_argument("--variants", nargs="+", default=["raw", "onnxsim", "int8dyn"])
    return p.parse_args()


def make_onnxsim(src: Path, dst: Path) -> None:
    import onnx
    import onnxsim
    model = onnx.load(str(src))
    simplified, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError("onnxsim validity check failed")
    onnx.save(simplified, str(dst))


def make_int8dyn(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


class _CalibReader:
    """Minimal CalibrationDataReader for the speed-only static-INT8 probe.

    Yields a few random-spectrogram / zero-state samples -- enough to give the
    static quantizer non-degenerate activation ranges. Calibration *quality* is
    irrelevant here: this probe measures INT8 latency, not INT8 accuracy.
    """

    def __init__(self, sidecar: dict, n_samples: int = 8) -> None:
        import numpy as np
        io, geom = sidecar["io"], sidecar["geometry"]
        f_bins, t_export = int(geom["freq_size"]), int(geom["T_export"])
        state_names = list(io["state_names"])
        state_shapes = [tuple(s) for s in io["state_shapes"]]
        rng = np.random.default_rng(0)
        self._samples = []
        for _ in range(n_samples):
            feed = {k: rng.standard_normal((1, f_bins, t_export)).astype(np.float32)
                    for k in ("bcs_mag", "bcs_pha", "acs_mag", "acs_pha")}
            for nm, shp in zip(state_names, state_shapes):
                feed[nm] = np.zeros(shp, dtype=np.float32)
            self._samples.append(feed)
        self._i = 0

    def get_next(self):
        if self._i >= len(self._samples):
            return None
        s = self._samples[self._i]
        self._i += 1
        return s

    def rewind(self) -> None:
        self._i = 0


def make_int8static(src: Path, dst: Path, sidecar: dict) -> None:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    quantize_static(
        str(src), str(dst), _CalibReader(sidecar),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )


def bench(onnx_path: Path, sidecar: dict, threads: int, warmup: int, measure: int,
          providers=None, provider_options=None):
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    kwargs = {"sess_options": so, "providers": providers or ["CPUExecutionProvider"]}
    if provider_options is not None:
        kwargs["provider_options"] = provider_options
    sess = ort.InferenceSession(str(onnx_path), **kwargs)

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
        for name, arr in zip(state_names, states):
            feed[name] = arr
        outputs = sess.run(None, feed)
        for i, s in enumerate(outputs[n_nso:]):
            states[i] = s

    for _ in range(warmup):
        step()
    dts = []
    for _ in range(measure):
        t0 = time.perf_counter_ns()
        step()
        dts.append(time.perf_counter_ns() - t0)
    return np.asarray(dts, dtype=np.float64) / 1.0e6


def main() -> int:
    args = parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    import onnx

    WORK.mkdir(parents=True, exist_ok=True)
    chunk, t_chunk_ms = TIER[args.tier]
    base = REPO_ROOT / "results" / "onnx" / "c09_ort_cpu" / f"{args.tier}ms" / f"bafnetplus_{args.tier}ms_fp32.onnx"
    if not base.exists():
        print(f"ERROR: base ONNX missing: {base} (run bench_local_ort_cpu.py first)", file=sys.stderr)
        return 3
    sidecar = json.load(open(f"{base}.json"))

    variant_path = {"raw": base}
    variant_ep: dict = {}  # variant -> (providers, provider_options); default = CPU EP
    if "onnxsim" in args.variants:
        p = WORK / f"{args.tier}ms_onnxsim.onnx"
        if not p.exists():
            print("  building onnxsim variant ...", flush=True)
            make_onnxsim(base, p)
        variant_path["onnxsim"] = p
    if "int8dyn" in args.variants:
        p = WORK / f"{args.tier}ms_int8dyn.onnx"
        if not p.exists():
            print("  building int8dyn variant ...", flush=True)
            make_int8dyn(base, p)
        variant_path["int8dyn"] = p
    if "int8static" in args.variants:
        p = WORK / f"{args.tier}ms_int8static.onnx"
        if not p.exists():
            print("  building int8static variant (QDQ, dummy calibration) ...", flush=True)
            make_int8static(base, p, sidecar)
        variant_path["int8static"] = p
    if "openvino" in args.variants:
        # Same FP32 ONNX; only the execution provider changes (OpenVINO EP, CPU).
        variant_path["openvino"] = base
        variant_ep["openvino"] = (
            ["OpenVINOExecutionProvider"],
            [{"device_type": "CPU", "num_of_threads": str(args.threads)}],
        )

    print(f"### CPU speedup probe — tier {args.tier} ms, ORT CPU EP, intra_op={args.threads}", flush=True)
    results = {}
    for v in args.variants:
        p = variant_path[v]
        nodes = len(onnx.load(str(p)).graph.node)
        size_mb = p.stat().st_size / 1.0e6
        providers, provider_options = variant_ep.get(v, (None, None))
        a = bench(p, sidecar, args.threads, args.warmup, args.measure, providers, provider_options)
        results[v] = float(a.mean())
        print(f"[{v:9s}] {nodes:5d} nodes | {size_mb:6.1f} MB | "
              f"mean={a.mean():7.2f}ms p50={np.percentile(a, 50):7.2f} "
              f"p99={np.percentile(a, 99):7.2f} | RTF={a.mean() / t_chunk_ms:.3f}", flush=True)

    if "raw" in results:
        for v in args.variants:
            if v != "raw":
                print(f"  speedup {v:9s} vs raw: {results['raw'] / results[v]:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
