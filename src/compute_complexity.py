"""Model complexity measurement for BAFNet+ paper (CR5).

Computes per-model Params, MACs, and model size for all trained experiments
under `results/experiments/`. Also supports optional real-time factor (RTF)
measurement on CPU and GPU.

Usage:
    # All experiments
    python -m src.compute_complexity --all \\
        --output_json paper_work/tables/complexity.json

    # Single experiment
    python -m src.compute_complexity --experiment bafnetplus_50ms

    # Include RTF (requires free GPU)
    python -m src.compute_complexity --all --measure_rtf \\
        --rtf_duration 10 --rtf_warmup 5 --rtf_runs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stft import mag_pha_stft  # noqa: E402
from src.utils import load_model, load_checkpoint, get_stft_args_from_config  # noqa: E402


POD_PREFIX = "/workspace"
LOCAL_PREFIX = "/home/yskim/workspace"

logger = logging.getLogger(__name__)


def _rewrite_paths(cfg: OmegaConf) -> OmegaConf:
    """Rewrite pod paths to local paths in-place inside config."""
    text = OmegaConf.to_yaml(cfg).replace(POD_PREFIX, LOCAL_PREFIX)
    return OmegaConf.create(text)


def _load_config(exp_dir: Path) -> OmegaConf:
    cfg_path = exp_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    return _rewrite_paths(cfg)


def _dummy_stft_inputs(
    duration_sec: float,
    stft_args: Dict[str, Any],
    device: str,
) -> torch.Tensor:
    """Return a dummy complex spectrogram shaped [1, F, T, 2]."""
    sr = 16000
    n_samples = int(duration_sec * sr)
    wav = torch.randn(1, n_samples, device=device)
    _, _, com = mag_pha_stft(wav, **stft_args)
    return com


def _count_params(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _model_size_mb(total_params: int, dtype_bytes: int = 4) -> float:
    return total_params * dtype_bytes / (1024 ** 2)


def _measure_macs(
    model: torch.nn.Module,
    dummy_input: Any,
    input_type: str,
) -> Optional[float]:
    """Compute MACs using thop (returns MACs as float). Returns None on failure."""
    from thop import profile

    model.eval()
    try:
        with torch.no_grad():
            # thop expects inputs=(x,) for a single-arg forward.
            # For tuple-forward models (BAFNet/BAFNetPlus), wrap tuple inside tuple.
            if input_type == "acs+bcs":
                macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
            else:
                macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(macs)
    except Exception as exc:
        logger.warning(f"thop profile failed: {exc}")
        return None


def _measure_rtf(
    model: torch.nn.Module,
    dummy_input: Any,
    device: str,
    duration_sec: float,
    warmup: int = 5,
    runs: int = 20,
) -> Dict[str, float]:
    """Measure end-to-end forward-pass latency and RTF.

    RTF = inference_time / audio_duration. RTF<1.0 means real-time capable.
    """
    model.eval()
    use_cuda = device == "cuda" and torch.cuda.is_available()

    def _sync() -> None:
        if use_cuda:
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            _sync()

        times: List[float] = []
        for _ in range(runs):
            _sync()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            _sync()
            times.append(time.perf_counter() - t0)

    mean = sum(times) / len(times)
    sorted_times = sorted(times)
    median = sorted_times[len(sorted_times) // 2]
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return {
        "rtf_mean": mean / duration_sec,
        "rtf_median": median / duration_sec,
        "latency_mean_ms": mean * 1000,
        "latency_median_ms": median * 1000,
        "latency_std_ms": std * 1000,
        "runs": runs,
    }


def analyze_experiment(
    name: str,
    exp_root: Path,
    measure_rtf: bool = False,
    rtf_duration: float = 10.0,
    rtf_warmup: int = 5,
    rtf_runs: int = 20,
    device: str = "cuda",
) -> Dict[str, Any]:
    exp_dir = exp_root / name
    cfg = _load_config(exp_dir)
    model_args = cfg.model
    input_type = model_args.input_type

    device_eff = device if device != "cuda" or torch.cuda.is_available() else "cpu"

    model = load_model(
        model_args.model_lib,
        model_args.model_class,
        model_args.param,
        device_eff,
    )
    model = load_checkpoint(model, str(exp_dir), "best.th", device_eff)
    model.eval()

    total, trainable = _count_params(model)
    size_mb = _model_size_mb(total)

    stft_args = get_stft_args_from_config(model_args)
    if hasattr(cfg, "compress_factor"):
        stft_args["compress_factor"] = cfg.compress_factor

    # 1-second dummy audio for MACs measurement (per-second MACs is the norm)
    com_1s = _dummy_stft_inputs(1.0, stft_args, device_eff)
    if input_type == "acs+bcs":
        dummy_macs = (com_1s, com_1s.clone())
    else:
        dummy_macs = com_1s

    macs = _measure_macs(model, dummy_macs, input_type)

    result: Dict[str, Any] = {
        "name": name,
        "model_class": str(model_args.model_class),
        "input_type": input_type,
        "params_total": total,
        "params_trainable": trainable,
        "params_M": total / 1e6,
        "size_MB_fp32": size_mb,
        "macs_per_sec": macs,
        "gmacs_per_sec": (macs / 1e9) if macs is not None else None,
    }

    if measure_rtf:
        com = _dummy_stft_inputs(rtf_duration, stft_args, device_eff)
        if input_type == "acs+bcs":
            dummy_rtf = (com, com.clone())
        else:
            dummy_rtf = com

        rtf_cpu: Dict[str, float] = {}
        rtf_gpu: Dict[str, float] = {}

        try:
            cpu_model = model.cpu()
            if isinstance(dummy_rtf, tuple):
                cpu_input = tuple(t.cpu() for t in dummy_rtf)
            else:
                cpu_input = dummy_rtf.cpu()
            rtf_cpu = _measure_rtf(
                cpu_model, cpu_input, "cpu", rtf_duration, rtf_warmup, rtf_runs
            )
        except Exception as exc:
            logger.warning(f"CPU RTF failed for {name}: {exc}")

        if torch.cuda.is_available():
            try:
                gpu_model = model.cuda()
                if isinstance(dummy_rtf, tuple):
                    gpu_input = tuple(t.cuda() for t in dummy_rtf)
                else:
                    gpu_input = dummy_rtf.cuda()
                rtf_gpu = _measure_rtf(
                    gpu_model, gpu_input, "cuda", rtf_duration, rtf_warmup, rtf_runs
                )
            except Exception as exc:
                logger.warning(f"GPU RTF failed for {name}: {exc}")

        result["rtf_cpu"] = rtf_cpu
        result["rtf_gpu"] = rtf_gpu
        result["rtf_duration_sec"] = rtf_duration

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def format_table(results: List[Dict[str, Any]], include_rtf: bool = False) -> str:
    headers = ["name", "model_class", "input", "Params(M)", "Size(MB)", "GMACs/s"]
    if include_rtf:
        headers += ["CPU RTF", "GPU RTF"]

    rows: List[List[str]] = []
    for r in results:
        row = [
            r["name"],
            r["model_class"],
            r["input_type"],
            f"{r['params_M']:.3f}",
            f"{r['size_MB_fp32']:.2f}",
            (
                f"{r['gmacs_per_sec']:.3f}"
                if r.get("gmacs_per_sec") is not None
                else "n/a"
            ),
        ]
        if include_rtf:
            cpu = r.get("rtf_cpu", {})
            gpu = r.get("rtf_gpu", {})
            row.append(
                f"{cpu.get('rtf_mean'):.4f}" if cpu.get("rtf_mean") is not None else "n/a"
            )
            row.append(
                f"{gpu.get('rtf_mean'):.4f}" if gpu.get("rtf_mean") is not None else "n/a"
            )
        rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = []
    lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model complexity measurement")
    parser.add_argument("--experiment", type=str, default=None, help="Single experiment name")
    parser.add_argument("--all", action="store_true", help="Measure all experiments")
    parser.add_argument(
        "--exp_root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "experiments"),
        help="Root directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(PROJECT_ROOT / "paper_work" / "tables" / "complexity.json"),
        help="Path to save results as JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model loading (cuda|cpu)",
    )
    parser.add_argument("--measure_rtf", action="store_true", help="Also measure RTF")
    parser.add_argument("--rtf_duration", type=float, default=10.0, help="Audio duration for RTF (s)")
    parser.add_argument("--rtf_warmup", type=int, default=5, help="RTF warmup runs")
    parser.add_argument("--rtf_runs", type=int, default=20, help="RTF timed runs")
    parser.add_argument("--skip", type=str, nargs="*", default=["archive"], help="Experiment dirs to skip")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    exp_root = Path(args.exp_root)

    if args.all:
        names = sorted(
            p.name
            for p in exp_root.iterdir()
            if p.is_dir() and p.name not in args.skip and (p / ".hydra" / "config.yaml").exists()
        )
    elif args.experiment:
        names = [args.experiment]
    else:
        parser.error("Specify --experiment or --all")

    results: List[Dict[str, Any]] = []
    for name in names:
        logger.info(f"Analyzing {name} ...")
        try:
            r = analyze_experiment(
                name=name,
                exp_root=exp_root,
                measure_rtf=args.measure_rtf,
                rtf_duration=args.rtf_duration,
                rtf_warmup=args.rtf_warmup,
                rtf_runs=args.rtf_runs,
                device=args.device,
            )
            results.append(r)
        except Exception as exc:
            logger.error(f"Failed on {name}: {exc}")
            results.append({"name": name, "error": str(exc)})

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    print()
    print(format_table([r for r in results if "error" not in r], include_rtf=args.measure_rtf))
    errors = [r for r in results if "error" in r]
    if errors:
        print()
        print(f"Errors on {len(errors)} experiments:")
        for e in errors:
            print(f"  {e['name']}: {e['error']}")


if __name__ == "__main__":
    main()
