"""Verify LaCoSENet functional ONNX export: PT-vs-ORT FP32 corr >= 0.9980 (TAPS).

Phase 1 of plan-mode-happy-summit.md cleanup cycle.

Runs the new functional StatefulExportableNNCore (PT, chunk-by-chunk) and
ORT FP32 session in lockstep over TAPS BCS test utterances and checks that
per-chunk est_mask correlation is at or above the legacy LaCoSENet anchor.

Usage:
    PYTHONPATH=. python scripts/verify_lacosenet_corr.py \\
        --chkpt_dir results/experiments/bm_map_50ms \\
        --onnx /tmp/lacosenet_func_50ms/bm_map_50ms_c8.onnx \\
        --utt_indices 0,1,2,3,4,5,6,7,8,9 \\
        --chunks 30 \\
        --corr_threshold 0.9980
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import onnxruntime as ort  # noqa: E402

from src.models.streaming.onnx.export_onnx import (  # noqa: E402
    StatefulExportableNNCore,
    prepare_from_checkpoint,
)
from src.models.streaming.onnx.functional_core import (  # noqa: E402
    FunctionalStateRegistry,
    build_single_backbone_state_entries,
)
from src.stft import mag_pha_stft  # noqa: E402

SR = 16000
N_FFT = 400
HOP = 100
WIN = 400
COMPRESS = 0.3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chkpt_dir", type=str, required=True)
    p.add_argument("--chkpt_file", type=str, default="best.th")
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--utt_indices", type=str, default="0,1,2,3,4,5,6,7,8,9",
                   help="Comma-separated TAPS test split indices.")
    p.add_argument("--chunks", type=int, default=30)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--encoder_lookahead", type=int, default=3)
    p.add_argument("--decoder_lookahead", type=int, default=3)
    p.add_argument("--corr_threshold", type=float, default=0.9980)
    p.add_argument("--atol", type=float, default=1e-4,
                   help="Per-chunk max abs diff tolerance (sanity check).")
    return p.parse_args()


def load_taps_audio(utt_idx: int) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset(
        "yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test"
    )
    sample = ds[utt_idx]
    arr = sample["audio.throat_microphone"]["array"]
    if sample["audio.throat_microphone"]["sampling_rate"] != SR:
        raise ValueError(f"TAPS BCS utt_idx={utt_idx} not 16kHz")
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def stft_chunk(chunk_with_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mag, pha, _ = mag_pha_stft(
        chunk_with_context.unsqueeze(0),
        n_fft=N_FFT, hop_size=HOP, win_size=WIN,
        compress_factor=COMPRESS, center=False,
    )
    return mag, pha


def corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    if af.std() < 1e-12 or bf.std() < 1e-12:
        return 1.0 if np.allclose(af, bf, atol=1e-8) else 0.0
    return float(np.corrcoef(af, bf)[0, 1])


def main() -> int:
    args = parse_args()
    utt_indices: List[int] = [int(x) for x in args.utt_indices.split(",")]

    print("=" * 64)
    print("Phase 1: LaCoSENet PT-vs-ORT FP32 corr check")
    print("=" * 64)
    print(f"  chkpt_dir       : {args.chkpt_dir}")
    print(f"  onnx            : {args.onnx}")
    print(f"  utt_indices     : {utt_indices}")
    print(f"  chunks per utt  : {args.chunks}")
    print(f"  chunk_size      : {args.chunk_size}")
    print(f"  encoder_la      : {args.encoder_lookahead}")
    print(f"  decoder_la      : {args.decoder_lookahead}")
    print(f"  corr_threshold  : {args.corr_threshold}")
    print()

    # --- Build PT wrapper via new functional API ---
    model, model_info = prepare_from_checkpoint(
        chkpt_dir=args.chkpt_dir, chkpt_file=args.chkpt_file,
        device="cpu", verbose=False,
    )
    n_fft = model_info["n_fft"]
    freq_bins = n_fft // 2 + 1
    freq_size = model_info["freq_size_encoded"]
    export_time_frames = args.chunk_size + max(args.encoder_lookahead, args.decoder_lookahead)

    entries = build_single_backbone_state_entries(
        model, freq_bins, freq_size, export_time_frames,
    )
    state_registry = FunctionalStateRegistry(entries)
    wrapper = StatefulExportableNNCore(
        model=model, freq_size=freq_size, chunk_size=args.chunk_size,
        infer_type=model_info["infer_type"], state_registry=state_registry,
    ).eval()

    # --- ORT session ---
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_state_inputs = [i.name for i in sess.get_inputs() if i.name.startswith("state_")]
    assert sorted(wrapper.state_names) == sorted(onnx_state_inputs), \
        f"state name mismatch: PT={sorted(wrapper.state_names)} vs ONNX={sorted(onnx_state_inputs)}"
    print(f"  states matched : {len(wrapper.state_names)}")

    # Sample frames per chunk match check_onnx_parity.py: stft_context (WIN/2)
    # prepended to chunk_samples (=SAMPLES_PER_CHUNK) for stft_center=False
    # boundary, advancing input by OUTPUT_SAMPLES (chunk_size * hop).
    samples_per_chunk = (export_time_frames - 1) * HOP + WIN // 2  # 1200 for 50ms
    output_samples = args.chunk_size * HOP                          # 800

    all_mask_corrs: List[float] = []
    all_phase_real_corrs: List[float] = []
    all_phase_imag_corrs: List[float] = []
    all_max_diffs: List[float] = []

    for utt_idx in utt_indices:
        audio = load_taps_audio(utt_idx)
        flush = samples_per_chunk * 2
        padded = torch.cat([audio, torch.zeros(flush)])

        # Initialize states (cold start)
        pt_states = [torch.zeros(*state_registry[n]) for n in wrapper.state_names]
        onnx_states = {n: np.zeros(state_registry[n], dtype=np.float32) for n in wrapper.state_names}

        input_buffer = torch.tensor([], dtype=torch.float32)
        stft_context = torch.zeros(WIN // 2)
        feed_idx = 0
        chunk_done = 0

        utt_mask_corrs: List[float] = []
        while chunk_done < args.chunks:
            new_chunk = padded[feed_idx * output_samples : (feed_idx + 1) * output_samples]
            feed_idx += 1
            if new_chunk.numel() == 0:
                break
            input_buffer = torch.cat([input_buffer, new_chunk])
            if input_buffer.numel() < samples_per_chunk:
                continue
            chunk_samples = input_buffer[:samples_per_chunk]
            chunk_with_context = torch.cat([stft_context, chunk_samples])

            adv = output_samples
            ctx_size = WIN // 2
            if adv >= ctx_size:
                stft_context = input_buffer[adv - ctx_size : adv].clone()
            else:
                need = ctx_size - adv
                stft_context = torch.cat([stft_context[-need:], input_buffer[:adv]]).clone()
            input_buffer = input_buffer[output_samples:]

            mag, pha = stft_chunk(chunk_with_context)

            # PT forward
            with torch.no_grad():
                pt_out = wrapper(mag, pha, *pt_states)
            pt_mask = pt_out[0].numpy()
            pt_phase_real = pt_out[1].numpy()
            pt_phase_imag = pt_out[2].numpy()
            pt_next = list(pt_out[3:])

            # ORT forward
            ort_inputs = {"mag": mag.numpy(), "pha": pha.numpy()}
            ort_inputs.update(onnx_states)
            ort_outs = sess.run(None, ort_inputs)
            ort_mask = ort_outs[0]
            ort_phase_real = ort_outs[1]
            ort_phase_imag = ort_outs[2]
            ort_next = ort_outs[3:]

            # Compare
            mask_corr = corr(pt_mask, ort_mask)
            pr_corr = corr(pt_phase_real, ort_phase_real)
            pi_corr = corr(pt_phase_imag, ort_phase_imag)
            max_diff = float(np.abs(pt_mask - ort_mask).max())

            utt_mask_corrs.append(mask_corr)
            all_mask_corrs.append(mask_corr)
            all_phase_real_corrs.append(pr_corr)
            all_phase_imag_corrs.append(pi_corr)
            all_max_diffs.append(max_diff)

            # Carry state
            pt_states = [t.detach() for t in pt_next]
            for j, n in enumerate(wrapper.state_names):
                onnx_states[n] = ort_next[j]
            chunk_done += 1

        utt_avg = float(np.mean(utt_mask_corrs)) if utt_mask_corrs else float("nan")
        utt_min = float(np.min(utt_mask_corrs)) if utt_mask_corrs else float("nan")
        print(f"  utt_idx={utt_idx:2d}  chunks={len(utt_mask_corrs):2d}  "
              f"mask_corr avg={utt_avg:.6f} min={utt_min:.6f}")

    overall_mask_corr = float(np.mean(all_mask_corrs))
    overall_pr_corr = float(np.mean(all_phase_real_corrs))
    overall_pi_corr = float(np.mean(all_phase_imag_corrs))
    max_diff_overall = float(np.max(all_max_diffs))

    print()
    print("Aggregate (mean corr across all utts/chunks):")
    print(f"  est_mask    : {overall_mask_corr:.6f}")
    print(f"  phase_real  : {overall_pr_corr:.6f}")
    print(f"  phase_imag  : {overall_pi_corr:.6f}")
    print(f"  max abs diff (mask): {max_diff_overall:.6e}")
    print()

    fail = False
    if overall_mask_corr < args.corr_threshold:
        print(f"FAIL: est_mask corr {overall_mask_corr:.6f} < threshold {args.corr_threshold}")
        fail = True
    if max_diff_overall > args.atol * 100:  # generous numerical sanity
        print(f"WARN: max abs diff {max_diff_overall:.6e} unusually large (atol*100 = {args.atol * 100})")

    if fail:
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
