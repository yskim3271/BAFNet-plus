"""Compute TAPS train per-frame log-energy ratio between ACS and BCS branches.

phi_t = log(mean_k |X_acs|^2 + eps) - log(mean_k |X_bcs|^2 + eps), with the same
STFT settings the BAFNet+ paper uses (W=N=400, H=100, fs=16k, Hann, center=True).
Outputs Paper_TASLP/tables/calib_bound_stats.{csv,md} with 5/25/50/75/95
percentiles and the fraction of frames violating the calibration bound
G_max^Delta = 1.0 nats per branch (joint |phi| > 2.0 nats == ~8.686 dB).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "BAFNetPlus"))

N_FFT, HOP, WIN, SR = 400, 100, 400, 16000
EPS = 1e-8
NATS_TO_DB = 10.0 / math.log(10.0)
G_MAX_NATS = 1.0
SAT_NATS = 2.0 * G_MAX_NATS  # joint bound on |phi_t|


def per_frame_phi(bcs: np.ndarray, acs: np.ndarray) -> np.ndarray:
    win = torch.hann_window(WIN)

    def mag(y):
        spec = torch.stft(torch.from_numpy(y.astype(np.float32)), N_FFT, hop_length=HOP,
                          win_length=WIN, window=win, center=True, pad_mode="reflect",
                          normalized=False, return_complex=True)
        return spec.abs()

    e_b = mag(bcs).pow(2).mean(dim=0)
    e_a = mag(acs).pow(2).mean(dim=0)
    return (torch.log(e_a + EPS) - torch.log(e_b + EPS)).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num_utts", type=int, default=1000, help="Subsample N utts (-1 = full).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "Paper_TASLP" / "tables"))
    args = ap.parse_args()

    from datasets import load_dataset
    print("Loading TAPS train...")
    ds = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="train")
    n_total = len(ds)
    rng = np.random.default_rng(args.seed)
    idxs = (np.arange(n_total) if args.num_utts < 0
            else np.sort(rng.choice(n_total, size=min(args.num_utts, n_total), replace=False)))
    print(f"  total={n_total}; sampling={len(idxs)} (seed={args.seed})")

    parts = []
    for i, k in enumerate(idxs):
        item = ds[int(k)]
        bcs = np.asarray(item["audio.throat_microphone"]["array"], dtype=np.float32)
        acs = np.asarray(item["audio.acoustic_microphone"]["array"], dtype=np.float32)
        sr_b = item["audio.throat_microphone"]["sampling_rate"]
        sr_a = item["audio.acoustic_microphone"]["sampling_rate"]
        if sr_b != SR or sr_a != SR:
            raise RuntimeError(f"sr mismatch idx={k}: bcs={sr_b}, acs={sr_a}")
        parts.append(per_frame_phi(bcs, acs))
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(idxs)}] frames so far: {sum(p.size for p in parts)}")

    phi = np.concatenate(parts)
    n_frames = phi.size
    p_nats = np.percentile(phi, [5, 25, 50, 75, 95])
    p_db = p_nats * NATS_TO_DB
    sat_rate = float(np.mean(np.abs(phi) > SAT_NATS))
    sat_db = SAT_NATS * NATS_TO_DB

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_lines = ["metric,value", f"num_utterances,{len(idxs)}", f"num_frames,{n_frames}"]
    for p, n, d in zip([5, 25, 50, 75, 95], p_nats, p_db):
        csv_lines += [f"phi_p{p}_nats,{n:.6f}", f"phi_p{p}_dB,{d:.6f}"]
    csv_lines += [f"saturation_rate,{sat_rate:.6f}",
                  f"saturation_threshold_nats,{SAT_NATS}",
                  f"saturation_threshold_dB,{sat_db:.6f}"]
    (out_dir / "calib_bound_stats.csv").write_text("\n".join(csv_lines) + "\n")

    rows = "".join(f"| p{p} | {n:+.4f} | {d:+.3f} |\n"
                   for p, n, d in zip([5, 25, 50, 75, 95], p_nats, p_db))
    summary = (f"On clean TAPS train ({len(idxs)} utts, {n_frames} frames), per-frame "
               f"10 log10(E_acs/E_bcs) has 5/25/50/75/95-pct = "
               f"{p_db[0]:+.2f} / {p_db[1]:+.2f} / {p_db[2]:+.2f} / {p_db[3]:+.2f} / "
               f"{p_db[4]:+.2f} dB; only {sat_rate * 100:.2f}% of frames exceed the "
               f"calibration bound (|.| > {sat_db:.2f} dB), justifying "
               f"G_max^Delta = {G_MAX_NATS} nats.")
    (out_dir / "calib_bound_stats.md").write_text(
        f"# TAPS train per-frame log-energy ratio (ACS vs BCS)\n\n"
        f"STFT W=N={WIN}, H={HOP}, fs={SR}, Hann, center=True; uncompressed |X|^2; eps={EPS:g}. "
        f"Clean TAPS pairs only (no noise/RIR).\n\n"
        f"Sampled: **{len(idxs)}** of {n_total} train utts (seed={args.seed}); "
        f"frames: **{n_frames}**.\n\n"
        f"phi_t = log(mean_k |X_acs|^2+eps) - log(mean_k |X_bcs|^2+eps); dB = phi * {NATS_TO_DB:.4f}.\n\n"
        f"## Percentiles\n\n| pct | phi (nats) | 10 log10(E_acs/E_bcs) (dB) |\n| --- | --- | --- |\n{rows}\n"
        f"## Saturation\n\nBound G_max^Delta = {G_MAX_NATS} nats per branch implies "
        f"|phi_t| <= {SAT_NATS} nats (|10 log10(E_acs/E_bcs)| <= {sat_db:.3f} dB). "
        f"Frames violating: **{sat_rate * 100:.3f}%** "
        f"({int(np.sum(np.abs(phi) > SAT_NATS))}/{n_frames}).\n\n"
        f"## One-liner\n\n> {summary}\n"
    )

    print(f"\n=== utts={len(idxs)}; frames={n_frames}")
    print("dB: " + ", ".join(f"p{p}={d:+.3f}" for p, d in zip([5, 25, 50, 75, 95], p_db)))
    print(f"sat (|phi|>{SAT_NATS} nats / {sat_db:.3f} dB): {sat_rate * 100:.3f}%")
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
