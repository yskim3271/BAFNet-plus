"""Phase 2 — Generate fixture pair for EnhancerIntegrityTest.

Drives the bundled ``model_qdq.onnx`` (CPU EP) over TAPS BCS test ``utt_idx=0``
first ``--duration_s`` seconds using the same chunk-by-chunk streaming sequence
the recording-app's ``StreamingEnhancer.processChunk`` replicates on device,
and writes the resulting fixture quad next to ``recording-app/src/androidTest/
assets/fixtures/``:

    input_wideband.wav      BCS throat_microphone first N seconds (16 kHz mono PCM16)
    reference_enhanced.wav  ONNX streaming output (CPU EP), produced 800 samples / chunk
    clean.wav               Acoustic mic first N seconds (PESQ/STOI golden, not used in tolerance)
    fixture.json            sample_rate, n_chunks, checkpoint_md5, model_md5, git_commit, ...

Usage (from repo root):

    python BAFNetPlus/scripts/make_enhancer_golden.py \\
        --onnx Android_projects/recording-app/src/main/assets/model_qdq.onnx \\
        --config Android_projects/recording-app/src/main/assets/streaming_config.json \\
        --chkpt_dir BAFNetPlus/results/experiments/bm_map_50ms \\
        --out_dir Android_projects/recording-app/src/androidTest/assets/fixtures

The output reference is deterministic given a fixed ``model_qdq.onnx`` — host
parity passing at MAE < 1e-4 vs the device implies the same fixture round-trips
through the on-device pipeline within the EnhancerIntegrityTest tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BAFNETPLUS_ROOT = REPO_ROOT / "BAFNetPlus"
if str(BAFNETPLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(BAFNETPLUS_ROOT))

import onnxruntime as ort  # noqa: E402
from src.stft import mag_pha_stft, manual_istft_ola  # noqa: E402

# Streaming export constants (bm_map_50ms). These mirror Phase 0
# check_onnx_parity.py exactly so the fixture is bit-equivalent to the host
# reference that was used during H2 diagnosis.
CHUNK_FRAMES = 8
ENC_LOOKAHEAD = 3
EXPORT_FRAMES = 11
N_FFT = 400
HOP = 100
WIN = 400
COMPRESS = 0.3
SR = 16000
SAMPLES_PER_CHUNK = (EXPORT_FRAMES - 1) * HOP + WIN // 2  # 1200 samples (stft_center)
OUTPUT_SAMPLES = CHUNK_FRAMES * HOP  # 800 samples / chunk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--onnx",
        default=str(REPO_ROOT / "Android_projects/recording-app/src/main/assets/model_qdq.onnx"),
        help="Bundled ONNX model that ships with recording-app.",
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "Android_projects/recording-app/src/main/assets/streaming_config.json"),
        help="streaming_config.json bundled with the recording-app.",
    )
    p.add_argument(
        "--chkpt_dir",
        default=str(BAFNETPLUS_ROOT / "results/experiments/bm_map_50ms"),
        help="PyTorch checkpoint directory (only used to compute checkpoint_md5).",
    )
    p.add_argument("--chkpt_file", default="best.th")
    p.add_argument("--utt_idx", type=int, default=0,
                   help="TAPS test split index; default 0 matches Phase 0 default.")
    p.add_argument("--duration_s", type=float, default=4.0,
                   help="Duration to capture from utt_idx; default 4 s.")
    p.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "Android_projects/recording-app/src/androidTest/assets/fixtures"),
    )
    return p.parse_args()


def load_taps_pair(utt_idx: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load one TAPS test sample (BCS + ACS), trimmed to ``samples``."""
    from datasets import load_dataset

    ds = load_dataset(
        "yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test"
    )
    item = ds[utt_idx]
    bcs = np.asarray(item["audio.throat_microphone"]["array"], dtype=np.float32)
    acs = np.asarray(item["audio.acoustic_microphone"]["array"], dtype=np.float32)
    bcs_sr = item["audio.throat_microphone"]["sampling_rate"]
    acs_sr = item["audio.acoustic_microphone"]["sampling_rate"]
    if bcs_sr != SR or acs_sr != SR:
        raise RuntimeError(
            f"TAPS utt_idx={utt_idx} not 16 kHz (bcs_sr={bcs_sr}, acs_sr={acs_sr})"
        )
    if len(bcs) < samples or len(acs) < samples:
        raise RuntimeError(
            f"TAPS utt_idx={utt_idx} too short for duration_s "
            f"(bcs={len(bcs)}, acs={len(acs)}, need={samples})"
        )
    return bcs[:samples], acs[:samples]


def write_pcm16_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    """Write a 16-bit PCM mono WAV. Audio is clipped to [-1, 1]."""
    from scipy.io import wavfile

    a = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm = (a * 32767.0).round().astype(np.int16)
    wavfile.write(str(path), sr, pcm)


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_short(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def stft_chunk(chunk_with_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mag/pha STFT with center=False and 200-sample left context."""
    mag, pha, _ = mag_pha_stft(
        chunk_with_context.unsqueeze(0),
        n_fft=N_FFT, hop_size=HOP, win_size=WIN,
        compress_factor=COMPRESS, center=False,
    )
    return mag, pha  # [1, F, 11]


def apply_mask_and_istft(
    mag: torch.Tensor,
    est_mask: torch.Tensor,
    phase_real: torch.Tensor,
    phase_imag: torch.Tensor,
    infer_type: str,
    ola_buffer: torch.Tensor,
    ola_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror StreamingEnhancer.processChunk's mask-apply / iSTFT path."""
    est_mag = est_mask if infer_type == "mapping" else mag * est_mask
    est_pha = torch.atan2(phase_imag + 1e-8, phase_real + 1e-8)
    est_mag = est_mag[:, :, :CHUNK_FRAMES]
    est_pha = est_pha[:, :, :CHUNK_FRAMES]
    out, new_buf, new_norm = manual_istft_ola(
        est_mag, est_pha,
        n_fft=N_FFT, hop_size=HOP, win_size=WIN, compress_factor=COMPRESS,
        ola_buffer=ola_buffer, ola_norm=ola_norm,
    )
    return out[:OUTPUT_SAMPLES], new_buf, new_norm


def stream_onnx_reference(
    audio: np.ndarray,
    sess: ort.InferenceSession,
    state_names: List[str],
    state_shapes: dict,
    infer_type: str,
) -> np.ndarray:
    """Drive ``audio`` through the ONNX session in 800-sample push increments
    and return the concatenated 800-sample-per-chunk enhanced output. Pushes
    that fall before the buffer fills produce no output (mirrors Android's
    processChunk → null path).
    """
    audio_t = torch.from_numpy(audio.astype(np.float32))
    n_pushes = len(audio_t) // OUTPUT_SAMPLES
    if n_pushes < 2:
        raise RuntimeError(
            f"audio too short: {len(audio_t)} samples → {n_pushes} pushes "
            f"(need ≥ 2 to produce any enhanced output)."
        )

    onnx_states = {n: np.zeros(state_shapes[n], dtype=np.float32) for n in state_names}
    ola_buf = torch.zeros(WIN - HOP)
    ola_norm = torch.zeros(WIN - HOP)
    stft_context = torch.zeros(WIN // 2)
    input_buffer = torch.tensor([], dtype=torch.float32)

    enhanced_chunks: List[np.ndarray] = []

    for push_idx in range(n_pushes):
        new_chunk = audio_t[push_idx * OUTPUT_SAMPLES:(push_idx + 1) * OUTPUT_SAMPLES]
        input_buffer = torch.cat([input_buffer, new_chunk])
        if len(input_buffer) < SAMPLES_PER_CHUNK:
            continue

        chunk_samples = input_buffer[:SAMPLES_PER_CHUNK]
        chunk_with_context = torch.cat([stft_context, chunk_samples])
        mag, pha = stft_chunk(chunk_with_context)

        adv = OUTPUT_SAMPLES
        ctx_size = WIN // 2
        if adv >= ctx_size:
            stft_context = input_buffer[adv - ctx_size:adv].clone()
        else:
            need = ctx_size - adv
            stft_context = torch.cat([stft_context[-need:], input_buffer[:adv]]).clone()
        input_buffer = input_buffer[OUTPUT_SAMPLES:]

        ort_inputs = {"mag": mag.numpy(), "pha": pha.numpy()}
        ort_inputs.update(onnx_states)
        ort_out = sess.run(None, ort_inputs)
        mask, phase_real, phase_imag = ort_out[0], ort_out[1], ort_out[2]
        next_states = ort_out[3:]
        assert len(next_states) == len(state_names), (
            f"output state count mismatch: got {len(next_states)} "
            f"vs {len(state_names)} state inputs."
        )
        onnx_states = {n: next_states[i] for i, n in enumerate(state_names)}

        chunk_audio, ola_buf, ola_norm = apply_mask_and_istft(
            mag, torch.from_numpy(mask),
            torch.from_numpy(phase_real), torch.from_numpy(phase_imag),
            infer_type, ola_buf, ola_norm,
        )
        enhanced_chunks.append(chunk_audio.numpy())

    if not enhanced_chunks:
        raise RuntimeError("No enhanced chunks produced; audio too short.")
    return np.concatenate(enhanced_chunks).astype(np.float32)


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx).resolve()
    config_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir).resolve()
    chkpt = Path(args.chkpt_dir, args.chkpt_file).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"make_enhancer_golden.py")
    print(f"  onnx       : {onnx_path}")
    print(f"  config     : {config_path}")
    print(f"  chkpt      : {chkpt}")
    print(f"  utt_idx    : {args.utt_idx}")
    print(f"  duration_s : {args.duration_s}")
    print(f"  out_dir    : {out_dir}")

    with open(config_path) as f:
        cfg = json.load(f)
    state_names: List[str] = cfg["state_info"]["state_names"]
    infer_type: str = cfg["model_info"]["infer_type"]
    cfg_checkpoint_md5 = cfg["export_info"]["checkpoint_md5"]
    cfg_git_commit = cfg["export_info"]["git_commit"]
    assert len(state_names) == 80

    samples = int(round(args.duration_s * SR))
    bcs, acs = load_taps_pair(args.utt_idx, samples)

    # Save inputs as PCM16 WAV first so md5(input_wideband.wav) ↔ Android-side
    # WavFileAudioSource reads bit-equivalent floats. We then run the streaming
    # ONNX over the same int16-roundtripped audio (not the float32 array) for
    # fixture self-consistency.
    input_path = out_dir / "input_wideband.wav"
    clean_path = out_dir / "clean.wav"
    write_pcm16_wav(input_path, bcs)
    write_pcm16_wav(clean_path, acs)

    from scipy.io import wavfile
    sr_back, bcs_pcm = wavfile.read(str(input_path))
    assert sr_back == SR
    bcs_round = bcs_pcm.astype(np.float32) / 32768.0  # mirror PcmFormat.shortToFloat

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    state_shapes = {
        inp.name: tuple(inp.shape)
        for inp in sess.get_inputs() if inp.name in set(state_names)
    }
    missing = set(state_names) - set(state_shapes)
    if missing:
        raise RuntimeError(f"ONNX missing state inputs: {sorted(missing)[:5]}...")

    enhanced = stream_onnx_reference(
        bcs_round, sess, state_names, state_shapes, infer_type,
    )
    n_chunks = len(enhanced) // OUTPUT_SAMPLES
    print(f"  produced enhanced: {len(enhanced)} samples = {n_chunks} chunks of {OUTPUT_SAMPLES}")

    enhanced_path = out_dir / "reference_enhanced.wav"
    write_pcm16_wav(enhanced_path, enhanced)

    fixture = {
        "fixture_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": SR,
        "duration_s": args.duration_s,
        "samples_per_chunk": OUTPUT_SAMPLES,
        "n_pushes": int(samples // OUTPUT_SAMPLES),
        "n_chunks": int(n_chunks),
        "input_wav": "input_wideband.wav",
        "reference_wav": "reference_enhanced.wav",
        "clean_wav": "clean.wav",
        "input_wideband_md5": md5_of_file(input_path),
        "reference_enhanced_md5": md5_of_file(enhanced_path),
        "clean_md5": md5_of_file(clean_path),
        "checkpoint_md5": md5_of_file(chkpt),
        "model_md5": md5_of_file(onnx_path),
        "git_commit": git_commit_short(BAFNETPLUS_ROOT),
        "config_checkpoint_md5": cfg_checkpoint_md5,
        "config_git_commit": cfg_git_commit,
        "infer_type": infer_type,
        "taps": {
            "split": "test",
            "utt_idx": args.utt_idx,
            "channel_input": "audio.throat_microphone",
            "channel_clean": "audio.acoustic_microphone",
        },
    }

    if fixture["checkpoint_md5"] != cfg_checkpoint_md5:
        print(
            f"  WARNING: checkpoint_md5 mismatch (chkpt={fixture['checkpoint_md5']}, "
            f"streaming_config={cfg_checkpoint_md5}). The exported ONNX was built from a "
            "different checkpoint than the one referenced here."
        )

    fixture_path = out_dir / "fixture.json"
    with open(fixture_path, "w") as f:
        json.dump(fixture, f, indent=2, sort_keys=True)
    print(f"  wrote {fixture_path}")
    print()
    print("fixture summary:")
    for k in (
        "n_pushes", "n_chunks", "checkpoint_md5", "model_md5", "git_commit",
        "config_checkpoint_md5",
    ):
        print(f"  {k}: {fixture[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
