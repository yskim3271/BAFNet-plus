"""Compare BAFNetPlus offline forward vs BAFNetPlusStreaming.process_audio_fast.

Stage 1 diagnostic CLI. Loads two checkpoints, builds both offline and streaming
models with shared (deepcopy'd) fusion weights, runs synthetic audio through each,
and reports RMS / max difference.

Exits 0 on parity success (RMS < 1e-4, max < 1e-3), 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on sys.path when invoked directly as a script
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from src.models.streaming.bafnetplus_streaming import BAFNetPlusStreaming
from src.stft import mag_pha_stft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chkpt_dir_mapping", default="results/experiments/bm_map_50ms")
    parser.add_argument("--chkpt_dir_masking", default="results/experiments/bm_mask_50ms")
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--encoder_lookahead", type=int, default=3)
    parser.add_argument("--decoder_lookahead", type=int, default=3)
    parser.add_argument("--audio_length", type=int, default=16000, help="Samples (default 1s at 16kHz)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rms_tol", type=float, default=1e-4)
    parser.add_argument("--max_tol", type=float, default=1e-3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"[compare] building BAFNetPlusStreaming from {args.chkpt_dir_mapping} / {args.chkpt_dir_masking}")
    streaming, offline = BAFNetPlusStreaming.from_checkpoint(
        chkpt_dir_mapping=args.chkpt_dir_mapping,
        chkpt_dir_masking=args.chkpt_dir_masking,
        chunk_size=args.chunk_size,
        encoder_lookahead=args.encoder_lookahead,
        decoder_lookahead=args.decoder_lookahead,
        device=args.device,
        verbose=args.verbose,
        return_offline=True,
    )
    print(f"[compare] latency_ms={streaming.latency_ms:.2f}, samples_per_chunk={streaming.samples_per_chunk}")

    bcs = torch.randn(args.audio_length)
    acs = torch.randn(args.audio_length)

    # Compare at spectrogram level with matched STFT convention (center=False + matching
    # context prepend / flush pad). Wav-level comparison is confounded by STFT convention
    # differences between offline (center=True default) and streaming (manual center=False
    # context). See tests/test_bafnetplus_streaming.py::test_offline_vs_streaming_parity.
    print("[compare] running offline BAFNetPlus.forward (center=False, matched context)...")
    ctx = streaming.win_size // 2
    flush = streaming.n_fft
    bcs_pad = torch.cat([torch.zeros(ctx), bcs, torch.zeros(flush)])
    acs_pad = torch.cat([torch.zeros(ctx), acs, torch.zeros(flush)])
    with torch.no_grad():
        _, _, bcs_com = mag_pha_stft(
            bcs_pad.unsqueeze(0),
            n_fft=streaming.n_fft,
            hop_size=streaming.hop_size,
            win_size=streaming.win_size,
            compress_factor=streaming.compress_factor,
            center=False,
        )
        _, _, acs_com = mag_pha_stft(
            acs_pad.unsqueeze(0),
            n_fft=streaming.n_fft,
            hop_size=streaming.hop_size,
            win_size=streaming.win_size,
            compress_factor=streaming.compress_factor,
            center=False,
        )
        off_mag, off_pha, _ = offline((bcs_com, acs_com))

    # Streaming spec (re-run streaming primitives against the same spec input)
    print("[compare] running BAFNetPlusStreaming primitives (spec capture)...")
    from src.models.streaming.utils import StateFramesContext

    streaming.reset_state()
    stream_mag_parts, stream_pha_parts = [], []
    T_total = bcs_com.shape[-2]
    with torch.inference_mode():
        for i in range(0, T_total, streaming.chunk_size):
            end = min(i + streaming.chunk_size + streaming.encoder_lookahead, T_total)
            bs = bcs_com[:, :, i:end, :]
            as_ = acs_com[:, :, i:end, :]
            T_spec = bs.shape[2]
            vf = min(T_spec, streaming.chunk_size)
            with StateFramesContext(vf):
                bm, bto, _ = streaming._process_encoder(
                    bs,
                    streaming.model.mapping.dense_encoder,
                    streaming.streaming_tsblocks_mapping,
                    streaming._tsblock_states_mapping,
                )
                am, ato, _ = streaming._process_encoder(
                    as_,
                    streaming.model.masking.dense_encoder,
                    streaming.streaming_tsblocks_masking,
                    streaming._tsblock_states_masking,
                )
            streaming._buffer_features(bto, bm, vf, streaming.bcs_feature_buffer)
            streaming._buffer_features(ato, am, vf, streaming.acs_feature_buffer)
            streaming._buffered_frames += vf
            if streaming._buffered_frames < streaming.chunk_size + streaming.decoder_lookahead:
                continue
            bef, bem = streaming._extract_extended(streaming.bcs_feature_buffer)
            aef, aem = streaming._extract_extended(streaming.acs_feature_buffer)
            with StateFramesContext(streaming.chunk_size):
                b_mag, _, b_com = streaming._decode_branch(
                    bef,
                    bem,
                    streaming.model.mapping.mask_decoder,
                    streaming.model.mapping.phase_decoder,
                    "mapping",
                    False,
                )
                a_mag, _, a_com, a_mask = streaming._decode_branch(
                    aef,
                    aem,
                    streaming.model.masking.mask_decoder,
                    streaming.model.masking.phase_decoder,
                    "masking",
                    True,
                )
                if streaming.model.use_calibration:
                    bcs_cal, acs_cal = streaming._apply_calibration_streaming(
                        b_com,
                        a_com,
                        b_mag,
                        a_mag,
                        a_mask,
                    )
                else:
                    bcs_cal, acs_cal = b_com, a_com
                em, ep = streaming._fuse(bcs_cal, acs_cal, a_mask)
            r = streaming._slide_feature_buffer(streaming.bcs_feature_buffer, streaming.chunk_size)
            streaming._slide_feature_buffer(streaming.acs_feature_buffer, streaming.chunk_size)
            streaming._buffered_frames -= r
            stream_mag_parts.append(em)
            stream_pha_parts.append(ep)
    stream_mag = torch.cat(stream_mag_parts, dim=2)
    stream_pha = torch.cat(stream_pha_parts, dim=2)

    L = min(off_mag.shape[-1], stream_mag.shape[-1])
    diff_mag = (off_mag[:, :, :L] - stream_mag[:, :, :L]).abs()
    diff_pha = (off_pha[:, :, :L] - stream_pha[:, :, :L]).abs()
    rms = diff_mag.pow(2).mean().sqrt().item()
    mx = diff_mag.max().item()

    print(f"\n[compare] mag RMS  diff = {rms:.6e}  (tol {args.rms_tol:.1e})")
    print(f"[compare] mag max  diff = {mx:.6e}  (tol {args.max_tol:.1e})")
    print(f"[compare] pha RMS  diff = {diff_pha.pow(2).mean().sqrt().item():.6e}")
    print(f"[compare] pha max  diff = {diff_pha.max().item():.6e}")
    print(f"[compare] offline T = {off_mag.shape[-1]}, streaming T = {stream_mag.shape[-1]}, compared L = {L}")

    passed = rms < args.rms_tol and mx < args.max_tol
    if passed:
        print("\n[compare] PASS")
        return 0
    print("\n[compare] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
