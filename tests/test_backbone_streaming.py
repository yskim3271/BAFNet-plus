"""Tests for the PyTorch streaming wrapper (``streaming/backbone_streaming.py``).

S3 exit gate (model-level): chunked PT streaming ``BackboneStreaming`` output
equals the PT non-streaming ``Backbone.forward`` output on complete spectrograms,
within float32 tolerance, after the documented align/trim, for **both**
``infer_type='mapping'`` and ``infer_type='masking'``. The gate runs on a
synthetic small-channel ``Backbone`` (fast) and — when present locally — on the
real ``bm_map_50ms`` / ``bm_mask_50ms`` checkpoints.

Tolerances. The "float32 tolerance" here is deliberately generous on the *phase*
and *complex* outputs because the model's ``atan2`` phase decoder is inherently
ill-conditioned (the non-streaming model has the same conditioning): a ~1e-6
chunk-boundary perturbation propagates to ~1e-4 in phase through the 64-channel
decoder. The *magnitude* path (encoder + TS-blocks + mask decoder) is the tight
correctness check (~1e-6 reordering floor). Observed maxima (real 50 ms ckpt,
spectrogram-domain): ``|dmag| ~ 7e-7``, ``|dcom| ~ 3e-5``, ``|dpha_wrapped| ~
3e-4``, ``corr = 1.000000``. The asserted bounds give ~10x+ margin.

The chunk-level correlation / max-abs prints are diagnostic only — the gate is
full-spectrogram parity.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.checkpoint import load_checkpoint, load_model
from src.models.backbone import Backbone
from src.models.streaming.backbone_streaming import BackboneStreaming
from src.stft import mag_pha_stft, mag_pha_to_complex

# --- float32-tolerance bounds for the parity gate (see module docstring) ---
MAG_TOL = 1e-4  # tight: encoder + TS-blocks + mask decoder; observed ~7e-7
COM_TOL = 1e-3  # looser: complex output folds in the ill-conditioned phase; observed ~3e-5
PHA_TOL = 2e-3  # loosest: atan2 phase decoder conditioning; observed ~3e-4 (wrapped)
CORR_MIN = 0.9999  # observed 1.000000

# 50 ms BAFNet+ Backbone anchors (re-verified at runtime via compute_lookahead).
CHUNK_SIZE = 8
N_FFT, HOP, WIN, COMPRESS = 400, 100, 400, 0.3

_REAL_CKPTS = {
    "mapping": Path(__file__).resolve().parents[1] / "results" / "experiments" / "bm_map_50ms",
    "masking": Path(__file__).resolve().parents[1] / "results" / "experiments" / "bm_mask_50ms",
}


# --------------------------------------------------------------------------- helpers
def _synthetic_backbone(infer_type: str, dense_channel: int = 8) -> Backbone:
    """Small ``Backbone`` with the 50 ms padding ratios + ``dense_depth=4`` / causal TS.

    ``dense_channel`` is shrunk for speed, but ``dense_depth=4`` and
    ``causal_ts_block=True`` are kept so the lookahead (``L_enc == L_dec == 3``)
    and the streaming state behaviour are real.
    """
    return Backbone(
        n_fft=N_FFT,
        hop_size=HOP,
        win_size=WIN,
        dense_channel=dense_channel,
        sigmoid_beta=2.0,
        compress_factor=COMPRESS,
        dense_depth=4,
        num_tsblock=2,
        time_dw_kernel_size=3,
        time_block_kernel=[3, 5, 7, 11],
        freq_block_kernel=[3, 11, 23, 31],
        time_block_num=2,
        freq_block_num=2,
        causal_ts_block=True,
        encoder_padding_ratio=(0.9, 0.1),
        decoder_padding_ratio=(0.9, 0.1),
        sca_kernel_size=11,
        infer_type=infer_type,
    ).eval()


def _real_backbone(infer_type: str) -> Backbone:
    """Load a real ``bm_*_50ms`` Backbone from its Hydra config + ``best.th``."""
    from omegaconf import OmegaConf

    ckpt_dir = _REAL_CKPTS[infer_type]
    conf = OmegaConf.load(ckpt_dir / ".hydra" / "config.yaml")
    bb = load_model(conf.model.model_lib, conf.model.model_class, conf.model.param, "cpu")
    bb = load_checkpoint(bb, str(ckpt_dir), "best.th", "cpu")
    return bb.eval()


def _wrapped_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``|atan2(sin(a-b), cos(a-b))|`` — phase difference modulo 2π."""
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d)).abs()


def _assert_spectrogram_parity(backbone: Backbone, streaming: BackboneStreaming, noisy_com: torch.Tensor, tag: str):
    """Compare streaming vs non-streaming on the SAME ``noisy_com`` with the documented trim.

    Documented align/trim: streaming output frame ``k`` == reference frame ``k``
    (no positional shift); the last ``total_lookahead`` produced frames are
    excluded (they used zero-padded lookahead because the input ran out). Prints
    the diagnostic maxima/correlation; asserts the float32-tolerance bounds.
    """
    with torch.no_grad():
        ref_mag, ref_pha, ref_com = backbone(noisy_com)
    str_mag, str_pha, str_com = streaming.process_spectrogram(noisy_com)

    t_stream = str_mag.shape[2]
    assert t_stream > streaming.total_lookahead, f"{tag}: too few streamed frames ({t_stream})"
    safe_t = t_stream - streaming.total_lookahead

    rm, rp, rc = ref_mag[:, :, :safe_t], ref_pha[:, :, :safe_t], ref_com[:, :, :safe_t, :]
    sm, sp, sc = str_mag[:, :, :safe_t], str_pha[:, :, :safe_t], str_com[:, :, :safe_t, :]

    dmag = (rm - sm).abs().max().item()
    dcom = (rc - sc).abs().max().item()
    dpha = _wrapped_abs_diff(rp, sp).max().item()
    corr = float(np.corrcoef(rc.flatten().numpy(), sc.flatten().numpy())[0, 1])
    print(
        f"[{tag}] spectrogram-domain (T_total={noisy_com.shape[2]}, T_stream={t_stream}, safe_T={safe_t}): "
        f"max|dmag|={dmag:.3e} max|dcom|={dcom:.3e} max|dpha_wrapped|={dpha:.3e} est_com_corr={corr:.8f}"
    )

    assert math.isfinite(dmag) and dmag < MAG_TOL, f"{tag}: |dmag|={dmag:.3e} >= {MAG_TOL}"
    assert math.isfinite(dcom) and dcom < COM_TOL, f"{tag}: |dcom|={dcom:.3e} >= {COM_TOL}"
    assert math.isfinite(dpha) and dpha < PHA_TOL, f"{tag}: |dpha_wrapped|={dpha:.3e} >= {PHA_TOL}"
    assert corr > CORR_MIN, f"{tag}: est_com corr={corr:.8f} <= {CORR_MIN}"


# --------------------------------------------------------------- geometry / config
def test_chunk_geometry_matches_50ms_anchors():
    """``from_model`` on a 50 ms-config Backbone derives the locked geometry."""
    streaming = BackboneStreaming.from_model(_synthetic_backbone("mapping"), chunk_size=CHUNK_SIZE, device="cpu")
    assert (streaming.encoder_lookahead, streaming.decoder_lookahead) == (3, 3)
    assert streaming.total_lookahead == 6
    assert streaming.total_frames_needed == CHUNK_SIZE + 3 == 11
    assert streaming.samples_per_chunk == (11 - 1) * HOP + WIN // 2 == 1200
    assert streaming.output_frames_per_chunk == CHUNK_SIZE == 8
    assert streaming.output_samples_per_chunk == CHUNK_SIZE * HOP == 800
    assert streaming.latency_samples == 6 * HOP + WIN // 2 == 800
    assert streaming.latency_ms == pytest.approx(50.0)
    assert streaming.infer_type == "mapping"
    # streaming_config exposes the same geometry + provenance.
    cfg = streaming.streaming_config
    assert cfg["source"] == "from_model" and cfg["T_export_planned"] == CHUNK_SIZE + 6 == 14
    assert [r for _, r in cfg["encoder_breakdown"]] == [0, 0, 1, 2]
    assert [r for _, r in cfg["decoder_breakdown"]] == [0, 0, 1, 2]


def test_ref_frame_count_matches_center_true_stft():
    """``process_audio``'s ``ref_T = 1 + len(audio)//hop`` == ``mag_pha_stft(center=True)`` frames."""
    for length in (8000, 12345, 16000, 16400):
        audio = torch.zeros(1, length)
        n_frames = mag_pha_stft(audio, N_FFT, HOP, WIN, compress_factor=COMPRESS, center=True)[2].shape[2]
        assert n_frames == 1 + length // HOP


def test_from_model_leaves_source_backbone_non_streaming():
    """``from_model`` deep-copies — the passed Backbone stays usable as the non-streaming ref."""
    bb = _synthetic_backbone("masking")
    _ = BackboneStreaming.from_model(bb, chunk_size=CHUNK_SIZE, device="cpu")
    # No stateful conv leaked into bb (it would break the plain forward path).
    from src.models.streaming.layers.stateful_conv import StatefulAsymmetricConv2d

    assert not any(isinstance(m, StatefulAsymmetricConv2d) for m in bb.modules())
    audio = torch.randn(1, 4000)
    noisy_com = mag_pha_stft(audio, N_FFT, HOP, WIN, compress_factor=COMPRESS, center=True)[2]
    with torch.no_grad():
        out = bb(noisy_com)
    assert out[0].shape[2] == noisy_com.shape[2]


def test_from_model_rejects_non_module():
    with pytest.raises(TypeError):
        BackboneStreaming.from_model(object(), chunk_size=CHUNK_SIZE)  # type: ignore[arg-type]


# ----------------------------------------------------------- the S3 exit gate
@pytest.mark.parametrize("infer_type", ["mapping", "masking"])
def test_spectrogram_parity_synthetic(infer_type):
    """PT streaming == PT non-streaming on a complete spectrogram (synthetic Backbone)."""
    torch.manual_seed(2039)
    backbone = _synthetic_backbone(infer_type)
    streaming = BackboneStreaming.from_model(backbone, chunk_size=CHUNK_SIZE, device="cpu")
    audio = torch.randn(1, 16000)
    noisy_com = mag_pha_stft(audio, N_FFT, HOP, WIN, compress_factor=COMPRESS, center=True)[2]
    _assert_spectrogram_parity(backbone, streaming, noisy_com, f"synthetic/{infer_type}")


@pytest.mark.parametrize("infer_type", ["mapping", "masking"])
def test_spectrogram_parity_real_checkpoint(infer_type):
    """PT streaming == PT non-streaming on a complete spectrogram (real bm_*_50ms ckpt)."""
    ckpt_dir = _REAL_CKPTS[infer_type]
    if not (ckpt_dir / "best.th").exists() or not (ckpt_dir / ".hydra" / "config.yaml").exists():
        pytest.skip(f"real checkpoint not present: {ckpt_dir}")
    torch.manual_seed(2039)
    backbone = _real_backbone(infer_type)
    # Build via from_checkpoint (the production path) and cross-check it matches from_model.
    streaming = BackboneStreaming.from_checkpoint(
        str(ckpt_dir), "best.th", chunk_size=CHUNK_SIZE, device="cpu", verbose=False
    )
    assert streaming.infer_type == infer_type
    assert (streaming.encoder_lookahead, streaming.decoder_lookahead) == (3, 3)
    audio = torch.randn(1, 16000) * 0.05  # low-amplitude noise; the gate holds for any input
    noisy_com = mag_pha_stft(audio, N_FFT, HOP, WIN, compress_factor=COMPRESS, center=True)[2]
    _assert_spectrogram_parity(backbone, streaming, noisy_com, f"real/{infer_type}")

    streaming_fm = BackboneStreaming.from_model(_real_backbone(infer_type), chunk_size=CHUNK_SIZE, device="cpu")
    a, _, c = streaming.process_spectrogram(noisy_com)
    a2, _, c2 = streaming_fm.process_spectrogram(noisy_com)
    assert torch.equal(a, a2) and torch.equal(c, c2)


# ------------------------------------------------------- process_samples / process_audio
@pytest.mark.parametrize("infer_type", ["mapping", "masking"])
def test_process_samples_warmup_and_shapes(infer_type):
    """``process_samples`` returns ``None`` while warming up, then matured triples of the right shape."""
    torch.manual_seed(2039)
    backbone = _synthetic_backbone(infer_type)
    streaming = BackboneStreaming.from_model(backbone, chunk_size=CHUNK_SIZE, device="cpu")
    streaming.reset_state()
    f_bins = N_FFT // 2 + 1
    audio = torch.randn(16000)
    flush = streaming.samples_per_chunk * (streaming.total_lookahead + 2)
    padded = torch.cat([audio, torch.zeros(flush)])

    warmup_nones = 0
    first_out_shape = None
    n_outputs = 0
    for i in range(0, len(padded), streaming.output_samples_per_chunk):
        chunk = padded[i : i + streaming.output_samples_per_chunk]
        if len(chunk) == 0:
            break
        result = streaming.process_samples(chunk)
        if result is None:
            if n_outputs == 0:
                warmup_nones += 1
            continue
        n_outputs += 1
        est_mag, est_pha, est_com = result
        if first_out_shape is None:
            first_out_shape = (tuple(est_mag.shape), tuple(est_pha.shape), tuple(est_com.shape))
        assert est_mag.shape == (1, f_bins, CHUNK_SIZE)
        assert est_pha.shape == (1, f_bins, CHUNK_SIZE)
        assert est_com.shape == (1, f_bins, CHUNK_SIZE, 2)
        # est_com is exactly mag_pha_to_complex(est_mag, est_pha).
        assert torch.equal(est_com, mag_pha_to_complex(est_mag, est_pha, stack_dim=-1))

    print(
        f"[{infer_type}] process_samples: {warmup_nones} warm-up None(s) before first output; "
        f"{n_outputs} matured chunks; first output shapes={first_out_shape}"
    )
    assert warmup_nones >= 1  # the lookahead pipeline must delay the first output
    assert n_outputs >= 1


@pytest.mark.parametrize("infer_type", ["mapping", "masking"])
def test_process_audio_equals_chunked_process_samples(infer_type):
    """``process_audio`` == manual ``process_samples`` loop (cat'd, then trimmed to ref_T)."""
    torch.manual_seed(2039)
    streaming = BackboneStreaming.from_model(_synthetic_backbone(infer_type), chunk_size=CHUNK_SIZE, device="cpu")
    audio = torch.randn(1, 12000)
    pa_mag, pa_pha, pa_com = streaming.process_audio(audio)

    # Replicate process_audio's loop manually.
    streaming.reset_state()
    flush = streaming.samples_per_chunk * (streaming.total_lookahead + 2)
    padded = torch.cat([audio.squeeze(0), torch.zeros(flush)])
    mags, phas, coms = [], [], []
    for i in range(0, len(padded), streaming.output_samples_per_chunk):
        chunk = padded[i : i + streaming.output_samples_per_chunk]
        if len(chunk) == 0:
            break
        result = streaming.process_samples(chunk)
        if result is not None:
            mags.append(result[0])
            phas.append(result[1])
            coms.append(result[2])
    cat_mag = torch.cat(mags, dim=2)
    cat_pha = torch.cat(phas, dim=2)
    cat_com = torch.cat(coms, dim=2)
    ref_t = 1 + audio.shape[1] // HOP
    assert pa_mag.shape[2] == min(cat_mag.shape[2], ref_t)
    assert torch.equal(pa_mag, cat_mag[:, :, : pa_mag.shape[2]])
    assert torch.equal(pa_pha, cat_pha[:, :, : pa_pha.shape[2]])
    assert torch.equal(pa_com, cat_com[:, :, : pa_com.shape[2], :])


@pytest.mark.parametrize("infer_type", ["mapping", "masking"])
def test_process_audio_matches_center_true_backbone(infer_type):
    """``process_audio`` ≈ ``Backbone(mag_pha_stft(audio, center=True))`` (host STFT-context emulation).

    Uses ``win//2`` leading + trailing zeros so the ``center=True`` reflect-pad
    equals the streaming wrapper's zero past-context / zero flush, making the
    host STFT emulation bit-exact-ish (the residual is the ``center=True`` vs
    ``center=False`` STFT-rounding floor, amplified through the deep net + atan2).
    """
    torch.manual_seed(2039)
    backbone = _synthetic_backbone(infer_type)
    streaming = BackboneStreaming.from_model(backbone, chunk_size=CHUNK_SIZE, device="cpu")
    core = torch.randn(16000)
    audio = torch.cat([torch.zeros(WIN // 2), core, torch.zeros(WIN // 2)])
    noisy_com = mag_pha_stft(audio.unsqueeze(0), N_FFT, HOP, WIN, compress_factor=COMPRESS, center=True)[2]
    with torch.no_grad():
        ref_mag, ref_pha, ref_com = backbone(noisy_com)
    str_mag, str_pha, str_com = streaming.process_audio(audio)

    assert str_mag.shape[2] == ref_mag.shape[2] == noisy_com.shape[2]
    safe_t = str_mag.shape[2] - streaming.total_lookahead
    dmag = (ref_mag[:, :, :safe_t] - str_mag[:, :, :safe_t]).abs().max().item()
    dcom = (ref_com[:, :, :safe_t, :] - str_com[:, :, :safe_t, :]).abs().max().item()
    dpha = _wrapped_abs_diff(ref_pha[:, :, :safe_t], str_pha[:, :, :safe_t]).max().item()
    print(
        f"[{infer_type}] process_audio vs Backbone(center=True): "
        f"max|dmag|={dmag:.3e} max|dcom|={dcom:.3e} max|dpha_wrapped|={dpha:.3e}"
    )
    assert dmag < MAG_TOL
    assert dcom < COM_TOL
    assert dpha < PHA_TOL


@pytest.mark.parametrize("infer_type", ["mapping", "masking"])
def test_reset_state_restarts_stream(infer_type):
    """``reset_state`` fully clears streaming state — re-processing the same audio is identical."""
    torch.manual_seed(2039)
    streaming = BackboneStreaming.from_model(_synthetic_backbone(infer_type), chunk_size=CHUNK_SIZE, device="cpu")
    audio = torch.randn(1, 8000)
    out1 = streaming.process_audio(audio)  # process_audio resets internally
    out2 = streaming.process_audio(audio)
    for a, b in zip(out1, out2):
        assert torch.equal(a, b)
    # Also: process_spectrogram resets too.
    noisy_com = mag_pha_stft(audio, N_FFT, HOP, WIN, compress_factor=COMPRESS, center=True)[2]
    s1 = streaming.process_spectrogram(noisy_com)
    s2 = streaming.process_spectrogram(noisy_com)
    for a, b in zip(s1, s2):
        assert torch.equal(a, b)


def test_process_samples_rejects_batched_input():
    streaming = BackboneStreaming.from_model(_synthetic_backbone("masking"), chunk_size=CHUNK_SIZE, device="cpu")
    with pytest.raises(ValueError):
        streaming.process_samples(torch.randn(2, 1000))


def test_process_spectrogram_rejects_batched_input():
    streaming = BackboneStreaming.from_model(_synthetic_backbone("masking"), chunk_size=CHUNK_SIZE, device="cpu")
    with pytest.raises(ValueError):
        streaming.process_spectrogram(torch.randn(2, N_FFT // 2 + 1, 40, 2))
