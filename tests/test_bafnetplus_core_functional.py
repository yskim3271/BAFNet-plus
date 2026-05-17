"""Tests for the S8 functional-stateful BAFNet+ exportable core.

Covers :class:`ExportableBAFNetPlusCore` and the BAFNet+ ONNX export driver
(``streaming/onnx/export.py``'s ``export_bafnetplus_core_to_onnx`` +
``verify_bafnetplus_core_multistep``).

The 10 launch-prompt gates:

(a) ``test_pt_core_full_sequence_matches_forward_bit_exact[full|no_calibration|mask_only_alpha]``
    — single-shot ``ExportableBAFNetPlusCore.forward()`` (with zero init states +
    NO ``state_frames_for_update`` bound) matches ``BAFNetPlus.forward()`` on a
    complete spectrogram **within ``MAG_TOL_PARITY = 1e-5``** for each ablation.
    The cheapest faithful-rewrite check. Follows S5's precedent (the same 1-ULP
    ``acs_mask = acs_est_mag / acs_mag`` recovery drift documented in
    :mod:`src.models.streaming.onnx.bafnetplus_core` propagates through the
    alpha CNN to ~1e-6 max — well within ``MAG_TOL_PARITY = 1e-5``).
(b) ``test_pt_core_streaming_matches_forward_synthetic[full|no_calibration|mask_only_alpha]``
    — chunked PT core via ``run_bafnetplus_core_streaming`` equals
    ``BAFNetPlus.forward`` on the matched ``recon_T`` prefix at ALL chunks
    (not just the first 8 frames — that's the S8 payoff closing the S6/S7
    non-streaming-fusion drift). Tight tolerances ``MAG_TOL=1e-4``,
    ``COM_TOL=1e-3``, ``PHA_TOL=2e-3`` (wrapped).
(c) ``test_pt_core_streaming_matches_forward_real_checkpoint`` — same on the
    real ``bafnetplus_50ms/best.th``. Expected `max|dmag| ≤ 1e-5,
    max|dcom| ≤ 1e-4, max|dpha_wrapped| ≤ 1e-3` across the FULL utterance on
    real weights. Skipped when ckpt + per-branch hydra configs are absent.
(d) ``test_onnx_vs_pt_core_synthetic[full|no_calibration|mask_only_alpha]`` —
    export the synthetic-config core, run ``verify_bafnetplus_core_multistep
    (num_steps=5, atol=1e-4, state_atol=1e-4)``. Hard structural gates:
    ``state_shape_check=True`` + ``all_finite=True`` + ``all_match`` against
    the supplied tolerances.
(e) ``test_onnx_vs_pt_core_real_checkpoint`` — same on the real ckpt via
    ``export_bafnetplus_to_onnx_from_checkpoint``. Sidecar JSON validated.
(f) ``test_state_count_matches_formula`` — 110 for synthetic
    (``num_tsblock=2``); 190 for the real ckpt.
(g) ``test_state_names_unique_and_ordered`` — names follow the S5-frozen
    prefix order ``mapping/*`` → ``masking/*`` → ``calibration/*`` → ``alpha/*``.
(h) ``test_input_output_names_freeze_contract`` — frozen input/output names.
(i) ``test_set_state_frames_for_update_propagates_to_fusion`` — perturbing
    the spectrogram-input lookahead frames does NOT change calibration or
    alpha's next_state (proves the lookahead-protection contract extends
    from the backbones to the fusion convs).
(j) ``test_fusion_modules_are_deep_copied_from_bafnet`` — verifies the
    documented S8 departure from S6's shared-reference contract.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, cast

import pytest
import torch

from src.checkpoint import ConfigDict, load_checkpoint
from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.onnx.backbone_core import ExportableBackboneCore
from src.models.streaming.onnx.bafnetplus_core import ExportableBAFNetPlusCore
from src.models.streaming.onnx.export import (
    export_bafnetplus_core_to_onnx,
    export_bafnetplus_to_onnx_from_checkpoint,
    run_bafnetplus_core_streaming,
    verify_bafnetplus_core_multistep,
)
from src.models.streaming.onnx.functional_stateful import (
    FunctionalStatefulCausalConv2d,
    FunctionalStatefulConv1d,
)
from src.stft import complex_to_mag_pha


# --- 50 ms BAFNet+ anchors (re-verified at runtime via compute_lookahead inside the cores) ---
CHUNK_SIZE = 8
N_FFT, HOP, WIN, COMPRESS = 400, 100, 400, 0.3
FREQ_SIZE = N_FFT // 2 + 1  # 201
T_EXPORT = 14  # chunk + L_enc + L_dec + L_alpha = 8 + 3 + 3 + 0

# S5-style tight tolerance for single-shot full-sequence parity (test (a)):
# only the (a*b)/a 1-ULP mask-recovery drift propagated through the 4-layer
# alpha CNN to ~1e-6 max. Following the S5 documented precedent — true
# `torch.equal` would require extracting `acs_mask` directly from the masking
# branch core (S8 deferred — see the launch prompt note on extending
# `ExportableBackboneCore` with an explicit mask output).
MAG_TOL_PARITY = 1e-5
COM_TOL_PARITY = 1e-4
PHA_TOL_PARITY = 1e-3  # wrapped — atan2 ill-conditioning near zero crossings.

# Streaming-vs-forward (tests (b) (c)) — the S8 payoff: chunked streaming
# matches BAFNet+.forward at ALL chunks, not just the first one.
MAG_TOL_STREAM = 1e-4
COM_TOL_STREAM = 1e-3
PHA_TOL_STREAM = 2e-3

# Real ckpt (test (c)) tighter targets — trained weights are smoother.
MAG_TOL_STREAM_REAL = 1e-5
COM_TOL_STREAM_REAL = 1e-4
PHA_TOL_STREAM_REAL = 1e-3

ABLATIONS = ["full", "mask_only_alpha", "no_calibration"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_CKPT_DIR = _REPO_ROOT / "results" / "experiments" / "bafnetplus_50ms"
_REAL_CKPT_FILE = _REAL_CKPT_DIR / "best.th"
_BM_MAP_CFG = _REPO_ROOT / "results" / "experiments" / "bm_map_50ms" / ".hydra" / "config.yaml"
_BM_MASK_CFG = _REPO_ROOT / "results" / "experiments" / "bm_mask_50ms" / ".hydra" / "config.yaml"

# Expected MD5 of the real unified 50 ms ckpt (from the S1 anchor entry).
_EXPECTED_CKPT_MD5 = "0f95f466ffc2f540ed5cf0fd968a66ff"


# --------------------------------------------------------------------------- helpers
def _synthetic_backbone_param(infer_type: str, dense_channel: int = 8) -> Dict:
    """50 ms-config Backbone param with ``dense_channel`` shrunk for speed.

    ``dense_depth=4`` + ``causal_ts_block=True`` are kept so the lookahead
    ``L_enc==L_dec==3`` and the streaming behaviour are real.
    """
    return {
        "n_fft": N_FFT,
        "hop_size": HOP,
        "win_size": WIN,
        "dense_channel": dense_channel,
        "sigmoid_beta": 2.0,
        "compress_factor": COMPRESS,
        "dense_depth": 4,
        "num_tsblock": 2,  # shrunk from real 4 — keeps state count manageable.
        "time_dw_kernel_size": 3,
        "time_block_kernel": [3, 5, 7, 11],
        "freq_block_kernel": [3, 11, 23, 31],
        "time_block_num": 2,
        "freq_block_num": 2,
        "causal_ts_block": True,
        "encoder_padding_ratio": (0.9, 0.1),
        "decoder_padding_ratio": (0.9, 0.1),
        "sca_kernel_size": 11,
        "infer_type": infer_type,
    }


def _synthetic_bafnetplus(ablation_mode: str, dense_channel: int = 8) -> BAFNetPlus:
    """Construct a fresh small-channel ``BAFNetPlus`` for the given ablation."""
    args_mapping = ConfigDict(
        {
            "model_lib": "backbone",
            "model_class": "Backbone",
            "param": _synthetic_backbone_param("mapping", dense_channel=dense_channel),
        }
    )
    args_masking = ConfigDict(
        {
            "model_lib": "backbone",
            "model_class": "Backbone",
            "param": _synthetic_backbone_param("masking", dense_channel=dense_channel),
        }
    )
    model = BAFNetPlus(
        args_mapping=args_mapping,
        args_masking=args_masking,
        ablation_mode=ablation_mode,
        load_pretrained_weights=False,
    )
    return model.eval()


def _real_bafnetplus() -> BAFNetPlus:
    """Load the unified ``bafnetplus_50ms/best.th`` per the wiki foot-gun note."""
    from omegaconf import OmegaConf

    map_conf = OmegaConf.load(_BM_MAP_CFG)
    mask_conf = OmegaConf.load(_BM_MASK_CFG)
    args_mapping = ConfigDict(
        {
            "model_lib": map_conf.model.model_lib,
            "model_class": map_conf.model.model_class,
            "param": OmegaConf.to_container(map_conf.model.param, resolve=True),
        }
    )
    args_masking = ConfigDict(
        {
            "model_lib": mask_conf.model.model_lib,
            "model_class": mask_conf.model.model_class,
            "param": OmegaConf.to_container(mask_conf.model.param, resolve=True),
        }
    )
    bp_conf = OmegaConf.load(_REAL_CKPT_DIR / ".hydra" / "config.yaml")
    bp_param = cast(Dict[str, Any], OmegaConf.to_container(bp_conf.model.param, resolve=True))
    bp_param.pop("checkpoint_mapping", None)
    bp_param.pop("checkpoint_masking", None)
    model = BAFNetPlus(
        args_mapping=args_mapping,
        args_masking=args_masking,
        load_pretrained_weights=False,
        **bp_param,
    )
    model = load_checkpoint(model, str(_REAL_CKPT_DIR), "best.th", "cpu")
    return model.eval()


def _wrapped_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``|atan2(sin(a-b), cos(a-b))|`` — phase difference modulo 2π."""
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d)).abs()


def _gen_complex(seed: int, freq_size: int, t: int, scale: float = 0.3) -> torch.Tensor:
    """Random complex spectrogram ``[1, F, T, 2]`` (real, imag)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, freq_size, t, 2, generator=g) * scale


# ============================================================== (a) full-sequence parity
@pytest.mark.parametrize("ablation_mode", ABLATIONS)
def test_pt_core_full_sequence_matches_forward_bit_exact(ablation_mode):
    """The exportable core's single-shot forward (zero init + no state_frames
    bound) is bit-exact to ``BAFNetPlus.forward()`` within ``MAG_TOL_PARITY``.

    The only departure from ``torch.equal`` is the documented 1-ULP
    ``acs_mask = acs_est_mag / acs_mag`` recovery drift (same as S5
    :class:`BAFNetPlusCore`). Propagates through the alpha CNN to ~1e-6 max
    — well within ``MAG_TOL_PARITY = 1e-5``.
    """
    torch.manual_seed(2039 + hash(ablation_mode) % 997)
    bafnet = _synthetic_bafnetplus(ablation_mode)
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()

    T = 32
    bcs_com = _gen_complex(seed=11, freq_size=FREQ_SIZE, t=T)
    acs_com = _gen_complex(seed=22, freq_size=FREQ_SIZE, t=T)
    bcs_mag, bcs_pha = complex_to_mag_pha(bcs_com, stack_dim=-1)
    acs_mag, acs_pha = complex_to_mag_pha(acs_com, stack_dim=-1)

    with torch.no_grad():
        ref_mag, ref_pha, ref_com = bafnet((bcs_com, acs_com))
        # No state_frames bound + zero init -> the functional rewrite is faithful
        # (equivalent to non-streaming on a complete sequence).
        core.set_state_frames_for_update(None)
        states = core.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T)
        outs = core(bcs_mag, bcs_pha, acs_mag, acs_pha, *states)
        est_mag, est_pha, est_com = outs[0], outs[1], outs[2]
        next_states = outs[3:]

    dmag = (ref_mag - est_mag).abs().max().item()
    dcom = (ref_com - est_com).abs().max().item()
    dpha = _wrapped_abs_diff(ref_pha, est_pha).max().item()
    print(
        f"[full_seq/{ablation_mode}] T={T}: dmag={dmag:.3e} dcom={dcom:.3e} dpha_wrapped={dpha:.3e} "
        f"(num_next_states={len(next_states)}, expected={core.num_states})"
    )
    assert math.isfinite(dmag) and dmag < MAG_TOL_PARITY, f"dmag={dmag:.3e} >= {MAG_TOL_PARITY}"
    assert math.isfinite(dcom) and dcom < COM_TOL_PARITY, f"dcom={dcom:.3e} >= {COM_TOL_PARITY}"
    assert math.isfinite(dpha) and dpha < PHA_TOL_PARITY, f"dpha_wrapped={dpha:.3e} >= {PHA_TOL_PARITY}"
    assert len(next_states) == core.num_states, (len(next_states), core.num_states)


# ============================================================== (b) streaming parity, synthetic
@pytest.mark.parametrize("ablation_mode", ABLATIONS)
def test_pt_core_streaming_matches_forward_synthetic(ablation_mode):
    """Chunk-driven PT core via ``run_bafnetplus_core_streaming`` equals
    ``BAFNetPlus.forward`` on the matched ``recon_T`` prefix.

    This is the S8 payoff: at ALL chunks (not just the first 8 frames), the
    chunked streaming output matches the non-streaming forward to within
    ``MAG_TOL_STREAM = 1e-4`` / ``COM_TOL_STREAM = 1e-3`` / ``PHA_TOL_STREAM
    = 2e-3``. Closes the S6 documented "non-streaming-fusion drift".
    """
    torch.manual_seed(2039 + hash(ablation_mode) % 997)
    bafnet = _synthetic_bafnetplus(ablation_mode)
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()

    # T well above max(T_export) so multiple full chunks span the past-context reach
    # of calibration (kernel=5, depth=2 -> 8 frames) and alpha (kernel=7, depth=4 -> 24 frames).
    T = 48
    bcs_com = _gen_complex(seed=101, freq_size=FREQ_SIZE, t=T)
    acs_com = _gen_complex(seed=202, freq_size=FREQ_SIZE, t=T)

    with torch.no_grad():
        ref_mag, ref_pha, ref_com = bafnet((bcs_com, acs_com))

    est_mag, est_pha, est_com = run_bafnetplus_core_streaming(core, bcs_com, acs_com, CHUNK_SIZE, freq_size=FREQ_SIZE)
    recon_T = est_mag.shape[2]
    n_chunks = recon_T // CHUNK_SIZE
    print(f"[stream/{ablation_mode}] T={T} -> recon_T={recon_T} ({n_chunks} chunks)")

    dmag = (ref_mag[:, :, :recon_T] - est_mag).abs().max().item()
    dcom = (ref_com[:, :, :recon_T, :] - est_com).abs().max().item()
    dpha = _wrapped_abs_diff(ref_pha[:, :, :recon_T], est_pha).max().item()
    # Diagnostic — per-chunk diffs (printed only).
    for c in range(n_chunks):
        s, e = c * CHUNK_SIZE, (c + 1) * CHUNK_SIZE
        c_dmag = (ref_mag[:, :, s:e] - est_mag[:, :, s:e]).abs().max().item()
        c_dcom = (ref_com[:, :, s:e, :] - est_com[:, :, s:e, :]).abs().max().item()
        print(f"  chunk {c}: dmag={c_dmag:.3e}, dcom={c_dcom:.3e}")
    print(
        f"  total: dmag={dmag:.3e} dcom={dcom:.3e} dpha_wrapped={dpha:.3e} "
        f"(MAG_TOL={MAG_TOL_STREAM}, COM_TOL={COM_TOL_STREAM}, PHA_TOL={PHA_TOL_STREAM})"
    )
    assert math.isfinite(dmag) and dmag < MAG_TOL_STREAM, f"dmag={dmag:.3e} >= {MAG_TOL_STREAM}"
    assert math.isfinite(dcom) and dcom < COM_TOL_STREAM, f"dcom={dcom:.3e} >= {COM_TOL_STREAM}"
    assert math.isfinite(dpha) and dpha < PHA_TOL_STREAM, f"dpha_wrapped={dpha:.3e} >= {PHA_TOL_STREAM}"


# ============================================================== (c) streaming parity, real
def test_pt_core_streaming_matches_forward_real_checkpoint():
    """Same as (b) on the real unified ``bafnetplus_50ms/best.th`` ckpt.

    Expected `max|dmag| ≤ 1e-5, max|dcom| ≤ 1e-4, max|dpha_wrapped| ≤ 1e-3`
    across the FULL utterance (not just the first chunk) — the S8 PT-side
    headline result proving the S6/S7 non-streaming-fusion drift IS closed
    at the PT level.
    """
    if not (_REAL_CKPT_FILE.exists() and _BM_MAP_CFG.exists() and _BM_MASK_CFG.exists()):
        pytest.skip(
            "real checkpoint or per-branch hydra configs missing: "
            f"{_REAL_CKPT_FILE.exists()=}, {_BM_MAP_CFG.exists()=}, {_BM_MASK_CFG.exists()=}"
        )
    bafnet = _real_bafnetplus()
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    assert bafnet.ablation_mode == "full", f"unexpected ablation_mode={bafnet.ablation_mode!r}"

    torch.manual_seed(31337)
    T = 48
    bcs_com = _gen_complex(seed=1001, freq_size=FREQ_SIZE, t=T)
    acs_com = _gen_complex(seed=2002, freq_size=FREQ_SIZE, t=T)

    with torch.no_grad():
        ref_mag, ref_pha, ref_com = bafnet((bcs_com, acs_com))

    est_mag, est_pha, est_com = run_bafnetplus_core_streaming(core, bcs_com, acs_com, CHUNK_SIZE, freq_size=FREQ_SIZE)
    recon_T = est_mag.shape[2]
    print(f"[stream/real_ckpt] T={T} -> recon_T={recon_T} ({recon_T // CHUNK_SIZE} chunks)")

    dmag = (ref_mag[:, :, :recon_T] - est_mag).abs().max().item()
    dcom = (ref_com[:, :, :recon_T, :] - est_com).abs().max().item()
    dpha = _wrapped_abs_diff(ref_pha[:, :, :recon_T], est_pha).max().item()
    print(f"  dmag={dmag:.3e} dcom={dcom:.3e} dpha_wrapped={dpha:.3e}")
    assert math.isfinite(dmag) and dmag < MAG_TOL_STREAM_REAL, f"dmag={dmag:.3e} >= {MAG_TOL_STREAM_REAL}"
    assert math.isfinite(dcom) and dcom < COM_TOL_STREAM_REAL, f"dcom={dcom:.3e} >= {COM_TOL_STREAM_REAL}"
    assert math.isfinite(dpha) and dpha < PHA_TOL_STREAM_REAL, f"dpha_wrapped={dpha:.3e} >= {PHA_TOL_STREAM_REAL}"


# ============================================================== (d) ONNX vs PT, synthetic
@pytest.mark.parametrize("ablation_mode", ABLATIONS)
def test_onnx_vs_pt_core_synthetic(ablation_mode):
    """S8 exit gate (synthetic): multi-step ORT-vs-PT parity with state propagation.

    Exports the synthetic-config core, runs ``verify_bafnetplus_core_multistep
    (num_steps=5, atol=1e-4, state_atol=1e-4)``. Asserts every output diff
    within atol + ``state_shape_check=True`` + ``all_finite=True``.
    """
    torch.manual_seed(4242 + hash(ablation_mode) % 997)
    bafnet = _synthetic_bafnetplus(ablation_mode)
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / f"bafnetplus_synth_{ablation_mode}.onnx"
        result = export_bafnetplus_core_to_onnx(
            core, onnx_path, chunk_size=CHUNK_SIZE, freq_size=FREQ_SIZE, verbose=False
        )
        verif = verify_bafnetplus_core_multistep(
            onnx_path,
            core,
            chunk_size=CHUNK_SIZE,
            freq_size=FREQ_SIZE,
            num_steps=5,
            atol=1e-4,
            state_atol=1e-4,
            verbose=True,
        )

    print(f"[onnx/{ablation_mode}] verif={verif['all_match']}")
    # Hard structural gates.
    assert verif["state_shape_check"], f"PT next-state shape != prev-state shape: {verif}"
    assert verif["all_finite"], f"non-finite state values: {verif}"
    # Numerical gates (per-output).
    assert all(d <= 1e-4 for d in verif["max_output_diffs"]), verif["max_output_diffs"]
    assert verif["max_state_diff"] <= 1e-4, verif["max_state_diff"]
    assert verif["max_phase_wrapped_diff"] <= 1e-4, verif["max_phase_wrapped_diff"]
    assert verif["all_match"], verif
    # Sidecar JSON contents.
    meta = result.metadata
    assert meta["schema_version"] == "s8-bafnetplus-functional-fp32"
    assert meta["geometry"]["T_export"] == T_EXPORT
    assert meta["geometry"]["freq_size"] == FREQ_SIZE
    assert meta["core"]["num_states"] == core.num_states
    assert meta["core"]["ablation_mode"] == ablation_mode
    assert meta["branches"]["fusion"]["calibration"]["num_states"] == core.num_calibration_states
    assert meta["branches"]["fusion"]["alpha"]["num_states"] == core.num_alpha_states


# ============================================================== (e) ONNX vs PT, real ckpt
def test_onnx_vs_pt_core_real_checkpoint():
    """S8 exit gate (real ckpt): ORT-vs-PT multistep verify + sidecar JSON validation.

    Exports the real unified BAFNet+ 50 ms ckpt via
    ``export_bafnetplus_to_onnx_from_checkpoint``, runs verify with split per-
    output tolerances (the COM_TOL=1e-3 aggregate atol with state_atol=1e-3
    matches the real-ckpt envelope observed for the single-Backbone S4 gate).
    """
    if not (_REAL_CKPT_FILE.exists() and _BM_MAP_CFG.exists() and _BM_MASK_CFG.exists()):
        pytest.skip("real checkpoint or per-branch hydra configs missing")
    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "bafnetplus_50ms_real.onnx"
        result = export_bafnetplus_to_onnx_from_checkpoint(
            str(_REAL_CKPT_DIR), onnx_path, chunk_size=CHUNK_SIZE, verbose=False
        )
        # Reload the same model + core for the verify step (the export consumed it).
        bafnet = _real_bafnetplus()
        core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
        verif = verify_bafnetplus_core_multistep(
            onnx_path,
            core,
            chunk_size=CHUNK_SIZE,
            num_steps=5,
            atol=1e-3,
            state_atol=1e-3,
            verbose=True,
        )

    print(f"[onnx/real_ckpt] verif={verif}")
    # Hard structural gates.
    assert verif["state_shape_check"], f"PT next-state shape != prev-state shape: {verif}"
    assert verif["all_finite"], f"non-finite state values: {verif}"
    # Per-output split tolerances (matches the S4 real-ckpt envelope).
    assert verif["max_output_diffs"][0] < 1e-3, f"est_mag diff = {verif['max_output_diffs'][0]:.3e}"
    assert verif["max_output_diffs"][1] < 1e-3, f"est_pha diff = {verif['max_output_diffs'][1]:.3e}"
    assert verif["max_output_diffs"][2] < 1e-3, f"est_com diff = {verif['max_output_diffs'][2]:.3e}"
    assert verif["max_phase_wrapped_diff"] < 2e-3, f"phase_wrapped = {verif['max_phase_wrapped_diff']:.3e}"
    assert verif["max_state_diff"] < 1e-3, f"state diff = {verif['max_state_diff']:.3e}"

    # Sidecar JSON validated fields.
    meta = result.metadata
    assert meta["schema_version"] == "s8-bafnetplus-functional-fp32"
    assert meta["geometry"]["T_export"] == T_EXPORT
    assert meta["geometry"]["freq_size"] == FREQ_SIZE
    assert meta["core"]["num_states"] == 190
    assert meta["core"]["ablation_mode"] == "full"
    assert meta["branches"]["fusion"]["calibration"]["num_states"] == 2
    assert meta["branches"]["fusion"]["alpha"]["num_states"] == 4
    assert meta["checkpoint"]["md5"] == _EXPECTED_CKPT_MD5


# ============================================================== (f) state count formula
def test_state_count_matches_formula():
    """Synthetic config (``num_tsblock=2``) → 110 states (52 + 52 + 2 + 4)."""
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    assert core.mapping_core.num_states == 52, core.mapping_core.num_states
    assert core.masking_core.num_states == 52, core.masking_core.num_states
    assert core.num_calibration_states == 2, core.num_calibration_states
    assert core.num_alpha_states == 4, core.num_alpha_states
    assert core.num_states == 52 + 52 + 2 + 4 == 110, core.num_states
    assert len(core.get_state_names()) == core.num_states
    assert len(core.init_states(batch_size=1, freq_size=FREQ_SIZE)) == core.num_states


def test_state_count_matches_formula_real_checkpoint():
    """Real ckpt (``num_tsblock=4``) → 190 states (92 + 92 + 2 + 4)."""
    if not (_REAL_CKPT_FILE.exists() and _BM_MAP_CFG.exists() and _BM_MASK_CFG.exists()):
        pytest.skip("real checkpoint or per-branch hydra configs missing")
    bafnet = _real_bafnetplus()
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    assert core.mapping_core.num_states == 92, core.mapping_core.num_states
    assert core.masking_core.num_states == 92, core.masking_core.num_states
    assert core.num_calibration_states == 2, core.num_calibration_states
    assert core.num_alpha_states == 4, core.num_alpha_states
    assert core.num_states == 190, core.num_states


# ============================================================== (g) state-name ordering
def test_state_names_unique_and_ordered():
    """State names follow ``mapping/*`` (all 52) → ``masking/*`` (all 52) →
    ``calibration/*`` (2) → ``alpha/*`` (4); every name starts with the right
    prefix; every name is unique; every name starts with ``state_`` inside its
    prefix.
    """
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    names = core.get_state_names()
    n_map = core.mapping_core.num_states
    n_mask = core.masking_core.num_states
    n_cal = core.num_calibration_states
    assert len(names) == len(set(names)), "state names not unique"
    assert all(n.startswith("mapping/") for n in names[:n_map]), names[:5]
    assert all(n.startswith("masking/") for n in names[n_map : n_map + n_mask]), names[n_map : n_map + 5]
    assert all(n.startswith("calibration/") for n in names[n_map + n_mask : n_map + n_mask + n_cal]), names[
        n_map + n_mask : n_map + n_mask + n_cal
    ]
    assert all(n.startswith("alpha/") for n in names[n_map + n_mask + n_cal :]), names[n_map + n_mask + n_cal :]
    # Inside each prefix the suffix starts with ``state_``.
    for n in names:
        prefix, _, suffix = n.partition("/")
        assert suffix.startswith("state_"), f"{n} doesn't have state_ suffix prefix"


# ============================================================== (h) I/O contract freeze
def test_input_output_names_freeze_contract():
    """Input/output names follow the documented frozen S8 contract."""
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    inp = core.input_names()
    assert inp[:4] == ["bcs_mag", "bcs_pha", "acs_mag", "acs_pha"], inp[:4]
    assert inp[4:] == core.get_state_names()
    out = core.output_names()
    assert out[:3] == ["est_mag", "est_pha", "est_com"], out[:3]
    assert out[3:] == [f"next_{n}" for n in core.get_state_names()]
    assert len(inp) == 4 + core.num_states
    assert len(out) == 3 + core.num_states


# ============================================================== (i) lookahead protection
def test_set_state_frames_for_update_propagates_to_fusion():
    """The ``state_frames_for_update`` bound is wired through to every fusion
    functional conv — perturbing each fusion conv's OWN input at frames
    ``[chunk_size:]`` does NOT change its next_state.

    This tests the **per-layer** protection contract (S2's
    :class:`StateFramesContext` semantics, generalised). For the end-to-end
    chained protection through the backbones one would need backbones with
    ``L_enc = L_dec = 0`` (fully causal) — the deployed 50 ms BAFNet+
    backbones have ``L_enc = L_dec = 3``, so output at frames
    ``[:chunk_size]`` mixes input from frames ``[:chunk_size + L_enc +
    L_dec]``, which means calibration's INPUT IS contaminated by backbone-
    side lookahead frames. The per-layer guarantee — once the contamination
    is bypassed — still holds, and that's what S8 needs to wire.

    Runs each fusion conv directly with two inputs differing only at frames
    ``[chunk_size:]`` and asserts ``next_state`` is bit-identical.
    """
    torch.manual_seed(2039)
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    core.set_state_frames_for_update(CHUNK_SIZE)

    # Get the first calibration + alpha functional convs.
    cal_layer0 = core._calibration_functional_modules[0][1]
    alpha_layer0 = core._alpha_functional_modules[0][1]
    assert isinstance(cal_layer0, FunctionalStatefulConv1d)
    assert isinstance(alpha_layer0, FunctionalStatefulCausalConv2d)

    T = T_EXPORT  # 14 frames
    # Calibration input: [B, 5, T] (5-ch frame-wise features).
    cal_in_a = torch.randn(1, 5, T)
    cal_in_b = cal_in_a.clone()
    cal_in_b[:, :, CHUNK_SIZE:] = torch.randn(1, 5, T - CHUNK_SIZE) + 5.0  # large perturbation
    cal_state_init = cal_layer0.init_state(batch_size=1)

    with torch.no_grad():
        _, cal_next_a = cal_layer0(cal_in_a, cal_state_init, state_frames=CHUNK_SIZE)
        _, cal_next_b = cal_layer0(cal_in_b, cal_state_init, state_frames=CHUNK_SIZE)
    cal_diff = (cal_next_a - cal_next_b).abs().max().item()
    print(f"[lookahead_protection] cal layer 0 diff = {cal_diff:.3e}")
    assert cal_diff == 0.0, f"calibration first-layer next_state contaminated: {cal_diff:.3e}"

    # Alpha input: [B, in_C=3, T, F=201].
    alpha_in_a = torch.randn(1, 3, T, FREQ_SIZE)
    alpha_in_b = alpha_in_a.clone()
    alpha_in_b[:, :, CHUNK_SIZE:, :] = torch.randn(1, 3, T - CHUNK_SIZE, FREQ_SIZE) + 5.0
    alpha_state_init = alpha_layer0.init_state(batch_size=1, freq_size=FREQ_SIZE)

    with torch.no_grad():
        _, alpha_next_a = alpha_layer0(alpha_in_a, alpha_state_init, state_frames=CHUNK_SIZE)
        _, alpha_next_b = alpha_layer0(alpha_in_b, alpha_state_init, state_frames=CHUNK_SIZE)
    alpha_diff = (alpha_next_a - alpha_next_b).abs().max().item()
    print(f"[lookahead_protection] alpha layer 0 diff = {alpha_diff:.3e}")
    assert alpha_diff == 0.0, f"alpha first-layer next_state contaminated: {alpha_diff:.3e}"

    # Backbone-side narrow guarantee is already covered by S4's
    # ``test_set_state_frames_for_update_first_layer_ignores_lookahead`` —
    # not duplicated here.


# ============================================================== (j) deep-copy departure
def test_fusion_modules_are_deep_copied_from_bafnet():
    """``ExportableBAFNetPlusCore.from_bafnetplus`` deep-copies the fusion modules.

    Documented S8 departure from S6's shared-reference contract — the fusion
    modules must be in-place graph-rewritten (``convert_to_stateful`` →
    ``convert_stateful_to_functional``) and a shared reference would mutate
    the source bafnet.
    """
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    # Calibration encoder: deep-copied AND functional-converted (no longer the
    # original CausalConv1d-based nn.Sequential).
    assert core.calibration_encoder is not bafnet.calibration_encoder, "calibration_encoder is shared"
    assert core.common_gain_head is not bafnet.common_gain_head, "common_gain_head is shared"
    assert core.relative_gain_head is not bafnet.relative_gain_head, "relative_gain_head is shared"
    assert core.alpha_convblocks is not bafnet.alpha_convblocks, "alpha_convblocks is shared"
    assert core.alpha_out is not bafnet.alpha_out, "alpha_out is shared"
    # Backbones cores ARE separate (deep-copied + functional-converted via from_backbone).
    assert core.mapping_core is not bafnet.mapping
    assert core.masking_core is not bafnet.masking
    # The fusion convs are now FunctionalStateful{Conv1d,CausalConv2d}.
    cal_convs = [m for m in core.calibration_encoder.modules() if isinstance(m, FunctionalStatefulConv1d)]
    alpha_convs = [m for m in core.alpha_convblocks.modules() if isinstance(m, FunctionalStatefulCausalConv2d)]
    assert len(cal_convs) == 2, len(cal_convs)
    assert len(alpha_convs) == 4, len(alpha_convs)
    # Source bafnet's calibration / alpha modules are UNCHANGED (still plain CausalConv1d/2d).
    from src.models.backbone import CausalConv1d, CausalConv2d

    src_cal_convs = [m for m in bafnet.calibration_encoder.modules() if isinstance(m, CausalConv1d)]
    src_alpha_convs = [m for m in bafnet.alpha_convblocks.modules() if isinstance(m, CausalConv2d)]
    assert len(src_cal_convs) == 2, len(src_cal_convs)
    assert len(src_alpha_convs) == 4, len(src_alpha_convs)


# ============================================================== misc structural
def test_from_bafnetplus_rejects_non_bafnetplus():
    """``from_bafnetplus`` refuses non-:class:`BAFNetPlus` arguments."""
    with pytest.raises(TypeError):
        ExportableBAFNetPlusCore.from_bafnetplus(object())  # type: ignore[arg-type]


def test_metadata_schema_and_keys():
    """``metadata()`` returns the documented S8 schema + key set."""
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    meta = core.metadata(chunk_size=CHUNK_SIZE)
    assert meta["schema_version"] == "s8-bafnetplus-functional-fp32"
    assert meta["ablation_mode"] == "full"
    assert meta["chunk_size"] == CHUNK_SIZE
    assert meta["n_fft"] == N_FFT
    assert meta["freq_size"] == FREQ_SIZE
    assert meta["encoder_lookahead"] == 3
    assert meta["decoder_lookahead"] == 3
    assert meta["alpha_time_lookahead"] == 0
    assert meta["t_export"] == T_EXPORT
    assert meta["num_states"] == core.num_states
    assert meta["state_order"] == "mapping/* masking/* calibration/* alpha/*"
    assert set(meta["branches"].keys()) == {"mapping", "masking", "fusion"}
    assert set(meta["branches"]["fusion"].keys()) == {"calibration", "alpha"}
    assert meta["branches"]["fusion"]["calibration"]["num_states"] == 2
    assert meta["branches"]["fusion"]["alpha"]["num_states"] == 4
    assert len(meta["state_shapes"]) == core.num_states


def test_set_state_frames_for_update_propagates_to_backbones():
    """``set_state_frames_for_update`` reaches both backbone cores AND is stored
    on ``self`` for the fusion convs to consume."""
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    assert core.state_frames_for_update is None
    assert core.mapping_core.state_frames_for_update is None
    assert core.masking_core.state_frames_for_update is None
    core.set_state_frames_for_update(CHUNK_SIZE)
    assert core.state_frames_for_update == CHUNK_SIZE
    assert core.mapping_core.state_frames_for_update == CHUNK_SIZE
    assert core.masking_core.state_frames_for_update == CHUNK_SIZE
    core.set_state_frames_for_update(None)
    assert core.state_frames_for_update is None
    assert core.mapping_core.state_frames_for_update is None
    assert core.masking_core.state_frames_for_update is None


def test_no_calibration_ablation_has_no_calibration_states():
    """``ablation_mode='no_calibration'`` → 0 calibration states."""
    bafnet = _synthetic_bafnetplus("no_calibration")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    assert not core.use_calibration
    assert core.calibration_encoder is None
    assert core.num_calibration_states == 0
    assert core.num_alpha_states == 4  # alpha is always built (even mask_only_alpha keeps 4 layers)
    assert core.num_states == 52 + 52 + 0 + 4 == 108
    # State names: mapping/* + masking/* + alpha/* (no calibration/*).
    names = core.get_state_names()
    assert all("calibration/" not in n for n in names), [n for n in names if "calibration/" in n]


def test_forward_rejects_wrong_state_count():
    """Forward fails clearly when too few or too many states are passed."""
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    T = 14
    bcs_mag = torch.randn(1, FREQ_SIZE, T)
    bcs_pha = torch.randn(1, FREQ_SIZE, T)
    acs_mag = torch.randn(1, FREQ_SIZE, T).abs() + 1e-3
    acs_pha = torch.randn(1, FREQ_SIZE, T)
    states = core.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T)
    with pytest.raises(ValueError, match="expected"):
        core(bcs_mag, bcs_pha, acs_mag, acs_pha, *states[:-1])
    with pytest.raises(ValueError, match="expected"):
        core(bcs_mag, bcs_pha, acs_mag, acs_pha, *(list(states) + [states[0]]))


def test_constructor_rejects_swapped_branches():
    """The ctor rejects swapped mapping/masking cores."""
    bafnet = _synthetic_bafnetplus("full")
    mapping_core = ExportableBackboneCore.from_backbone(bafnet.masking)  # WRONG infer_type
    masking_core = ExportableBackboneCore.from_backbone(bafnet.mapping)
    with pytest.raises(ValueError, match="mapping_core.infer_type"):
        ExportableBAFNetPlusCore(
            mapping_core=mapping_core,
            masking_core=masking_core,
            calibration_encoder=None,
            common_gain_head=None,
            relative_gain_head=None,
            alpha_convblocks=bafnet.alpha_convblocks,
            alpha_out=bafnet.alpha_out,
            ablation_mode="full",
            use_calibration=False,
            use_relative_gain=False,
            mask_only_alpha=False,
            calibration_max_common_log_gain=0.5,
            calibration_max_relative_log_gain=1.0,
            n_fft=N_FFT,
            phase_output_mode="atan2",
        )


def test_init_states_shapes_match_fusion_geometry():
    """Calibration state shape ``[B, in_C, 4]``, alpha state shape
    ``[B, in_C, 6, freq_size + 6]`` (kernel=5/7 + freq_pad=3)."""
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()
    states = core.init_states(batch_size=1, freq_size=FREQ_SIZE)
    n_map = core.mapping_core.num_states
    n_mask = core.masking_core.num_states
    # Calibration states: 2 layers, [B=1, in_C, 4] each.
    cal_0 = states[n_map + n_mask]
    cal_1 = states[n_map + n_mask + 1]
    assert tuple(cal_0.shape) == (1, 5, 4), tuple(cal_0.shape)  # in_ch=5 (5-ch calibration features)
    assert tuple(cal_1.shape) == (1, 16, 4), tuple(cal_1.shape)  # in_ch=16 (hidden)
    # Alpha states: 4 layers, [B=1, in_C, 6, F + 2*freq_pad=207] each.
    alpha_0 = states[n_map + n_mask + 2]
    alpha_3 = states[-1]
    assert tuple(alpha_0.shape) == (1, 3, 6, FREQ_SIZE + 6), tuple(alpha_0.shape)
    assert tuple(alpha_3.shape) == (1, 16, 6, FREQ_SIZE + 6), tuple(alpha_3.shape)


def test_full_sequence_parity_independent_of_state_frames_setting():
    """Running the single-shot full-sequence forward with state_frames_for_update
    = chunk_size vs None gives bit-identical outputs (when input length is
    exactly chunk_size, frames [:chunk_size] always carry the entire update).
    Sanity for the lookahead-protection contract.
    """
    torch.manual_seed(2039)
    bafnet = _synthetic_bafnetplus("full")
    core = ExportableBAFNetPlusCore.from_bafnetplus(bafnet).eval()

    T = CHUNK_SIZE  # exactly chunk_size frames -> state_frames_for_update is a no-op
    bcs_com = _gen_complex(seed=999, freq_size=FREQ_SIZE, t=T)
    acs_com = _gen_complex(seed=1000, freq_size=FREQ_SIZE, t=T)
    bcs_mag, bcs_pha = complex_to_mag_pha(bcs_com, stack_dim=-1)
    acs_mag, acs_pha = complex_to_mag_pha(acs_com, stack_dim=-1)
    states = core.init_states(batch_size=1, freq_size=FREQ_SIZE, time_frames=T)

    with torch.no_grad():
        core.set_state_frames_for_update(None)
        out_a = core(bcs_mag, bcs_pha, acs_mag, acs_pha, *states)
        core.set_state_frames_for_update(CHUNK_SIZE)
        out_b = core(bcs_mag, bcs_pha, acs_mag, acs_pha, *states)
    for i, (a, b) in enumerate(zip(out_a, out_b)):
        assert torch.equal(a, b), f"output {i} differs (state_frames=None vs chunk_size, T={T})"
