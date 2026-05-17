"""FP32 ONNX export driver + parity verification for the single-Backbone exportable core.

S4 of the streaming-ONNX rebuild (Stage 1 pt.3 — Stage-1 exit gate).
Builds an :class:`~src.models.streaming.onnx.backbone_core.ExportableBackboneCore`,
runs ``torch.onnx.export`` against it with the locked S4 settings, writes a
sidecar JSON metadata stub, and verifies ORT == PT-core parity over multiple
chunks with state propagation.

Locked export settings (the contract S8/S9/S11 re-bind to)
----------------------------------------------------------
- ``torch.onnx.export(... opset_version=17, dynamo=False)`` (torch 2.9.1 →
  legacy exporter; ``dynamo=True`` depends on onnxscript and is avoided per
  ``BAFNetPlus/CLAUDE.md``).
- **Static** time dim ``T_export = chunk_size + L_enc + L_dec`` (conservative
  monolithic-graph default; the S5 ``T_export`` proof harness may shrink to
  ``chunk + max(L_enc, L_dec)`` only if full-sequence parity holds for the
  actual configured backbone — that's *not* this session).
- ``state_frames_for_update = chunk_size`` for every stateful conv family (set
  on the core before export so the trace bakes it in).
- FP32 throughout — no INT8 / QDQ / dynamic-axes hedges. ``ConvTranspose2d``
  exports natively as ``ConvTranspose`` on torch 2.9.1 / ORT 1.18.1 (no
  ``conv_transpose_wrapper`` decomposition — see ``streaming/onnx/__init__.py``).
- Sidecar JSON sits next to the ``.onnx`` and records checkpoint
  path/md5, model config, ``chunk_size``, ``L_enc``, ``L_dec``, ``T_export``,
  STFT params, I/O names, state names+shapes, phase mode, expected sample rate
  (the *single-Backbone* version of the metadata; the full BAFNet+ sidecar is S8).

The verify harness :func:`verify_backbone_core_multistep` is the S4 **exit
gate**: it runs N random ``(mag, pha)`` chunks through both ORT and the PT
exportable core with state propagation, asserts every PT next-state shape
equals the matching input-state shape, asserts all state tensors are finite,
and compares output / state tensors within ``atol``. Correlation / max-abs are
diagnostic only — the gate is multi-chunk ORT-vs-PT parity.

Ported from LaCoSENet ``src/models/onnx_export/stateful_core.py``
(``export_stateful_nncore_to_onnx``) + ``verify_utils.py``
(``verify_stateful_onnx_multistep``) + ``streaming_wrapper.py``
(``from_checkpoint``-style loading), adjusted for BAFNet+ paths, the 3-output
atan2 contract (vs LaCoSENet's 2-output ``mask, pha``), and the FP32-only +
static-shapes scope.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.onnx
from torch import Tensor

from src.checkpoint import ConfigDict, load_checkpoint, load_model
from src.models.backbone import Backbone
from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.onnx.backbone_core import ExportableBackboneCore
from src.models.streaming.onnx.bafnetplus_core import (
    BAFNetPlusHeadCore,
    BAFNetPlusTrunkCore,
    ExportableBAFNetPlusCore,
)
from src.stft import complex_to_mag_pha

logger = logging.getLogger(__name__)

# Frozen non-state output counts (the contract S8/S9/S11 re-bind to).
_NUM_NON_STATE_OUTPUTS_ATAN2 = 3  # est_mag, est_pha, est_com
_NUM_NON_STATE_OUTPUTS_COMPLEX = 3  # est_mag, phase_real, phase_imag

# Sidecar schema versions for the two BAFNetPlus FP32 graph variants.
_BAFNETPLUS_SCHEMA_FP32_ATAN2 = "s8-bafnetplus-functional-fp32"
_BAFNETPLUS_SCHEMA_FP32_COMPLEX = "s20-bafnetplus-functional-fp32-complex"

# Sidecar schema versions for the S21 split-graph variant (B2 deployable track).
# The trunk is FP32 by default; ``quantize_bafnetplus_qdq`` flips its schema to
# ``s21-bafnetplus-trunk-int8-qdq`` after PTQ. The head stays FP32 (its job is
# to carry the INT8-hostile cluster — atan2/softmax/sqrt — outside the trunk).
_BAFNETPLUS_SCHEMA_TRUNK_FP32 = "s21-bafnetplus-trunk-fp32"
_BAFNETPLUS_SCHEMA_TRUNK_INT8 = "s21-bafnetplus-trunk-int8-qdq"
_BAFNETPLUS_SCHEMA_HEAD_FP32 = "s21-bafnetplus-head-fp32"
_BAFNETPLUS_SCHEMA_HEAD_INT8 = "s21-bafnetplus-head-int8-qdq"
_BAFNETPLUS_SCHEMA_SPLIT_COMBINED = "s21-bafnetplus-split-v1"

# Reshape-free TSBlock has been the only path since the cycle-13 cleanup that
# promoted Path β. The s8/s17/s20/s21 sidecar tokens are reused — the graph
# contract (input/output/state names, shapes, T_export, num_states) is
# unchanged; only the trunk TSBlock state layout switches from 3D
# ``[B*F_enc, C, padding]`` to 4D ``[B, C, padding, F_enc]`` and the
# corresponding op count drops (Reshape 216→56, Transpose 66→42 in FP32;
# Reshape 32→0, Transpose 32→8 in INT8). The cycle-11/12 ``s26-*-reshape-free``
# transitional tokens are retired with the cleanup.

# Boundary tensor names — frozen contract between trunk and head ONNX files.
_SPLIT_BOUNDARY_TENSOR_NAMES = (
    "bcs_est_mag",
    "bcs_phase_real",
    "bcs_phase_imag",
    "acs_est_mag",
    "acs_phase_real",
    "acs_phase_imag",
    "acs_mask",
)


@dataclass
class ExportResult:
    """Return value of :func:`export_backbone_core_to_onnx`.

    Attributes:
        onnx_path: Path to the written ``.onnx`` file.
        metadata_path: Path to the sidecar ``.json``.
        metadata: The metadata dict that was written (same content as the file).
        time_frames: ``T_export`` actually used.
        freq_size: ``n_fft // 2 + 1``.
        chunk_size: The chunk size used to set ``state_frames_for_update``.
    """

    onnx_path: str
    metadata_path: str
    metadata: Dict[str, Any]
    time_frames: int
    freq_size: int
    chunk_size: int


# ---------------------------------------------------------------------- loading


def load_backbone_from_checkpoint(
    chkpt_dir: Union[str, Path],
    chkpt_file: str = "best.th",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[Backbone, Dict[str, Any]]:
    """Load a single-Backbone Hydra-managed experiment dir into an instantiated ``Backbone``.

    Intended for the pre-joint single-Backbone checkpoints (e.g.
    ``results/experiments/bm_map_50ms`` / ``bm_mask_50ms``). Reads
    ``<chkpt_dir>/.hydra/config.yaml#model`` (``model_lib`` / ``model_class`` /
    ``param``), instantiates the model, loads the ``model`` state dict from
    ``<chkpt_dir>/<chkpt_file>``, computes the checkpoint MD5 (for the sidecar
    JSON), and returns ``(backbone, checkpoint_info)``.

    Do **not** point this at the unified ``bafnetplus_50ms`` checkpoint (that is
    S6) and do **not** pass ``checkpoint_mapping`` / ``checkpoint_masking`` from
    the Hydra config — those recorded paths are non-local.

    Args:
        chkpt_dir: Experiment directory with ``.hydra/config.yaml`` + checkpoint.
        chkpt_file: Checkpoint filename (default ``best.th``).
        device: Device for loading (export runs on CPU; pass ``"cpu"``).
        verbose: Print a one-line load summary.

    Returns:
        ``(backbone, info)`` where ``info`` has keys
        ``path`` / ``file`` / ``md5`` / ``model_lib`` / ``model_class`` /
        ``model_params``.

    Raises:
        FileNotFoundError: If the Hydra config or checkpoint file is missing.
    """
    from omegaconf import OmegaConf

    chkpt_dir = Path(chkpt_dir)
    cfg_path = chkpt_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {cfg_path}")
    ckpt_path = chkpt_dir / chkpt_file
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    conf = OmegaConf.load(cfg_path)
    model_cfg = conf.model
    if verbose:
        infer_type = model_cfg.param.get("infer_type", "masking") if hasattr(model_cfg.param, "get") else "masking"
        print(
            f"Loading Backbone from {chkpt_dir} "
            f"({model_cfg.model_lib}.{model_cfg.model_class}, infer_type={infer_type})"
        )

    backbone = load_model(model_cfg.model_lib, model_cfg.model_class, model_cfg.param, device)
    backbone = load_checkpoint(backbone, str(chkpt_dir), chkpt_file, device)

    md5 = hashlib.md5(ckpt_path.read_bytes()).hexdigest()
    info: Dict[str, Any] = {
        "path": str(ckpt_path.resolve()),
        "file": chkpt_file,
        "md5": md5,
        "model_lib": str(model_cfg.model_lib),
        "model_class": str(model_cfg.model_class),
        "model_params": OmegaConf.to_container(model_cfg.param, resolve=True),
    }
    return backbone.eval(), info


# -------------------------------------------------------------- streaming driver


@torch.inference_mode()
def run_core_streaming(
    core: ExportableBackboneCore,
    noisy_com: Tensor,
    chunk_size: int,
    *,
    states: Optional[List[Tensor]] = None,
    freq_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Drive the PT exportable core chunk-by-chunk over a complete spectrogram.

    Mirrors the **monolithic-graph** streaming pattern S9 will run (encoder +
    TS-blocks + decoder all in one ``core.forward`` per chunk, advancing by
    ``chunk_size`` frames and consuming ``T_export = chunk_size +
    encoder_lookahead + decoder_lookahead`` frames of input). For each full
    chunk: feed ``T_export`` frames, keep the first ``chunk_size`` output frames
    (all non-degraded — the trailing ``L_enc + L_dec`` frames used some zero-
    right-padding inside the convs, so they are discarded). The partial tail
    (fewer than ``T_export`` frames left in ``noisy_com``) is **not** produced —
    callers compare against ``Backbone.forward(noisy_com)[:, :, :recon_T]``,
    where ``recon_T = ((T - T_export) // chunk_size + 1) * chunk_size``.

    State updates inside the core are bounded to ``chunk_size`` via
    :meth:`~src.models.streaming.onnx.backbone_core.ExportableBackboneCore.set_state_frames_for_update`
    so the trailing lookahead frames feed only the conv outputs, not the
    next-chunk recurrent state. Streaming frame ``k`` ≡ ``Backbone.forward``
    frame ``k`` (no positional shift), so reconstruction concatenation is just
    ``cat([keep_first_chunk_size, ...], dim=2)``.

    Args:
        core: The PT exportable core (must be in ``eval`` mode — typical after
            :meth:`ExportableBackboneCore.from_backbone`).
        noisy_com: ``[B, F, T, 2]`` (``B == 1``) or ``[F, T, 2]`` complex
            spectrogram (e.g. ``mag_pha_stft(audio, center=True)[2]``).
        chunk_size: Output frames per chunk (= input-buffer advance).
        states: Initial states (defaults to ``core.init_states(...)``).
        freq_size: ``n_fft // 2 + 1`` (defaults to that).

    Returns:
        ``(est_mag, est_pha, est_com)``, each shaped ``[1, F, recon_T]`` /
        ``[1, F, recon_T, 2]`` with ``recon_T = n_full_chunks * chunk_size``.

    Raises:
        ValueError: If ``noisy_com`` has a batch dimension != 1, or if its time
            dim is too short for even one full ``T_export`` chunk.
    """
    if noisy_com.dim() == 3:
        noisy_com = noisy_com.unsqueeze(0)
    if noisy_com.shape[0] != 1:
        raise ValueError(f"run_core_streaming requires batch size 1, got {noisy_com.shape[0]}")

    f = noisy_com.shape[1]
    t_total = noisy_com.shape[2]
    total_lookahead = core.total_lookahead
    t_export = chunk_size + total_lookahead
    if t_total < t_export:
        raise ValueError(
            f"noisy_com has {t_total} frames but T_export = chunk_size({chunk_size}) + "
            f"L_enc({core.encoder_lookahead}) + L_dec({core.decoder_lookahead}) = {t_export}; "
            "input is too short for even one full chunk."
        )

    if freq_size is None:
        freq_size = f
    if states is None:
        states = core.init_states(batch_size=1, freq_size=freq_size, time_frames=t_export)

    # The export uses state_frames_for_update = chunk_size; mirror that here.
    prev_state_frames = core.state_frames_for_update
    core.set_state_frames_for_update(chunk_size)
    try:
        mags: List[Tensor] = []
        phas: List[Tensor] = []
        coms: List[Tensor] = []
        current_states = list(states)
        for t in range(0, t_total - t_export + 1, chunk_size):
            chunk = noisy_com[:, :, t : t + t_export, :]  # [1, F, T_export, 2]
            mag_chunk, pha_chunk = complex_to_mag_pha(chunk, stack_dim=-1)
            outs = core(mag_chunk, pha_chunk, *current_states)
            est_mag, est_pha, est_com = outs[0], outs[1], outs[2]
            mags.append(est_mag[:, :, :chunk_size])
            phas.append(est_pha[:, :, :chunk_size])
            coms.append(est_com[:, :, :chunk_size, :])
            current_states = list(outs[3:])
    finally:
        core.set_state_frames_for_update(prev_state_frames)

    return torch.cat(mags, dim=2), torch.cat(phas, dim=2), torch.cat(coms, dim=2)


# ---------------------------------------------------------------- ONNX export


def export_backbone_core_to_onnx(
    core: ExportableBackboneCore,
    output_path: Union[str, Path],
    *,
    chunk_size: int,
    time_frames: Optional[int] = None,
    freq_size: Optional[int] = None,
    batch_size: int = 1,
    opset_version: int = 17,
    sample_rate: int = 16000,
    hop_size: int = 100,
    win_size: int = 400,
    compress_factor: float = 0.3,
    checkpoint_info: Optional[Dict[str, Any]] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> ExportResult:
    """Export ``core`` to a static-shape FP32 ONNX, write the sidecar JSON.

    Args:
        core: The PT exportable core (``ExportableBackboneCore.from_backbone``).
        output_path: ``.onnx`` output path. The sidecar JSON is written to
            ``<output_path>.json``.
        chunk_size: Streaming output frames per call. Used to (a) compute
            ``T_export = chunk_size + L_enc + L_dec`` when ``time_frames`` is
            ``None`` and (b) set ``core.state_frames_for_update`` before export
            (so the trace bakes ``state_frames=chunk_size`` into every stateful
            conv).
        time_frames: ``T_export`` override (default = ``chunk_size +
            total_lookahead``).
        freq_size: Static frequency bins for the dummy inputs (default = ``n_fft
            // 2 + 1``).
        batch_size: Static batch (default 1; the deployed runtime is B=1).
        opset_version: ONNX opset (frozen at 17 — torch 2.9.1 / ORT 1.18.1).
        sample_rate / hop_size / win_size / compress_factor: STFT params for the
            sidecar JSON (host responsibility — not used in the graph).
        checkpoint_info: ``load_backbone_from_checkpoint`` output, or any
            ``{"path", "md5", ...}`` dict — recorded in the sidecar. ``None``
            for in-memory cores (e.g. synthetic test fixtures).
        metadata_extra: Optional extra keys to merge into the sidecar JSON.
        verbose: Print a one-line export summary.

    Returns:
        :class:`ExportResult` (paths + metadata + frame geometry).
    """
    core.eval()

    freq_size_val: int = freq_size if freq_size is not None else (core.n_fft // 2 + 1)
    time_frames_val: int = time_frames if time_frames is not None else (chunk_size + core.total_lookahead)
    if time_frames_val < chunk_size + core.total_lookahead:
        logger.warning(
            "T_export=%d is shorter than the conservative chunk+L_enc+L_dec=%d; "
            "some chunks will have fewer non-degraded output frames than chunk_size.",
            time_frames_val,
            chunk_size + core.total_lookahead,
        )

    # Lock the state-update window for streaming (S2's StateFramesContext role, but in-graph).
    core.set_state_frames_for_update(chunk_size)

    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype

    # Dummy inputs at the static export geometry.
    mag = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype)
    pha = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype)
    states = core.init_states(
        batch_size=batch_size,
        freq_size=freq_size_val,
        time_frames=time_frames_val,
        device=device,
        dtype=dtype,
    )

    input_names = core.input_names()
    output_names = core.output_names()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"Exporting ExportableBackboneCore -> {output_path}\n"
            f"  geometry: batch={batch_size} freq={freq_size_val} T_export={time_frames_val} "
            f"(chunk={chunk_size}, L_enc={core.encoder_lookahead}, L_dec={core.decoder_lookahead})\n"
            f"  num_states={core.num_states}, phase_output_mode={core.phase_output_mode!r}, "
            f"opset={opset_version}, dynamo=False"
        )

    torch.onnx.export(
        core,
        (mag, pha, *states),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
        # Static shapes throughout — no dynamic_axes. The deployed runtime is B=1,
        # T_export=14 (50ms anchor), freq=201. Stage 5's T_export proof harness may
        # re-export with a shrunk T_export later.
    )

    # Build sidecar JSON.
    core_meta = core.metadata(freq_size=freq_size_val)
    metadata: Dict[str, Any] = {
        "schema_version": "s4-single-backbone-fp32",
        "produced_by": "src.models.streaming.onnx.export.export_backbone_core_to_onnx",
        "torch_onnx": {"opset_version": int(opset_version), "dynamo": False},
        "geometry": {
            "batch_size": int(batch_size),
            "chunk_size": int(chunk_size),
            "encoder_lookahead": int(core.encoder_lookahead),
            "decoder_lookahead": int(core.decoder_lookahead),
            "total_lookahead": int(core.total_lookahead),
            "T_export": int(time_frames_val),
            "T_export_formula": "chunk_size + encoder_lookahead + decoder_lookahead",
            "freq_size": int(freq_size_val),
        },
        "stft": {
            "n_fft": int(core.n_fft),
            "hop_size": int(hop_size),
            "win_size": int(win_size),
            "compress_factor": float(compress_factor),
            "center": True,  # the host emulates center=True via center=False + win/2 context.
            "sample_rate": int(sample_rate),
        },
        "core": {
            "infer_type": core_meta["infer_type"],
            "phase_output_mode": core_meta["phase_output_mode"],
            "n_fft": core_meta["n_fft"],
            "state_frames_for_update": core_meta["state_frames_for_update"],
            "num_states": core_meta["num_states"],
        },
        "io": {
            "input_names": core_meta["input_names"],
            "output_names": core_meta["output_names"],
            "state_names": core_meta["state_names"],
            "state_shapes": core_meta["state_shapes"],
            "num_non_state_outputs": (
                _NUM_NON_STATE_OUTPUTS_COMPLEX if core.phase_output_mode == "complex" else _NUM_NON_STATE_OUTPUTS_ATAN2
            ),
        },
        "checkpoint": checkpoint_info,
        "onnx_file": output_path.name,
        "onnx_size_bytes": output_path.stat().st_size,
    }
    if metadata_extra:
        metadata["extra"] = metadata_extra

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=False))
    if verbose:
        print(f"  wrote sidecar: {metadata_path}  ({metadata['onnx_size_bytes'] / (1024 * 1024):.2f} MB onnx)")

    return ExportResult(
        onnx_path=str(output_path),
        metadata_path=str(metadata_path),
        metadata=metadata,
        time_frames=int(time_frames_val),
        freq_size=int(freq_size_val),
        chunk_size=int(chunk_size),
    )


def export_backbone_to_onnx_from_checkpoint(
    chkpt_dir: Union[str, Path],
    output_path: Union[str, Path],
    *,
    chunk_size: int,
    chkpt_file: str = "best.th",
    phase_output_mode: str = "atan2",
    time_frames: Optional[int] = None,
    sample_rate: int = 16000,
    hop_size: int = 100,
    win_size: int = 400,
    compress_factor: float = 0.3,
    opset_version: int = 17,
    verbose: bool = True,
) -> ExportResult:
    """One-shot: load a ``bm_*`` experiment dir, build the core, export, write sidecar.

    Wraps :func:`load_backbone_from_checkpoint` +
    :meth:`ExportableBackboneCore.from_backbone` +
    :func:`export_backbone_core_to_onnx`. The checkpoint info is included in the
    sidecar JSON. STFT params are read off the backbone (the explicit kwargs are
    fallbacks for the sidecar's host-side STFT metadata when the model's are
    missing — they should never differ in practice for the 50 ms anchor).

    Args:
        chkpt_dir: Experiment directory (see
            :func:`load_backbone_from_checkpoint`).
        output_path: ``.onnx`` output path.
        chunk_size: Streaming output frames per call.
        chkpt_file: Checkpoint filename (default ``best.th``).
        phase_output_mode: ``'atan2'`` (default) or ``'complex'``.
        time_frames: ``T_export`` override; default = ``chunk_size +
            total_lookahead`` (computed from ``compute_lookahead`` on the
            loaded backbone).
        sample_rate / hop_size / win_size / compress_factor / opset_version:
            See :func:`export_backbone_core_to_onnx`. ``hop_size`` /
            ``win_size`` / ``compress_factor`` are overridden by the loaded
            backbone's values if present (they are required and present for any
            real ``Backbone``).
        verbose: Print loading + export summaries.

    Returns:
        :class:`ExportResult`.
    """
    backbone, info = load_backbone_from_checkpoint(chkpt_dir, chkpt_file, device="cpu", verbose=verbose)
    # Prefer the model's own STFT params (always present on a real Backbone) for the sidecar.
    n_fft_attr = getattr(backbone, "n_fft", None)
    hop_attr = getattr(backbone, "hop_size", None)
    win_attr = getattr(backbone, "win_size", None)
    cf_attr = getattr(backbone, "compress_factor", None)
    if hop_attr is not None:
        hop_size = int(hop_attr)
    if win_attr is not None:
        win_size = int(win_attr)
    if cf_attr is not None:
        compress_factor = float(cf_attr)
    _ = n_fft_attr  # used by ExportableBackboneCore directly; just here to flag it's read off the model.

    core = ExportableBackboneCore.from_backbone(backbone, phase_output_mode=phase_output_mode)
    return export_backbone_core_to_onnx(
        core,
        output_path,
        chunk_size=chunk_size,
        time_frames=time_frames,
        sample_rate=sample_rate,
        hop_size=hop_size,
        win_size=win_size,
        compress_factor=compress_factor,
        opset_version=opset_version,
        checkpoint_info=info,
        verbose=verbose,
    )


# ---------------------------------------------------------------- ORT verification


def _wrapped_phase_max_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Max ``|atan2(sin(a-b), cos(a-b))|`` — wrapped phase difference, modulo 2π."""
    d = a - b
    return float(np.abs(np.arctan2(np.sin(d), np.cos(d))).max())


def verify_backbone_core_multistep(
    onnx_path: Union[str, Path],
    core: ExportableBackboneCore,
    *,
    chunk_size: int,
    time_frames: Optional[int] = None,
    freq_size: Optional[int] = None,
    batch_size: int = 1,
    num_steps: int = 5,
    atol: float = 1e-4,
    state_atol: Optional[float] = None,
    seed: int = 17,
    verbose: bool = False,
) -> Dict[str, Any]:
    """**S4 exit gate**: multi-chunk ORT-vs-PT-core parity with state propagation.

    Feeds ``num_steps`` random ``(mag, pha)`` chunks of shape
    ``[batch_size, freq_size, T_export]`` through both the PT exportable core
    and the ORT session. State is carried across steps in BOTH paths (PT and
    ORT use the SAME initial zeros, so any divergence is purely numerical).
    Verifies every PT next-state shape equals the matching input-state shape,
    every state tensor is finite, and aggregates max abs diffs per non-state
    output and per state.

    Per the launch prompt's S4 spec: "every next-state shape == the matching
    input-state shape, all state tensors finite for every chunk... Diagnostic
    corr / max-abs are diagnostic only — the gate is multi-chunk ORT-vs-PT
    parity." So the **structural** checks (shape, finite) are hard gates; the
    **numerical** ``all_match`` uses the supplied ``atol``. Tests pick the
    right ``atol`` for their model: ~1e-5 for a synthetic small-channel
    backbone, ~1e-3 for the real ``bm_*_50ms`` checkpoint (whose ``atan2`` phase
    decoder is ill-conditioned — see S3's ``MAG_TOL=1e-4`` / ``PHA_TOL=2e-3`` /
    ``COM_TOL=1e-3`` split). Per-output diffs are returned individually so
    tests can split-tolerance them; ``est_pha`` in atan2 mode also has a
    wrapped-phase diff available.

    Args:
        onnx_path: Path to the exported ``.onnx``.
        core: The SAME core the ONNX was exported from. Its
            ``state_frames_for_update`` is set to ``chunk_size`` here too so PT
            matches the in-graph constant baked into the export.
        chunk_size: Must match the export's ``state_frames_for_update``.
        time_frames: ``T_export`` (default = ``chunk_size + total_lookahead``).
        freq_size: Static freq dim (default = ``n_fft // 2 + 1``).
        batch_size: Static batch (default 1).
        num_steps: Number of chunks.
        atol: Aggregate tolerance for ``all_match`` (applied uniformly to every
            non-state output; ``state_atol`` overrides for states).
        state_atol: Tolerance for states (defaults to ``atol``).
        seed: RNG seed for reproducibility.
        verbose: Print a one-line PASS / FAIL summary.

    Returns:
        Dict with keys:

        - ``all_match`` (bool): every step's non-state outputs within ``atol``
          AND every state within ``state_atol`` AND ``state_shape_check`` AND
          ``all_finite``.
        - ``max_output_diffs``: per-output max abs diff over all steps
          (length 3 — atan2 mode: ``[est_mag, est_pha, est_com]``; complex
          mode: ``[est_mag, phase_real, phase_imag]``).
        - ``max_phase_wrapped_diff``: atan2 mode only — max wrapped diff for
          ``est_pha`` over all steps (``None`` in complex mode).
        - ``max_state_diff``: max abs diff over all states and all steps.
        - ``state_shape_check`` (bool): every PT next-state shape == the
          input-state shape (every step).
        - ``all_finite`` (bool): every PT/ORT state finite (every step).
        - ``steps``: per-step detail dicts.
        - ``num_states`` / ``num_non_state_outputs``.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}

    if state_atol is None:
        state_atol = atol

    core.eval()
    core.set_state_frames_for_update(chunk_size)

    freq_size_val: int = freq_size if freq_size is not None else (core.n_fft // 2 + 1)
    time_frames_val: int = time_frames if time_frames is not None else (chunk_size + core.total_lookahead)

    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype

    pt_states: List[Tensor] = core.init_states(
        batch_size=batch_size,
        freq_size=freq_size_val,
        time_frames=time_frames_val,
        device=device,
        dtype=dtype,
    )
    ort_states: List[np.ndarray] = [s.detach().cpu().numpy() for s in pt_states]
    state_names = core.get_state_names()

    is_atan2 = core.phase_output_mode != "complex"
    n_outs = _NUM_NON_STATE_OUTPUTS_ATAN2 if is_atan2 else _NUM_NON_STATE_OUTPUTS_COMPLEX
    # In atan2 mode, output index 1 is est_pha — compare wrapped (the ill-conditioned axis).
    pha_idx = 1 if is_atan2 else None

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rng = torch.Generator(device="cpu").manual_seed(seed)
    steps_info: List[Dict[str, Any]] = []
    max_output_diffs: List[float] = [0.0] * n_outs
    max_phase_wrapped_diff: float = 0.0
    max_state_diff: float = 0.0
    state_shape_check = True
    all_finite = True

    for step in range(num_steps):
        mag = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)
        pha = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)

        with torch.no_grad():
            pt_outputs = core(mag, pha, *pt_states)
        pt_non_state = [t.detach().cpu().numpy() for t in pt_outputs[:n_outs]]
        pt_next_states = list(pt_outputs[n_outs:])

        # Structural gates: shape + finiteness on the PT side (the core is the gate's authoritative model).
        for i, (prev, nxt) in enumerate(zip(pt_states, pt_next_states)):
            if tuple(prev.shape) != tuple(nxt.shape):
                state_shape_check = False
                logger.error(
                    "step %d state %d shape mismatch: prev=%s next=%s",
                    step,
                    i,
                    tuple(prev.shape),
                    tuple(nxt.shape),
                )
            if not torch.isfinite(nxt).all():
                all_finite = False
                logger.error("step %d state %d (PT) contains non-finite values", step, i)

        ort_inputs: Dict[str, np.ndarray] = {
            "mag": mag.detach().cpu().numpy(),
            "pha": pha.detach().cpu().numpy(),
        }
        for name, arr in zip(state_names, ort_states):
            ort_inputs[name] = arr
        ort_outputs = sess.run(None, ort_inputs)
        ort_non_state = ort_outputs[:n_outs]
        ort_next_states = ort_outputs[n_outs:]

        out_diffs = [float(np.abs(pt_non_state[i] - ort_non_state[i]).max()) for i in range(n_outs)]
        state_diffs = [
            float(np.abs(pt_next_states[i].detach().cpu().numpy() - ort_next_states[i]).max())
            for i in range(len(pt_next_states))
        ]
        max_output_diffs = [max(a, b) for a, b in zip(max_output_diffs, out_diffs)]
        if state_diffs:
            max_state_diff = max(max_state_diff, max(state_diffs))
        wrapped_pha: Optional[float] = None
        if pha_idx is not None:
            wrapped_pha = _wrapped_phase_max_abs(pt_non_state[pha_idx], ort_non_state[pha_idx])
            max_phase_wrapped_diff = max(max_phase_wrapped_diff, wrapped_pha)

        for i, arr in enumerate(ort_next_states):
            if not np.isfinite(arr).all():
                all_finite = False
                logger.error("step %d state %d (ORT) contains non-finite values", step, i)

        steps_info.append(
            {
                "step": step,
                "output_max_diffs": out_diffs,
                "state_max_diffs": state_diffs,
                "phase_wrapped_max_diff": wrapped_pha,
            }
        )

        # Propagate states (PT keeps its PT next_states; ORT keeps its ORT next_states).
        pt_states = pt_next_states
        ort_states = list(ort_next_states)

    all_outputs_close = all(d <= atol for d in max_output_diffs)
    all_states_close = max_state_diff <= state_atol
    all_match = all_outputs_close and all_states_close and state_shape_check and all_finite

    result: Dict[str, Any] = {
        "all_match": all_match,
        "max_output_diffs": max_output_diffs,
        "max_phase_wrapped_diff": max_phase_wrapped_diff if pha_idx is not None else None,
        "max_state_diff": float(max_state_diff),
        "state_shape_check": state_shape_check,
        "all_finite": all_finite,
        "num_states": core.num_states,
        "num_non_state_outputs": n_outs,
        "atol": float(atol),
        "state_atol": float(state_atol),
        "num_steps": num_steps,
        "steps": steps_info,
    }
    if verbose:
        status = "PASS" if all_match else "FAIL"
        out_str = " ".join(f"{d:.3e}" for d in max_output_diffs)
        wrapped_str = f"  max_phase_wrapped_diff: {max_phase_wrapped_diff:.3e}\n" if pha_idx is not None else ""
        print(
            f"verify_backbone_core_multistep ({num_steps} steps, atol={atol:.1e}, state_atol={state_atol:.1e}): "
            f"{status}\n"
            f"  max_output_diffs: {out_str}\n"
            f"{wrapped_str}"
            f"  max_state_diff: {max_state_diff:.3e}\n"
            f"  state_shape_check: {state_shape_check}, all_finite: {all_finite}"
        )
    return result


# ============================================================================
# S8 — BAFNet+ FP32 ONNX export + parity verification.
# ============================================================================
#
# Mirrors the single-Backbone path above (S4) for the full BAFNet+ graph
# (mapping + masking branch cores + functional-stateful calibration + alpha
# fusion). The export contract is the same: opset 17, dynamo=False, static
# T_export = chunk_size + L_enc + L_dec + L_alpha, state_frames_for_update =
# chunk_size, FP32 throughout. Sidecar JSON records the unified ckpt path/md5,
# the BAFNet+ model params (with checkpoint_mapping/masking popped), the
# 190-state I/O contract (mapping/* masking/* calibration/* alpha/*), and the
# per-component branches dict (mapping + masking + fusion{calibration, alpha}).


def load_bafnetplus_from_checkpoint(
    chkpt_dir: Union[str, Path],
    chkpt_file: str = "best.th",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[BAFNetPlus, Dict[str, Any]]:
    """Load the unified BAFNet+ checkpoint dir into an instantiated :class:`BAFNetPlus`.

    Mirrors :class:`~src.models.streaming.bafnetplus_streaming.BAFNetPlusStreaming.from_checkpoint`
    — reads ``<chkpt_dir>/.hydra/config.yaml#model`` (expects
    ``model_class=BAFNetPlus``), derives the per-branch ``bm_*`` Hydra
    experiment names from the recorded ``param.checkpoint_mapping`` /
    ``checkpoint_masking`` path basenames (those are non-local
    ``/workspace/...`` paths in general; the dirname basenames give the
    experiment names locally), pops the non-local paths, then instantiates
    BAFNet+ via the wiki foot-gun pattern
    (``load_pretrained_weights=False`` + ``load_state_dict(unified_ckpt['model'])``).
    Returns ``(bafnet, info)`` where ``info`` carries the unified ckpt MD5
    and the (popped) per-branch experiment names + the BAFNet+ model params
    (with ``checkpoint_mapping`` / ``checkpoint_masking`` removed) for the
    sidecar JSON.

    Args:
        chkpt_dir: Directory containing ``.hydra/config.yaml`` + the unified
            ``best.th`` (e.g. ``results/experiments/bafnetplus_50ms``).
        chkpt_file: Checkpoint filename (default ``"best.th"``).
        device: Device for loading (export runs on CPU; pass ``"cpu"``).
        verbose: Print a one-line load summary.

    Returns:
        ``(bafnet, info)``. ``info`` keys: ``path``, ``file``, ``md5``,
        ``model_lib``, ``model_class``, ``model_params``,
        ``per_branch.mapping`` / ``per_branch.masking`` (each with
        ``experiment_name`` + ``model_params``).

    Raises:
        FileNotFoundError: If the unified Hydra config, ckpt file, or either
            per-branch ``bm_*`` Hydra config is absent locally.
        ValueError: If the unified ckpt's model_class is not :class:`BAFNetPlus`.
    """
    from omegaconf import OmegaConf

    chkpt_dir = Path(chkpt_dir)
    cfg_path = chkpt_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {cfg_path}")
    ckpt_path = chkpt_dir / chkpt_file
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    conf = OmegaConf.load(cfg_path)
    model_cfg = conf.model
    if str(model_cfg.model_class) != "BAFNetPlus":
        raise ValueError(
            f"load_bafnetplus_from_checkpoint expects model_class=BAFNetPlus, "
            f"got {model_cfg.model_class!r} ({cfg_path})"
        )

    bp_param = OmegaConf.to_container(model_cfg.param, resolve=True)
    if not isinstance(bp_param, dict):
        raise ValueError(f"unexpected model.param shape in {cfg_path}: {type(bp_param).__name__}")
    bp_param_dict: Dict[str, Any] = cast(Dict[str, Any], bp_param)
    ckpt_mapping_path = bp_param_dict.pop("checkpoint_mapping", None)
    ckpt_masking_path = bp_param_dict.pop("checkpoint_masking", None)
    if ckpt_mapping_path is None or ckpt_masking_path is None:
        raise ValueError(
            f"unified ckpt's model.param is missing checkpoint_mapping / checkpoint_masking "
            f"(needed to derive per-branch experiment names): {cfg_path}"
        )
    experiments_dir = chkpt_dir.parent
    bm_map_name = Path(str(ckpt_mapping_path)).parent.name
    bm_mask_name = Path(str(ckpt_masking_path)).parent.name
    bm_map_cfg_path = experiments_dir / bm_map_name / ".hydra" / "config.yaml"
    bm_mask_cfg_path = experiments_dir / bm_mask_name / ".hydra" / "config.yaml"
    if not bm_map_cfg_path.exists() or not bm_mask_cfg_path.exists():
        raise FileNotFoundError(
            f"per-branch Hydra config(s) missing: mapping={bm_map_cfg_path} "
            f"(exists={bm_map_cfg_path.exists()}), masking={bm_mask_cfg_path} "
            f"(exists={bm_mask_cfg_path.exists()})."
        )

    map_conf = OmegaConf.load(bm_map_cfg_path)
    mask_conf = OmegaConf.load(bm_mask_cfg_path)
    map_param = OmegaConf.to_container(map_conf.model.param, resolve=True)
    mask_param = OmegaConf.to_container(mask_conf.model.param, resolve=True)
    args_mapping = ConfigDict(
        {
            "model_lib": str(map_conf.model.model_lib),
            "model_class": str(map_conf.model.model_class),
            "param": map_param,
        }
    )
    args_masking = ConfigDict(
        {
            "model_lib": str(mask_conf.model.model_lib),
            "model_class": str(mask_conf.model.model_class),
            "param": mask_param,
        }
    )

    if verbose:
        print(
            f"Loading BAFNetPlus from {chkpt_dir} "
            f"(ablation_mode={bp_param_dict.get('ablation_mode', 'full')}, "
            f"mapping={bm_map_name}, masking={bm_mask_name})"
        )
    bafnet = BAFNetPlus(
        args_mapping=args_mapping,
        args_masking=args_masking,
        load_pretrained_weights=False,
        **bp_param_dict,
    )
    bafnet = load_checkpoint(bafnet, str(chkpt_dir), chkpt_file, device)
    bafnet.eval()

    md5 = hashlib.md5(ckpt_path.read_bytes()).hexdigest()
    info: Dict[str, Any] = {
        "path": str(ckpt_path.resolve()),
        "file": chkpt_file,
        "md5": md5,
        "model_lib": str(model_cfg.model_lib),
        "model_class": str(model_cfg.model_class),
        "model_params": bp_param_dict,
        "per_branch": {
            "mapping": {"experiment_name": bm_map_name, "model_params": map_param},
            "masking": {"experiment_name": bm_mask_name, "model_params": mask_param},
        },
    }
    return bafnet, info


@torch.inference_mode()
def run_bafnetplus_core_streaming(
    core: ExportableBAFNetPlusCore,
    bcs_com: Tensor,
    acs_com: Tensor,
    chunk_size: int,
    *,
    init_states: Optional[List[Tensor]] = None,
    freq_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Drive the PT exportable BAFNet+ core chunk-by-chunk over paired spectrograms.

    Mirrors :func:`run_core_streaming` but for the full BAFNet+ graph with 4
    spectrogram inputs (bcs_mag / bcs_pha / acs_mag / acs_pha). For each
    chunk: feed ``T_export = chunk_size + L_enc + L_dec`` frames, keep the
    first ``chunk_size`` output frames (all non-degraded), carry all 190
    states across chunks via :meth:`ExportableBAFNetPlusCore.set_state_frames_for_update`
    bounded to ``chunk_size``. Partial tail (fewer than ``T_export`` frames
    left) is **not** produced.

    Args:
        core: The PT exportable BAFNet+ core.
        bcs_com: ``[B, F, T, 2]`` (B=1) or ``[F, T, 2]`` BCS complex
            spectrogram.
        acs_com: ``[B, F, T, 2]`` (B=1) or ``[F, T, 2]`` ACS complex
            spectrogram. Must have the same time/freq dims as ``bcs_com``.
        chunk_size: Output frames per chunk (= input-buffer advance).
        init_states: Initial states (defaults to ``core.init_states(...)``).
        freq_size: ``n_fft // 2 + 1`` (defaults to that).

    Returns:
        ``(est_mag [1, F, recon_T], est_pha [1, F, recon_T], est_com [1, F, recon_T, 2])``
        with ``recon_T = n_full_chunks * chunk_size``.

    Raises:
        ValueError: If ``bcs_com`` and ``acs_com`` shapes do not match, batch
            != 1, or the time dim is too short for even one full ``T_export``
            chunk.
    """
    if bcs_com.dim() == 3:
        bcs_com = bcs_com.unsqueeze(0)
    if acs_com.dim() == 3:
        acs_com = acs_com.unsqueeze(0)
    if bcs_com.shape != acs_com.shape:
        raise ValueError(f"bcs_com / acs_com shape mismatch: {bcs_com.shape} vs {acs_com.shape}")
    if bcs_com.shape[0] != 1:
        raise ValueError(f"run_bafnetplus_core_streaming requires batch size 1, got {bcs_com.shape[0]}")

    f = bcs_com.shape[1]
    t_total = bcs_com.shape[2]
    total_lookahead = core.total_lookahead
    t_export = chunk_size + total_lookahead
    if t_total < t_export:
        raise ValueError(
            f"input has {t_total} frames but T_export = chunk_size({chunk_size}) + "
            f"L_enc({core.mapping_core.encoder_lookahead}) + L_dec({core.mapping_core.decoder_lookahead}) "
            f"= {t_export}; input is too short for even one full chunk."
        )

    if freq_size is None:
        freq_size = f
    if init_states is None:
        init_states = core.init_states(batch_size=1, freq_size=freq_size, time_frames=t_export)

    prev_state_frames = core.state_frames_for_update
    core.set_state_frames_for_update(chunk_size)
    try:
        mags: List[Tensor] = []
        phas: List[Tensor] = []
        coms: List[Tensor] = []
        current_states = list(init_states)
        for t in range(0, t_total - t_export + 1, chunk_size):
            bcs_chunk = bcs_com[:, :, t : t + t_export, :]  # [1, F, T_export, 2]
            acs_chunk = acs_com[:, :, t : t + t_export, :]
            bcs_mag_c, bcs_pha_c = complex_to_mag_pha(bcs_chunk, stack_dim=-1)
            acs_mag_c, acs_pha_c = complex_to_mag_pha(acs_chunk, stack_dim=-1)
            outs = core(bcs_mag_c, bcs_pha_c, acs_mag_c, acs_pha_c, *current_states)
            est_mag, est_pha, est_com = outs[0], outs[1], outs[2]
            mags.append(est_mag[:, :, :chunk_size])
            phas.append(est_pha[:, :, :chunk_size])
            coms.append(est_com[:, :, :chunk_size, :])
            current_states = list(outs[3:])
    finally:
        core.set_state_frames_for_update(prev_state_frames)

    return torch.cat(mags, dim=2), torch.cat(phas, dim=2), torch.cat(coms, dim=2)


def export_bafnetplus_core_to_onnx(
    core: ExportableBAFNetPlusCore,
    output_path: Union[str, Path],
    *,
    chunk_size: int,
    time_frames: Optional[int] = None,
    freq_size: Optional[int] = None,
    batch_size: int = 1,
    opset_version: int = 17,
    sample_rate: int = 16000,
    hop_size: int = 100,
    win_size: int = 400,
    compress_factor: float = 0.3,
    checkpoint_info: Optional[Dict[str, Any]] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> ExportResult:
    """Export the full BAFNet+ ``ExportableBAFNetPlusCore`` to a static-shape FP32 ONNX.

    Mirrors :func:`export_backbone_core_to_onnx` but with 4 spectrogram inputs
    + all 190 state tensors. Writes the matching sidecar JSON with the S8
    schema (``schema_version='s8-bafnetplus-functional-fp32'``,
    ``branches.fusion`` populated, ``num_states`` correct, etc.).

    Args:
        core: The PT exportable BAFNet+ core.
        output_path: ``.onnx`` output path. Sidecar JSON at
            ``<output>.onnx.json``.
        chunk_size: Streaming output frames per call. Sets
            ``core.set_state_frames_for_update(chunk_size)``.
        time_frames: ``T_export`` override (default = ``chunk_size +
            total_lookahead + L_alpha`` from the core).
        freq_size: Static frequency bins for dummy inputs (default = ``n_fft
            // 2 + 1``).
        batch_size: Static batch (default 1).
        opset_version: ONNX opset (frozen at 17 for torch 2.9.1 / ORT 1.18.1).
        sample_rate / hop_size / win_size / compress_factor: STFT params for
            the sidecar (host-side; not used in the graph).
        checkpoint_info: ``load_bafnetplus_from_checkpoint`` output, or any
            ``{"path", "md5", ...}`` dict.
        metadata_extra: Extra keys to merge into the sidecar.
        verbose: Print a one-line export summary.

    Returns:
        :class:`ExportResult`.
    """
    core.eval()

    freq_size_val: int = freq_size if freq_size is not None else (core.n_fft // 2 + 1)
    t_export_default = chunk_size + core.total_lookahead
    time_frames_val: int = time_frames if time_frames is not None else t_export_default
    if time_frames_val < t_export_default:
        logger.warning(
            "T_export=%d is shorter than the conservative chunk+L_enc+L_dec=%d; "
            "some chunks will have fewer non-degraded output frames than chunk_size.",
            time_frames_val,
            t_export_default,
        )

    # Lock the state-update window for streaming.
    core.set_state_frames_for_update(chunk_size)

    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype

    # Dummy inputs at the static export geometry. acs_mag must be strictly positive
    # (the BAFNet+ forward uses it for mask recovery — host's STFT enforces sqrt(...+1e-9));
    # the trace doesn't require numerical accuracy, only positivity to avoid 0-division at trace time.
    bcs_mag = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype)
    bcs_pha = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype)
    acs_mag = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype).abs() + 1e-3
    acs_pha = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype)
    states = core.init_states(
        batch_size=batch_size,
        freq_size=freq_size_val,
        time_frames=time_frames_val,
        device=device,
        dtype=dtype,
    )

    input_names = core.input_names()
    output_names = core.output_names()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        n_map = core.mapping_core.num_states
        n_mask = core.masking_core.num_states
        n_cal = core.num_calibration_states
        n_alpha = core.num_alpha_states
        print(
            f"Exporting ExportableBAFNetPlusCore -> {output_path}\n"
            f"  geometry: batch={batch_size} freq={freq_size_val} T_export={time_frames_val} "
            f"(chunk={chunk_size}, L_enc={core.mapping_core.encoder_lookahead}, "
            f"L_dec={core.mapping_core.decoder_lookahead})\n"
            f"  num_states={core.num_states} "
            f"(mapping={n_map}, masking={n_mask}, calibration={n_cal}, alpha={n_alpha})\n"
            f"  ablation_mode={core.ablation_mode!r}, phase_output_mode={core.phase_output_mode!r}, "
            f"opset={opset_version}, dynamo=False"
        )

    torch.onnx.export(
        core,
        (bcs_mag, bcs_pha, acs_mag, acs_pha, *states),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
        # Static shapes throughout — no dynamic_axes.
    )

    core_meta = core.metadata(chunk_size=chunk_size, freq_size=freq_size_val)
    schema_version = (
        _BAFNETPLUS_SCHEMA_FP32_COMPLEX if core.phase_output_mode == "complex" else _BAFNETPLUS_SCHEMA_FP32_ATAN2
    )
    metadata: Dict[str, Any] = {
        "schema_version": schema_version,
        "produced_by": "src.models.streaming.onnx.export.export_bafnetplus_core_to_onnx",
        "torch_onnx": {"opset_version": int(opset_version), "dynamo": False},
        "geometry": {
            "batch_size": int(batch_size),
            "chunk_size": int(chunk_size),
            "encoder_lookahead": int(core_meta["encoder_lookahead"]),
            "decoder_lookahead": int(core_meta["decoder_lookahead"]),
            "alpha_time_lookahead": int(core_meta["alpha_time_lookahead"]),
            "total_lookahead": int(core_meta["total_lookahead"]),
            "T_export": int(time_frames_val),
            "T_export_formula": "chunk_size + encoder_lookahead + decoder_lookahead + alpha_time_lookahead",
            "freq_size": int(freq_size_val),
        },
        "stft": {
            "n_fft": int(core.n_fft),
            "hop_size": int(hop_size),
            "win_size": int(win_size),
            "compress_factor": float(compress_factor),
            "center": True,
            "sample_rate": int(sample_rate),
        },
        "core": {
            "infer_type": "bafnetplus",
            "phase_output_mode": core_meta["phase_output_mode"],
            "n_fft": core_meta["n_fft"],
            "state_frames_for_update": core_meta["state_frames_for_update"],
            "num_states": core_meta["num_states"],
            "ablation_mode": core_meta["ablation_mode"],
            "use_calibration": core_meta["use_calibration"],
            "use_relative_gain": core_meta["use_relative_gain"],
            "mask_only_alpha": core_meta["mask_only_alpha"],
            "calibration_max_common_log_gain": core_meta["calibration_max_common_log_gain"],
            "calibration_max_relative_log_gain": core_meta["calibration_max_relative_log_gain"],
        },
        "io": {
            "input_names": core_meta["input_names"],
            "output_names": core_meta["output_names"],
            "state_names": core_meta["state_names"],
            "state_shapes": core_meta["state_shapes"],
            "num_non_state_outputs": _NUM_NON_STATE_OUTPUTS_ATAN2,
            "state_order": core_meta["state_order"],
        },
        "branches": core_meta["branches"],
        "checkpoint": checkpoint_info,
        "onnx_file": output_path.name,
        "onnx_size_bytes": output_path.stat().st_size,
    }
    if metadata_extra:
        metadata["extra"] = metadata_extra

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=False))
    if verbose:
        print(f"  wrote sidecar: {metadata_path}  ({metadata['onnx_size_bytes'] / (1024 * 1024):.2f} MB onnx)")

    return ExportResult(
        onnx_path=str(output_path),
        metadata_path=str(metadata_path),
        metadata=metadata,
        time_frames=int(time_frames_val),
        freq_size=int(freq_size_val),
        chunk_size=int(chunk_size),
    )


def export_bafnetplus_to_onnx_from_checkpoint(
    chkpt_dir: Union[str, Path],
    output_path: Union[str, Path],
    *,
    chunk_size: int,
    chkpt_file: str = "best.th",
    phase_output_mode: str = "atan2",
    time_frames: Optional[int] = None,
    sample_rate: int = 16000,
    hop_size: int = 100,
    win_size: int = 400,
    compress_factor: float = 0.3,
    opset_version: int = 17,
    verbose: bool = True,
) -> ExportResult:
    """One-shot wrapper: load unified BAFNet+ ckpt → build exportable core → export.

    Mirrors :func:`export_backbone_to_onnx_from_checkpoint`.

    Args:
        chkpt_dir: Unified BAFNet+ experiment directory (e.g.
            ``results/experiments/bafnetplus_50ms``).
        output_path: ``.onnx`` output path.
        chunk_size: Streaming output frames per call.
        chkpt_file: Checkpoint filename (default ``"best.th"``).
        phase_output_mode: ``'atan2'`` (default) or ``'complex'``.
        time_frames: ``T_export`` override.
        sample_rate / hop_size / win_size / compress_factor / opset_version:
            See :func:`export_bafnetplus_core_to_onnx`. STFT params are taken
            from the loaded mapping branch if present (always present for a
            real Backbone).
        verbose: Print loading + export summaries.

    Returns:
        :class:`ExportResult`.
    """
    bafnet, info = load_bafnetplus_from_checkpoint(chkpt_dir, chkpt_file, device="cpu", verbose=verbose)
    mapping_branch = bafnet.mapping
    hop_attr = getattr(mapping_branch, "hop_size", None)
    win_attr = getattr(mapping_branch, "win_size", None)
    cf_attr = getattr(mapping_branch, "compress_factor", None)
    if hop_attr is not None:
        hop_size = int(hop_attr)
    if win_attr is not None:
        win_size = int(win_attr)
    if cf_attr is not None:
        compress_factor = float(cf_attr)

    core = ExportableBAFNetPlusCore.from_bafnetplus(
        bafnet,
        phase_output_mode=phase_output_mode,
    )
    return export_bafnetplus_core_to_onnx(
        core,
        output_path,
        chunk_size=chunk_size,
        time_frames=time_frames,
        sample_rate=sample_rate,
        hop_size=hop_size,
        win_size=win_size,
        compress_factor=compress_factor,
        opset_version=opset_version,
        checkpoint_info=info,
        verbose=verbose,
    )


def verify_bafnetplus_core_multistep(
    onnx_path: Union[str, Path],
    core: ExportableBAFNetPlusCore,
    *,
    chunk_size: int,
    time_frames: Optional[int] = None,
    freq_size: Optional[int] = None,
    batch_size: int = 1,
    num_steps: int = 5,
    atol: float = 1e-4,
    state_atol: Optional[float] = None,
    seed: int = 2039,
    verbose: bool = False,
) -> Dict[str, Any]:
    """**S8 exit gate**: multi-chunk ORT-vs-PT BAFNet+ core parity with state propagation.

    Mirrors :func:`verify_backbone_core_multistep` for the full BAFNet+ graph:
    feeds ``num_steps`` random ``(bcs_mag, bcs_pha, acs_mag, acs_pha)`` chunks
    through both the PT exportable core and the ORT session, propagates all
    190 states across steps in both paths (zero init in both, so divergence is
    purely numerical), and asserts:

    - Every PT next-state shape equals the matching input-state shape.
    - Every state tensor is finite on both sides at every step.
    - Per-non-state-output (``est_mag`` / ``est_pha`` / ``est_com``) max abs
      diff ≤ ``atol`` aggregated across all steps; per-state max abs diff ≤
      ``state_atol`` (defaults to ``atol``).

    ``acs_mag`` is generated as ``randn().abs() + 1e-3`` so the mask-recovery
    division (``acs_mask = acs_est_mag / acs_mag``) stays well-defined; the host
    STFT's ``sqrt(... + 1e-9)`` floor enforces this in production.

    Args:
        onnx_path: Path to the exported ``.onnx``.
        core: The SAME core the ONNX was exported from.
            ``state_frames_for_update`` is set to ``chunk_size`` here too so PT
            matches the in-graph constant baked into the export.
        chunk_size: Must match the export's ``state_frames_for_update``.
        time_frames: ``T_export`` (default = ``chunk_size + total_lookahead``).
        freq_size: Static freq dim (default = ``n_fft // 2 + 1``).
        batch_size: Static batch (default 1).
        num_steps: Number of chunks.
        atol: Aggregate tolerance for ``all_match`` (per non-state output).
        state_atol: Tolerance for states (defaults to ``atol``).
        seed: RNG seed for reproducibility.
        verbose: Print a one-line PASS / FAIL summary.

    Returns:
        Dict mirroring :func:`verify_backbone_core_multistep`'s — with
        ``num_non_state_outputs=3`` and 190 states by default for the real
        50 ms ckpt.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}

    if state_atol is None:
        state_atol = atol

    core.eval()
    core.set_state_frames_for_update(chunk_size)

    freq_size_val: int = freq_size if freq_size is not None else (core.n_fft // 2 + 1)
    time_frames_val: int = time_frames if time_frames is not None else (chunk_size + core.total_lookahead)

    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype

    pt_states: List[Tensor] = core.init_states(
        batch_size=batch_size,
        freq_size=freq_size_val,
        time_frames=time_frames_val,
        device=device,
        dtype=dtype,
    )
    ort_states: List[np.ndarray] = [s.detach().cpu().numpy() for s in pt_states]
    state_names = core.get_state_names()

    n_outs = _NUM_NON_STATE_OUTPUTS_ATAN2
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rng = torch.Generator(device="cpu").manual_seed(seed)
    steps_info: List[Dict[str, Any]] = []
    max_output_diffs: List[float] = [0.0] * n_outs
    max_phase_wrapped_diff: float = 0.0
    max_state_diff: float = 0.0
    state_shape_check = True
    all_finite = True

    for step in range(num_steps):
        bcs_mag = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)
        bcs_pha = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)
        acs_mag_raw = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)
        acs_mag = acs_mag_raw.abs() + 1e-3
        acs_pha = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)

        with torch.no_grad():
            pt_outputs = core(bcs_mag, bcs_pha, acs_mag, acs_pha, *pt_states)
        pt_non_state = [t.detach().cpu().numpy() for t in pt_outputs[:n_outs]]
        pt_next_states = list(pt_outputs[n_outs:])

        for i, (prev, nxt) in enumerate(zip(pt_states, pt_next_states)):
            if tuple(prev.shape) != tuple(nxt.shape):
                state_shape_check = False
                logger.error(
                    "step %d state %d (%s) shape mismatch: prev=%s next=%s",
                    step,
                    i,
                    state_names[i],
                    tuple(prev.shape),
                    tuple(nxt.shape),
                )
            if not torch.isfinite(nxt).all():
                all_finite = False
                logger.error("step %d state %d (%s, PT) contains non-finite values", step, i, state_names[i])

        ort_inputs: Dict[str, np.ndarray] = {
            "bcs_mag": bcs_mag.detach().cpu().numpy(),
            "bcs_pha": bcs_pha.detach().cpu().numpy(),
            "acs_mag": acs_mag.detach().cpu().numpy(),
            "acs_pha": acs_pha.detach().cpu().numpy(),
        }
        for name, arr in zip(state_names, ort_states):
            ort_inputs[name] = arr
        ort_outputs = sess.run(None, ort_inputs)
        ort_non_state = ort_outputs[:n_outs]
        ort_next_states = ort_outputs[n_outs:]

        out_diffs = [float(np.abs(pt_non_state[i] - ort_non_state[i]).max()) for i in range(n_outs)]
        state_diffs = [
            float(np.abs(pt_next_states[i].detach().cpu().numpy() - ort_next_states[i]).max())
            for i in range(len(pt_next_states))
        ]
        max_output_diffs = [max(a, b) for a, b in zip(max_output_diffs, out_diffs)]
        if state_diffs:
            max_state_diff = max(max_state_diff, max(state_diffs))
        # Wrapped-phase comparison is only meaningful in atan2 mode where
        # output[1] is est_pha (a wrapped angle). In complex mode output[1]
        # is phase_real (a raw conv output) — wrapped diff is undefined.
        if core.phase_output_mode == "complex":
            wrapped_pha: Optional[float] = None
        else:
            wrapped_pha = _wrapped_phase_max_abs(pt_non_state[1], ort_non_state[1])
            max_phase_wrapped_diff = max(max_phase_wrapped_diff, wrapped_pha)

        for i, arr in enumerate(ort_next_states):
            if not np.isfinite(arr).all():
                all_finite = False
                logger.error("step %d state %d (%s, ORT) contains non-finite values", step, i, state_names[i])

        steps_info.append(
            {
                "step": step,
                "output_max_diffs": out_diffs,
                "state_max_diffs": state_diffs,
                "phase_wrapped_max_diff": wrapped_pha,
            }
        )

        pt_states = pt_next_states
        ort_states = list(ort_next_states)

    all_outputs_close = all(d <= atol for d in max_output_diffs)
    all_states_close = max_state_diff <= state_atol
    all_match = all_outputs_close and all_states_close and state_shape_check and all_finite

    result: Dict[str, Any] = {
        "all_match": all_match,
        "max_output_diffs": max_output_diffs,
        "max_phase_wrapped_diff": max_phase_wrapped_diff,
        "max_state_diff": float(max_state_diff),
        "state_shape_check": state_shape_check,
        "all_finite": all_finite,
        "num_states": core.num_states,
        "num_non_state_outputs": n_outs,
        "atol": float(atol),
        "state_atol": float(state_atol),
        "num_steps": num_steps,
        "steps": steps_info,
    }
    if verbose:
        status = "PASS" if all_match else "FAIL"
        out_str = " ".join(f"{d:.3e}" for d in max_output_diffs)
        print(
            f"verify_bafnetplus_core_multistep ({num_steps} steps, atol={atol:.1e}, "
            f"state_atol={state_atol:.1e}): {status}\n"
            f"  max_output_diffs: {out_str}\n"
            f"  max_phase_wrapped_diff: {max_phase_wrapped_diff:.3e}\n"
            f"  max_state_diff: {max_state_diff:.3e}\n"
            f"  state_shape_check: {state_shape_check}, all_finite: {all_finite}"
        )
    return result


# ============================================================================
# S17 — INT8 QDQ quantization on the S8 functional-stateful BAFNet+ FP32 ONNX.
# ============================================================================
#
# Reconstructs the legacy ``export_onnx.py --quantize_qdq`` branch (deleted in
# the 2026-05-13 cleanup) on top of the post-S9.6 functional-stateful core.
# Reads a directory of calibration ``.npz`` files (produced by
# ``BAFNetPlus/scripts/make_calibration_npz.py``) — each contains
# ``bcs_mag / bcs_pha / acs_mag / acs_pha`` shaped ``[1, F=201, T=14]`` — and
# feeds them to ``onnxruntime.quantization.quantize_static`` with a custom
# :class:`CalibrationDataReader` that zero-fills the 190 state inputs (matches
# the deployed cold-start convention). Activation + weight types default to
# ``QUInt8`` (per-tensor), matching the legacy ``int8_qdq_quint8_quint8``
# token that the Kotlin :class:`BackendSelector` already accepts on the
# ``contains("int8")`` branch (see
# ``Android_projects/lacosenet-streaming/.../backend/BackendSelector.kt:67-86``).
#
# The output sidecar carries
# ``schema_version = "s17-bafnetplus-functional-int8-qdq"`` and inherits the
# S8 sidecar's geometry / state contract; the only changes are the
# ``core.quantization`` token + the ``onnx_file`` / ``onnx_size_bytes`` fields.


def _load_calibration_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load a single calibration NPZ and return all its arrays as a dict.

    Whatever keys are present in the NPZ become audio inputs to the
    quantizer; state tensors are zero-filled separately by the reader so
    the calibration ONNX session sees the cold-start state distribution.
    For the canonical trunk path the NPZ contains
    ``bcs_mag / bcs_pha / acs_mag / acs_pha`` each shaped ``[1, F=201, T=14]``;
    for the cycle-6 H1 head path it contains the 7 boundary tensors
    ``bcs_est_mag / bcs_phase_real / bcs_phase_imag / acs_est_mag /
    acs_phase_real / acs_phase_imag / acs_mask``.
    """
    arr = np.load(npz_path)
    return {k: arr[k].astype(np.float32, copy=False) for k in arr.files}


class _BAFNetPlusCalibrationDataReader:
    """``onnxruntime.quantization.CalibrationDataReader`` for the BAFNet+ graphs.

    Iterates a directory of ``calib_*.npz`` files; for each file produces one
    full input dict containing the per-NPZ audio tensors + zero-init state
    tensors. Audio input names are auto-detected from the NPZ keys (so the
    same reader serves the canonical trunk path — keys
    ``bcs_mag/bcs_pha/acs_mag/acs_pha`` — and the cycle-6 H1 head path —
    keys = the 7 trunk-boundary tensors). State shapes are read once from
    the ONNX session at construction time, so the reader is geometry-agnostic.
    """

    def __init__(
        self,
        npz_dir: Union[str, Path],
        onnx_path: Union[str, Path],
        *,
        max_samples: Optional[int] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is required for INT8 QDQ calibration"
            ) from e
        self._npz_dir = Path(npz_dir)
        if not self._npz_dir.is_dir():
            raise FileNotFoundError(f"calibration dir not found: {self._npz_dir}")
        files = sorted(self._npz_dir.glob("calib_*.npz"))
        if not files:
            raise FileNotFoundError(
                f"no calib_*.npz files under {self._npz_dir} — run scripts/make_calibration_npz.py first."
            )
        if max_samples is not None:
            files = files[: int(max_samples)]
        self._files: List[Path] = files

        # Auto-detect audio input names from the first NPZ's keys — works for
        # both the canonical trunk corpus (4 keys) and the head corpus (7 keys).
        sample = np.load(self._files[0])
        self._audio_input_names: Tuple[str, ...] = tuple(sample.files)
        sample.close()

        # Read state shapes once; cache them keyed by state name.
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        sess_input_names = {meta.name for meta in sess.get_inputs()}
        missing_audio = [n for n in self._audio_input_names if n not in sess_input_names]
        if missing_audio:
            raise KeyError(
                f"calibration NPZ keys not present as ONNX inputs: {missing_audio}"
            )
        self._state_inputs: List[Tuple[str, Tuple[int, ...]]] = []
        for meta in sess.get_inputs():
            if meta.name in self._audio_input_names:
                continue
            shape = tuple(int(d) if isinstance(d, int) else 1 for d in meta.shape)
            self._state_inputs.append((meta.name, shape))
        del sess

        self._idx = 0
        logger.info(
            "calibration reader: %d audio dicts (%d-tensor inputs), %d state tensors (zero-filled per call)",
            len(self._files),
            len(self._audio_input_names),
            len(self._state_inputs),
        )

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._idx >= len(self._files):
            return None
        path = self._files[self._idx]
        self._idx += 1
        audio = _load_calibration_npz(path)
        # Append zero-filled state tensors (matches the cold-start convention).
        feed: Dict[str, np.ndarray] = dict(audio)
        for name, shape in self._state_inputs:
            feed[name] = np.zeros(shape, dtype=np.float32)
        return feed

    def rewind(self) -> None:
        self._idx = 0


def _auto_precision_sensitive_nodes(fp32_onnx_path: Union[str, Path]) -> List[str]:
    """Identify graph nodes that should run in FP32 inside the QDQ graph.

    The 190-state functional-stateful BAFNet+ graph has several op
    families whose INT8 quantization-noise floor dominates the model
    output (S17 calibration matrix: all-INT8 U8/U8/U16 attempts yielded
    PESQ |Δ| 2.28-3.17 vs FP32). Excluding these from quantization (so
    the QDQ graph runs them in FP32) is the targeted fix:

    * ``Atan`` — phase decoder atan2 (3 nodes; output ±π wraps INT8
      precision into noise).
    * ``Softmax`` — alpha fusion blend (1 node; probabilities sum to 1
      but logits span the full FP range).
    * Top-level ``Pow``/``Sqrt`` (not inside ``/norm/`` LayerNorm
      blocks) — the calibration encoder energy features
      (``complex_to_mag_pha`` sqrt + atan2 path).

    LayerNorm-internal Pow/Sqrt/Div (192 nodes) are NOT excluded —
    they're per-block stable distributions that quantize well.
    SimpleGate Div (32 nodes) also stays quantized.

    On the **S21 trunk graph** (schema_version
    :data:`_BAFNETPLUS_SCHEMA_TRUNK_FP32`) the trunk has no top-level
    Atan/Softmax/Pow/Sqrt — those live in the FP32 head. The per-branch
    phase-decoder cluster (``/phase_conv/phase_conv.{0,1,2}/{ConvTranspose,
    BatchNormalization,PRelu}`` and ``/{,mapping_core/}phase_conv_{r,i}/Conv``)
    must instead be FP-fallbacked: on the unified S17 graph these ops
    are effectively FP32 because their downstream Atan2 is excluded
    (the QDQ pass surfaces the FP-fallback region upstream), but on the
    trunk the same ops feed graph-boundary tensors and reach the device
    as INT8 nodes. Hexagon V79 rejects them at HTP graph prepare time
    (S23 Phase C pre-flight, 2026-05-15: error 1002 / RouterFastRPC
    graph prepare failed 12 on
    ``/phase_conv_r/Conv_token_4679_2``, ``/mapping_core/phase_conv_r/
    Conv_token_4745_2``, ``/phase_conv/phase_conv.2/PRelu_3``). Adding
    these patterns to the trunk's auto-exclude restores the de-facto
    FP-fallback region that exists on the unified path.

    S19 note: the matching ``init_overrides`` experiment (symmetric=True
    on these tensors via ``config.extra_options['TensorQuantOverrides']``)
    produced PESQ_ort bit-identical to the no-overrides baseline (idx 0
    `2.337599` vs canonical `2.3375649`), so the targeted-tensor approach
    is rejected as a B1-style mitigation. INT8 deployable on this graph
    requires either B2 (graph splitting — FP32 head + INT8 trunk) or
    C (QAT) or D (atan2-free re-export via ``phase_output_mode='complex'``).
    """
    import onnx  # noqa: PLC0415 — lazy import (heavy)
    fp32_path = Path(fp32_onnx_path)
    model = onnx.load(str(fp32_path))

    is_trunk = False
    is_head = False
    sidecar_path = fp32_path.with_suffix(fp32_path.suffix + ".json")
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text())
            schema = sidecar.get("schema_version")
            is_trunk = schema == _BAFNETPLUS_SCHEMA_TRUNK_FP32
            is_head = schema == _BAFNETPLUS_SCHEMA_HEAD_FP32
        except (OSError, json.JSONDecodeError):
            pass

    trunk_phase_decoder_substrings: tuple = (
        "phase_conv_r/",
        "phase_conv_i/",
        "phase_conv/phase_conv.",
    )
    # On the S21 head graph (S25 cycle-6 H1, 2026-05-16): the three "boundary
    # Convs" — common_gain_head/Conv, relative_gain_head/Conv, alpha_out/Conv —
    # produce narrow-channel outputs (out_C ∈ {1, 2}) that feed directly into a
    # downstream FP-fallback op (Softmax or Tanh-shaped gain composition). Per-
    # tensor INT8 quantization compresses these into noise (initial H1 attempt
    # without this pattern: idx 0 |dRO| 3.45 vs FP32 head 0.35 — catastrophic).
    # Adding them to autofp brings the head INT8 |dRO| back to the trunk's
    # deployable band.
    head_boundary_conv_substrings: tuple = (
        "common_gain_head/",
        "relative_gain_head/",
        "alpha_out/",
    )

    sensitive: List[str] = []
    for n in model.graph.node:
        op = n.op_type
        name = n.name
        if op == "Atan":
            sensitive.append(name)
        elif op == "Softmax":
            sensitive.append(name)
        elif op in ("Pow", "Sqrt") and "/norm" not in name:
            sensitive.append(name)
        elif is_trunk and any(s in name for s in trunk_phase_decoder_substrings):
            sensitive.append(name)
        elif is_head and any(s in name for s in head_boundary_conv_substrings):
            sensitive.append(name)
    return sensitive



def quantize_bafnetplus_qdq(
    fp32_onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    calibration_dir: Union[str, Path],
    activation_type: str = "QUInt16",
    weight_type: str = "QUInt8",
    per_channel: bool = False,
    nodes_to_exclude: Optional[List[str]] = None,
    auto_exclude_sensitive: bool = True,
    max_calibration_samples: Optional[int] = None,
    verbose: bool = True,
    init_overrides: Optional[Dict[str, Any]] = None,
) -> ExportResult:
    """Produce the S17 INT8 QDQ asset from the S8 FP32 ONNX.

    Runs the QNN HTP-aware quantizer (``get_qnn_qdq_config`` + ``quantize``)
    with the BAFNet+ :class:`_BAFNetPlusCalibrationDataReader`. Default
    ``activation_type=QUInt16`` (mixed-precision INT8 weight + UINT16
    activation) — the 190-state functional-stateful BAFNet+ graph has wide
    activation dynamic ranges (atan2 phase, softmax alpha, sqrt energy
    feature) that a flat ``QUInt8`` activation collapses into noise (the
    pre-S17 sub-launch experiment showed PESQ |Δ| ~ 2.9 with ``QUInt8``
    activations even under per-channel weight quantization). The legacy
    166-state BAFNet+ INT8 QDQ pipeline supported the same UINT16
    activation knob via ``--qdq_activation_type`` in the deleted
    ``export_onnx.py`` (see ``BAFNetPlus`` commit ``bc43acf``). The
    geometry / state / I/O contract stays identical to the input FP32 graph
    — only the precision changes. Writes a new sidecar JSON at
    ``<output>.onnx.json`` with
    ``schema_version="s17-bafnetplus-functional-int8-qdq"`` and the
    ``model_info.quantization="int8_qdq_{act}_{wt}"`` token the Kotlin
    :class:`BackendSelector` already accepts on the ``contains("int8")``
    branch.

    Args:
        fp32_onnx_path: The S8 FP32 ONNX (must have a sidecar JSON alongside).
        output_path: Output INT8 ``.onnx`` path. Sidecar JSON written to
            ``<output>.onnx.json``.
        calibration_dir: Directory of ``calib_*.npz`` files (produced by
            ``scripts/make_calibration_npz.py``).
        activation_type: ``"QUInt16"`` (default, S17 recipe) or ``"QUInt8"``.
            ``QUInt16`` is required for the 190-state graph; ``QUInt8`` is
            kept as a knob for ablation but is known to produce a broken
            model on this architecture.
        weight_type: ``"QUInt8"`` (default) or ``"QInt8"``.
        per_channel: ``False`` (canonical S17 default — matches the 6.89 MB
            asset md5 ``0d972368…``). Setting True bloats the asset to
            ~10.25 MB (S19 regression — QDQ pass falls back the LN gamma/beta
            initializers to the activation type QUInt16 because the QNN HTP
            compat pass cannot find a ``LayerNormalization`` anchor on this
            expanded-LN graph) and collapses PESQ_ort from ~2.34 to ~1.14.
        nodes_to_exclude: Optional list of node names to keep in FP32.
        auto_exclude_sensitive: If True (default), additionally exclude the
            atan2/softmax/top-level Pow/Sqrt cluster identified by
            :func:`_auto_precision_sensitive_nodes`. On the S21 trunk
            graph (``schema_version`` :data:`_BAFNETPLUS_SCHEMA_TRUNK_FP32`)
            the same pass additionally excludes the per-branch phase-decoder
            cluster (``phase_conv_{r,i}`` Conv + ``phase_conv/phase_conv.{0,1,2}``
            ConvTranspose/BN/PRelu) — the unified path FP-fallbacks these
            transitively via the downstream Atan2 exclusion, but the trunk
            graph has no Atan2 so the exclusion has to be explicit
            (S23 Phase C pre-flight mini-fix A1, 2026-05-15: Hexagon V79
            rejects them at HTP graph prepare with err 1002 otherwise).
        max_calibration_samples: Cap on calibration file count (default: use
            all files in the dir).
        verbose: Print summary lines.

    Returns:
        :class:`ExportResult` (``onnx_path`` / ``metadata_path`` /
        ``metadata`` / ``time_frames`` / ``freq_size`` / ``chunk_size``).
    """
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantType,
            quantize,
        )
        from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime.quantization (incl. execution_providers.qnn) is required for INT8 QDQ export"
        ) from e

    fp32_path = Path(fp32_onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = fp32_path.with_suffix(fp32_path.suffix + ".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"FP32 sidecar not found: {sidecar_path}")
    fp32_sidecar = json.loads(sidecar_path.read_text())

    act_qtype = getattr(QuantType, activation_type)
    wt_qtype = getattr(QuantType, weight_type)

    # Build exclusion list — combine caller-provided list with auto-detected
    # precision-sensitive ops (atan2 / softmax / top-level sqrt/pow).
    exclude_list: List[str] = list(nodes_to_exclude or [])
    if auto_exclude_sensitive:
        auto_excluded = _auto_precision_sensitive_nodes(fp32_path)
        for name in auto_excluded:
            if name not in exclude_list:
                exclude_list.append(name)

    if verbose:
        print(
            f"Quantizing {fp32_path} -> {output_path}\n"
            f"  calibration_dir : {calibration_dir}\n"
            f"  max_samples     : {max_calibration_samples or 'all'}\n"
            f"  weight type     : {weight_type} (per_channel={per_channel})\n"
            f"  activation type : {activation_type} (QNN HTP recipe)\n"
            f"  format          : QDQ\n"
            f"  config          : get_qnn_qdq_config (QNN HTP-aware)\n"
            f"  nodes_excluded  : {len(exclude_list)} total"
        )

    # Pre-process: run shape inference + ORT graph optimizations on the FP32
    # graph BEFORE feeding it to the calibrator. The S8 functional-stateful
    # graph contains ``ConstantOfShape`` ops whose downstream ReduceMax
    # (auto-injected by the calibrator) trips a runtime exception without
    # this pre-pass. Recommended by ORT's calibration pipeline (the
    # "Please consider to run pre-processing" warning).
    preproc_path = output_path.with_name(output_path.stem + ".preproc.onnx")
    try:
        quant_pre_process(
            input_model_path=str(fp32_path),
            output_model_path=str(preproc_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            verbose=0,
        )
    except TypeError:
        # Older ORT API — fall back to the minimal signature.
        quant_pre_process(str(fp32_path), str(preproc_path))

    reader = _BAFNetPlusCalibrationDataReader(
        npz_dir=calibration_dir,
        onnx_path=preproc_path,
        max_samples=max_calibration_samples,
    )
    assert isinstance(reader, CalibrationDataReader) or hasattr(reader, "get_next"), (
        "calibration reader missing get_next/rewind contract"
    )

    # Use the QNN HTP-aware QDQ config (per-channel weight quantization for
    # conv layers, op-type allowlist, activation handling matching HTP) —
    # plain `quantize_static(..., per_channel=False)` collapses the 190-state
    # graph's wide activation dynamic ranges into a single tensor-wide scale
    # and produces a broken model (PESQ |Δ| > 3 on TAPS idx 0 in initial S17
    # experiment); the QNN config restores the legacy 80-state QDQ recipe
    # that achieved ~5e-3 PESQ delta after TAPS-aware calibration.
    qnn_config = get_qnn_qdq_config(
        model_input=str(preproc_path),
        calibration_data_reader=reader,  # type: ignore[arg-type]
        activation_type=act_qtype,
        weight_type=wt_qtype,
        per_channel=per_channel,
    )
    # Apply node-exclusion list (atan2 / softmax / top-level sqrt/pow stay in FP32).
    if exclude_list:
        qnn_config.nodes_to_exclude = list(exclude_list)
    # Cycle 18b Option β: inject QAT-learned per-tensor (scale, zp) overrides via
    # ORT's ``TensorQuantOverrides`` extra option. Calibration-derived scales for
    # any tensor present in ``init_overrides`` are replaced at QDQ-insertion time.
    if init_overrides:
        existing = dict(getattr(qnn_config, "extra_options", {}) or {})
        existing["TensorQuantOverrides"] = init_overrides
        qnn_config.extra_options = existing
        if verbose:
            print(f"  TensorQuantOverrides: {len(init_overrides)} tensor overrides applied")
    quantize(
        model_input=str(preproc_path),
        model_output=str(output_path),
        quant_config=qnn_config,
    )

    # Capture the source FP32 file digest BEFORE cleanup.
    source_fp32_name = fp32_path.name
    source_fp32_md5 = hashlib.md5(fp32_path.read_bytes()).hexdigest()

    # Best-effort cleanup of the intermediate pre-processed graph (only needed
    # as upstream input for the QDQ pipeline).
    try:
        if preproc_path.exists():
            preproc_path.unlink()
    except OSError:
        pass

    # Sidecar — inherit FP32 contract; flip schema_version + add quantization token.
    # The output schema depends on which FP32 variant we quantize:
    # * S8 atan2 (canonical) → S17 INT8 (the original 190-state INT8 path).
    # * S20 atan2-free (complex output mode) → S20 INT8-complex (a separate token
    #   so the host wrapper / parity script can detect the contract change).
    # * S21 trunk (split-graph 184-state variant) → S21 trunk-INT8 (the B2
    #   deployable target — trunk gets PTQ'd, head stays FP32 in a separate file).
    metadata = dict(fp32_sidecar)
    source_schema = fp32_sidecar.get("schema_version")
    if source_schema == _BAFNETPLUS_SCHEMA_TRUNK_FP32:
        metadata["schema_version"] = _BAFNETPLUS_SCHEMA_TRUNK_INT8
    elif source_schema == _BAFNETPLUS_SCHEMA_HEAD_FP32:
        metadata["schema_version"] = _BAFNETPLUS_SCHEMA_HEAD_INT8
    elif source_schema == _BAFNETPLUS_SCHEMA_FP32_COMPLEX:
        metadata["schema_version"] = "s20-bafnetplus-functional-int8-qdq-complex"
    else:
        metadata["schema_version"] = "s17-bafnetplus-functional-int8-qdq"
    metadata["produced_by"] = "src.models.streaming.onnx.export.quantize_bafnetplus_qdq"
    # Add a top-level model_info compatible with Kotlin BackendSelector's gate
    # (it lowercases + .contains("int8") at lacosenet-streaming/.../BackendSelector.kt:73).
    core_meta = dict(metadata.get("core", {}))
    quant_token = f"int8_qdq_{activation_type.lower()}_{weight_type.lower()}"
    core_meta["quantization"] = quant_token
    metadata["core"] = core_meta
    model_info: Dict[str, Any] = {
        "quantization": quant_token,
        "weight_type": weight_type,
        "activation_type": activation_type,
        "per_channel": bool(per_channel),
        "format": "QDQ",
        "calibration_dir": str(calibration_dir),
        "calibration_samples_used": len(reader._files),  # noqa: SLF001 — diagnostic only.
        "source_fp32_onnx": source_fp32_name,
        "source_fp32_md5": source_fp32_md5,
    }
    metadata["model_info"] = model_info
    metadata["onnx_file"] = output_path.name
    metadata["onnx_size_bytes"] = output_path.stat().st_size

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=False))

    if verbose:
        size_mb = metadata["onnx_size_bytes"] / (1024 * 1024)
        print(f"  wrote sidecar: {metadata_path}  ({size_mb:.2f} MB onnx)")

    geometry = metadata.get("geometry", {})
    return ExportResult(
        onnx_path=str(output_path),
        metadata_path=str(metadata_path),
        metadata=metadata,
        time_frames=int(geometry.get("T_export", 0)),
        freq_size=int(geometry.get("freq_size", 0)),
        chunk_size=int(geometry.get("chunk_size", 0)),
    )


# ============================================================================
# S21 — Split-graph BAFNet+ export (B2 deployable: FP32 head + INT8 trunk).
# ============================================================================
#
# Adds three exporters parallel to ``export_bafnetplus_core_to_onnx``:
#
# * :func:`export_bafnetplus_trunk_to_onnx` — exports a
#   :class:`BAFNetPlusTrunkCore` to ``.onnx`` with the
#   :data:`_BAFNETPLUS_SCHEMA_TRUNK_FP32` sidecar.
# * :func:`export_bafnetplus_head_to_onnx` — exports a
#   :class:`BAFNetPlusHeadCore` to ``.onnx`` with the
#   :data:`_BAFNETPLUS_SCHEMA_HEAD_FP32` sidecar.
# * :func:`export_bafnetplus_split_to_onnx_from_checkpoint` — one-shot
#   wrapper that loads the unified ckpt and exports both trunk + head plus
#   a combined sidecar (:data:`_BAFNETPLUS_SCHEMA_SPLIT_COMBINED`) describing
#   both files and the boundary contract.
#
# Plus :func:`verify_bafnetplus_split_multistep`, the S21 exit gate — chains
# the two ORT sessions across ``num_steps`` random chunks with shared init
# states and compares the chained ORT output against the chained PT path.
# Mirrors :func:`verify_bafnetplus_core_multistep` structurally.
#
# The trunk's PTQ path is the existing :func:`quantize_bafnetplus_qdq` — it
# is graph-agnostic (audio inputs identified by name; states zero-filled).
# When fed the trunk FP32 ONNX, it emits the trunk INT8 ONNX with schema
# :data:`_BAFNETPLUS_SCHEMA_TRUNK_INT8`. See the function's docstring for
# how schema detection branches on the source sidecar's ``schema_version``.


def export_bafnetplus_trunk_to_onnx(
    core: BAFNetPlusTrunkCore,
    output_path: Union[str, Path],
    *,
    chunk_size: int,
    time_frames: Optional[int] = None,
    freq_size: Optional[int] = None,
    batch_size: int = 1,
    opset_version: int = 17,
    sample_rate: int = 16000,
    hop_size: int = 100,
    win_size: int = 400,
    compress_factor: float = 0.3,
    checkpoint_info: Optional[Dict[str, Any]] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> ExportResult:
    """Export the BAFNet+ trunk subgraph to a static-shape FP32 ONNX.

    The trunk graph has 4 audio inputs + 184 backbone states; it returns 7
    boundary tensors + 184 next-states. ``T_export`` reduces to
    ``chunk_size + encoder_lookahead + decoder_lookahead`` (no alpha
    contribution; the alpha conv stack lives in the head). Sidecar schema
    is :data:`_BAFNETPLUS_SCHEMA_TRUNK_FP32`.

    Args:
        core: The PT trunk core (``BAFNetPlusTrunkCore.from_bafnetplus``).
        output_path: ``.onnx`` output path. Sidecar at ``<output>.onnx.json``.
        chunk_size: Streaming output frames per call. Used to derive
            ``T_export`` and sets ``core.set_state_frames_for_update``.
        time_frames: ``T_export`` override (default = trunk's conservative
            ``chunk + L_enc + L_dec``).
        freq_size: Static frequency bins (default = ``n_fft // 2 + 1``).
        batch_size: Static batch (default 1).
        opset_version: ONNX opset (frozen at 17 for torch 2.9.1 / ORT 1.18.1).
        sample_rate / hop_size / win_size / compress_factor: STFT params
            recorded in the sidecar (host-side; not used in the graph).
        checkpoint_info: Provenance block, recorded in the sidecar.
        metadata_extra: Extra keys merged into the sidecar.
        verbose: Print a one-line export summary.

    Returns:
        :class:`ExportResult`.
    """
    core.eval()

    freq_size_val: int = freq_size if freq_size is not None else (core.n_fft // 2 + 1)
    t_export_default = chunk_size + core.total_lookahead
    time_frames_val: int = time_frames if time_frames is not None else t_export_default
    if time_frames_val < t_export_default:
        logger.warning(
            "Trunk T_export=%d is shorter than chunk+L_enc+L_dec=%d; "
            "some chunks will have fewer non-degraded output frames than chunk_size.",
            time_frames_val,
            t_export_default,
        )

    core.set_state_frames_for_update(chunk_size)
    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype

    # Dummy inputs at the static export geometry. acs_mag must be strictly positive
    # (the masking branch's mask_decoder LearnableSigmoid_2d output is non-zero in
    # practice; in complex mode the trunk does not divide by acs_mag, but we keep
    # the same convention as the canonical export for consistency).
    bcs_mag = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype)
    bcs_pha = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype)
    acs_mag = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype).abs() + 1e-3
    acs_pha = torch.randn(batch_size, freq_size_val, time_frames_val, device=device, dtype=dtype)
    states = core.init_states(
        batch_size=batch_size,
        freq_size=freq_size_val,
        time_frames=time_frames_val,
        device=device,
        dtype=dtype,
    )

    input_names = core.input_names()
    output_names = core.output_names()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"Exporting BAFNetPlusTrunkCore -> {output_path}\n"
            f"  geometry: batch={batch_size} freq={freq_size_val} T_export={time_frames_val} "
            f"(chunk={chunk_size}, L_enc={core.mapping_core.encoder_lookahead}, "
            f"L_dec={core.mapping_core.decoder_lookahead})\n"
            f"  num_states={core.num_states} "
            f"(mapping={core.mapping_core.num_states}, masking={core.masking_core.num_states})\n"
            f"  phase_output_mode='complex' (raw phase_real/phase_imag at boundary), "
            f"opset={opset_version}, dynamo=False"
        )

    torch.onnx.export(
        core,
        (bcs_mag, bcs_pha, acs_mag, acs_pha, *states),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
    )

    core_meta = core.metadata(chunk_size=chunk_size, freq_size=freq_size_val)
    num_non_state_outputs = len(_SPLIT_BOUNDARY_TENSOR_NAMES)
    metadata: Dict[str, Any] = {
        "schema_version": _BAFNETPLUS_SCHEMA_TRUNK_FP32,
        "produced_by": "src.models.streaming.onnx.export.export_bafnetplus_trunk_to_onnx",
        "torch_onnx": {"opset_version": int(opset_version), "dynamo": False},
        "geometry": {
            "batch_size": int(batch_size),
            "chunk_size": int(chunk_size),
            "encoder_lookahead": int(core_meta["encoder_lookahead"]),
            "decoder_lookahead": int(core_meta["decoder_lookahead"]),
            "total_lookahead": int(core_meta["total_lookahead"]),
            "T_export": int(time_frames_val),
            "T_export_formula": "chunk_size + encoder_lookahead + decoder_lookahead",
            "freq_size": int(freq_size_val),
        },
        "stft": {
            "n_fft": int(core.n_fft),
            "hop_size": int(hop_size),
            "win_size": int(win_size),
            "compress_factor": float(compress_factor),
            "center": True,
            "sample_rate": int(sample_rate),
        },
        "core": {
            "infer_type": "bafnetplus-trunk",
            "phase_output_mode": core_meta["phase_output_mode"],
            "n_fft": core_meta["n_fft"],
            "num_states": core_meta["num_states"],
        },
        "io": {
            "input_names": core_meta["input_names"],
            "output_names": core_meta["output_names"],
            "state_names": core_meta["state_names"],
            "state_shapes": core_meta["state_shapes"],
            "num_non_state_outputs": num_non_state_outputs,
            "state_order": core_meta["state_order"],
            "boundary_tensor_names": list(_SPLIT_BOUNDARY_TENSOR_NAMES),
        },
        "branches": core_meta["branches"],
        "checkpoint": checkpoint_info,
        "onnx_file": output_path.name,
        "onnx_size_bytes": output_path.stat().st_size,
    }
    if metadata_extra:
        metadata["extra"] = metadata_extra

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=False))
    if verbose:
        size_mb = metadata["onnx_size_bytes"] / (1024 * 1024)
        print(f"  wrote sidecar: {metadata_path}  ({size_mb:.2f} MB onnx)")

    return ExportResult(
        onnx_path=str(output_path),
        metadata_path=str(metadata_path),
        metadata=metadata,
        time_frames=int(time_frames_val),
        freq_size=int(freq_size_val),
        chunk_size=int(chunk_size),
    )


def export_bafnetplus_head_to_onnx(
    core: BAFNetPlusHeadCore,
    output_path: Union[str, Path],
    *,
    chunk_size: int,
    time_frames: Optional[int] = None,
    freq_size: Optional[int] = None,
    batch_size: int = 1,
    opset_version: int = 17,
    sample_rate: int = 16000,
    hop_size: int = 100,
    win_size: int = 400,
    compress_factor: float = 0.3,
    checkpoint_info: Optional[Dict[str, Any]] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> ExportResult:
    """Export the BAFNet+ head subgraph to a static-shape FP32 ONNX.

    The head graph has 7 boundary inputs + 6 fusion states; it returns the
    canonical S8/S17 atan2 triple + 6 next-states. ``T_export`` matches the
    trunk's ``T_export`` (= ``chunk_size + L_enc + L_dec``) because the head
    is fed the trunk's full-T_export boundary tensors; the host wrapper trims
    to ``chunk_size`` at the matured output. Sidecar schema is
    :data:`_BAFNETPLUS_SCHEMA_HEAD_FP32`.

    The ``time_frames`` argument (default = ``chunk_size + L_enc + L_dec``
    matching the canonical full-graph ``T_export``) is the head's static T
    dim. The head itself has zero structural lookahead — feeding it
    ``chunk_size`` frames would also be valid, but matching the trunk's
    boundary is simpler.

    Args:
        core: The PT head core (``BAFNetPlusHeadCore.from_bafnetplus``).
        output_path: ``.onnx`` output path.
        chunk_size: Output chunk size for the host trim (used by the wrapper).
        time_frames: Static T dim of the head's inputs. Defaults to
            ``chunk_size + L_enc_trunk + L_dec_trunk`` if the trunk's
            lookahead is supplied; otherwise just ``chunk_size``.
        freq_size: Static F dim (default = ``n_fft // 2 + 1``).
        batch_size: Static batch (default 1).
        opset_version / sample_rate / hop_size / win_size / compress_factor /
            checkpoint_info / metadata_extra / verbose: See
            :func:`export_bafnetplus_trunk_to_onnx`.

    Returns:
        :class:`ExportResult`.
    """
    core.eval()

    freq_size_val: int = freq_size if freq_size is not None else (core.n_fft // 2 + 1)
    if time_frames is None:
        # The head defaults to chunk_size + alpha_time_lookahead — but in the
        # B2 split design it must accept the trunk's T_export-frame boundary.
        # The caller (split orchestrator) passes time_frames = trunk T_export
        # to enforce that contract.
        time_frames_val = chunk_size + core.total_lookahead
    else:
        time_frames_val = time_frames

    core.set_state_frames_for_update(chunk_size)
    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype

    # Dummy boundary inputs at the static export geometry. ``acs_mag`` (which
    # gets fed into _build_calibration_features via the mask_mean/mask_var
    # path) does not exist in the head; only ``acs_mask`` and the per-branch
    # est_mag / phase_real / phase_imag tensors do.
    boundary_shape = (batch_size, freq_size_val, time_frames_val)
    bcs_est_mag = torch.randn(*boundary_shape, device=device, dtype=dtype).abs() + 1e-3
    bcs_phase_real = torch.randn(*boundary_shape, device=device, dtype=dtype)
    bcs_phase_imag = torch.randn(*boundary_shape, device=device, dtype=dtype)
    acs_est_mag = torch.randn(*boundary_shape, device=device, dtype=dtype).abs() + 1e-3
    acs_phase_real = torch.randn(*boundary_shape, device=device, dtype=dtype)
    acs_phase_imag = torch.randn(*boundary_shape, device=device, dtype=dtype)
    # acs_mask in (0, 1) — match the LearnableSigmoid_2d output range; we add a
    # tiny floor so var() / .pow(2) ops don't see degenerate zeros at trace time.
    acs_mask = torch.sigmoid(torch.randn(*boundary_shape, device=device, dtype=dtype))
    states = core.init_states(
        batch_size=batch_size,
        freq_size=freq_size_val,
        time_frames=time_frames_val,
        device=device,
        dtype=dtype,
    )

    input_names = core.input_names()
    output_names = core.output_names()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"Exporting BAFNetPlusHeadCore -> {output_path}\n"
            f"  geometry: batch={batch_size} freq={freq_size_val} T_export={time_frames_val} "
            f"(chunk={chunk_size}, L_alpha={core.total_lookahead})\n"
            f"  num_states={core.num_states} "
            f"(calibration={core.num_calibration_states}, alpha={core.num_alpha_states})\n"
            f"  ablation_mode={core.ablation_mode!r}, phase_output_mode='atan2' (final fusion), "
            f"opset={opset_version}, dynamo=False"
        )

    torch.onnx.export(
        core,
        (
            bcs_est_mag,
            bcs_phase_real,
            bcs_phase_imag,
            acs_est_mag,
            acs_phase_real,
            acs_phase_imag,
            acs_mask,
            *states,
        ),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
    )

    core_meta = core.metadata(chunk_size=chunk_size, freq_size=freq_size_val)
    metadata: Dict[str, Any] = {
        "schema_version": _BAFNETPLUS_SCHEMA_HEAD_FP32,
        "produced_by": "src.models.streaming.onnx.export.export_bafnetplus_head_to_onnx",
        "torch_onnx": {"opset_version": int(opset_version), "dynamo": False},
        "geometry": {
            "batch_size": int(batch_size),
            "chunk_size": int(chunk_size),
            "alpha_time_lookahead": int(core_meta["alpha_time_lookahead"]),
            "total_lookahead": int(core_meta["total_lookahead"]),
            "T_export": int(time_frames_val),
            "T_export_formula": "trunk's T_export (chunk_size + L_enc + L_dec) — passed in by the split orchestrator",
            "freq_size": int(freq_size_val),
        },
        "stft": {
            "n_fft": int(core.n_fft),
            "hop_size": int(hop_size),
            "win_size": int(win_size),
            "compress_factor": float(compress_factor),
            "center": True,
            "sample_rate": int(sample_rate),
        },
        "core": {
            "infer_type": "bafnetplus-head",
            "phase_output_mode": core_meta["phase_output_mode"],
            "n_fft": core_meta["n_fft"],
            "state_frames_for_update": core_meta["state_frames_for_update"],
            "num_states": core_meta["num_states"],
            "ablation_mode": core_meta["ablation_mode"],
            "use_calibration": core_meta["use_calibration"],
            "use_relative_gain": core_meta["use_relative_gain"],
            "mask_only_alpha": core_meta["mask_only_alpha"],
            "calibration_max_common_log_gain": core_meta["calibration_max_common_log_gain"],
            "calibration_max_relative_log_gain": core_meta["calibration_max_relative_log_gain"],
        },
        "io": {
            "input_names": core_meta["input_names"],
            "output_names": core_meta["output_names"],
            "state_names": core_meta["state_names"],
            "state_shapes": core_meta["state_shapes"],
            "num_non_state_outputs": _NUM_NON_STATE_OUTPUTS_ATAN2,
            "state_order": core_meta["state_order"],
            "boundary_tensor_names": list(_SPLIT_BOUNDARY_TENSOR_NAMES),
        },
        "fusion": core_meta["fusion"],
        "checkpoint": checkpoint_info,
        "onnx_file": output_path.name,
        "onnx_size_bytes": output_path.stat().st_size,
    }
    if metadata_extra:
        metadata["extra"] = metadata_extra

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=False))
    if verbose:
        size_mb = metadata["onnx_size_bytes"] / (1024 * 1024)
        print(f"  wrote sidecar: {metadata_path}  ({size_mb:.2f} MB onnx)")

    return ExportResult(
        onnx_path=str(output_path),
        metadata_path=str(metadata_path),
        metadata=metadata,
        time_frames=int(time_frames_val),
        freq_size=int(freq_size_val),
        chunk_size=int(chunk_size),
    )


def export_bafnetplus_split_to_onnx_from_checkpoint(
    chkpt_dir: Union[str, Path],
    *,
    trunk_output_path: Union[str, Path],
    head_output_path: Union[str, Path],
    combined_sidecar_path: Optional[Union[str, Path]] = None,
    chunk_size: int,
    chkpt_file: str = "best.th",
    time_frames: Optional[int] = None,
    sample_rate: int = 16000,
    hop_size: int = 100,
    win_size: int = 400,
    compress_factor: float = 0.3,
    opset_version: int = 17,
    verbose: bool = True,
) -> Dict[str, Any]:
    """One-shot: load unified BAFNet+ ckpt → build trunk+head → export both ONNX + combined sidecar.

    Mirrors :func:`export_bafnetplus_to_onnx_from_checkpoint` for the S21 B2
    split design. Loads the unified ckpt, builds a :class:`BAFNetPlusTrunkCore`
    and :class:`BAFNetPlusHeadCore` from the SAME bafnet instance, then
    exports each as a separate ONNX file with its own sidecar JSON. Finally
    writes a combined sidecar at ``combined_sidecar_path`` (or
    ``<trunk_output_path>.parent / "bafnetplus_split.json"`` if not set)
    that describes both files + the boundary contract + checkpoint provenance.

    The trunk's ``T_export = chunk_size + L_enc + L_dec`` is also the head's
    ``T_export`` — the head accepts the trunk's full T_export-frame boundary
    outputs and emits the same shape; the host wrapper trims to ``chunk_size``
    only at the very end.

    Args:
        chkpt_dir: Unified BAFNet+ experiment directory.
        trunk_output_path: ``.onnx`` output path for the trunk.
        head_output_path: ``.onnx`` output path for the head.
        combined_sidecar_path: Combined sidecar JSON path. Defaults to
            ``<trunk_output_path>.parent / "bafnetplus_split.json"``.
        chunk_size: Streaming output frames per call (50 ms anchor: 8).
        chkpt_file: Checkpoint filename (default ``"best.th"``).
        time_frames: ``T_export`` override for BOTH trunk + head (default =
            ``chunk_size + L_enc + L_dec`` from the loaded mapping branch).
        sample_rate / hop_size / win_size / compress_factor / opset_version:
            See :func:`export_bafnetplus_to_onnx_from_checkpoint`.
        verbose: Print loading + export summaries.

    Returns:
        Dict with keys ``trunk`` (ExportResult), ``head`` (ExportResult),
        ``combined_sidecar_path`` (str), ``combined_metadata`` (dict).
    """
    bafnet, info = load_bafnetplus_from_checkpoint(chkpt_dir, chkpt_file, device="cpu", verbose=verbose)
    mapping_branch = bafnet.mapping
    hop_attr = getattr(mapping_branch, "hop_size", None)
    win_attr = getattr(mapping_branch, "win_size", None)
    cf_attr = getattr(mapping_branch, "compress_factor", None)
    if hop_attr is not None:
        hop_size = int(hop_attr)
    if win_attr is not None:
        win_size = int(win_attr)
    if cf_attr is not None:
        compress_factor = float(cf_attr)

    trunk = BAFNetPlusTrunkCore.from_bafnetplus(bafnet)
    head = BAFNetPlusHeadCore.from_bafnetplus(bafnet)

    if time_frames is None:
        time_frames_val = chunk_size + trunk.total_lookahead
    else:
        time_frames_val = int(time_frames)

    trunk_result = export_bafnetplus_trunk_to_onnx(
        trunk,
        trunk_output_path,
        chunk_size=chunk_size,
        time_frames=time_frames_val,
        sample_rate=sample_rate,
        hop_size=hop_size,
        win_size=win_size,
        compress_factor=compress_factor,
        opset_version=opset_version,
        checkpoint_info=info,
        verbose=verbose,
    )
    head_result = export_bafnetplus_head_to_onnx(
        head,
        head_output_path,
        chunk_size=chunk_size,
        time_frames=time_frames_val,
        sample_rate=sample_rate,
        hop_size=hop_size,
        win_size=win_size,
        compress_factor=compress_factor,
        opset_version=opset_version,
        checkpoint_info=info,
        verbose=verbose,
    )

    trunk_path_p = Path(trunk_result.onnx_path)
    head_path_p = Path(head_result.onnx_path)
    if combined_sidecar_path is None:
        combined_sidecar_path = trunk_path_p.parent / "bafnetplus_split.json"
    combined_sidecar_path = Path(combined_sidecar_path)
    combined_sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    combined_metadata: Dict[str, Any] = {
        "schema_version": _BAFNETPLUS_SCHEMA_SPLIT_COMBINED,
        "produced_by": "src.models.streaming.onnx.export.export_bafnetplus_split_to_onnx_from_checkpoint",
        "checkpoint": info,
        "geometry": {
            "chunk_size": int(chunk_size),
            "encoder_lookahead": int(trunk.mapping_core.encoder_lookahead),
            "decoder_lookahead": int(trunk.mapping_core.decoder_lookahead),
            "total_lookahead": int(trunk.total_lookahead),
            "alpha_time_lookahead": int(head.total_lookahead),
            "T_export": int(time_frames_val),
            "T_export_formula": "chunk_size + encoder_lookahead + decoder_lookahead "
            "(shared by trunk + head; alpha_time_lookahead = 0 for the deployed ckpt)",
            "freq_size": int(trunk_result.freq_size),
        },
        "stft": {
            "n_fft": int(trunk.n_fft),
            "hop_size": int(hop_size),
            "win_size": int(win_size),
            "compress_factor": float(compress_factor),
            "center": True,
            "sample_rate": int(sample_rate),
        },
        "boundary": {
            "tensor_names": list(_SPLIT_BOUNDARY_TENSOR_NAMES),
            "shape": [1, int(trunk_result.freq_size), int(time_frames_val)],
            "dtype": "float32",
            "ordering_note": (
                "Per-branch trunk outputs in mapping-then-masking order; "
                "acs_mask is the masking branch's post-LearnableSigmoid raw mask."
            ),
        },
        "state_partition": {
            "trunk": {
                "num_states": int(trunk.num_states),
                "names_prefix": ["mapping/", "masking/"],
            },
            "head": {
                "num_states": int(head.num_states),
                "names_prefix": ["calibration/", "alpha/"],
            },
            "total": int(trunk.num_states + head.num_states),
        },
        "trunk": {
            "onnx_file": trunk_path_p.name,
            "sidecar_file": Path(trunk_result.metadata_path).name,
            "onnx_size_bytes": trunk_result.metadata.get("onnx_size_bytes"),
            "schema_version": trunk_result.metadata["schema_version"],
            "num_states": int(trunk.num_states),
            "input_names": trunk_result.metadata["io"]["input_names"],
            "output_names": trunk_result.metadata["io"]["output_names"],
        },
        "head": {
            "onnx_file": head_path_p.name,
            "sidecar_file": Path(head_result.metadata_path).name,
            "onnx_size_bytes": head_result.metadata.get("onnx_size_bytes"),
            "schema_version": head_result.metadata["schema_version"],
            "num_states": int(head.num_states),
            "input_names": head_result.metadata["io"]["input_names"],
            "output_names": head_result.metadata["io"]["output_names"],
        },
    }
    combined_sidecar_path.write_text(json.dumps(combined_metadata, indent=2, sort_keys=False))
    if verbose:
        print(f"  wrote combined sidecar: {combined_sidecar_path}")

    return {
        "trunk": trunk_result,
        "head": head_result,
        "combined_sidecar_path": str(combined_sidecar_path),
        "combined_metadata": combined_metadata,
    }


@torch.inference_mode()
def run_bafnetplus_split_streaming(
    trunk: BAFNetPlusTrunkCore,
    head: BAFNetPlusHeadCore,
    bcs_com: Tensor,
    acs_com: Tensor,
    chunk_size: int,
    *,
    init_trunk_states: Optional[List[Tensor]] = None,
    init_head_states: Optional[List[Tensor]] = None,
    freq_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Drive the PT split (trunk → head) chain chunk-by-chunk over paired spectrograms.

    Mirrors :func:`run_bafnetplus_core_streaming` for the split design. For each
    chunk: feed ``T_export = chunk_size + L_enc + L_dec`` frames to the trunk
    (with the trunk's state-update window bounded to ``chunk_size``), feed the
    trunk's 7 boundary outputs to the head (with the head's state-update
    window bounded to ``chunk_size``), keep the first ``chunk_size`` output
    frames. Carry trunk states (184) and head states (6) across chunks
    independently — there is no cross-flow between them by construction.

    Args:
        trunk: :class:`BAFNetPlusTrunkCore`.
        head: :class:`BAFNetPlusHeadCore`.
        bcs_com: ``[B, F, T, 2]`` (B=1) BCS complex spectrogram.
        acs_com: ``[B, F, T, 2]`` (B=1) ACS complex spectrogram.
        chunk_size: Output frames per chunk.
        init_trunk_states: Initial trunk states (defaults to zero-init).
        init_head_states: Initial head states (defaults to zero-init).
        freq_size: ``n_fft // 2 + 1`` (defaults to that).

    Returns:
        ``(est_mag [1, F, recon_T], est_pha [1, F, recon_T], est_com [1, F, recon_T, 2])``
        with ``recon_T = n_full_chunks * chunk_size``.
    """
    if bcs_com.dim() == 3:
        bcs_com = bcs_com.unsqueeze(0)
    if acs_com.dim() == 3:
        acs_com = acs_com.unsqueeze(0)
    if bcs_com.shape != acs_com.shape:
        raise ValueError(f"bcs_com / acs_com shape mismatch: {bcs_com.shape} vs {acs_com.shape}")
    if bcs_com.shape[0] != 1:
        raise ValueError(f"run_bafnetplus_split_streaming requires batch size 1, got {bcs_com.shape[0]}")

    f = bcs_com.shape[1]
    t_total = bcs_com.shape[2]
    total_lookahead = trunk.total_lookahead
    t_export = chunk_size + total_lookahead
    if t_total < t_export:
        raise ValueError(
            f"input has {t_total} frames but T_export={t_export}; input is too short for even one full chunk."
        )

    if freq_size is None:
        freq_size = f
    if init_trunk_states is None:
        init_trunk_states = trunk.init_states(batch_size=1, freq_size=freq_size, time_frames=t_export)
    if init_head_states is None:
        init_head_states = head.init_states(batch_size=1, freq_size=freq_size, time_frames=t_export)

    prev_trunk_window = trunk.mapping_core.state_frames_for_update
    prev_head_window = head.state_frames_for_update
    trunk.set_state_frames_for_update(chunk_size)
    head.set_state_frames_for_update(chunk_size)
    try:
        mags: List[Tensor] = []
        phas: List[Tensor] = []
        coms: List[Tensor] = []
        current_trunk = list(init_trunk_states)
        current_head = list(init_head_states)
        for t in range(0, t_total - t_export + 1, chunk_size):
            bcs_chunk = bcs_com[:, :, t : t + t_export, :]
            acs_chunk = acs_com[:, :, t : t + t_export, :]
            bcs_mag_c, bcs_pha_c = complex_to_mag_pha(bcs_chunk, stack_dim=-1)
            acs_mag_c, acs_pha_c = complex_to_mag_pha(acs_chunk, stack_dim=-1)
            trunk_outs = trunk(bcs_mag_c, bcs_pha_c, acs_mag_c, acs_pha_c, *current_trunk)
            boundary = trunk_outs[: len(_SPLIT_BOUNDARY_TENSOR_NAMES)]
            current_trunk = list(trunk_outs[len(_SPLIT_BOUNDARY_TENSOR_NAMES) :])
            head_outs = head(*boundary, *current_head)
            est_mag, est_pha, est_com = head_outs[0], head_outs[1], head_outs[2]
            current_head = list(head_outs[3:])
            mags.append(est_mag[:, :, :chunk_size])
            phas.append(est_pha[:, :, :chunk_size])
            coms.append(est_com[:, :, :chunk_size, :])
    finally:
        trunk.set_state_frames_for_update(prev_trunk_window)
        head.set_state_frames_for_update(prev_head_window)

    return torch.cat(mags, dim=2), torch.cat(phas, dim=2), torch.cat(coms, dim=2)


def verify_bafnetplus_split_multistep(
    trunk_onnx_path: Union[str, Path],
    head_onnx_path: Union[str, Path],
    trunk: BAFNetPlusTrunkCore,
    head: BAFNetPlusHeadCore,
    *,
    chunk_size: int,
    time_frames: Optional[int] = None,
    freq_size: Optional[int] = None,
    batch_size: int = 1,
    num_steps: int = 5,
    atol: float = 1e-4,
    state_atol: Optional[float] = None,
    seed: int = 2039,
    verbose: bool = False,
) -> Dict[str, Any]:
    """**S21 split exit gate**: chained ORT-vs-PT split parity across ``num_steps``.

    Mirrors :func:`verify_bafnetplus_core_multistep` for the split design.
    Drives both the PT split chain (trunk → head) and the ORT split chain
    (trunk session → head session) on the SAME random ``(bcs_mag, bcs_pha,
    acs_mag, acs_pha)`` inputs with the SAME initial zero states. Compares
    the final ``(est_mag, est_pha, est_com)`` outputs + all next-states
    (trunk + head, 190 total) per step.

    The trunk's intermediate boundary tensors (the 7 outputs of the trunk
    ONNX) are NOT cross-compared between PT and ORT directly — only their
    aggregated effect on the head's outputs is policed. If a boundary-level
    drift surfaces, it shows up in the final est_mag / est_pha / est_com.

    Args:
        trunk_onnx_path / head_onnx_path: Paths to the exported trunk + head ONNX.
        trunk / head: The PT cores the ONNX files were exported from.
        chunk_size / time_frames / freq_size / batch_size / num_steps / atol /
            state_atol / seed / verbose: Same semantics as
            :func:`verify_bafnetplus_core_multistep`.

    Returns:
        Dict with the same keys as :func:`verify_bafnetplus_core_multistep`
        plus ``trunk_state_diff`` and ``head_state_diff`` for trunk vs head
        state-drift attribution.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}

    if state_atol is None:
        state_atol = atol

    trunk.eval()
    head.eval()
    trunk.set_state_frames_for_update(chunk_size)
    head.set_state_frames_for_update(chunk_size)

    freq_size_val: int = freq_size if freq_size is not None else (trunk.n_fft // 2 + 1)
    time_frames_val: int = time_frames if time_frames is not None else (chunk_size + trunk.total_lookahead)

    device = next(trunk.parameters()).device
    dtype = next(trunk.parameters()).dtype

    pt_trunk_states: List[Tensor] = trunk.init_states(
        batch_size=batch_size, freq_size=freq_size_val, time_frames=time_frames_val, device=device, dtype=dtype
    )
    pt_head_states: List[Tensor] = head.init_states(
        batch_size=batch_size, freq_size=freq_size_val, time_frames=time_frames_val, device=device, dtype=dtype
    )
    ort_trunk_states: List[np.ndarray] = [s.detach().cpu().numpy() for s in pt_trunk_states]
    ort_head_states: List[np.ndarray] = [s.detach().cpu().numpy() for s in pt_head_states]
    trunk_state_names = trunk.get_state_names()
    head_state_names = head.get_state_names()

    n_outs = _NUM_NON_STATE_OUTPUTS_ATAN2
    trunk_sess = ort.InferenceSession(str(trunk_onnx_path), providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(str(head_onnx_path), providers=["CPUExecutionProvider"])

    rng = torch.Generator(device="cpu").manual_seed(seed)
    steps_info: List[Dict[str, Any]] = []
    max_output_diffs: List[float] = [0.0] * n_outs
    max_phase_wrapped_diff: float = 0.0
    max_trunk_state_diff: float = 0.0
    max_head_state_diff: float = 0.0
    state_shape_check = True
    all_finite = True

    for step in range(num_steps):
        bcs_mag = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)
        bcs_pha = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)
        acs_mag_raw = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)
        acs_mag = acs_mag_raw.abs() + 1e-3
        acs_pha = torch.randn(batch_size, freq_size_val, time_frames_val, generator=rng, dtype=dtype).to(device)

        # --- PT split chain ---
        with torch.no_grad():
            pt_trunk_outs = trunk(bcs_mag, bcs_pha, acs_mag, acs_pha, *pt_trunk_states)
        pt_boundary = list(pt_trunk_outs[: len(_SPLIT_BOUNDARY_TENSOR_NAMES)])
        pt_trunk_next = list(pt_trunk_outs[len(_SPLIT_BOUNDARY_TENSOR_NAMES) :])
        with torch.no_grad():
            pt_head_outs = head(*pt_boundary, *pt_head_states)
        pt_non_state = [t.detach().cpu().numpy() for t in pt_head_outs[:n_outs]]
        pt_head_next = list(pt_head_outs[n_outs:])

        # State integrity (shape + finiteness) on the PT side.
        for i, (prev, nxt) in enumerate(zip(pt_trunk_states, pt_trunk_next)):
            if tuple(prev.shape) != tuple(nxt.shape):
                state_shape_check = False
                logger.error(
                    "step %d trunk state %d (%s) shape mismatch: prev=%s next=%s",
                    step, i, trunk_state_names[i], tuple(prev.shape), tuple(nxt.shape),
                )
            if not torch.isfinite(nxt).all():
                all_finite = False
        for i, (prev, nxt) in enumerate(zip(pt_head_states, pt_head_next)):
            if tuple(prev.shape) != tuple(nxt.shape):
                state_shape_check = False
                logger.error(
                    "step %d head state %d (%s) shape mismatch: prev=%s next=%s",
                    step, i, head_state_names[i], tuple(prev.shape), tuple(nxt.shape),
                )
            if not torch.isfinite(nxt).all():
                all_finite = False

        # --- ORT split chain ---
        trunk_inputs: Dict[str, np.ndarray] = {
            "bcs_mag": bcs_mag.detach().cpu().numpy(),
            "bcs_pha": bcs_pha.detach().cpu().numpy(),
            "acs_mag": acs_mag.detach().cpu().numpy(),
            "acs_pha": acs_pha.detach().cpu().numpy(),
        }
        for name, arr in zip(trunk_state_names, ort_trunk_states):
            trunk_inputs[name] = arr
        trunk_outputs = trunk_sess.run(None, trunk_inputs)
        ort_boundary = trunk_outputs[: len(_SPLIT_BOUNDARY_TENSOR_NAMES)]
        ort_trunk_next = trunk_outputs[len(_SPLIT_BOUNDARY_TENSOR_NAMES) :]

        head_inputs: Dict[str, np.ndarray] = {
            name: arr for name, arr in zip(_SPLIT_BOUNDARY_TENSOR_NAMES, ort_boundary)
        }
        for name, arr in zip(head_state_names, ort_head_states):
            head_inputs[name] = arr
        head_outputs = head_sess.run(None, head_inputs)
        ort_non_state = head_outputs[:n_outs]
        ort_head_next = head_outputs[n_outs:]

        # Compare PT vs ORT final outputs + states.
        out_diffs = [float(np.abs(pt_non_state[i] - ort_non_state[i]).max()) for i in range(n_outs)]
        trunk_state_diffs = [
            float(np.abs(pt_trunk_next[i].detach().cpu().numpy() - ort_trunk_next[i]).max())
            for i in range(len(pt_trunk_next))
        ]
        head_state_diffs = [
            float(np.abs(pt_head_next[i].detach().cpu().numpy() - ort_head_next[i]).max())
            for i in range(len(pt_head_next))
        ]
        max_output_diffs = [max(a, b) for a, b in zip(max_output_diffs, out_diffs)]
        if trunk_state_diffs:
            max_trunk_state_diff = max(max_trunk_state_diff, max(trunk_state_diffs))
        if head_state_diffs:
            max_head_state_diff = max(max_head_state_diff, max(head_state_diffs))
        wrapped_pha = _wrapped_phase_max_abs(pt_non_state[1], ort_non_state[1])
        max_phase_wrapped_diff = max(max_phase_wrapped_diff, wrapped_pha)

        for i, arr in enumerate(ort_trunk_next):
            if not np.isfinite(arr).all():
                all_finite = False
                logger.error("step %d trunk state %d (ORT) contains non-finite values", step, i)
        for i, arr in enumerate(ort_head_next):
            if not np.isfinite(arr).all():
                all_finite = False
                logger.error("step %d head state %d (ORT) contains non-finite values", step, i)

        steps_info.append(
            {
                "step": step,
                "output_max_diffs": out_diffs,
                "trunk_state_max_diffs": trunk_state_diffs,
                "head_state_max_diffs": head_state_diffs,
                "phase_wrapped_max_diff": wrapped_pha,
            }
        )

        # Carry states forward.
        pt_trunk_states = pt_trunk_next
        pt_head_states = pt_head_next
        ort_trunk_states = list(ort_trunk_next)
        ort_head_states = list(ort_head_next)

    max_state_diff = max(max_trunk_state_diff, max_head_state_diff)
    all_outputs_close = all(d <= atol for d in max_output_diffs)
    all_states_close = max_state_diff <= state_atol
    all_match = all_outputs_close and all_states_close and state_shape_check and all_finite

    result: Dict[str, Any] = {
        "all_match": all_match,
        "max_output_diffs": max_output_diffs,
        "max_phase_wrapped_diff": max_phase_wrapped_diff,
        "max_trunk_state_diff": float(max_trunk_state_diff),
        "max_head_state_diff": float(max_head_state_diff),
        "max_state_diff": float(max_state_diff),
        "state_shape_check": state_shape_check,
        "all_finite": all_finite,
        "num_trunk_states": trunk.num_states,
        "num_head_states": head.num_states,
        "num_non_state_outputs": n_outs,
        "atol": float(atol),
        "state_atol": float(state_atol),
        "num_steps": num_steps,
        "steps": steps_info,
    }
    if verbose:
        status = "PASS" if all_match else "FAIL"
        out_str = " ".join(f"{d:.3e}" for d in max_output_diffs)
        print(
            f"verify_bafnetplus_split_multistep ({num_steps} steps, atol={atol:.1e}): {status}\n"
            f"  max_output_diffs: {out_str}\n"
            f"  max_phase_wrapped_diff: {max_phase_wrapped_diff:.3e}\n"
            f"  max_trunk_state_diff: {max_trunk_state_diff:.3e}\n"
            f"  max_head_state_diff: {max_head_state_diff:.3e}\n"
            f"  state_shape_check: {state_shape_check}, all_finite: {all_finite}"
        )
    return result


__all__ = [
    "ExportResult",
    "load_backbone_from_checkpoint",
    "run_core_streaming",
    "export_backbone_core_to_onnx",
    "export_backbone_to_onnx_from_checkpoint",
    "verify_backbone_core_multistep",
    # S8 BAFNet+ export driver.
    "load_bafnetplus_from_checkpoint",
    "run_bafnetplus_core_streaming",
    "export_bafnetplus_core_to_onnx",
    "export_bafnetplus_to_onnx_from_checkpoint",
    "verify_bafnetplus_core_multistep",
    # S17 INT8 QDQ driver.
    "quantize_bafnetplus_qdq",
    # S21 split-graph driver (B2 deployable track).
    "export_bafnetplus_trunk_to_onnx",
    "export_bafnetplus_head_to_onnx",
    "export_bafnetplus_split_to_onnx_from_checkpoint",
    "run_bafnetplus_split_streaming",
    "verify_bafnetplus_split_multistep",
]
