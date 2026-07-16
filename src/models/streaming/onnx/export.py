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
- FP32 throughout with static shapes. ``ConvTranspose2d``
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
from src.models.streaming.onnx.bafnetplus_core import ExportableBAFNetPlusCore
from src.stft import complex_to_mag_pha

logger = logging.getLogger(__name__)

# Frozen non-state output counts (the contract S8/S9/S11 re-bind to).
_NUM_NON_STATE_OUTPUTS_ATAN2 = 3  # est_mag, est_pha, est_com
_NUM_NON_STATE_OUTPUTS_COMPLEX = 3  # est_mag, phase_real, phase_imag

# Sidecar schema versions for the two BAFNetPlus FP32 graph variants.
_BAFNETPLUS_SCHEMA_FP32_ATAN2 = "s8-bafnetplus-functional-fp32"
_BAFNETPLUS_SCHEMA_FP32_COMPLEX = "s20-bafnetplus-functional-fp32-complex"
# Cycle 18b-8 debug variant of the FP32 complex graph: adds 4 alpha-stack
# intermediate outputs (debug_alpha_features + first-block conv/BN/PReLU) for
# sub-op divergence forensics. Production output contract is preserved — the
# debug outputs trail the production outputs in graph order.
_BAFNETPLUS_SCHEMA_FP32_COMPLEX_DEBUG_ALPHA = "s20-bafnetplus-functional-fp32-complex-debug-alpha-v1"
_NUM_DEBUG_ALPHA_OUTPUTS = 4  # debug_alpha_features + conv0_conv + conv0_bn + conv0_prelu
# Cycle 18b-10 debug variant of the FP32 complex graph: adds 9 mapping-branch
# decoder intermediate outputs (sequence_block output + 4 mask_decoder
# dense_block iter outputs + 4 phase_decoder dense_block iter outputs) for
# mapping-cascade sub-op divergence forensics. Production output contract is
# preserved — the debug outputs trail the production outputs in graph order.
_BAFNETPLUS_SCHEMA_FP32_COMPLEX_DEBUG_MAPPING = "s20-bafnetplus-functional-fp32-complex-debug-mapping-v1"
_NUM_DEBUG_MAPPING_OUTPUTS = 9  # seq_block_out + 4 mask_dense + 4 phase_dense
# Cycle 18b-11 Tier 1A debug variant of the FP32 complex graph: adds 8
# mapping-branch TSBlock-internal intermediate outputs from the LAST
# TSBlock's freq_stage (SimpleGate + SCA multiply + 4× GPKFFN attn*conv +
# GPKFFN proj_last + final state_84 input) for Codex's narrowed sub-op
# divergence forensics. Production output contract is preserved — the
# debug outputs trail the production outputs in graph order. Mutually
# exclusive with cycle 18b-8 debug-alpha, cycle 18b-10 debug-mapping, and
# cycle 18b-9 alpha workaround variants.
_BAFNETPLUS_SCHEMA_FP32_COMPLEX_DEBUG_TSBLOCK = "s20-bafnetplus-functional-fp32-complex-debug-tsblock-v1"
_NUM_DEBUG_TSBLOCK_OUTPUTS = 8  # simplegate + sca_mult + 4× attnconv + proj_last + state_84_input
# Cycle 18b-12 Track (v) boundary-only TSBlock probe variant: emits one
# debug output per TSBlock (block output captured post time_stage +
# freq_stage + residual) without any intra-block code change. For the
# 50ms BAFNet+ recipe (num_tsblock=4) the count is 4; the constant below
# tracks that expected anchor. Production schema
# ``s20-bafnetplus-functional-fp32-complex`` stays bound to the
# 0-debug-output combination so the deployed D3 sidecar md5
# (804f7b1d…) stays byte-identical.
_BAFNETPLUS_SCHEMA_FP32_COMPLEX_DEBUG_TSBLOCK_BOUNDARY = (
    "s20-bafnetplus-functional-fp32-complex-debug-tsblock-boundary-v1"
)
_NUM_DEBUG_TSBLOCK_BOUNDARY_OUTPUTS = 4  # one per TSBlock on the 50ms recipe
# Cycle 18b-11 Tier 1B FP16-safe epsilon variant of the FP32 complex
# graph: same op set / output contract as production, but the in-graph
# ``_phase_components_to_complex`` uses raised epsilons (default 1e-3 /
# 1e-6) that survive FP16 ``flush-to-zero`` on the QNN HTP-FP16 delegate.
# Codex's host FP16 emulation predicts the cycle 18b-8 alpha 2/8442
# nonfinite spots vanish under this variant. Production schema
# ``s20-bafnetplus-functional-fp32-complex`` stays bound to the
# 1e-8 / 1e-16 defaults so the deployed D3 sidecar md5 (804f7b1d…) stays
# byte-identical.
_BAFNETPLUS_SCHEMA_FP32_COMPLEX_FP16_SAFE_EPSILON = (
    "s20-bafnetplus-functional-fp32-complex-fp16-safe-epsilon-v1"
)
# Cycle 18b-9 alpha-feature numerical workaround variants of the FP32 complex
# graph. The production output contract (3 non-state + 190 next-state) is
# preserved; only the in-graph arithmetic of the module-level
# ``_build_alpha_features`` is modified at export time. The schema token
# disambiguates the variants for downstream tooling. Production schema
# ``s20-bafnetplus-functional-fp32-complex`` stays bound to the
# ``alpha_clamp_threshold=None`` + ``alpha_safe_sqrt=False`` combination so
# the deployed D3 sidecar md5 (804f7b1d…) stays byte-identical.
_BAFNETPLUS_SCHEMA_FP32_COMPLEX_CLAMP = "s20-bafnetplus-functional-fp32-complex-clamp-v1"
_BAFNETPLUS_SCHEMA_FP32_COMPLEX_SAFE_SQRT = "s20-bafnetplus-functional-fp32-complex-safe-sqrt-v1"
_BAFNETPLUS_SCHEMA_FP32_COMPLEX_CLAMP_SAFE_SQRT = (
    "s20-bafnetplus-functional-fp32-complex-clamp-safe-sqrt-v1"
)

# Reshape-free TSBlock has been the only path since the cycle-13 cleanup that
# promoted Path β. The FP32 sidecar contract (input/output/state names, shapes,
# T_export, num_states) is unchanged; only the TSBlock state layout switches
# from 3D
# ``[B*F_enc, C, padding]`` to 4D ``[B, C, padding, F_enc]`` and the
# corresponding op count drops (Reshape 216→56, Transpose 66→42 in FP32;
# the transitional cycle-11/12 schema tokens are retired with the cleanup.


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
    emit_debug_alpha = bool(getattr(core, "emit_debug_alpha", False))
    emit_debug_mapping_decoder = bool(getattr(core, "emit_debug_mapping_decoder", False))
    emit_debug_mapping_tsblock_internals = bool(
        getattr(core, "emit_debug_mapping_tsblock_internals", False)
    )
    emit_debug_mapping_tsblock_boundaries = bool(
        getattr(core, "emit_debug_mapping_tsblock_boundaries", False)
    )
    alpha_clamp_threshold = getattr(core, "alpha_clamp_threshold", None)
    alpha_safe_sqrt = bool(getattr(core, "alpha_safe_sqrt", False))
    has_alpha_workaround = (alpha_clamp_threshold is not None) or alpha_safe_sqrt
    phase_norm_eps = float(getattr(core, "phase_normalization_epsilon", 1e-8))
    phase_norm_sq_eps = float(getattr(core, "phase_normalization_squared_epsilon", 1e-16))
    has_fp16_safe_eps = (phase_norm_eps != 1e-8) or (phase_norm_sq_eps != 1e-16)
    if emit_debug_mapping_decoder:
        # Cycle 18b-10: debug-mapping schema only applies on top of the FP32
        # complex graph (the production deploy variant we're forensically
        # inspecting). Mutually exclusive with cycle 18b-8 debug-alpha and
        # cycle 18b-9 alpha workaround knobs — pick one debug variant per
        # exported ONNX.
        if core.phase_output_mode != "complex":
            raise ValueError(
                "emit_debug_mapping_decoder=True requires phase_output_mode='complex' "
                f"(got {core.phase_output_mode!r})"
            )
        if emit_debug_alpha:
            raise ValueError(
                "emit_debug_mapping_decoder=True is incompatible with emit_debug_alpha=True; "
                "pick one debug variant."
            )
        if has_alpha_workaround:
            raise ValueError(
                "emit_debug_mapping_decoder=True is incompatible with alpha workaround "
                "options (alpha_clamp_threshold / alpha_safe_sqrt); pick one."
            )
        if has_fp16_safe_eps:
            raise ValueError(
                "emit_debug_mapping_decoder=True is incompatible with FP16-safe epsilons; "
                "pick one variant."
            )
        schema_version = _BAFNETPLUS_SCHEMA_FP32_COMPLEX_DEBUG_MAPPING
    elif emit_debug_mapping_tsblock_internals:
        # Cycle 18b-11 Tier 1A: debug-tsblock schema only applies on top
        # of the FP32 complex graph. Mutually exclusive with all other
        # debug / workaround variants — pick one per exported ONNX.
        if core.phase_output_mode != "complex":
            raise ValueError(
                "emit_debug_mapping_tsblock_internals=True requires phase_output_mode='complex' "
                f"(got {core.phase_output_mode!r})"
            )
        if emit_debug_alpha:
            raise ValueError(
                "emit_debug_mapping_tsblock_internals=True is incompatible with emit_debug_alpha=True; "
                "pick one debug variant."
            )
        if has_alpha_workaround:
            raise ValueError(
                "emit_debug_mapping_tsblock_internals=True is incompatible with alpha workaround "
                "options (alpha_clamp_threshold / alpha_safe_sqrt); pick one."
            )
        if has_fp16_safe_eps:
            raise ValueError(
                "emit_debug_mapping_tsblock_internals=True is incompatible with FP16-safe epsilons; "
                "pick one variant."
            )
        schema_version = _BAFNETPLUS_SCHEMA_FP32_COMPLEX_DEBUG_TSBLOCK
    elif emit_debug_mapping_tsblock_boundaries:
        # Cycle 18b-12 Track (v): boundary-only TSBlock schema applies
        # on top of the FP32 complex graph. Mutually exclusive with all
        # other debug / workaround variants.
        if core.phase_output_mode != "complex":
            raise ValueError(
                "emit_debug_mapping_tsblock_boundaries=True requires phase_output_mode='complex' "
                f"(got {core.phase_output_mode!r})"
            )
        if emit_debug_alpha:
            raise ValueError(
                "emit_debug_mapping_tsblock_boundaries=True is incompatible with emit_debug_alpha=True; "
                "pick one debug variant."
            )
        if has_alpha_workaround:
            raise ValueError(
                "emit_debug_mapping_tsblock_boundaries=True is incompatible with alpha workaround "
                "options (alpha_clamp_threshold / alpha_safe_sqrt); pick one."
            )
        if has_fp16_safe_eps:
            raise ValueError(
                "emit_debug_mapping_tsblock_boundaries=True is incompatible with FP16-safe epsilons; "
                "pick one variant."
            )
        schema_version = _BAFNETPLUS_SCHEMA_FP32_COMPLEX_DEBUG_TSBLOCK_BOUNDARY
    elif has_fp16_safe_eps:
        # Cycle 18b-11 Tier 1B: FP16-safe epsilon variant of the FP32
        # complex graph only. Mutually exclusive with all debug + alpha
        # workaround variants — pick one per exported ONNX.
        if core.phase_output_mode != "complex":
            raise ValueError(
                "FP16-safe epsilons require phase_output_mode='complex' "
                f"(got {core.phase_output_mode!r})"
            )
        if emit_debug_alpha:
            raise ValueError(
                "FP16-safe epsilons are incompatible with emit_debug_alpha=True; pick one variant."
            )
        if has_alpha_workaround:
            raise ValueError(
                "FP16-safe epsilons are incompatible with alpha workaround "
                "options (alpha_clamp_threshold / alpha_safe_sqrt); pick one."
            )
        schema_version = _BAFNETPLUS_SCHEMA_FP32_COMPLEX_FP16_SAFE_EPSILON
    elif emit_debug_alpha:
        # Cycle 18b-8: debug-alpha schema only applies on top of the FP32 complex
        # graph (the production deploy variant we're forensically inspecting).
        if core.phase_output_mode != "complex":
            raise ValueError(
                "emit_debug_alpha=True requires phase_output_mode='complex' "
                f"(got {core.phase_output_mode!r})"
            )
        if has_alpha_workaround:
            raise ValueError(
                "emit_debug_alpha=True is incompatible with alpha workaround "
                "options (alpha_clamp_threshold / alpha_safe_sqrt); pick one."
            )
        schema_version = _BAFNETPLUS_SCHEMA_FP32_COMPLEX_DEBUG_ALPHA
    elif has_alpha_workaround:
        # Cycle 18b-9: numerical workaround variants of the FP32 complex graph
        # only — the atan2 path is not affected by the FP16 overflow root sub-op
        # (which appears only on the device's FP16 NNAPI delegate against the
        # complex deploy graph).
        if core.phase_output_mode != "complex":
            raise ValueError(
                "alpha workaround (alpha_clamp_threshold / alpha_safe_sqrt) requires "
                f"phase_output_mode='complex' (got {core.phase_output_mode!r})"
            )
        if (alpha_clamp_threshold is not None) and alpha_safe_sqrt:
            schema_version = _BAFNETPLUS_SCHEMA_FP32_COMPLEX_CLAMP_SAFE_SQRT
        elif alpha_clamp_threshold is not None:
            schema_version = _BAFNETPLUS_SCHEMA_FP32_COMPLEX_CLAMP
        else:
            schema_version = _BAFNETPLUS_SCHEMA_FP32_COMPLEX_SAFE_SQRT
    else:
        schema_version = (
            _BAFNETPLUS_SCHEMA_FP32_COMPLEX if core.phase_output_mode == "complex" else _BAFNETPLUS_SCHEMA_FP32_ATAN2
        )
    io_block: Dict[str, Any] = {
        "input_names": core_meta["input_names"],
        "output_names": core_meta["output_names"],
        "state_names": core_meta["state_names"],
        "state_shapes": core_meta["state_shapes"],
        "num_non_state_outputs": _NUM_NON_STATE_OUTPUTS_ATAN2,
        "state_order": core_meta["state_order"],
    }
    if emit_debug_mapping_decoder:
        debug_names = core.debug_mapping_decoder_output_names()
        io_block["debug_output_names"] = debug_names
        io_block["num_debug_outputs"] = _NUM_DEBUG_MAPPING_OUTPUTS
    elif emit_debug_mapping_tsblock_internals:
        debug_names = core.debug_mapping_tsblock_output_names()
        io_block["debug_output_names"] = debug_names
        io_block["num_debug_outputs"] = _NUM_DEBUG_TSBLOCK_OUTPUTS
    elif emit_debug_mapping_tsblock_boundaries:
        debug_names = core.debug_mapping_tsblock_boundary_output_names()
        io_block["debug_output_names"] = debug_names
        io_block["num_debug_outputs"] = len(debug_names)
    elif emit_debug_alpha:
        debug_names = core.debug_alpha_output_names()
        io_block["debug_output_names"] = debug_names
        io_block["num_debug_outputs"] = _NUM_DEBUG_ALPHA_OUTPUTS
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
        "io": io_block,
        "branches": core_meta["branches"],
        "checkpoint": checkpoint_info,
        "onnx_file": output_path.name,
        "onnx_size_bytes": output_path.stat().st_size,
    }
    if emit_debug_alpha:
        # Surfaced only on the debug variant so the production sidecar JSON
        # stays byte-identical (md5 804f7b1d... on the deployed 50 ms anchor).
        metadata["core"]["emit_debug_alpha"] = True
    if emit_debug_mapping_decoder:
        # Cycle 18b-10: surfaced only on the debug-mapping variant so the
        # production sidecar JSON stays byte-identical.
        metadata["core"]["emit_debug_mapping_decoder"] = True
    if emit_debug_mapping_tsblock_internals:
        # Cycle 18b-11 Tier 1A: surfaced only on the debug-tsblock variant
        # so the production sidecar JSON stays byte-identical.
        metadata["core"]["emit_debug_mapping_tsblock_internals"] = True
    if emit_debug_mapping_tsblock_boundaries:
        # Cycle 18b-12 Track (v): surfaced only on the boundary variant
        # so the production sidecar JSON stays byte-identical.
        metadata["core"]["emit_debug_mapping_tsblock_boundaries"] = True
    if alpha_clamp_threshold is not None:
        # Cycle 18b-9: surfaced only on the workaround variants so the
        # production sidecar JSON stays byte-identical.
        metadata["core"]["alpha_clamp_threshold"] = float(alpha_clamp_threshold)
    if alpha_safe_sqrt:
        metadata["core"]["alpha_safe_sqrt"] = True
    if has_fp16_safe_eps:
        # Cycle 18b-11 Tier 1B: surfaced only on the fp16-safe-epsilon
        # variant so the production sidecar JSON stays byte-identical.
        metadata["core"]["phase_normalization_epsilon"] = phase_norm_eps
        metadata["core"]["phase_normalization_squared_epsilon"] = phase_norm_sq_eps
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
    emit_debug_alpha: bool = False,
    alpha_clamp_threshold: Optional[float] = None,
    alpha_safe_sqrt: bool = False,
    emit_debug_mapping_decoder: bool = False,
    emit_debug_mapping_tsblock_internals: bool = False,
    emit_debug_mapping_tsblock_boundaries: bool = False,
    phase_normalization_epsilon: float = 1e-8,
    phase_normalization_squared_epsilon: float = 1e-16,
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
        emit_debug_alpha=emit_debug_alpha,
        alpha_clamp_threshold=alpha_clamp_threshold,
        alpha_safe_sqrt=alpha_safe_sqrt,
        emit_debug_mapping_decoder=emit_debug_mapping_decoder,
        emit_debug_mapping_tsblock_internals=emit_debug_mapping_tsblock_internals,
        emit_debug_mapping_tsblock_boundaries=emit_debug_mapping_tsblock_boundaries,
        phase_normalization_epsilon=phase_normalization_epsilon,
        phase_normalization_squared_epsilon=phase_normalization_squared_epsilon,
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
]
