"""PyTorch streaming wrapper for the unified BAFNet+ model (audio → audio).

S6 of the streaming-ONNX rebuild (Stage 3 pt.1). :class:`BAFNetPlusStreaming`
runs an instantiated :class:`~src.models.bafnetplus.BAFNetPlus` chunk by chunk
over a paired ``(bcs_audio, acs_audio)`` stream, producing the fused, enhanced
mono waveform. It composes two :class:`~src.models.streaming.backbone_streaming.BackboneStreaming`
instances (one per branch) + the BAFNet+ fusion algebra (calibration + alpha
softmax blend, both kept non-streaming at S6 — that becomes functional in S8)
+ a shared iSTFT/OLA buffer for the monophonic fused output.

What this is / is not
---------------------
- It IS the BAFNet+ analog of :class:`BackboneStreaming`, but with TWO
  branches + fusion + iSTFT-OLA. The buffer model mirrors LaCoSENet (cf.
  ``docs/wiki/concepts/lacosenet-backbone-streaming.md`` §4) per-branch for
  (a) the STFT-context buffer and (b) the feature buffer — the latter lives
  INSIDE each ``BackboneStreaming`` and is not re-implemented here. There is
  ONE shared iSTFT/OLA buffer because the fused output is monophonic.
- It IS the audio-correct PT streaming reference for BAFNet+. The
  spectrogram-level S5 decomposed core (:class:`~src.models.streaming.onnx.bafnetplus_core.BAFNetPlusCore`)
  is the non-streaming parity gate; this wrapper is the streaming reference
  the S7 PESQ gate and the S8/S9 ONNX/ORT exports will re-bind to.
- It is **NOT** the ONNX-exportable graph — calibration and alpha layers run
  as plain ``nn.Module`` over each matured chunk's first ``chunk_size`` frames,
  with their built-in left-only causal padding zero-padding every chunk. The
  resulting "non-streaming-fusion" drift across chunks is bounded by the
  layers' reach (cf. *Non-streaming fusion drift envelope* below); S8 will
  turn calibration / alpha functional-stateful and close that gap.
- It does **NOT** compute PESQ — that is S7.

Host buffers (cf. ``docs/wiki/concepts/lacosenet-backbone-streaming.md`` §4)
---------------------------------------------------------------------------
(a) **Per-stream input sample buffer** ``_input_buffer_bcs`` / ``_input_buffer_acs``.
    ``process_samples`` fills both and only acts when BOTH have ≥
    ``samples_per_chunk`` samples. Anything that desyncs the two streams (e.g.
    feeding ``bcs`` and ``acs`` chunks of different lengths) is silently
    coalesced into the next aligned step — the wrapper does NOT raise if a
    transient size mismatch occurs; it only emits matured output when both
    streams have crossed the same per-chunk threshold.
(b) **Per-stream STFT-context buffer** ``_stft_context_bcs`` / ``_stft_context_acs``,
    each ``win_size // 2 = 200`` samples. ``center=True`` emulation, identical
    semantics to :class:`BackboneStreaming` but maintained per branch (so the
    two host STFTs are independent — the underlying audio streams are
    independent until fusion).
(c) **Per-branch feature buffer**: lives INSIDE each ``BackboneStreaming``
    (the encoder-output / decoder-lookahead bridge). Not re-implemented here.
    Synchronisation: both branches share the 50 ms anchor's topology
    (``L_enc == L_dec == 3``), so they go ``None → matured`` simultaneously.
    The constructor asserts ``bcs_streaming.total_lookahead ==
    acs_streaming.total_lookahead`` so a future topology change would surface
    immediately, and :meth:`process_samples` asserts the two branches stay
    in sync (i.e. ``(bcs_result is None) == (acs_result is None)``).
(d) **Shared iSTFT-OLA buffer** ``_ola_buffer`` / ``_ola_norm``, each
    ``win_size - hop_size = 300`` samples. The fused output is monophonic, so
    one OLA buffer pair suffices. Reuses :func:`src.stft.manual_istft_ola`
    (the same routine BAFNet+ training uses for ``center=False`` iSTFT) for
    bit-equivalent cross-chunk OLA carry-over.

Plus a small mag FIFO (``_input_mag_fifo``) holding ``(bcs_mag_valid,
acs_mag_valid)`` tuples — the first ``chunk_size`` frames of each per-chunk
STFT output. The feature buffer inside each branch delays the matured output
by ``L_dec`` frames; the FIFO supplies the *aligned* ``acs_mag`` (and ``bcs_mag``)
to feed the fusion algebra's mask-recovery + calibration features. Sized
1–2 entries (popped once per matured chunk).

Chunk geometry (for the 50 ms BAFNet+ Backbone: same as
:class:`BackboneStreaming`'s — both branches share topology, so the wrapper
inherits a single set of values, sourced from
``bcs_streaming.streaming_config``).

Mask recovery convention (same as S5)
-------------------------------------
``BAFNetPlus.forward`` calls ``self.masking(acs_com, return_mask=True)`` to
recover the raw mask. ``BackboneStreaming.process_spectrogram_buffered`` does
NOT expose the masking branch's mask separately — it returns
``(est_mag, est_pha, est_com)`` with ``est_mag = acs_mag * mask`` (for
``infer_type='masking'``). S6 follows S5 in recovering
``acs_mask = acs_est_mag / acs_mag_input`` where ``acs_mag_input`` is the
matured chunk's input ACS magnitude (sourced from the FIFO described above).
This is exact within IEEE 754 ``(a*b)/a`` 1-ULP drift; the host's ``acs_mag``
has the ``sqrt(... + 1e-9)`` magnitude floor from :func:`src.stft.mag_pha_stft`
so the division is always safe.

Non-streaming fusion drift envelope (S6 known limitation, closed by S8)
------------------------------------------------------------------------
Calibration (``CausalConv1d`` × 2, kernel=5) and alpha (``CausalConv2d`` × 4,
kernel=7 in time) both have left-only causal padding. In a non-streaming
``nn.Module`` call over each matured chunk they zero-pad on the left in EVERY
chunk — so calibration output for the FIRST chunk is bit-identical to
``BAFNet+.forward``'s first ``chunk_size`` frames, but later chunks lose
``BAFNet+.forward``'s full-sequence left-context. Drift bound (per layer):
- calibration: depth=2, kernel=5 → past-context reach ``2 * (5-1) = 8`` frames;
- alpha: depth=4, kernel=7 → past-context reach ``4 * (7-1) = 24`` frames.

So roughly the first 24 frames of every matured chunk past chunk 0 are
"wrong" vs. ``BAFNet+.forward``. With ``chunk_size = 8``, this means most
matured chunks past 0 differ. The magnitude depends strongly on weights —
random kaiming-init synthetic weights produce large spectrogram drift (max
~1, since calibration's ``tanh`` gain heads can swing exp(±0.75) and alpha's
softmax can flip branch dominance), while trained weights produce much
smaller drift (the real unified 50 ms ckpt's steady-state audio-level
``max|d| ~ 1e-3``, ``rms ~ 2e-4`` — empirically; see the test module's
documented tolerances). The first ``chunk_size`` matured frames are always
bit-identical to ``BAFNet+.forward`` (zero left-context on both sides). S7's
PESQ gate must tolerate the steady-state drift; S8 makes calibration/alpha
functional-stateful and closes the gap.

Public API
----------
- :meth:`process_samples(bcs_chunk, acs_chunk) → Optional[Tensor]`: chunked
  input, returns ``None`` while warming up else 800 mature samples (50 ms).
- :meth:`process_audio(bcs_audio, acs_audio) → Tensor`: whole-utterance audio
  in, whole-utterance enhanced audio out (with flush + length trim).
- :meth:`process_spectrogram(noisy_bcs_com, noisy_acs_com) → SpectrogramChunk`:
  the host-STFT-free spectrogram→spectrogram driver (no iSTFT). Mirrors
  :meth:`BackboneStreaming.process_spectrogram`.
- :meth:`forward(bcs_audio, acs_audio) == process_audio(bcs_audio, acs_audio)`.
- :meth:`reset_state`: clears every host buffer + calls
  :func:`reset_streaming_state` on both branches.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.checkpoint import ConfigDict, load_checkpoint
from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.backbone_streaming import BackboneStreaming, SpectrogramChunk
from src.models.streaming.converters import reset_streaming_state
from src.models.streaming.onnx.bafnetplus_core import (
    _build_alpha_features,
    _build_calibration_features,
    audit_alpha_time_lookahead,
    audit_calibration_is_causal,
)
from src.stft import complex_to_mag_pha, mag_pha_to_complex, manual_istft_ola

logger = logging.getLogger(__name__)


class BAFNetPlusStreaming(nn.Module):
    """Chunk-by-chunk streaming wrapper around an instantiated :class:`BAFNetPlus`.

    Use :meth:`from_checkpoint` (load the unified ``bafnetplus_*ms`` checkpoint
    via the wiki foot-gun pattern + the matching ``bm_*`` per-branch configs)
    or :meth:`from_model` (an already-loaded :class:`BAFNetPlus`) to build one.
    Both internally create two :class:`BackboneStreaming` instances (deep-copies
    of the BCS mapping branch + the ACS masking branch, each stateful-converted
    and streaming-enabled) and store **shared references** to the fusion-path
    modules (calibration encoder + gain heads + alpha conv stack + ``alpha_out``).

    Attributes:
        bcs_streaming / acs_streaming: The two :class:`BackboneStreaming` instances
            (``infer_type='mapping'`` and ``'masking'`` respectively).
        calibration_encoder / common_gain_head / relative_gain_head /
            alpha_convblocks / alpha_out: SHARED REFERENCES to the BAFNet+ instance
            (registered submodules so they appear in ``state_dict`` /
            ``parameters()``).
        ablation_mode / use_calibration / use_relative_gain / mask_only_alpha:
            Algebra flags mirrored from the BAFNet+ instance.
        chunk_size / encoder_lookahead / decoder_lookahead / total_lookahead /
            total_frames_needed / samples_per_chunk / output_samples_per_chunk /
            latency_samples / latency_ms / n_fft / hop_size / win_size /
            compress_factor / sample_rate: Streaming geometry inherited from the
            BCS branch (asserted equal to the ACS branch's).
        _input_buffer_bcs / _input_buffer_acs: Per-stream raw audio buffer.
        _stft_context_bcs / _stft_context_acs: Per-stream ``win//2``-sample
            past-context buffer (``center=True`` emulation).
        _ola_buffer / _ola_norm: ``win - hop``-sample shared iSTFT-OLA carry-over.
        _input_mag_fifo: FIFO of ``(bcs_mag_valid, acs_mag_valid)`` for the
            mask-recovery + calibration features alignment.

    Audits stored at init (for downstream diagnostics):
        _audit_calibration: ``audit_calibration_is_causal`` result on the
            BAFNet+ ``calibration_encoder`` (``None`` if no calibration).
        _audit_alpha_lookahead: ``audit_alpha_time_lookahead`` on the alpha
            conv stack (must be 0 for the deployed 50 ms ckpt).
    """

    def __init__(
        self,
        model: BAFNetPlus,
        bcs_streaming: BackboneStreaming,
        acs_streaming: BackboneStreaming,
        chunk_size: int = 8,
    ) -> None:
        """Initialize from a :class:`BAFNetPlus` + two prepared :class:`BackboneStreaming`.

        Args:
            model: The :class:`BAFNetPlus` instance whose fusion-path modules
                (calibration / alpha) are SHARED REFERENCES into the wrapper.
                Its ``mapping`` / ``masking`` branches are NOT touched (those
                were deep-copied into the two ``BackboneStreaming`` instances).
            bcs_streaming: ``BackboneStreaming(infer_type='mapping')`` over the
                BCS branch (already prepared via ``BackboneStreaming.from_model``).
            acs_streaming: ``BackboneStreaming(infer_type='masking')`` over the
                ACS branch.
            chunk_size: STFT frames per output step. Must match both branches'.

        Raises:
            ValueError: If the two branches' ``infer_type`` or chunk geometry
                disagree, or if ``model.use_calibration`` requires modules that
                are missing on ``model``.
        """
        super().__init__()
        if bcs_streaming.infer_type != "mapping":
            raise ValueError(f"bcs_streaming.infer_type must be 'mapping', got {bcs_streaming.infer_type!r}")
        if acs_streaming.infer_type != "masking":
            raise ValueError(f"acs_streaming.infer_type must be 'masking', got {acs_streaming.infer_type!r}")
        if bcs_streaming.chunk_size != chunk_size or acs_streaming.chunk_size != chunk_size:
            raise ValueError(
                f"chunk_size mismatch: wrapper={chunk_size}, bcs={bcs_streaming.chunk_size}, "
                f"acs={acs_streaming.chunk_size}"
            )
        if bcs_streaming.total_lookahead != acs_streaming.total_lookahead:
            raise ValueError(
                f"branch total_lookahead mismatch (would desync the per-branch feature buffers): "
                f"bcs={bcs_streaming.total_lookahead}, acs={acs_streaming.total_lookahead}"
            )
        for attr in ("hop_size", "n_fft", "win_size", "compress_factor", "sample_rate"):
            if getattr(bcs_streaming, attr) != getattr(acs_streaming, attr):
                raise ValueError(
                    f"branch STFT/sample_rate mismatch on {attr!r}: "
                    f"bcs={getattr(bcs_streaming, attr)}, acs={getattr(acs_streaming, attr)}"
                )

        self.bcs_streaming = bcs_streaming
        self.acs_streaming = acs_streaming

        # Shared references to BAFNet+ fusion-path modules. These are registered as
        # nn.Module submodules so they appear in state_dict() / parameters().
        self.calibration_encoder = getattr(model, "calibration_encoder", None) if model.use_calibration else None
        self.common_gain_head = getattr(model, "common_gain_head", None) if model.use_calibration else None
        self.relative_gain_head = (
            getattr(model, "relative_gain_head", None) if (model.use_calibration and model.use_relative_gain) else None
        )
        self.alpha_convblocks = model.alpha_convblocks
        self.alpha_out = model.alpha_out

        self.ablation_mode: str = model.ablation_mode
        self.use_calibration: bool = bool(model.use_calibration)
        self.use_relative_gain: bool = bool(model.use_relative_gain)
        self.mask_only_alpha: bool = bool(model.mask_only_alpha)
        self.calibration_max_common_log_gain: float = float(model.calibration_max_common_log_gain)
        self.calibration_max_relative_log_gain: float = float(model.calibration_max_relative_log_gain)

        # Geometry, inherited from the BCS branch (asserted equal to ACS above).
        self.chunk_size = int(chunk_size)
        self.encoder_lookahead = int(bcs_streaming.encoder_lookahead)
        self.decoder_lookahead = int(bcs_streaming.decoder_lookahead)
        self.total_lookahead = int(bcs_streaming.total_lookahead)
        self.total_frames_needed = int(bcs_streaming.total_frames_needed)
        self.samples_per_chunk = int(bcs_streaming.samples_per_chunk)
        self.output_samples_per_chunk = int(bcs_streaming.output_samples_per_chunk)
        self.latency_samples = int(bcs_streaming.latency_samples)
        self.latency_ms = float(bcs_streaming.latency_ms)
        self.n_fft = int(bcs_streaming.n_fft)
        self.hop_size = int(bcs_streaming.hop_size)
        self.win_size = int(bcs_streaming.win_size)
        self.compress_factor = float(bcs_streaming.compress_factor)
        self.sample_rate = int(bcs_streaming.sample_rate)
        self.ola_tail_size = self.win_size - self.hop_size

        # Causality audits — surface a non-causal regression immediately.
        if self.use_calibration and self.calibration_encoder is not None:
            self._audit_calibration: Optional[Dict[str, Any]] = audit_calibration_is_causal(self.calibration_encoder)
            if not self._audit_calibration["causal"]:
                raise ValueError(
                    f"calibration_encoder is not causal: total_right_pad="
                    f"{self._audit_calibration['total_right_pad']}; "
                    "S6 fusion path assumes left-only causal padding."
                )
        else:
            self._audit_calibration = None
        self._audit_alpha_lookahead: int = audit_alpha_time_lookahead(self.alpha_convblocks)
        if self._audit_alpha_lookahead != 0:
            logger.warning(
                "alpha_convblocks has non-zero time-axis right-padding (L_alpha=%d); "
                "S6 buffer sizing assumes L_alpha=0. Consider extending the wrapper.",
                self._audit_alpha_lookahead,
            )

        self._streaming_config: Dict[str, Any] = {}
        self._reset_host_buffers()

    # ------------------------------------------------------------------ buffers
    def _reset_host_buffers(self) -> None:
        """Reset every host-level buffer (input + STFT-context × 2, OLA, FIFO)."""
        device = torch.device("cpu")
        self._input_buffer_bcs: Tensor = torch.tensor([], dtype=torch.float32, device=device)
        self._input_buffer_acs: Tensor = torch.tensor([], dtype=torch.float32, device=device)
        self._stft_context_bcs: Tensor = torch.zeros(self.win_size // 2, dtype=torch.float32, device=device)
        self._stft_context_acs: Tensor = torch.zeros(self.win_size // 2, dtype=torch.float32, device=device)
        self._ola_buffer: Tensor = torch.zeros(self.ola_tail_size, dtype=torch.float32, device=device)
        self._ola_norm: Tensor = torch.zeros(self.ola_tail_size, dtype=torch.float32, device=device)
        # FIFO of (bcs_mag_valid [1,F,chunk_size], acs_mag_valid [1,F,chunk_size])
        # for fusion-time alignment with the matured backbone outputs.
        self._input_mag_fifo: Deque[Tuple[Tensor, Tensor]] = deque()

    def reset_state(self) -> None:
        """Reset all streaming state for a new utterance.

        Clears every host-level buffer (input × 2, STFT context × 2, OLA pair,
        mag FIFO) and calls :func:`reset_streaming_state` on both branches'
        underlying models — every stateful conv's ``_state`` → ``None``.
        """
        self._reset_host_buffers()
        # Re-zero the BackboneStreaming-internal host buffers + clear conv state.
        self.bcs_streaming.reset_state()
        self.acs_streaming.reset_state()
        # Also clear any lingering conv state inside the shared fusion modules.
        # Calibration / alpha are non-streaming in S6 — no stateful convs to reset
        # there yet (S8 territory), but call the helper defensively in case a
        # future S8 prototype shares this wrapper.
        if self.calibration_encoder is not None:
            reset_streaming_state(self.calibration_encoder)
        reset_streaming_state(self.alpha_convblocks)

    @property
    def device(self) -> torch.device:
        """Device of the underlying BAFNet+ fusion-path parameters.

        Pulled off ``alpha_convblocks`` (always present); both backbones share
        device with their respective ``BackboneStreaming`` instances.
        """
        return next(self.alpha_convblocks.parameters()).device

    @property
    def streaming_config(self) -> Dict[str, Any]:
        """Geometry + provenance summary (for export metadata / debugging)."""
        cfg: Dict[str, Any] = {
            **self._streaming_config,
            "ablation_mode": self.ablation_mode,
            "use_calibration": self.use_calibration,
            "use_relative_gain": self.use_relative_gain,
            "mask_only_alpha": self.mask_only_alpha,
            "chunk_size": self.chunk_size,
            "encoder_lookahead": self.encoder_lookahead,
            "decoder_lookahead": self.decoder_lookahead,
            "total_lookahead": self.total_lookahead,
            "total_frames_needed": self.total_frames_needed,
            "samples_per_chunk": self.samples_per_chunk,
            "output_samples_per_chunk": self.output_samples_per_chunk,
            "latency_samples": self.latency_samples,
            "latency_ms": self.latency_ms,
            "n_fft": self.n_fft,
            "hop_size": self.hop_size,
            "win_size": self.win_size,
            "compress_factor": self.compress_factor,
            "sample_rate": self.sample_rate,
            "ola_tail_size": self.ola_tail_size,
            "alpha_time_lookahead": self._audit_alpha_lookahead,
            "calibration_causal": self._audit_calibration["causal"] if self._audit_calibration else None,
        }
        return cfg

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_model(
        cls,
        bafnet: BAFNetPlus,
        chunk_size: int = 8,
        device: str = "cpu",
        verbose: bool = False,
    ) -> "BAFNetPlusStreaming":
        """Build a :class:`BAFNetPlusStreaming` from an already-loaded :class:`BAFNetPlus`.

        Internally builds two ``BackboneStreaming.from_model`` instances — one
        per branch — sharing ``chunk_size`` and ``device``. The wrapper then
        stores SHARED REFERENCES to the BAFNet+ instance's fusion-path modules.

        Args:
            bafnet: An instantiated :class:`BAFNetPlus`. Its ``mapping`` /
                ``masking`` submodules are passed through
                :meth:`BackboneStreaming.from_model` (which deep-copies and
                stateful-converts each branch, leaving ``bafnet.mapping`` /
                ``bafnet.masking`` non-streaming and usable as the reference
                non-streaming forward).
            chunk_size: STFT frames per output step (50 ms variant: 8).
            device: Device for the prepared streaming models.
            verbose: Log a one-line geometry summary.

        Returns:
            A ready-to-stream :class:`BAFNetPlusStreaming` (state already reset).

        Raises:
            TypeError: If ``bafnet`` is not a :class:`BAFNetPlus`.
        """
        if not isinstance(bafnet, BAFNetPlus):
            raise TypeError(f"from_model expects a BAFNetPlus, got {type(bafnet).__name__}")

        bcs_streaming = BackboneStreaming.from_model(bafnet.mapping, chunk_size=chunk_size, device=device)
        acs_streaming = BackboneStreaming.from_model(bafnet.masking, chunk_size=chunk_size, device=device)

        instance = cls(model=bafnet, bcs_streaming=bcs_streaming, acs_streaming=acs_streaming, chunk_size=chunk_size)
        instance._streaming_config = {
            "source": "from_model",
            "model_class": type(bafnet).__name__,
            "T_export_planned": chunk_size + bcs_streaming.total_lookahead + instance._audit_alpha_lookahead,
            "encoder_breakdown": bcs_streaming.streaming_config.get("encoder_breakdown"),
            "decoder_breakdown": bcs_streaming.streaming_config.get("decoder_breakdown"),
        }
        # Move the fusion modules + audit-derived buffers to the requested device.
        instance.to(device)
        instance._move_host_buffers_to(torch.device(device))

        if verbose:
            logger.info(
                "BAFNetPlusStreaming: ablation_mode=%s chunk_size=%d L_enc=%d L_dec=%d "
                "total_frames_needed=%d samples_per_chunk=%d output_samples_per_chunk=%d latency=%.1fms",
                instance.ablation_mode,
                chunk_size,
                instance.encoder_lookahead,
                instance.decoder_lookahead,
                instance.total_frames_needed,
                instance.samples_per_chunk,
                instance.output_samples_per_chunk,
                instance.latency_ms,
            )
        return instance

    @classmethod
    def from_checkpoint(
        cls,
        chkpt_dir: str,
        chkpt_file: str = "best.th",
        chunk_size: int = 8,
        device: str = "cpu",
        verbose: bool = True,
    ) -> "BAFNetPlusStreaming":
        """Build a :class:`BAFNetPlusStreaming` from the unified BAFNet+ checkpoint dir.

        Reads ``<chkpt_dir>/.hydra/config.yaml#model``: ``model_lib`` must be
        ``bafnetplus`` and ``model_class`` ``BAFNetPlus``. Locates the matching
        per-branch ``bm_*`` Hydra configs by reading the unified ckpt's
        ``param.checkpoint_mapping`` / ``checkpoint_masking`` paths (their
        dirname gives the experiment name even when the absolute path is
        non-local, e.g. ``/workspace/...``), then looking those up under
        ``<chkpt_dir>/../<exp_name>/.hydra/config.yaml`` locally. Pops the
        non-local ``checkpoint_mapping`` / ``checkpoint_masking`` keys before
        instantiating BAFNet+ (the wiki foot-gun: those keys cause BAFNet+ to
        try to load pre-joint Backbone weights from a non-local path; we want
        ``load_pretrained_weights=False`` + ``load_state_dict(unified_ckpt['model'])``
        on the WHOLE module instead).

        Args:
            chkpt_dir: Directory containing ``.hydra/config.yaml`` and the
                unified ``best.th`` (e.g. ``results/experiments/bafnetplus_50ms``).
            chkpt_file: Checkpoint filename (default ``"best.th"``).
            chunk_size: STFT frames per output step.
            device: Device for loading + streaming.
            verbose: Print/log a loading + geometry summary.

        Returns:
            A ready-to-stream :class:`BAFNetPlusStreaming`.

        Raises:
            FileNotFoundError: If the Hydra config, ckpt file, or either
                ``bm_*_50ms`` per-branch config is absent locally.
            ValueError: If the recorded model class is not :class:`BAFNetPlus`.
        """
        from omegaconf import OmegaConf

        chkpt_dir_path = Path(chkpt_dir)
        cfg_path = chkpt_dir_path / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Hydra config not found: {cfg_path}")
        ckpt_path = chkpt_dir_path / chkpt_file
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        conf = OmegaConf.load(cfg_path)
        model_cfg = conf.model
        if str(model_cfg.model_class) != "BAFNetPlus":
            raise ValueError(
                f"from_checkpoint expects model_class=BAFNetPlus, got {model_cfg.model_class!r} "
                f"({cfg_path}); pass a unified BAFNet+ experiment directory."
            )

        # Locate the two per-branch bm_*_50ms Hydra configs.
        experiments_dir = chkpt_dir_path.parent
        bp_param = OmegaConf.to_container(model_cfg.param, resolve=True)
        if not isinstance(bp_param, dict):
            raise ValueError(f"unexpected model.param shape in {cfg_path}: {type(bp_param).__name__}")
        bp_param_dict: Dict[str, Any] = cast(Dict[str, Any], bp_param)
        ckpt_mapping_path = bp_param_dict.pop("checkpoint_mapping", None)
        ckpt_masking_path = bp_param_dict.pop("checkpoint_masking", None)
        if ckpt_mapping_path is None or ckpt_masking_path is None:
            raise ValueError(
                f"unified ckpt's model.param is missing checkpoint_mapping / checkpoint_masking "
                f"(needed to derive the per-branch experiment names): {cfg_path}"
            )
        bm_map_name = Path(str(ckpt_mapping_path)).parent.name
        bm_mask_name = Path(str(ckpt_masking_path)).parent.name
        bm_map_cfg_path = experiments_dir / bm_map_name / ".hydra" / "config.yaml"
        bm_mask_cfg_path = experiments_dir / bm_mask_name / ".hydra" / "config.yaml"
        if not bm_map_cfg_path.exists() or not bm_mask_cfg_path.exists():
            raise FileNotFoundError(
                f"per-branch Hydra config(s) missing: "
                f"mapping={bm_map_cfg_path} (exists={bm_map_cfg_path.exists()}), "
                f"masking={bm_mask_cfg_path} (exists={bm_mask_cfg_path.exists()}). "
                f"The unified ckpt recorded non-local paths "
                f"({ckpt_mapping_path!r}, {ckpt_masking_path!r}) but the experiment names "
                f"({bm_map_name!r}, {bm_mask_name!r}) were used to look them up under "
                f"{experiments_dir!r}. Place the per-branch configs there, or pass a custom path."
            )

        map_conf = OmegaConf.load(bm_map_cfg_path)
        mask_conf = OmegaConf.load(bm_mask_cfg_path)
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

        if verbose:
            print(
                f"Loading BAFNetPlusStreaming from {chkpt_dir_path} "
                f"(ablation_mode={bp_param_dict.get('ablation_mode', 'full')}, chunk_size={chunk_size})"
            )
        bafnet = BAFNetPlus(
            args_mapping=args_mapping,
            args_masking=args_masking,
            load_pretrained_weights=False,
            **bp_param_dict,
        )
        bafnet = load_checkpoint(bafnet, str(chkpt_dir_path), chkpt_file, device)
        bafnet.eval()

        instance = cls.from_model(bafnet, chunk_size=chunk_size, device=device, verbose=verbose)
        instance._streaming_config["source"] = "from_checkpoint"
        instance._streaming_config["chkpt_dir"] = str(chkpt_dir_path)
        instance._streaming_config["chkpt_file"] = chkpt_file
        instance._streaming_config["bm_mapping"] = str(bm_map_cfg_path.parent.parent)
        instance._streaming_config["bm_masking"] = str(bm_mask_cfg_path.parent.parent)
        return instance

    # ----------------------------------------------------------------- host STFT
    def _stft(self, audio: Tensor) -> Tensor:
        """``center=False`` compressed-complex STFT — same as :meth:`BackboneStreaming._stft`.

        Reuses the BCS branch's STFT implementation to guarantee bit-identical
        spectral inputs to both branches (any future change to the STFT must
        change there too).
        """
        return self.bcs_streaming._stft(audio)

    # -------------------------------------------------------------- fusion algebra
    def _run_fusion(
        self,
        bcs_est_mag: Tensor,
        bcs_est_pha: Tensor,
        bcs_est_com: Tensor,
        acs_est_mag: Tensor,
        acs_est_pha: Tensor,
        acs_est_com: Tensor,
        bcs_mag_input: Tensor,
        acs_mag_input: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Run the BAFNet+ fusion algebra (calibration + alpha + softmax blend).

        Mirrors :meth:`BAFNetPlus.forward` and :meth:`BAFNetPlusCore.forward`
        (the S5 decomposed-core re-expression). All shapes are ``[1, F, chunk_size]``
        (or ``[1, F, chunk_size, 2]`` for complex spectrograms). Calibration
        and alpha layers run as **non-streaming** ``nn.Module`` over the
        chunk — their internal left-only causal padding zero-pads on every
        chunk, so chunk-to-chunk continuity is *not* preserved (the S6 known
        limitation; see the module docstring's *Non-streaming fusion drift
        envelope*).

        Args:
            bcs_est_mag / bcs_est_pha / bcs_est_com: BCS branch matured chunk
                outputs (``[1, F, chunk_size]`` / ``[1, F, chunk_size]`` /
                ``[1, F, chunk_size, 2]``).
            acs_est_mag / acs_est_pha / acs_est_com: ACS branch matured chunk
                outputs.
            bcs_mag_input / acs_mag_input: The matched-chunk input magnitudes
                (``[1, F, chunk_size]`` — popped from ``_input_mag_fifo``).
                ``acs_mag_input`` drives the mask-recovery
                (``acs_mask = acs_est_mag / acs_mag_input``).

        Returns:
            Tuple ``(est_mag [1, F, chunk_size], est_pha [1, F, chunk_size])``
            — the fused complex spectrogram's magnitude/phase, ready for
            iSTFT-OLA.
        """
        # Mask recovery — exact within IEEE 754 (a*b)/a 1-ULP drift; acs_mag_input is
        # strictly positive thanks to mag_pha_stft's sqrt(...+1e-9) floor.
        acs_mask = acs_est_mag / acs_mag_input

        # ----- calibration path
        # NB: BAFNet+.forward and BAFNetPlusCore.forward both pass the *branch outputs*
        # (bcs_est_mag, acs_est_mag) into _build_calibration_features — NOT the inputs.
        # bcs_est_mag is the mapping branch's predicted clean magnitude;
        # acs_est_mag = acs_mag_input * acs_mask. The 5-channel calibration feature
        # consumes both energy maps + their difference + mask mean/var.
        if self.use_calibration:
            calibration_feat = _build_calibration_features(bcs_est_mag, acs_est_mag, acs_mask)
            calibration_encoder = cast(nn.Module, self.calibration_encoder)
            calibration_hidden = calibration_encoder(calibration_feat)
            common_gain_head = cast(nn.Module, self.common_gain_head)
            common_log_gain = torch.tanh(common_gain_head(calibration_hidden)) * self.calibration_max_common_log_gain
            if self.use_relative_gain:
                relative_gain_head = cast(nn.Module, self.relative_gain_head)
                relative_log_gain = (
                    torch.tanh(relative_gain_head(calibration_hidden)) * self.calibration_max_relative_log_gain
                )
                bcs_gain = torch.exp(common_log_gain - 0.5 * relative_log_gain)
                acs_gain = torch.exp(common_log_gain + 0.5 * relative_log_gain)
            else:
                bcs_gain = acs_gain = torch.exp(common_log_gain)
            bcs_gain = bcs_gain.transpose(1, 2).unsqueeze(1)
            acs_gain = acs_gain.transpose(1, 2).unsqueeze(1)
            bcs_com_fused = bcs_est_com * bcs_gain
            acs_com_fused = acs_est_com * acs_gain
        else:
            bcs_com_fused = bcs_est_com
            acs_com_fused = acs_est_com

        # ----- alpha fusion
        if self.mask_only_alpha:
            alpha = acs_mask.unsqueeze(1).transpose(2, 3)  # [B, 1, T, F]
        else:
            alpha = _build_alpha_features(bcs_com_fused, acs_com_fused, acs_mask)
        for block in self.alpha_convblocks:
            alpha = block(alpha)
        alpha = self.alpha_out(alpha)
        alpha = alpha.transpose(2, 3)
        alpha = torch.softmax(alpha, dim=1)
        alpha_bcs = alpha[:, 0].unsqueeze(-1)
        alpha_acs = alpha[:, 1].unsqueeze(-1)
        est_com = bcs_com_fused * alpha_bcs + acs_com_fused * alpha_acs

        est_mag, est_pha = complex_to_mag_pha(est_com)
        # Suppress unused-var warnings (kept by name for documentation parity with S5).
        _ = (bcs_est_pha, acs_est_pha)
        return est_mag, est_pha

    # ----------------------------------------------------------- iSTFT-OLA
    def _manual_istft_ola(self, est_mag: Tensor, est_pha: Tensor) -> Tensor:
        """Cross-chunk iSTFT-OLA using ``manual_istft_ola`` (shared with training).

        Maintains the wrapper's ``_ola_buffer`` / ``_ola_norm`` carry-over pair
        — each call returns ``chunk_size * hop_size = 800`` (50 ms) mature
        samples + updates the ``win - hop = 300``-sample tail for the next
        chunk.

        Args:
            est_mag: ``[1, F, chunk_size]`` compressed magnitude (post-fusion).
            est_pha: ``[1, F, chunk_size]`` phase (post-fusion).

        Returns:
            ``[output_samples_per_chunk]`` (``800`` for 50 ms variant) mature
            audio samples.
        """
        if self._ola_buffer.device != est_mag.device:
            self._ola_buffer = self._ola_buffer.to(est_mag.device)
            self._ola_norm = self._ola_norm.to(est_mag.device)

        output, new_ola_buffer, new_ola_norm = manual_istft_ola(
            est_mag,
            est_pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            ola_buffer=self._ola_buffer,
            ola_norm=self._ola_norm,
        )
        self._ola_buffer = new_ola_buffer
        self._ola_norm = new_ola_norm
        return output[: self.output_samples_per_chunk]

    # ------------------------------------------------------------------ helpers
    def _move_host_buffers_to(self, target: torch.device) -> None:
        """Move all host-level buffers to ``target`` (called after ``.to(device)``)."""
        self._input_buffer_bcs = self._input_buffer_bcs.to(target)
        self._input_buffer_acs = self._input_buffer_acs.to(target)
        self._stft_context_bcs = self._stft_context_bcs.to(target)
        self._stft_context_acs = self._stft_context_acs.to(target)
        self._ola_buffer = self._ola_buffer.to(target)
        self._ola_norm = self._ola_norm.to(target)
        # FIFO entries are short-lived (popped after each matured chunk); leave them as-is.

    def _stft_chunk(
        self,
        chunk: Tensor,
        stft_context: Tensor,
        input_buffer: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute ``center=True``-emulated STFT for one chunk + the next context.

        Args:
            chunk: ``[samples_per_chunk]`` slice of one stream's input buffer.
            stft_context: The current per-stream ``win//2``-sample past context.
            input_buffer: The full per-stream input buffer (the chunk is its
                prefix; we read ``advance - context_size : advance`` from this
                for the next context).

        Returns:
            ``(spec, next_stft_context)`` where ``spec`` is ``[1, F, total_frames_needed, 2]``
            compressed complex and ``next_stft_context`` is the updated
            ``win//2``-sample past context.
        """
        context_size = self.win_size // 2
        chunk_with_context = torch.cat([stft_context.to(chunk.device), chunk])
        spec = self._stft(chunk_with_context)
        advance = self.output_samples_per_chunk
        if advance >= context_size:
            next_stft_context = input_buffer[advance - context_size : advance].clone()
        else:
            need_from_prev = context_size - advance
            prev_part = stft_context[len(stft_context) - need_from_prev :]
            curr_part = input_buffer[:advance]
            next_stft_context = torch.cat([prev_part, curr_part]).clone()
        return spec, next_stft_context

    # ------------------------------------------------------------- public APIs
    @torch.inference_mode()
    def process_samples(self, bcs_chunk: Tensor, acs_chunk: Tensor) -> Optional[Tensor]:
        """Feed paired audio chunks; return one matured 800-sample (50 ms) output or ``None``.

        Accumulates ``bcs_chunk`` and ``acs_chunk`` in the per-stream input
        buffers. Once BOTH have ≥ ``samples_per_chunk`` samples, runs both
        branches' streaming step + the fusion algebra + iSTFT-OLA, and returns
        the matured ``output_samples_per_chunk`` (800-sample, 50 ms) waveform.
        Returns ``None`` while the warm-up pipeline fills (typically the first
        2 calls for the 50 ms anchor — both branches go ``None → matured``
        simultaneously thanks to identical topology).

        Does **not** reset state between calls — use :meth:`reset_state` (or
        :meth:`process_audio`) for a new utterance.

        Args:
            bcs_chunk: ``[T]`` or ``[1, T]`` BCS audio (any length — buffered).
            acs_chunk: ``[T]`` or ``[1, T]`` ACS audio (any length — buffered).

        Returns:
            ``Tensor [output_samples_per_chunk]`` of mature fused-enhanced
            audio samples, or ``None`` while either input buffer / lookahead
            pipeline is still filling.

        Raises:
            ValueError: If either chunk is 2-D with a batch dimension != 1,
                or if the two branches' :class:`BackboneStreaming` returns
                desynchronise (one ``None``, the other matured — only possible
                if topology was modified post-init).
        """
        if bcs_chunk.dim() == 2:
            if bcs_chunk.shape[0] != 1:
                raise ValueError("batch size must be 1 for streaming")
            bcs_chunk = bcs_chunk.squeeze(0)
        if acs_chunk.dim() == 2:
            if acs_chunk.shape[0] != 1:
                raise ValueError("batch size must be 1 for streaming")
            acs_chunk = acs_chunk.squeeze(0)
        target_device = self.device
        bcs_chunk = bcs_chunk.to(target_device)
        acs_chunk = acs_chunk.to(target_device)

        if self._stft_context_bcs.device != target_device:
            self._move_host_buffers_to(target_device)

        # Append to per-stream input buffers.
        self._input_buffer_bcs = torch.cat([self._input_buffer_bcs.to(target_device), bcs_chunk])
        self._input_buffer_acs = torch.cat([self._input_buffer_acs.to(target_device), acs_chunk])

        # Only act when BOTH streams have ≥ samples_per_chunk samples.
        if len(self._input_buffer_bcs) < self.samples_per_chunk:
            return None
        if len(self._input_buffer_acs) < self.samples_per_chunk:
            return None

        # Per-stream chunk + STFT (center=True emulation).
        bcs_chunk_samples = self._input_buffer_bcs[: self.samples_per_chunk]
        acs_chunk_samples = self._input_buffer_acs[: self.samples_per_chunk]
        bcs_spec, next_stft_context_bcs = self._stft_chunk(
            bcs_chunk_samples, self._stft_context_bcs, self._input_buffer_bcs
        )
        acs_spec, next_stft_context_acs = self._stft_chunk(
            acs_chunk_samples, self._stft_context_acs, self._input_buffer_acs
        )
        self._stft_context_bcs = next_stft_context_bcs
        self._stft_context_acs = next_stft_context_acs

        # Push the per-stream input mag (valid frames only — the first chunk_size
        # of the per-call STFT) into the FIFO so we can align it with the matured
        # backbone outputs L_dec frames later.
        bcs_mag_call, _ = complex_to_mag_pha(bcs_spec, stack_dim=-1)
        acs_mag_call, _ = complex_to_mag_pha(acs_spec, stack_dim=-1)
        valid = self.chunk_size
        self._input_mag_fifo.append((bcs_mag_call[:, :, :valid].clone(), acs_mag_call[:, :, :valid].clone()))

        # Drive both branches' streaming step.
        bcs_result = self.bcs_streaming.process_spectrogram_buffered(bcs_spec)
        acs_result = self.acs_streaming.process_spectrogram_buffered(acs_spec)

        # Advance both input buffers by output_samples_per_chunk (regardless of
        # whether the lookahead pipeline matured — matches BackboneStreaming).
        self._input_buffer_bcs = self._input_buffer_bcs[self.output_samples_per_chunk :]
        self._input_buffer_acs = self._input_buffer_acs[self.output_samples_per_chunk :]

        # Sync guard: both branches share topology (L_enc = L_dec on both), so they
        # must mature together. A desync would silently drift the alignment.
        if (bcs_result is None) != (acs_result is None):
            raise RuntimeError(
                "branch desync inside process_spectrogram_buffered: "
                f"bcs_result={'None' if bcs_result is None else 'matured'}, "
                f"acs_result={'None' if acs_result is None else 'matured'} — "
                "both branches must go None→matured simultaneously."
            )

        if bcs_result is None or acs_result is None:
            # Warmup — both still buffering for decoder lookahead. The sync guard
            # above ensures both are None together (the `or` is for mypy's benefit).
            return None

        # Pop the aligned input-mag chunk for fusion (the OLDEST entry; the FIFO
        # length is bounded by ~1 + L_dec/chunk_size, popped per matured chunk).
        bcs_mag_input, acs_mag_input = self._input_mag_fifo.popleft()

        # Fusion algebra (non-streaming nn.Module forward).
        bcs_est_mag, bcs_est_pha, bcs_est_com = bcs_result
        acs_est_mag, acs_est_pha, acs_est_com = acs_result
        est_mag, est_pha = self._run_fusion(
            bcs_est_mag=bcs_est_mag,
            bcs_est_pha=bcs_est_pha,
            bcs_est_com=bcs_est_com,
            acs_est_mag=acs_est_mag,
            acs_est_pha=acs_est_pha,
            acs_est_com=acs_est_com,
            bcs_mag_input=bcs_mag_input,
            acs_mag_input=acs_mag_input,
        )

        # Cross-chunk iSTFT-OLA → 800 mature samples.
        return self._manual_istft_ola(est_mag, est_pha)

    @torch.inference_mode()
    def process_audio(self, bcs_audio: Tensor, acs_audio: Tensor) -> Tensor:
        """Stream a complete paired utterance and return the fused enhanced audio.

        Resets state, appends a zero flush long enough to drain the lookahead
        pipeline + the OLA tail, feeds both streams through :meth:`process_samples`
        in ``output_samples_per_chunk``-sized pieces, concatenates the matured
        output samples, and trims to ``len(bcs_audio)`` (BAFNet+ is
        length-preserving in the spectrogram domain and the OLA is mid-aligned
        with the input via the ``win//2`` STFT past-context emulation).

        If the two input audios have different lengths, the longer one is
        truncated to the shorter one's length before feeding — BAFNet+ assumes
        paired-modality input where BCS and ACS are sample-aligned. (A future
        revision could accept different-length pairs by zero-padding the
        shorter, but that's outside the S6 spec.)

        Args:
            bcs_audio: ``[T]`` or ``[1, T]`` BCS audio.
            acs_audio: ``[T]`` or ``[1, T]`` ACS audio (sample-aligned with BCS).

        Returns:
            ``Tensor [T']`` of enhanced audio with ``T' = min(produced,
            len(bcs_audio))``.
        """
        if bcs_audio.dim() == 2:
            bcs_audio = bcs_audio.squeeze(0)
        if acs_audio.dim() == 2:
            acs_audio = acs_audio.squeeze(0)
        target_device = self.device
        bcs_audio = bcs_audio.to(target_device)
        acs_audio = acs_audio.to(target_device)
        ref_length = min(len(bcs_audio), len(acs_audio))
        bcs_audio = bcs_audio[:ref_length]
        acs_audio = acs_audio[:ref_length]

        self.reset_state()

        flush_size = self.samples_per_chunk * (self.total_lookahead + 2)
        bcs_padded = torch.cat([bcs_audio, torch.zeros(flush_size, device=target_device)])
        acs_padded = torch.cat([acs_audio, torch.zeros(flush_size, device=target_device)])

        outputs: List[Tensor] = []
        for i in range(0, len(bcs_padded), self.output_samples_per_chunk):
            bcs_chunk = bcs_padded[i : i + self.output_samples_per_chunk]
            acs_chunk = acs_padded[i : i + self.output_samples_per_chunk]
            if len(bcs_chunk) == 0 or len(acs_chunk) == 0:
                break
            result = self.process_samples(bcs_chunk, acs_chunk)
            if result is not None and len(result) > 0:
                outputs.append(result)

        if not outputs:
            return torch.zeros(0, device=target_device)
        cat = torch.cat(outputs)
        if len(cat) > ref_length:
            cat = cat[:ref_length]
        return cat

    @torch.inference_mode()
    def process_spectrogram(self, noisy_bcs_com: Tensor, noisy_acs_com: Tensor) -> SpectrogramChunk:
        """Stream a complete paired spectrogram in chunks (spectrogram in → out, no iSTFT).

        The host-STFT/iSTFT-free path used by the S6 spectrogram-level parity
        gate. Resets, loops the time axis in ``chunk_size`` steps, feeding each
        ``noisy_*_com[:, :, t : t + chunk_size + L_enc, :]`` slice (overlapping
        the next chunk by ``L_enc``) through both branches'
        :meth:`BackboneStreaming.process_spectrogram_buffered`, runs the fusion
        algebra on each matured pair, and concatenates the matured fused
        spectrogram triples along the time axis. No internal flush: the last
        ``~chunk_size + total_lookahead`` reference frames are not drained.

        Args:
            noisy_bcs_com: ``[B, F, T, 2]`` (``B == 1``) or ``[F, T, 2]`` BCS
                compressed complex spectrogram (e.g.
                ``mag_pha_stft(bcs, ..., center=True)[2]``).
            noisy_acs_com: same shape for ACS.

        Returns:
            ``(est_mag, est_pha, est_com)`` for the streamed (matured) frames:
            ``est_mag`` / ``est_pha`` are ``[1, F, T_stream]`` and ``est_com``
            is ``[1, F, T_stream, 2]`` with ``T_stream < T``.

        Raises:
            ValueError: If either spectrogram has a batch dimension != 1, or
                the two have different shapes.
            RuntimeError: If the two branches' ``process_spectrogram_buffered``
                outputs desync (matured / None mismatch).
        """
        if noisy_bcs_com.dim() == 3:
            noisy_bcs_com = noisy_bcs_com.unsqueeze(0)
        if noisy_acs_com.dim() == 3:
            noisy_acs_com = noisy_acs_com.unsqueeze(0)
        if noisy_bcs_com.shape[0] != 1 or noisy_acs_com.shape[0] != 1:
            raise ValueError(
                f"streaming requires batch size 1, got bcs={noisy_bcs_com.shape[0]}, acs={noisy_acs_com.shape[0]}"
            )
        if noisy_bcs_com.shape[2] != noisy_acs_com.shape[2]:
            raise ValueError(
                f"BCS / ACS spectrograms must share T (frame count): "
                f"bcs.T={noisy_bcs_com.shape[2]}, acs.T={noisy_acs_com.shape[2]}"
            )

        self.reset_state()
        t_total = noisy_bcs_com.shape[2]
        mags: List[Tensor] = []
        phas: List[Tensor] = []
        coms: List[Tensor] = []
        for t in range(0, t_total, self.chunk_size):
            t_end = min(t + self.chunk_size + self.encoder_lookahead, t_total)
            bcs_chunk = noisy_bcs_com[:, :, t:t_end, :]
            acs_chunk = noisy_acs_com[:, :, t:t_end, :]
            # Snapshot the per-call input mag (valid frames only, NOT the L_enc
            # lookahead) into the FIFO — same trimming convention as process_samples.
            valid = min(self.chunk_size, t_end - t)
            bcs_mag_call, _ = complex_to_mag_pha(bcs_chunk, stack_dim=-1)
            acs_mag_call, _ = complex_to_mag_pha(acs_chunk, stack_dim=-1)
            self._input_mag_fifo.append((bcs_mag_call[:, :, :valid].clone(), acs_mag_call[:, :, :valid].clone()))

            bcs_result = self.bcs_streaming.process_spectrogram_buffered(bcs_chunk)
            acs_result = self.acs_streaming.process_spectrogram_buffered(acs_chunk)
            if (bcs_result is None) != (acs_result is None):
                raise RuntimeError(
                    "branch desync inside process_spectrogram: "
                    f"bcs_result={'None' if bcs_result is None else 'matured'}, "
                    f"acs_result={'None' if acs_result is None else 'matured'}"
                )
            if bcs_result is None or acs_result is None:
                continue
            bcs_mag_input, acs_mag_input = self._input_mag_fifo.popleft()
            bcs_est_mag, bcs_est_pha, bcs_est_com = bcs_result
            acs_est_mag, acs_est_pha, acs_est_com = acs_result
            est_mag, est_pha = self._run_fusion(
                bcs_est_mag=bcs_est_mag,
                bcs_est_pha=bcs_est_pha,
                bcs_est_com=bcs_est_com,
                acs_est_mag=acs_est_mag,
                acs_est_pha=acs_est_pha,
                acs_est_com=acs_est_com,
                bcs_mag_input=bcs_mag_input,
                acs_mag_input=acs_mag_input,
            )
            est_com = mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)
            mags.append(est_mag)
            phas.append(est_pha)
            coms.append(est_com)

        f_bins = self.n_fft // 2 + 1
        if not mags:
            empty = torch.zeros(1, f_bins, 0, device=noisy_bcs_com.device)
            return empty, empty.clone(), torch.zeros(1, f_bins, 0, 2, device=noisy_bcs_com.device)
        return torch.cat(mags, dim=2), torch.cat(phas, dim=2), torch.cat(coms, dim=2)

    def forward(self, bcs_audio: Tensor, acs_audio: Tensor) -> Tensor:
        """``nn.Module`` entry point — equivalent to :meth:`process_audio`."""
        return self.process_audio(bcs_audio, acs_audio)

    def __repr__(self) -> str:
        cfg = self._streaming_config
        return (
            f"{self.__class__.__name__}(\n"
            f"  source={cfg.get('source', 'direct')}, ablation_mode={self.ablation_mode},\n"
            f"  use_calibration={self.use_calibration}, use_relative_gain={self.use_relative_gain}, "
            f"mask_only_alpha={self.mask_only_alpha},\n"
            f"  chunk_size={self.chunk_size}, L_enc={self.encoder_lookahead}, L_dec={self.decoder_lookahead}, "
            f"L_alpha={self._audit_alpha_lookahead},\n"
            f"  total_frames_needed={self.total_frames_needed}, samples_per_chunk={self.samples_per_chunk}, "
            f"output_samples_per_chunk={self.output_samples_per_chunk},\n"
            f"  latency_ms={self.latency_ms:.2f}, ola_tail_size={self.ola_tail_size},\n"
            f")"
        )


__all__ = ["BAFNetPlusStreaming"]
