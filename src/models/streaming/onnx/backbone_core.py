"""Exportable single-``Backbone`` ONNX core (explicit-state, FP32).

S4 of the streaming-ONNX rebuild (Stage 1 pt.3 — Stage-1 exit gate).
:class:`ExportableBackboneCore` is a graph-internal :class:`torch.nn.Module` that
runs the BAFNet+ ``Backbone`` (DenseEncoder + TS_BLOCK + MaskDecoder +
PhaseDecoder) with **every conv state promoted to an explicit forward
arg/return**, built on the S2 ``FunctionalStatefulConv*`` layers via
:func:`convert_to_functional`. Conv state buffers live as graph I/O; the host
STFT / iSTFT-OLA / feature-buffer logic lives outside (S3
:class:`~src.models.streaming.backbone_streaming.BackboneStreaming` for the
PT-side reference, S9 for the ORT host wrapper).

What this is / is not
---------------------
- It is the **graph-internal model**: a pure function
  ``(mag, pha, *prev_states) -> (est_mag, est_pha, est_com, *next_states)``
  (atan2 mode) — bit-equivalent to ``Backbone.forward`` on a complete
  spectrogram (with zero init states + ``state_frames_for_update=None``), and
  chunk-by-chunk equivalent on ``T_export``-frame chunks (with
  ``state_frames_for_update=chunk_size``).
- It does **not** do the host STFT / iSTFT / feature-buffer / framing — that is
  S3 (``BackboneStreaming``) on the PT side, S9 on the ORT side.
- It does **not** do BAFNet+ fusion (mapping+masking calibration / alpha /
  unified-checkpoint load). That is S5+/S6/S8.

Graph boundary (cf. ``docs/wiki/concepts/lacosenet-backbone-streaming.md`` §6):
  ``Host FP32:`` audio → STFT → ``com`` → ``complex_to_mag_pha`` → ``mag, pha``
  ``ONNX FP32:`` ``(mag, pha, *prev_states)`` → core →
                 ``(est_mag, est_pha, est_com, *next_states)``
  ``Host FP32:`` ``est_mag``, ``est_pha`` → ``mag_pha_to_complex`` → iSTFT → audio

The host's ``mag, pha`` should be the same ``complex_to_mag_pha(com)`` output
that ``Backbone.forward`` would compute internally — i.e. round-trip
``mag_pha_stft(audio)[2]`` through ``complex_to_mag_pha`` (the
``sqrt(...+1e-8)`` / ``atan2(...+1e-8)`` floors must be applied), so feeding the
raw ``mag_pha_stft(audio)[:2]`` directly is **not** bit-equivalent. See S3
``BackboneStreaming`` (which does this round-trip in ``_process_encoder``) and
the test (a) parity check in ``tests/test_backbone_core.py``.

State-update window
-------------------
:meth:`ExportableBackboneCore.set_state_frames_for_update` plays the role of
:class:`~src.models.streaming.context.StateFramesContext` *inside the graph*: it
bounds each conv's state update to the leading ``state_frames`` frames so the
trailing lookahead frames feed the conv outputs but do not corrupt the next
chunk's recurrent state. The export driver sets this to ``chunk_size``;
``Backbone.forward``-equivalence on a complete sequence uses ``None`` (all
frames update state, which equals "no streaming" with zero init states).

Phase output mode
-----------------
``phase_output_mode="atan2"`` (default, FP32) computes ``atan2`` in the graph
and returns ``(est_mag, est_pha, est_com)`` — the same triple ``Backbone.forward``
returns. ``phase_output_mode="complex"`` instead emits ``(est_mag, x_r, x_i)``
where ``x_r``/``x_i`` are the raw ``phase_conv_r``/``phase_conv_i`` outputs;
the host then does ``est_pha = atan2(x_i+1e-8, x_r+1e-8)`` and recovers
``est_com``. The complex mode is an INT8-precision hedge (``atan2`` is hard to
quantise); FP32 has no precision difference, so default to atan2.

Ported from LaCoSENet ``src/models/onnx_export/exportable_core.py`` +
``stateful_core.py`` (the reshape-based path; the reshape-free variant
``stateful_core_rf.py`` is a separate NPU concern not ported here per the FP32
launch prompt). Adjusted for BAFNet+ module paths (``src.models.backbone`` /
``src.models.streaming.*``) and BAFNet+'s ``ChannelAttentionBlock`` /
``GroupPrimeKernelFFN`` ``fold_residual_scale`` / ``post_norm`` branches (which
LaCoSENet's variant pre-dated).

Frozen I/O contract (the contract S8/S9/S11 re-bind to)
-------------------------------------------------------
Inputs (in order):  ``mag [1, F, T]``, ``pha [1, F, T]``, then the state
tensors in :meth:`get_state_names` order (one per ``FunctionalStateful*``
module, DFS order from :meth:`torch.nn.Module.named_modules`, named
``state_<i>_<dotted_path_with_underscores>``).
Outputs (in order): atan2 mode ``est_mag [1, F, T]``, ``est_pha [1, F, T]``,
``est_com [1, F, T, 2]``; complex mode ``est_mag``, ``phase_real [1, F, T]``,
``phase_imag [1, F, T]``. Then the next-state tensors named
``next_<state_name>`` in the same order. The export driver freezes these in the
sidecar JSON (see :mod:`src.models.streaming.onnx.export`).

Anchors (50 ms BAFNet+ Backbone): ``chunk_size = 8``, ``L_enc = L_dec = 3``,
``T_export = chunk_size + L_enc + L_dec = 14``, ``num_states = 92`` (= 12
DS_DDB + 80 TS time-stage). The S5 ``T_export`` proof harness may shrink to
``chunk + max(L_enc, L_dec) = 11`` only if full-sequence parity holds — that's
**not** this session.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.models.backbone import Backbone
from src.models.streaming.converters import convert_to_stateful
from src.models.streaming.layers.stateful_conv import (
    StatefulAsymmetricConv2d,
    StatefulCausalConv1d,
    StatefulCausalConv2d,
)
from src.models.streaming.lookahead import compute_lookahead
from src.models.streaming.onnx.functional_stateful import (
    FunctionalStatefulCausalConv2d,
    FunctionalStatefulConv1d,
    FunctionalStatefulConv2d,
    convert_to_functional,
)
from src.models.streaming.onnx.reshape_free import FunctionalStatefulConv2dTimeAxis
from src.models.streaming.onnx.reshape_free_tsblock import (
    ReshapeFreeTSBlock,
    convert_sequence_block_to_reshape_free,
)
from src.stft import mag_pha_to_complex

logger = logging.getLogger(__name__)

_FUNCTIONAL_TYPES: Tuple[type, ...] = (
    FunctionalStatefulConv1d,
    FunctionalStatefulConv2d,
    FunctionalStatefulCausalConv2d,
    FunctionalStatefulConv2dTimeAxis,
)
_STATEFUL_TYPES: Tuple[type, ...] = (
    StatefulCausalConv1d,
    StatefulAsymmetricConv2d,
    StatefulCausalConv2d,
)
# Time-axis-aware functional conv. Used by init_states to pick the right
# ``effective_batch`` / ``freq_size`` recipe and by ``_forward_cab`` /
# ``_forward_gpkffn`` to dispatch through StateIterator. The 1D
# ``FunctionalStatefulConv1d`` time-axis variant was removed when reshape-free
# TSBlock became the default — only the 4D ``[B, C, T, F]`` time-axis conv
# appears inside the trunk's TSBlocks now.
_TIME_AXIS_FUNCTIONAL_TYPE: type = FunctionalStatefulConv2dTimeAxis


class StateIterator:
    """Sequential consumer/producer of conv-state tensors during a forward pass.

    Wraps ``prev_states`` (the forward's incoming list, in :meth:`get_state_names`
    order) and accumulates ``next_states``. Each functional stateful conv call
    goes through :meth:`run_layer`, which feeds the next previous-state and
    appends the returned next-state — keeping the read/write cursors paired.

    Attributes:
        next_states: The accumulated next-state list (read after the forward).
        consumed_count: Number of states consumed so far (sanity check).
    """

    def __init__(self, prev_states: List[Tensor]) -> None:
        """Initialise from the forward's incoming state list (registration order)."""
        self._prev_states: List[Tensor] = list(prev_states)
        self._next_states: List[Tensor] = []
        self._idx: int = 0

    def run_layer(
        self,
        layer: nn.Module,
        x: Tensor,
        state_frames: Optional[int] = None,
    ) -> Tensor:
        """Run a functional stateful conv: feed it the next prev-state, store its next-state.

        Args:
            layer: A ``FunctionalStatefulConv*`` (the only kind whose forward is
                ``(x, state, state_frames=None) -> (out, next_state)``).
            x: Conv input tensor.
            state_frames: ``state_frames`` to pass to the layer (typically
                ``ExportableBackboneCore.state_frames_for_update``).

        Returns:
            The conv's output tensor.

        Raises:
            IndexError: If more states are consumed than were provided.
        """
        if self._idx >= len(self._prev_states):
            raise IndexError(
                f"StateIterator: tried to consume state {self._idx} but only " f"{len(self._prev_states)} were provided"
            )
        prev = self._prev_states[self._idx]
        out, next_state = layer(x, prev, state_frames=state_frames)
        self._next_states.append(next_state)
        self._idx += 1
        return out

    @property
    def next_states(self) -> List[Tensor]:
        """The accumulated next-state list."""
        return self._next_states

    @property
    def consumed_count(self) -> int:
        """Number of states consumed so far."""
        return self._idx


def convert_stateful_to_functional(model: nn.Module) -> int:
    """Replace every ``Stateful*`` conv leaf in ``model`` with its functional counterpart.

    Walks ``model.named_modules()``, and for each ``StatefulCausalConv1d`` /
    ``StatefulAsymmetricConv2d`` / ``StatefulCausalConv2d`` calls
    :func:`~src.models.streaming.onnx.functional_stateful.convert_to_functional`
    and substitutes the result in the parent module (handling both attribute and
    indexed children).

    Args:
        model: A model whose conv leaves were already converted by
            :func:`~src.models.streaming.converters.convert_to_stateful`.

    Returns:
        Number of layers replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, _STATEFUL_TYPES):
            continue
        parts = name.split(".")
        parent: Any = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        functional = convert_to_functional(cast(nn.Module, module))
        attr = parts[-1]
        if attr.isdigit():
            parent[int(attr)] = functional
        else:
            setattr(parent, attr, functional)
        count += 1
    return count


class ExportableBackboneCore(nn.Module):
    """Explicit-state ONNX-exportable wrapper around a single BAFNet+ ``Backbone``.

    The conv leaves are :class:`FunctionalStatefulConv1d` /
    :class:`FunctionalStatefulConv2d` instances (no internal buffers, state is a
    forward arg/return). The forward routes states through every stateful conv
    in DFS order via :class:`StateIterator`, and reproduces ``Backbone.forward``
    in-graph (encoder ``mag/pha`` stack/permute, masking-vs-mapping
    ``est_mag = mag * mask`` / ``est_mag = mask`` branch, ``mag_pha_to_complex``
    for ``est_com``).

    Attributes:
        dense_encoder / sequence_block / mask_decoder / phase_decoder: Submodules
            extracted from the source ``Backbone`` (deep-copied during conversion).
        infer_type: ``'masking'`` or ``'mapping'``.
        n_fft: STFT size (used to derive the default ``freq_size = n_fft//2 + 1``).
        encoder_lookahead / decoder_lookahead: ``L_enc`` / ``L_dec`` from
            :func:`compute_lookahead` on the **original** (pre-conversion) backbone.
        phase_output_mode: ``'atan2'`` (default, FP32) or ``'complex'``.
        state_frames_for_update: Bound on which leading frames update conv state
            (``None`` = all frames; set to ``chunk_size`` for streaming export).
    """

    def __init__(
        self,
        dense_encoder: nn.Module,
        sequence_block: nn.Module,
        mask_decoder: nn.Module,
        phase_decoder: nn.Module,
        *,
        infer_type: str,
        n_fft: int,
        encoder_lookahead: int,
        decoder_lookahead: int,
        phase_output_mode: str = "atan2",
    ) -> None:
        """Build a core from already-functional submodules. Use :meth:`from_backbone` instead.

        Args:
            dense_encoder: The functional-stateful ``DenseEncoder``.
            sequence_block: The functional-stateful ``nn.Sequential`` of
                ``ReshapeFreeTSBlock`` blocks (the reshape-free TSBlock has been
                the default since the cycle-13 cleanup that promoted Path β).
            mask_decoder: The functional-stateful ``MaskDecoder``.
            phase_decoder: The functional-stateful ``PhaseDecoder``.
            infer_type: ``'masking'`` or ``'mapping'``.
            n_fft: FFT size.
            encoder_lookahead: ``L_enc`` (encoder right-padding frame sum).
            decoder_lookahead: ``L_dec`` (decoder right-padding frame sum).
            phase_output_mode: ``'atan2'`` or ``'complex'``.

        Raises:
            ValueError: If ``infer_type`` or ``phase_output_mode`` is not a
                recognised value.
        """
        super().__init__()
        if infer_type not in ("masking", "mapping"):
            raise ValueError(f"infer_type must be 'masking' or 'mapping', got {infer_type!r}")
        if phase_output_mode not in ("atan2", "complex"):
            raise ValueError(f"phase_output_mode must be 'atan2' or 'complex', got {phase_output_mode!r}")
        self.dense_encoder = dense_encoder
        self.sequence_block = sequence_block
        self.mask_decoder = mask_decoder
        self.phase_decoder = phase_decoder
        self.infer_type: str = infer_type
        self.n_fft: int = int(n_fft)
        self.encoder_lookahead: int = int(encoder_lookahead)
        self.decoder_lookahead: int = int(decoder_lookahead)
        self.phase_output_mode: str = phase_output_mode
        self.state_frames_for_update: Optional[int] = None
        self._functional_modules: List[Tuple[str, nn.Module]] = []
        self._collect_functional_modules()

    # ---------------------------------------------------------------- collection
    def _collect_functional_modules(self) -> None:
        """Walk ``self.named_modules()`` and record every functional stateful conv (DFS order)."""
        collected: List[Tuple[str, nn.Module]] = []
        for name, module in self.named_modules():
            if isinstance(module, _FUNCTIONAL_TYPES):
                collected.append((name, cast(nn.Module, module)))
        self._functional_modules = collected
        if logger.isEnabledFor(logging.INFO):
            logger.info("ExportableBackboneCore: collected %d functional stateful modules", self.num_states)

    @property
    def num_states(self) -> int:
        """Total number of state tensors (= number of functional stateful convs)."""
        return len(self._functional_modules)

    @property
    def total_lookahead(self) -> int:
        """``encoder_lookahead + decoder_lookahead`` (= the monolithic ``T_export`` extra)."""
        return self.encoder_lookahead + self.decoder_lookahead

    # --------------------------------------------------------------- state-frames
    def set_state_frames_for_update(self, state_frames: Optional[int]) -> None:
        """Bound how many leading input frames may update the conv states.

        Args:
            state_frames: Max number of leading frames per forward whose values
                feed the next-state computation. ``None`` = no bound (the entire
                input updates state, the right behaviour for a single-shot
                ``Backbone.forward``-equivalent run on a complete sequence).
                Set to ``chunk_size`` for streaming export so the trailing
                ``L_enc + L_dec`` lookahead frames do not corrupt state.
        """
        self.state_frames_for_update = None if state_frames is None else int(state_frames)

    # ------------------------------------------------------------------- naming
    def get_state_names(self) -> List[str]:
        """Return one unique name per state, in registration (DFS) order.

        Format: ``state_<i>_<dotted_module_path_with_underscores>``. Frozen as the
        ONNX state input name; the matching output is ``next_<that_name>``.
        """
        return [f"state_{i}_{name.replace('.', '_')}" for i, (name, _) in enumerate(self._functional_modules)]

    def input_names(self) -> List[str]:
        """Frozen ONNX input names: ``["mag", "pha", *state_names]``."""
        return ["mag", "pha"] + self.get_state_names()

    def output_names(self) -> List[str]:
        """Frozen ONNX output names (depend on :attr:`phase_output_mode`)."""
        nexts = [f"next_{n}" for n in self.get_state_names()]
        if self.phase_output_mode == "complex":
            return ["est_mag", "phase_real", "phase_imag"] + nexts
        return ["est_mag", "est_pha", "est_com"] + nexts

    # ---------------------------------------------------------------- init shapes
    def init_states(
        self,
        batch_size: int = 1,
        freq_size: Optional[int] = None,
        time_frames: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tensor]:
        """Initialise every state tensor (zeros) with architecture-aware shapes.

        For each functional conv:

        - Reshape-free time-axis 4D conv (TS-block time-stage): state shape
          ``[B, C, padding, F_enc]``.
        - 2D conv (DS_DDB ``AsymmetricConv2d``): batch dim is ``batch_size``;
          the freq dim is ``freq_size`` for encoder DS_DDB convs and ``freq_enc``
          for mask/phase decoder DS_DDB convs (the encoder's ``dense_conv_2``
          stride-2 halves the freq axis before the decoders see it).

        ``freq_enc = (freq_size - 1) // 2`` mirrors the
        ``Conv2d(., (1, 3), (1, 2))`` output-freq formula (e.g.
        ``201 -> 100`` for ``n_fft=400``).

        Args:
            batch_size: Logical batch size (typically 1 for streaming).
            freq_size: ``n_fft // 2 + 1``; defaults to that if ``None``.
            time_frames: Accepted for API parity with the LaCoSENet pattern.
                BAFNet+'s freq_stage has no stateful convs so the value is
                unused here; kept so downstream callers don't need to drop it.
            device: State device (defaults to the model's parameter device).
            dtype: State dtype (defaults to the model's parameter dtype).

        Returns:
            List of zero-init state tensors in :meth:`get_state_names` order.
        """
        del time_frames  # unused — see docstring
        if freq_size is None:
            freq_size = self.n_fft // 2 + 1
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        freq_enc = (freq_size - 1) // 2

        states: List[Tensor] = []
        for name, module in self._functional_modules:
            conv2d_freq = freq_size if "dense_encoder" in name else freq_enc

            if isinstance(module, FunctionalStatefulConv2dTimeAxis):
                # Reshape-free time-axis 4D conv inside ``sequence_block.*.time_stage.*``;
                # uses freq_enc as the trailing F axis (TSBlock's input/output freq dim).
                states.append(module.init_state(batch_size, freq_enc, device, dtype))
            elif isinstance(module, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
                states.append(module.init_state(batch_size, conv2d_freq, device, dtype))
            else:
                raise TypeError(f"unknown functional stateful module type: {type(module).__name__}")
        return states

    def get_state_shapes(
        self,
        batch_size: int = 1,
        freq_size: Optional[int] = None,
    ) -> List[Tuple[int, ...]]:
        """Return state shapes in :meth:`get_state_names` order (calls :meth:`init_states`).

        Args:
            batch_size: Logical batch size.
            freq_size: ``n_fft // 2 + 1`` if ``None``.

        Returns:
            Per-state shape tuples.
        """
        return [tuple(s.shape) for s in self.init_states(batch_size=batch_size, freq_size=freq_size)]

    # ----------------------------------------------------------------- metadata
    def metadata(self, *, freq_size: Optional[int] = None) -> Dict[str, Any]:
        """Snapshot the core's I/O contract for downstream consumers.

        Returned keys: ``infer_type``, ``phase_output_mode``, ``n_fft``,
        ``freq_size``, ``encoder_lookahead``, ``decoder_lookahead``,
        ``total_lookahead``, ``state_frames_for_update``, ``num_states``,
        ``input_names``, ``output_names``, ``state_names``, ``state_shapes``.
        Used by the export driver (S4) and S8/S9/S11 binders.
        """
        if freq_size is None:
            freq_size = self.n_fft // 2 + 1
        shapes = self.get_state_shapes(batch_size=1, freq_size=freq_size)
        return {
            "infer_type": self.infer_type,
            "phase_output_mode": self.phase_output_mode,
            "n_fft": self.n_fft,
            "freq_size": int(freq_size),
            "encoder_lookahead": self.encoder_lookahead,
            "decoder_lookahead": self.decoder_lookahead,
            "total_lookahead": self.total_lookahead,
            "state_frames_for_update": self.state_frames_for_update,
            "num_states": self.num_states,
            "input_names": self.input_names(),
            "output_names": self.output_names(),
            "state_names": self.get_state_names(),
            "state_shapes": [list(s) for s in shapes],
        }

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_backbone(
        cls,
        backbone: Backbone,
        *,
        phase_output_mode: str = "atan2",
    ) -> "ExportableBackboneCore":
        """Build an exportable core from an instantiated **original** ``Backbone``.

        Steps: capture ``compute_lookahead`` on the ORIGINAL backbone (the
        stateful/functional convs aren't ``AsymmetricConv2d`` instances — the
        lookahead must be read first); deep-copy the backbone; swap every
        ``TSBlock`` to its reshape-free 4D variant via
        :func:`~src.models.streaming.onnx.reshape_free_tsblock.convert_sequence_block_to_reshape_free`
        (the cycle-13 cleanup made this the only path);
        :func:`~src.models.streaming.converters.convert_to_stateful` it (in place
        on the copy); :func:`convert_stateful_to_functional` it (in place);
        construct the core from its submodules. The source ``backbone`` is
        unchanged and stays usable as the non-streaming reference.

        Args:
            backbone: An instantiated ``Backbone`` whose conv leaves are still
                the original ``CausalConv1d`` / ``AsymmetricConv2d``.
            phase_output_mode: ``'atan2'`` (default, FP32) or ``'complex'``.

        Returns:
            A ready ``ExportableBackboneCore`` (``state_frames_for_update``
            starts as ``None`` — set it via
            :meth:`set_state_frames_for_update` before streaming export).

        Raises:
            TypeError: If ``backbone`` is not a ``Backbone``.
            ValueError: If ``backbone.n_fft`` is unset, or if ``backbone`` has
                already been converted to stateful (``compute_lookahead`` would
                give ``L_enc=L_dec=0`` — pass the pre-conversion model).
        """
        if not isinstance(backbone, Backbone):
            raise TypeError(f"from_backbone expects a Backbone, got {type(backbone).__name__}")
        if backbone.n_fft is None:
            raise ValueError("Backbone is missing n_fft")
        if any(isinstance(m, _STATEFUL_TYPES + _FUNCTIONAL_TYPES) for m in backbone.modules()):
            raise ValueError(
                "from_backbone expects the ORIGINAL (non-stateful) Backbone; "
                "compute_lookahead would otherwise see no AsymmetricConv2d. "
                "Pass the pre-conversion module."
            )

        lookahead = compute_lookahead(backbone)  # MUST run on the original
        n_fft = int(backbone.n_fft)
        infer_type = getattr(backbone, "infer_type", "masking")

        model = copy.deepcopy(backbone).eval()

        # Reshape-free TSBlock swap MUST run BEFORE convert_to_stateful so the
        # stateful pass only sees the encoder/decoder ``CausalConv1d`` /
        # ``AsymmetricConv2d`` / ``CausalConv2d`` leaves (the RF TSBlocks carry
        # their own FunctionalStatefulConv2dTimeAxis convs from the converter,
        # which the stateful pass would not recognise and shouldn't touch).
        # The converter reads the ORIGINAL (non-stateful) TSBlock weights.
        model.sequence_block = convert_sequence_block_to_reshape_free(model.sequence_block)

        convert_to_stateful(model, verbose=False, inplace=True)
        n_func = convert_stateful_to_functional(model)
        if n_func == 0:
            raise RuntimeError("convert_stateful_to_functional found no Stateful* convs")

        return cls(
            dense_encoder=model.dense_encoder,
            sequence_block=model.sequence_block,
            mask_decoder=model.mask_decoder,
            phase_decoder=model.phase_decoder,
            infer_type=infer_type,
            n_fft=n_fft,
            encoder_lookahead=lookahead.encoder_lookahead,
            decoder_lookahead=lookahead.decoder_lookahead,
            phase_output_mode=phase_output_mode,
        )

    # ------------------------------------------------------------------ forward
    def _forward_impl(
        self,
        mag: Tensor,
        pha: Tensor,
        prev_states: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[Tensor]]:
        """Shared body for :meth:`forward` and :meth:`forward_with_mask`.

        Returns ``(est_mag, b, c, mask, next_states)`` where ``(b, c)`` is the
        ``(est_pha, est_com)`` pair in atan2 mode or the ``(phase_real,
        phase_imag)`` pair in complex mode. ``mask`` is the raw
        ``mask_decoder`` output (``[B, F, T]``) — the same tensor BAFNet+'s
        eval-reference :meth:`BAFNetPlus.forward` consumes via
        ``self.masking(acs_com, return_mask=True)`` for the calibration / alpha
        features. Exposing it lets the outer BAFNet+ core skip the
        ``acs_mask = acs_est_mag / acs_mag`` recovery (S5 design compromise)
        that introduces an ``(a*b)/a`` 1-ULP drift on the recovered mask.
        """
        state_iter = StateIterator(prev_states)

        # Encoder input layout matches Backbone.forward: stack(mag, pha) -> [B, 2, T, F].
        b_dim, _f_orig, _t = mag.shape
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)
        x = self._forward_dense_encoder(x, state_iter)  # [B, C, T, F_enc]
        _, _c, _t_enc, f_enc = x.shape

        x = self._forward_sequence_block(x, state_iter, b_dim, f_enc)  # [B, C, T, F_enc]

        mask = self._forward_mask_decoder(x, state_iter).squeeze(1).transpose(1, 2)  # [B, F, T]
        phase_out = self._forward_phase_decoder(x, state_iter)

        if self.infer_type == "masking":
            est_mag = mag * mask
        else:  # 'mapping'
            est_mag = mask

        if self.phase_output_mode == "complex":
            x_r, x_i = phase_out  # type: ignore[misc]
            x_r = x_r.squeeze(1).transpose(1, 2)  # [B, F, T]
            x_i = x_i.squeeze(1).transpose(1, 2)
            return est_mag, x_r, x_i, mask, list(state_iter.next_states)

        est_pha = phase_out.squeeze(1).transpose(1, 2)  # type: ignore[union-attr]   # [B, F, T]
        est_com = mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)  # [B, F, T, 2]
        return est_mag, est_pha, est_com, mask, list(state_iter.next_states)

    def forward(self, mag: Tensor, pha: Tensor, *prev_states: Tensor) -> Tuple[Tensor, ...]:
        """Run the explicit-state forward.

        Args:
            mag: Compressed magnitude ``[B, F, T]``.
            pha: Phase ``[B, F, T]``.
            *prev_states: ``num_states`` prev-state tensors in
                :meth:`get_state_names` order.

        Returns:
            Atan2 mode (default): ``(est_mag [B, F, T], est_pha [B, F, T],
            est_com [B, F, T, 2], *next_states)``. ``est_mag = mag * mask``
            (masking) or ``est_mag = mask`` (mapping) — matching
            ``Backbone.forward``.

            Complex mode: ``(est_mag, phase_real [B, F, T], phase_imag [B, F, T],
            *next_states)`` — host computes
            ``est_pha = atan2(phase_imag + 1e-8, phase_real + 1e-8)`` and
            ``est_com = mag_pha_to_complex(est_mag, est_pha)``.

        Raises:
            ValueError: If ``len(prev_states) != self.num_states``.
        """
        if len(prev_states) != self.num_states:
            raise ValueError(
                f"ExportableBackboneCore.forward: expected {self.num_states} prev_states, " f"got {len(prev_states)}"
            )
        a, b, c, _mask, next_states = self._forward_impl(mag, pha, list(prev_states))
        return (a, b, c) + tuple(next_states)

    def forward_with_mask(self, mag: Tensor, pha: Tensor, *prev_states: Tensor) -> Tuple[Tensor, ...]:
        """Forward variant that exposes the raw ``mask_decoder`` output.

        Same algebra as :meth:`forward` but appends ``mask [B, F, T]`` (the
        post-``LearnableSigmoid_2d`` mask, pre-multiplication) immediately
        before the next-state tensors. ``mask`` matches the
        ``return_mask=True`` branch of ``Backbone.forward`` to 1 ULP, so the
        outer BAFNet+ core can use it directly instead of recovering it via
        ``acs_mask = est_mag / mag`` (which introduces an ``(a*b)/a`` 1-ULP
        drift).

        Returns:
            Atan2 mode: ``(est_mag, est_pha, est_com, mask, *next_states)``.
            Complex mode: ``(est_mag, phase_real, phase_imag, mask, *next_states)``.
        """
        if len(prev_states) != self.num_states:
            raise ValueError(
                f"ExportableBackboneCore.forward_with_mask: expected {self.num_states} prev_states, "
                f"got {len(prev_states)}"
            )
        a, b, c, mask, next_states = self._forward_impl(mag, pha, list(prev_states))
        return (a, b, c, mask) + tuple(next_states)

    # ------------------------------------------------------------ submodule paths
    # The helpers below take ``Any`` for the submodule arg: typing them as
    # ``nn.Module`` makes mypy infer every ``submod.attr`` as ``Tensor | Module``
    # (the union of buffers vs children that ``nn.Module.__getattr__`` returns),
    # which doesn't help — these helpers already know the concrete classes they
    # operate on (DenseEncoder, DenseDilatedBlock, TSBlock, ChannelAttentionBlock,
    # GroupPrimeKernelFFN, MaskDecoder, PhaseDecoder from ``src.models.backbone``).
    def _forward_dense_encoder(self, x: Tensor, state_iter: StateIterator) -> Tensor:
        """Mirror of :meth:`src.models.backbone.DenseEncoder.forward` with state routing."""
        encoder: Any = self.dense_encoder
        x = encoder.dense_conv_1(x)
        x = self._forward_ds_ddb(x, encoder.dense_block, state_iter)
        x = encoder.dense_conv_2(x)
        return x

    def _forward_ds_ddb(self, x: Tensor, ds_ddb: Any, state_iter: StateIterator) -> Tensor:
        """Mirror of :meth:`src.models.backbone.DenseDilatedBlock.forward` with state routing.

        ``DenseDilatedBlock.forward`` runs ``self.dense_block[i](skip)`` for each
        ``i``, where ``self.dense_block[i]`` is an
        ``nn.Sequential(AsymmetricConv2d, Conv2d, BatchNorm2d, PReLU)`` — only
        the first leaf is stateful (now ``FunctionalStatefulConv2d``).
        """
        skip = x
        for dense_conv in ds_ddb.dense_block:  # nn.ModuleList
            layer_input = skip
            for layer in dense_conv:  # nn.Sequential
                if isinstance(layer, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
                    layer_input = state_iter.run_layer(layer, layer_input, state_frames=self.state_frames_for_update)
                else:
                    layer_input = layer(layer_input)
            x = layer_input
            skip = torch.cat([x, skip], dim=1)
        return x

    def _forward_sequence_block(
        self,
        x: Tensor,
        state_iter: StateIterator,
        b: int,
        f_enc: int,
    ) -> Tensor:
        """Mirror of ``Backbone.sequence_block`` (an ``nn.Sequential`` of TSBlocks)."""
        for ts_block in cast(nn.Sequential, self.sequence_block):
            x = self._forward_ts_block(x, ts_block, state_iter, b, f_enc)
        return x

    def _forward_ts_block(
        self,
        x: Tensor,
        ts_block: ReshapeFreeTSBlock,
        state_iter: StateIterator,
        b: int,
        f_enc: int,
    ) -> Tensor:
        """Mirror of :meth:`src.models.backbone.TSBlock.forward` with state routing.

        Operates on a :class:`ReshapeFreeTSBlock` only (the cycle-13 cleanup
        promoted Path β to the only path). Keeps the 4D ``[B, C, T, F_enc]``
        layout end-to-end across time and freq stages — the freq_stage runs as
        stateless ``Conv2d(kernel=(1,K))`` so no permute/reshape is needed at
        any boundary. ``beta_t`` / ``beta_f`` are already shaped ``[1, C, 1, 1]``
        by the converter.
        """
        del b, f_enc  # carried for API parity with _forward_sequence_block
        residual = x
        x = self._forward_stage(x, ts_block.time_stage, state_iter) + residual * ts_block.beta_t
        residual = x
        x = self._forward_stage(x, ts_block.freq_stage, state_iter) + residual * ts_block.beta_f
        return x

    def _forward_stage(self, x: Tensor, stage: nn.Sequential, state_iter: StateIterator) -> Tensor:
        """Apply one ``time_stage`` / ``freq_stage`` (an ``nn.Sequential`` of ``(CAB, GPKFFN)``)."""
        for block in stage:  # nn.Sequential of nn.Sequential(CAB, GPKFFN)
            for sub_block in cast(nn.Sequential, block):
                if hasattr(sub_block, "sca"):
                    x = self._forward_cab(x, sub_block, state_iter)
                elif hasattr(sub_block, "proj_first"):
                    x = self._forward_gpkffn(x, sub_block, state_iter)
                else:
                    x = sub_block(x)
        return x

    def _forward_cab(self, x: Tensor, cab: Any, state_iter: StateIterator) -> Tensor:
        """Mirror of :meth:`src.models.backbone.ChannelAttentionBlock.forward` with state routing.

        Handles both legacy (``beta`` residual scale) and ``fold_residual_scale``
        / ``post_norm`` branches. ``cab.dwconv`` is ``FunctionalStatefulConv1d``
        when causal (the time stage); ``cab.sca`` is ``nn.Sequential`` for both
        causal (``StatefulCausalConv1d`` -> ``Functional``) and non-causal
        (``AdaptiveAvgPool1d``) variants.
        """
        skip = x
        x = cab.norm(x)
        x = cab.pwconv1(x)
        if isinstance(cab.dwconv, _TIME_AXIS_FUNCTIONAL_TYPE):
            x = state_iter.run_layer(cab.dwconv, x, state_frames=self.state_frames_for_update)
        else:
            x = cab.dwconv(x)
        x = cab.sg(x)

        if isinstance(cab.sca, nn.Sequential):
            sca_out = x
            for layer in cab.sca:
                if isinstance(layer, _TIME_AXIS_FUNCTIONAL_TYPE):
                    sca_out = state_iter.run_layer(layer, sca_out, state_frames=self.state_frames_for_update)
                else:
                    sca_out = layer(sca_out)
            x = x * sca_out
        else:
            x = x * cab.sca(x)

        x = cab.pwconv2(x)
        if getattr(cab, "fold_residual_scale", False):
            x = skip + x
        else:
            x = skip + x * cab.beta
        if getattr(cab, "post_norm", False):
            x = cab.post_norm_layer(x)
        return cast(Tensor, x)

    def _forward_gpkffn(self, x: Tensor, gpkffn: Any, state_iter: StateIterator) -> Tensor:
        """Mirror of :meth:`src.models.backbone.GroupPrimeKernelFFN.forward` with state routing.

        ``attn_{k}`` is ``nn.Sequential(causal/non-causal conv, 1x1 Conv1d)``;
        only the first is functional-stateful (when causal). ``conv_{k}`` is the
        causal/non-causal conv directly.
        """
        shortcut = x
        x = gpkffn.norm(x)
        x = gpkffn.proj_first(x)
        expand_ratio = int(gpkffn.expand_ratio)
        kernel_list = list(gpkffn.kernel_list)

        x_chunks: List[Tensor] = list(torch.chunk(x, expand_ratio, dim=1))
        for i in range(expand_ratio):
            ks = kernel_list[i]
            attn_module = getattr(gpkffn, f"attn_{ks}")
            conv_module = getattr(gpkffn, f"conv_{ks}")

            attn_out = x_chunks[i]
            for layer in attn_module:
                if isinstance(layer, _TIME_AXIS_FUNCTIONAL_TYPE):
                    attn_out = state_iter.run_layer(layer, attn_out, state_frames=self.state_frames_for_update)
                else:
                    attn_out = layer(attn_out)

            if isinstance(conv_module, _TIME_AXIS_FUNCTIONAL_TYPE):
                conv_out = state_iter.run_layer(conv_module, x_chunks[i], state_frames=self.state_frames_for_update)
            else:
                conv_out = conv_module(x_chunks[i])

            x_chunks[i] = attn_out * conv_out

        x = torch.cat(x_chunks, dim=1)
        if getattr(gpkffn, "fold_residual_scale", False):
            x = gpkffn.proj_last(x) + shortcut
        else:
            x = gpkffn.proj_last(x) * gpkffn.scale + shortcut
        if getattr(gpkffn, "post_norm", False):
            x = gpkffn.post_norm_layer(x)
        return cast(Tensor, x)

    def _forward_mask_decoder(self, x: Tensor, state_iter: StateIterator) -> Tensor:
        """Mirror of :meth:`src.models.backbone.MaskDecoder.forward` with state routing.

        Returns the mask shaped ``[B, 1, T, F]`` (the caller squeezes/transposes
        to ``[B, F, T]``).
        """
        decoder: Any = self.mask_decoder
        x = self._forward_ds_ddb(x, decoder.dense_block, state_iter)
        x = decoder.mask_conv(x)
        x = x.squeeze(1).transpose(1, 2)
        x = decoder.lsigmoid(x).transpose(1, 2).unsqueeze(1)
        return cast(Tensor, x)

    def _forward_phase_decoder(
        self,
        x: Tensor,
        state_iter: StateIterator,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Mirror of :meth:`src.models.backbone.PhaseDecoder.forward` with state routing.

        Returns:
            Atan2 mode: ``[B, 1, T, F]`` phase tensor (``atan2(x_i+1e-8, x_r+1e-8)``).
            Complex mode: tuple ``(x_r, x_i)``, each ``[B, 1, T, F]`` (the raw
                ``phase_conv_r`` / ``phase_conv_i`` outputs — host atan2'd later).
        """
        decoder: Any = self.phase_decoder
        x = self._forward_ds_ddb(x, decoder.dense_block, state_iter)
        x = decoder.phase_conv(x)
        x_r = decoder.phase_conv_r(x)
        x_i = decoder.phase_conv_i(x)
        if self.phase_output_mode == "complex":
            return x_r, x_i
        # Same epsilons as src.models.backbone.PhaseDecoder.forward (gradient-stability floor).
        return torch.atan2(x_i + 1e-8, x_r + 1e-8)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(infer_type={self.infer_type!r}, "
            f"n_fft={self.n_fft}, L_enc={self.encoder_lookahead}, L_dec={self.decoder_lookahead}, "
            f"phase_output_mode={self.phase_output_mode!r}, num_states={self.num_states}, "
            f"state_frames_for_update={self.state_frames_for_update})"
        )



__all__ = [
    "ExportableBackboneCore",
    "StateIterator",
    "convert_stateful_to_functional",
]
