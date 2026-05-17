"""Exportable **decomposed** BAFNet+ core (PyTorch, non-streaming — S5).

S5 of the streaming-ONNX rebuild (Stage 2 — graph design freeze).
:class:`BAFNetPlusCore` is the PyTorch decomposed re-expression of
:class:`~src.models.bafnetplus.BAFNetPlus`: it runs the BCS mapping branch and
the ACS masking branch via two :class:`~src.models.streaming.onnx.backbone_core.ExportableBackboneCore`
instances, then the calibration encoder + alpha conv stack + softmax blend —
matching ``BAFNetPlus.forward()`` on a complete spectrogram. The fusion-path
``CausalConv1d`` (calibration) / ``CausalConv2d`` (alpha) layers stay
*non-functional* (no explicit state I/O) at S5; turning them functional is S8
when we export the actual fusion ONNX. State propagation across chunks is
therefore **not** wired through calibration/alpha in S5 — the S5 parity gate
is the non-streaming decomposed-core ≡ ``BAFNetPlus.forward`` re-expression.

What this is / is not
---------------------
- It IS a PyTorch ``nn.Module`` taking the host-prepared 4-tuple
  ``(bcs_mag, bcs_pha, acs_mag, acs_pha)`` (the same ``mag/pha`` boundary the
  S4 single-Backbone core uses, applied per branch) and producing the spectral
  triple ``(est_mag, est_pha, est_com)``.
- It IS a re-expression of ``BAFNetPlus.forward`` — the calibration / common
  gain / relative gain / alpha modules are **shared references** to the
  passed-in BAFNet+ instance (no deep-copies). The two backbone cores ARE
  deep-copied (via ``ExportableBackboneCore.from_backbone``).
- It IS the place the S5 ``T_export`` proof harness lives — see
  :func:`prove_t_export`.
- It is **NOT** the streaming export graph: calibration/alpha are not yet
  functional-stateful, and conv-state I/O is not threaded through the
  decomposed-core's forward. That is S8.
- It does **NOT** do host STFT / iSTFT / OLA — that is S6.

Graph boundary (per-branch; cf. ``docs/wiki/concepts/lacosenet-backbone-streaming.md`` §6):
  ``Host FP32:`` audio → STFT → ``com`` → ``complex_to_mag_pha`` →
                 ``(bcs_mag, bcs_pha, acs_mag, acs_pha)``
  ``ONNX FP32:`` ``(bcs_mag, bcs_pha, acs_mag, acs_pha, *prev_states)`` →
                 core → ``(est_mag, est_pha, est_com, *next_states)``
  ``Host FP32:`` ``est_mag, est_pha`` → ``mag_pha_to_complex`` → iSTFT → audio

Frozen I/O contract (the contract S8/S9/S11 re-bind to)
-------------------------------------------------------
Inputs (in order):
    ``bcs_mag [1, F, T]``, ``bcs_pha [1, F, T]``,
    ``acs_mag [1, F, T]``, ``acs_pha [1, F, T]``,
    then state tensors in :meth:`get_state_names` order — at S5 those are
    ``mapping/<sub-state>`` (one per mapping-branch functional stateful conv)
    followed by ``masking/<sub-state>`` (one per masking-branch). S8 will
    append ``calibration/*`` and ``alpha/*`` once those paths are made
    functional. The S5 ``num_states`` therefore covers backbones only.
Outputs (in order):
    atan2 mode (default): ``est_mag [1, F, T]``, ``est_pha [1, F, T]``,
    ``est_com [1, F, T, 2]``, then ``next_<state_name>`` per state.
    Complex mode (INT8 hedge — accepted by ``from_bafnetplus`` for forward
    compatibility but the BAFNetPlusCore output stays
    ``(est_mag, est_pha, est_com)`` — atan2 is computed internally from
    the per-branch ``phase_real`` / ``phase_imag``, since the fusion needs
    ``est_com`` for the alpha blend).

Frame geometry
--------------
At the 50 ms anchor (chunk_size=8, both backbones' ``L_enc=L_dec=3``):
:attr:`mapping_total_lookahead` = :attr:`masking_total_lookahead` =
``L_enc + L_dec = 6``. The conservative monolithic-graph default is
``T_export = chunk_size + L_enc + L_dec = 14`` (one shared geometry — both
branches share the same backbone topology). The smaller candidate
``chunk_size + max(L_enc, L_dec) = 11`` is the deployed-single-Backbone
geometry (LaCoSENet-style decoupled encoder/decoder via a feature buffer),
which is **not** valid for a monolithic fusion graph that fuses encoder+TS+
decoders+calibration+alpha in one shot. :func:`prove_t_export` runs the
one-chunk proof: feed the first ``T_export`` frames at ``t=0`` through the
decomposed core and compare its first ``chunk_size`` outputs against
``BAFNetPlus.forward()``'s first ``chunk_size`` outputs. The chosen
``T_export`` is the smallest geometry that matches within float32
tolerance — defaulting to ``chunk_size + L_enc + L_dec`` if neither
candidate passes (or if both pass, the smaller).

Mask recovery convention
------------------------
``BAFNetPlus.forward`` calls ``self.masking(acs_com, return_mask=True)`` to
get the raw mask ``acs_mask`` (post-``LearnableSigmoid_2d``, pre-multiplication).
``ExportableBackboneCore`` does **not** expose the mask as a separate output —
it only emits ``est_mag = mag * mask`` (for ``infer_type='masking'``). In S5
we recover ``acs_mask`` via the mathematical inverse
``acs_mask = acs_est_mag / acs_mag`` — exact within IEEE 754 ``(a*b)/a`` 1-ULP
drift, since the host's input ``acs_mag`` has the ``sqrt(... + 1e-9)`` floor
from :func:`src.stft.mag_pha_stft` and is always strictly positive. This is
the S5 design compromise: it costs ``torch.equal`` parity with
``BAFNetPlus.forward`` but holds tightly within ``MAG_TOL = 1e-5`` (the test
gate). S8 can extend ``ExportableBackboneCore`` with an explicit mask output
if a stricter S5/S6 spec emerges.

Causality audits (S5)
---------------------
- :func:`audit_calibration_is_causal` walks the calibration encoder and
  asserts every conv is a left-only-padded :class:`~src.models.backbone.CausalConv1d`
  (matches the wiki's claim — calibration adds **no** lookahead).
- :func:`audit_alpha_time_lookahead` walks the alpha conv stack and computes
  the time-axis right-padding sum per :class:`~src.models.backbone.CausalConv2d`
  (whose ``F.pad`` quartet is ``(left_freq, right_freq, top_time, bottom_time)``
  with ``bottom_time = 0`` for causal). Reports the alpha right-padding
  contribution ``L_alpha`` — for the deployed unified ckpt this is **0**
  because BAFNet+ alpha uses :class:`~src.models.backbone.CausalConv2d`, NOT a
  plain ``nn.Conv2d`` with symmetric padding. The S5 proof harness uses the
  audited value to form the third candidate
  ``T_export_c = chunk + L_enc + L_dec + L_alpha`` only when ``L_alpha > 0``.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.models.backbone import CausalConv1d, CausalConv2d
from src.models.bafnetplus import BAFNetPlus
from src.models.streaming.converters import convert_to_stateful
from src.models.streaming.onnx.backbone_core import (
    ExportableBackboneCore,
    StateIterator,
    convert_stateful_to_functional,
)
from src.models.streaming.onnx.functional_stateful import (
    FunctionalStatefulCausalConv2d,
    FunctionalStatefulConv1d,
)
from src.stft import complex_to_mag_pha

logger = logging.getLogger(__name__)


# ============================================================================ core


class BAFNetPlusCore(nn.Module):
    """Decomposed PyTorch core re-expressing :meth:`BAFNetPlus.forward`.

    Forward: ``(bcs_mag, bcs_pha, acs_mag, acs_pha) → (est_mag, est_pha, est_com)``.

    The two branch cores are deep-copied + functional-converted (via
    :meth:`ExportableBackboneCore.from_backbone`). The calibration encoder,
    common/relative gain heads, alpha conv stack and ``alpha_out`` are SHARED
    REFERENCES with the passed-in :class:`BAFNetPlus` instance — the design is
    a re-expression, not a parallel structure. ``set_state_frames_for_update``
    propagates to both backbone cores; calibration/alpha layers remain
    non-functional in S5.

    Attributes:
        mapping_core: Functional :class:`ExportableBackboneCore` (``infer_type='mapping'``).
        masking_core: Functional :class:`ExportableBackboneCore` (``infer_type='masking'``).
        calibration_encoder: Shared ref to BAFNet+'s calibration encoder (or ``None``).
        common_gain_head: Shared ref (or ``None``).
        relative_gain_head: Shared ref (or ``None``; ``None`` when
            ``use_relative_gain=False``).
        alpha_convblocks: Shared ref to BAFNet+'s alpha conv stack.
        alpha_out: Shared ref to BAFNet+'s alpha 1x1 conv.
        ablation_mode: One of the :attr:`BAFNetPlus.VALID_ABLATION_MODES`.
        use_calibration / use_relative_gain / mask_only_alpha: Algebraic flags
            (mirror the BAFNet+ instance).
        calibration_max_common_log_gain / calibration_max_relative_log_gain:
            ``tanh``-scaled max log gains (mirror the BAFNet+ instance).
        n_fft: STFT size (used to derive ``freq_size``).
        phase_output_mode: ``'atan2'`` (default, FP32) or ``'complex'``. Only
            affects the two underlying backbone cores' internal layout; the
            BAFNetPlusCore forward output is always
            ``(est_mag, est_pha, est_com)`` because the fusion algebra needs
            ``est_com`` for the alpha blend.
    """

    def __init__(
        self,
        mapping_core: ExportableBackboneCore,
        masking_core: ExportableBackboneCore,
        *,
        calibration_encoder: Optional[nn.Module],
        common_gain_head: Optional[nn.Module],
        relative_gain_head: Optional[nn.Module],
        alpha_convblocks: nn.ModuleList,
        alpha_out: nn.Module,
        ablation_mode: str,
        use_calibration: bool,
        use_relative_gain: bool,
        mask_only_alpha: bool,
        calibration_max_common_log_gain: float,
        calibration_max_relative_log_gain: float,
        n_fft: int,
        phase_output_mode: str = "atan2",
    ) -> None:
        super().__init__()
        if mapping_core.infer_type != "mapping":
            raise ValueError(f"mapping_core.infer_type must be 'mapping', got {mapping_core.infer_type!r}")
        if masking_core.infer_type != "masking":
            raise ValueError(f"masking_core.infer_type must be 'masking', got {masking_core.infer_type!r}")
        if phase_output_mode not in ("atan2", "complex"):
            raise ValueError(f"phase_output_mode must be 'atan2' or 'complex', got {phase_output_mode!r}")
        if mapping_core.phase_output_mode != phase_output_mode:
            raise ValueError(
                "mapping_core.phase_output_mode != phase_output_mode "
                f"({mapping_core.phase_output_mode!r} vs {phase_output_mode!r})"
            )
        if masking_core.phase_output_mode != phase_output_mode:
            raise ValueError(
                "masking_core.phase_output_mode != phase_output_mode "
                f"({masking_core.phase_output_mode!r} vs {phase_output_mode!r})"
            )
        if ablation_mode not in BAFNetPlus.VALID_ABLATION_MODES:
            raise ValueError(f"ablation_mode must be one of {BAFNetPlus.VALID_ABLATION_MODES}, got {ablation_mode!r}")
        if use_calibration and calibration_encoder is None:
            raise ValueError("use_calibration=True but calibration_encoder is None")
        if use_calibration and common_gain_head is None:
            raise ValueError("use_calibration=True but common_gain_head is None")
        if use_calibration and use_relative_gain and relative_gain_head is None:
            raise ValueError("use_relative_gain=True but relative_gain_head is None")

        self.mapping_core = mapping_core
        self.masking_core = masking_core

        # Shared references (NOT deep-copies); registered as submodules by nn.Module.__setattr__
        # so they appear in state_dict + parameters().
        if calibration_encoder is not None:
            self.calibration_encoder = calibration_encoder
        else:
            self.calibration_encoder = None  # type: ignore[assignment]
        if common_gain_head is not None:
            self.common_gain_head = common_gain_head
        else:
            self.common_gain_head = None  # type: ignore[assignment]
        if relative_gain_head is not None:
            self.relative_gain_head = relative_gain_head
        else:
            self.relative_gain_head = None  # type: ignore[assignment]
        self.alpha_convblocks = alpha_convblocks
        self.alpha_out = alpha_out

        self.ablation_mode: str = ablation_mode
        self.use_calibration: bool = bool(use_calibration)
        self.use_relative_gain: bool = bool(use_relative_gain)
        self.mask_only_alpha: bool = bool(mask_only_alpha)
        self.calibration_max_common_log_gain: float = float(calibration_max_common_log_gain)
        self.calibration_max_relative_log_gain: float = float(calibration_max_relative_log_gain)
        self.n_fft: int = int(n_fft)
        self.phase_output_mode: str = phase_output_mode

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_bafnetplus(
        cls,
        model: BAFNetPlus,
        *,
        phase_output_mode: str = "atan2",
    ) -> "BAFNetPlusCore":
        """Build a decomposed core from an instantiated :class:`BAFNetPlus`.

        Steps: deep-copy + functional-convert each branch (via
        :meth:`ExportableBackboneCore.from_backbone`), copy SHARED REFERENCES
        to ``model.calibration_encoder`` / ``model.common_gain_head`` /
        ``model.relative_gain_head`` / ``model.alpha_convblocks`` /
        ``model.alpha_out``, mirror the algebra flags
        (``use_calibration`` / ``use_relative_gain`` / ``mask_only_alpha``,
        ``calibration_max_common_log_gain`` / ``calibration_max_relative_log_gain``,
        ``ablation_mode``).

        Args:
            model: An instantiated :class:`BAFNetPlus` whose ``mapping`` /
                ``masking`` branches are still original ``Backbone`` instances
                (not stateful-converted). For the real ckpt path the caller
                should have done
                ``BAFNetPlus(args_mapping=..., args_masking=..., load_pretrained_weights=False)``
                followed by ``model.load_state_dict(unified_ckpt['model'])`` —
                see the wiki :doc:`docs/wiki/projects/reference-runtime` foot-gun
                note.
            phase_output_mode: ``'atan2'`` (default, FP32) or ``'complex'``.
                Forwarded to both backbone cores.

        Returns:
            A ready :class:`BAFNetPlusCore` (``state_frames_for_update`` on
            both backbone cores starts as ``None`` — set it via
            :meth:`set_state_frames_for_update` before streaming).

        Raises:
            TypeError: If ``model`` is not a :class:`BAFNetPlus`.
        """
        if not isinstance(model, BAFNetPlus):
            raise TypeError(f"from_bafnetplus expects a BAFNetPlus, got {type(model).__name__}")

        mapping_core = ExportableBackboneCore.from_backbone(model.mapping, phase_output_mode=phase_output_mode)
        masking_core = ExportableBackboneCore.from_backbone(model.masking, phase_output_mode=phase_output_mode)

        calibration_encoder = getattr(model, "calibration_encoder", None) if model.use_calibration else None
        common_gain_head = getattr(model, "common_gain_head", None) if model.use_calibration else None
        relative_gain_head = (
            getattr(model, "relative_gain_head", None) if (model.use_calibration and model.use_relative_gain) else None
        )

        n_fft = int(model.mapping.n_fft) if model.mapping.n_fft is not None else 0
        if n_fft == 0:
            raise ValueError("model.mapping.n_fft is not set")

        return cls(
            mapping_core=mapping_core,
            masking_core=masking_core,
            calibration_encoder=calibration_encoder,
            common_gain_head=common_gain_head,
            relative_gain_head=relative_gain_head,
            alpha_convblocks=model.alpha_convblocks,
            alpha_out=model.alpha_out,
            ablation_mode=model.ablation_mode,
            use_calibration=model.use_calibration,
            use_relative_gain=model.use_relative_gain,
            mask_only_alpha=model.mask_only_alpha,
            calibration_max_common_log_gain=model.calibration_max_common_log_gain,
            calibration_max_relative_log_gain=model.calibration_max_relative_log_gain,
            n_fft=n_fft,
            phase_output_mode=phase_output_mode,
        )

    # ------------------------------------------------------------------ state ops
    def set_state_frames_for_update(self, state_frames: Optional[int]) -> None:
        """Propagate ``state_frames_for_update`` to both backbone cores.

        Calibration/alpha layers remain non-functional at S5 — no state to
        bound. S8 will extend this method when those paths become functional.

        Args:
            state_frames: ``None`` for the single-shot full-sequence parity
                run (the S5 default); ``chunk_size`` for streaming export.
        """
        self.mapping_core.set_state_frames_for_update(state_frames)
        self.masking_core.set_state_frames_for_update(state_frames)

    @property
    def num_states(self) -> int:
        """Total number of state tensors across both backbone cores.

        S5 has no calibration/alpha states (those paths stay non-functional);
        S8 will add them, increasing this count.
        """
        return self.mapping_core.num_states + self.masking_core.num_states

    @property
    def total_lookahead(self) -> int:
        """Conservative monolithic-graph lookahead: max of the two branches'
        ``encoder_lookahead + decoder_lookahead`` (typically the same value).

        Calibration/alpha contribute zero lookahead by construction (both use
        :class:`CausalConv1d` / :class:`CausalConv2d` from
        :mod:`src.models.backbone`).
        """
        return max(self.mapping_core.total_lookahead, self.masking_core.total_lookahead)

    def get_state_names(self) -> List[str]:
        """Return state names in the frozen S8 export order: ``mapping/* masking/*``.

        Calibration/alpha state names will be appended after ``masking/*`` in
        S8 once those paths are functional — at S5 the list covers only
        backbone states.
        """
        names: List[str] = []
        for n in self.mapping_core.get_state_names():
            names.append(f"mapping/{n}")
        for n in self.masking_core.get_state_names():
            names.append(f"masking/{n}")
        return names

    def input_names(self) -> List[str]:
        """Frozen ONNX input names:
        ``[bcs_mag, bcs_pha, acs_mag, acs_pha, *state_names]``."""
        return ["bcs_mag", "bcs_pha", "acs_mag", "acs_pha"] + self.get_state_names()

    def output_names(self) -> List[str]:
        """Frozen ONNX output names — depend on :attr:`phase_output_mode`.

        Atan2 mode (default): ``[est_mag, est_pha, est_com, *next_state_names]``.
        The fusion ``est_com`` is the alpha-blended complex spectrogram and
        ``est_pha = atan2(est_imag+1e-8, est_real+1e-8)`` is computed in the
        graph via :func:`src.stft.complex_to_mag_pha`.

        Complex mode (S20 — atan2-free): ``[est_mag, phase_real, phase_imag,
        *next_state_names]``. The graph emits the FUSED complex spectrogram
        components directly (``phase_real``/``phase_imag`` are the real/imag
        parts of the alpha-blended ``est_com``), and ``est_mag`` is the
        sqrt-derived magnitude (``sqrt(real^2 + imag^2 + 1e-8)``). The host
        is responsible for ``est_pha = atan2(phase_imag+1e-8, phase_real+1e-8)``
        — i.e., atan2 is moved entirely OUT of the ONNX graph (both per-branch
        and the fused atan2 are eliminated). This is the S20 fix for the S18(A)
        atan2 single-chunk drift dominating the ORT-vs-PT divergence.
        """
        nexts = [f"next_{n}" for n in self.get_state_names()]
        if self.phase_output_mode == "complex":
            return ["est_mag", "phase_real", "phase_imag"] + nexts
        return ["est_mag", "est_pha", "est_com"] + nexts

    # ----------------------------------------------------------------- metadata
    def metadata(self, *, chunk_size: int = 8, freq_size: Optional[int] = None) -> Dict[str, Any]:
        """Snapshot the decomposed core's I/O contract + frame geometry.

        Returned keys: ``schema_version``, ``ablation_mode``, ``use_calibration``,
        ``use_relative_gain``, ``mask_only_alpha``,
        ``calibration_max_common_log_gain``,
        ``calibration_max_relative_log_gain``, ``phase_output_mode``,
        ``n_fft``, ``freq_size``, ``chunk_size``, ``encoder_lookahead``,
        ``decoder_lookahead``, ``total_lookahead``, ``t_export``,
        ``t_export_formula``, ``alpha_time_lookahead``, ``num_states``,
        ``input_names``, ``output_names``, ``state_names``, ``state_order``,
        ``branches`` (the per-branch :meth:`ExportableBackboneCore.metadata`).

        Used by S8 (export) and S9/S11 (host/Android binders).

        Args:
            chunk_size: Output chunk size; used to derive ``t_export``.
                Anchor for 50 ms BAFNet+ is ``chunk_size = 8``.
            freq_size: ``n_fft // 2 + 1`` if ``None``.
        """
        if freq_size is None:
            freq_size = self.n_fft // 2 + 1
        # Both branches share the same backbone topology in BAFNet+ — pick mapping_core for the
        # encoder/decoder lookahead; the metadata records them as a single shared value, and
        # cross-checks against masking_core inside the assertion below.
        l_enc_m = self.mapping_core.encoder_lookahead
        l_dec_m = self.mapping_core.decoder_lookahead
        if self.masking_core.encoder_lookahead != l_enc_m or self.masking_core.decoder_lookahead != l_dec_m:
            logger.warning(
                "metadata: mapping/masking lookahead mismatch (enc=%d/%d, dec=%d/%d) — " "using the maximum",
                l_enc_m,
                self.masking_core.encoder_lookahead,
                l_dec_m,
                self.masking_core.decoder_lookahead,
            )
        l_enc = max(l_enc_m, self.masking_core.encoder_lookahead)
        l_dec = max(l_dec_m, self.masking_core.decoder_lookahead)
        alpha_lookahead = audit_alpha_time_lookahead(self.alpha_convblocks)
        t_export = int(chunk_size) + l_enc + l_dec + alpha_lookahead
        return {
            "schema_version": "s5-bafnetplus-decomposed-fp32",
            "ablation_mode": self.ablation_mode,
            "use_calibration": self.use_calibration,
            "use_relative_gain": self.use_relative_gain,
            "mask_only_alpha": self.mask_only_alpha,
            "calibration_max_common_log_gain": self.calibration_max_common_log_gain,
            "calibration_max_relative_log_gain": self.calibration_max_relative_log_gain,
            "phase_output_mode": self.phase_output_mode,
            "n_fft": self.n_fft,
            "freq_size": int(freq_size),
            "chunk_size": int(chunk_size),
            "encoder_lookahead": int(l_enc),
            "decoder_lookahead": int(l_dec),
            "total_lookahead": int(l_enc + l_dec),
            "alpha_time_lookahead": int(alpha_lookahead),
            "t_export": int(t_export),
            "t_export_formula": ("chunk_size + encoder_lookahead + decoder_lookahead + alpha_time_lookahead"),
            "num_states": self.num_states,
            "input_names": self.input_names(),
            "output_names": self.output_names(),
            "state_names": self.get_state_names(),
            "state_order": "mapping/* masking/* (calibration/* alpha/* appended by S8)",
            "branches": {
                "mapping": self.mapping_core.metadata(freq_size=freq_size),
                "masking": self.masking_core.metadata(freq_size=freq_size),
            },
        }

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        bcs_mag: Tensor,
        bcs_pha: Tensor,
        acs_mag: Tensor,
        acs_pha: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Re-express :meth:`BAFNetPlus.forward` over the per-branch
        ``(mag, pha)`` boundary.

        At S5 the two backbone cores run with **zero init states** and the
        ``state_frames_for_update`` whatever the caller set (``None`` for the
        non-streaming parity gate, ``chunk_size`` for the
        :func:`prove_t_export` chunk test).

        Args:
            bcs_mag: BCS compressed magnitude ``[B, F, T]``.
            bcs_pha: BCS phase ``[B, F, T]``.
            acs_mag: ACS compressed magnitude ``[B, F, T]`` — produced by the
                host via :func:`src.stft.mag_pha_stft` (always strictly
                positive thanks to the ``sqrt(... + 1e-9)`` floor).
            acs_pha: ACS phase ``[B, F, T]``.

        Returns:
            Atan2 mode (default): ``(est_mag [B, F, T], est_pha [B, F, T],
            est_com [B, F, T, 2])`` — the fused complex spectrogram computed
            by softmax-blending the two branches' calibrated complex outputs,
            then :func:`src.stft.complex_to_mag_pha`-decomposed.

            Complex mode (S20 — atan2-free): ``(est_mag [B, F, T],
            phase_real [B, F, T], phase_imag [B, F, T])`` — the same fused
            complex spectrogram, but emitted as separate real/imag tensors
            with ``est_mag = sqrt(real^2 + imag^2 + 1e-8)``. atan2 is computed
            on the host instead of in-graph, eliminating the S18(A) atan2
            drift source on both sides of the alpha fusion.
        """
        bcs_est_mag, bcs_est_com, _bcs_mask = self._run_branch(self.mapping_core, bcs_mag, bcs_pha, expose_mask=False)
        # S9.6: route the masking branch through ``forward_with_mask`` so the
        # raw mask (post-LearnableSigmoid, pre-multiplication) is returned
        # explicitly — bit-equivalent to ``BAFNetPlus.forward``'s
        # ``self.masking(acs_com, return_mask=True)`` consumption. The pre-S9.6
        # path recovered ``acs_mask = acs_est_mag / acs_mag`` (a*b)/a, which
        # drifts by 1 ULP and propagates through the calibration / alpha
        # softmax blend; the explicit mask closes that drift at the source.
        acs_est_mag, acs_est_com, acs_mask = self._run_branch(self.masking_core, acs_mag, acs_pha, expose_mask=True)

        # ----- calibration path
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

        if self.phase_output_mode == "complex":
            # S20 atan2-free output: emit the fused complex components directly.
            # est_mag matches complex_to_mag_pha's mag formula (sqrt(re^2+im^2+1e-8))
            # so downstream consumers that need a magnitude get the same scale.
            fused_real = est_com[..., 0]
            fused_imag = est_com[..., 1]
            est_mag = torch.sqrt(fused_real**2 + fused_imag**2 + 1e-8)
            return est_mag, fused_real, fused_imag

        est_mag, est_pha = complex_to_mag_pha(est_com)
        return est_mag, est_pha, est_com

    def _run_branch(
        self,
        core: ExportableBackboneCore,
        mag: Tensor,
        pha: Tensor,
        *,
        expose_mask: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Run one branch core with zero init states and return
        ``(est_mag, est_com, raw_mask_or_None)``.

        Handles both ``phase_output_mode`` settings of the underlying core:

        - ``atan2``: backbone returns ``(est_mag, est_pha, est_com,
          *next_states)`` — we take ``est_mag`` and ``est_com`` directly.
        - ``complex`` (S20 atan2-free): backbone returns ``(est_mag,
          phase_real, phase_imag, *next_states)`` — we reconstruct ``est_com``
          via the algebraic identity
          ``mag_pha_to_complex(mag, atan2(im+eps, re+eps))
          = mag * (re+eps, im+eps) / sqrt((re+eps)^2 + (im+eps)^2)``.
          This keeps the per-branch est_com numerically equivalent to the
          atan2 path (up to libm cos/sin/atan2 precision) while keeping
          ``atan2`` OUT of the in-graph compute. The fused output is also
          atan2-free in complex mode (see :meth:`forward`).

        When ``expose_mask=True`` the call routes through
        :meth:`ExportableBackboneCore.forward_with_mask` so the raw
        ``mask_decoder`` output (the same tensor ``Backbone.forward(...,
        return_mask=True)`` returns) is returned alongside ``est_mag`` /
        ``est_com``. The caller can use it directly to skip the
        ``acs_mask = est_mag / mag`` recovery and its (a*b)/a 1-ULP drift.
        """
        states = core.init_states(
            batch_size=mag.shape[0],
            freq_size=mag.shape[1],
            device=mag.device,
            dtype=mag.dtype,
        )
        if expose_mask:
            outs = core.forward_with_mask(mag, pha, *states)
            est_mag = outs[0]
            if core.phase_output_mode == "complex":
                est_com = _phase_components_to_complex(est_mag, outs[1], outs[2])
            else:
                est_com = outs[2]
            mask = outs[3]
            return est_mag, est_com, mask

        outs = core(mag, pha, *states)
        est_mag = outs[0]
        if core.phase_output_mode == "complex":
            est_com = _phase_components_to_complex(est_mag, outs[1], outs[2])
        else:
            est_com = outs[2]
        return est_mag, est_com, None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ablation_mode={self.ablation_mode!r}, "
            f"use_calibration={self.use_calibration}, use_relative_gain={self.use_relative_gain}, "
            f"mask_only_alpha={self.mask_only_alpha}, "
            f"phase_output_mode={self.phase_output_mode!r}, "
            f"num_states={self.num_states}, total_lookahead={self.total_lookahead})"
        )


# ============================================================ S8 exportable core


class ExportableBAFNetPlusCore(nn.Module):
    """Functional-stateful BAFNet+ core for FP32 ONNX export (S8 — Stage 4 pt.1).

    Departs from :class:`BAFNetPlusCore` (S5) in two ways:

    1. **No shared references with the source BAFNet+**: the entire fusion path
       (calibration encoder / common+relative gain heads / alpha conv stack /
       ``alpha_out``) is **deep-copied** so the calibration / alpha layers can
       be ``convert_to_stateful`` → ``convert_stateful_to_functional`` in place
       without mutating the source bafnet. The S5 module's shared-reference
       semantics are incompatible with the in-place graph rewrite required for
       functional-stateful export.
    2. **Forward routes explicit state I/O through calibration + alpha** in
       addition to the two backbone cores. The 190-state graph (92 mapping + 92
       masking + 2 calibration + 4 alpha for the 50 ms anchor) is exported as a
       single ``(bcs_mag, bcs_pha, acs_mag, acs_pha, *all_states) → (est_mag,
       est_pha, est_com, *all_next_states)`` function.

    The class is paired with :class:`BAFNetPlusCore`: the latter (non-streaming,
    shared-reference) stays the bit-exact metric anchor and is the model
    :func:`prove_t_export` drives. This class is the **streaming-exportable**
    form — its single-shot full-sequence forward (with zero init states + no
    ``state_frames_for_update`` bound) is bit-exact (``torch.equal``) to
    :meth:`BAFNetPlus.forward` (modulo the 1-ULP mask-recovery drift already
    present in S5's ``acs_mask = acs_est_mag / acs_mag`` formula). Streaming the
    same core chunk by chunk with state propagation closes the
    S6/S7 non-streaming-fusion drift documented in
    :class:`~src.models.streaming.bafnetplus_streaming.BAFNetPlusStreaming`.

    Frozen I/O contract (the contract S9/S11 re-bind to)
    ----------------------------------------------------
    Inputs (in order):
        ``bcs_mag [B, F, T]``, ``bcs_pha [B, F, T]``,
        ``acs_mag [B, F, T]``, ``acs_pha [B, F, T]``,
        then state tensors in :meth:`get_state_names` order — frozen as
        ``mapping/*`` (all 92) → ``masking/*`` (all 92) → ``calibration/*`` (2)
        → ``alpha/*`` (4). Calibration / alpha states are appended after
        ``masking/*`` exactly as the S5 ``metadata()`` schema's
        ``t_export_formula`` anticipated.
    Outputs (in order):
        ``est_mag [B, F, T]``, ``est_pha [B, F, T]``, ``est_com [B, F, T, 2]``
        (atan2 mode), then ``next_<state_name>`` per state. Complex mode
        ``phase_output_mode='complex'`` is accepted for forward compatibility
        but the fused output stays the atan2 triple because the fusion needs
        ``est_com`` for the alpha blend.

    State count breakdown (50 ms BAFNet+ anchor)
    --------------------------------------------
    ``mapping`` (92) + ``masking`` (92) + ``calibration`` (2 × ``CausalConv1d``)
    + ``alpha`` (4 × ``CausalConv2d``) = **190 total**.

    Attributes:
        mapping_core: Functional :class:`ExportableBackboneCore`
            (``infer_type='mapping'``) deep-copied from the source.
        masking_core: Same but ``infer_type='masking'``.
        calibration_encoder: ``nn.Sequential`` of
            ``nn.Sequential(FunctionalStatefulConv1d, PReLU)`` blocks — or
            ``None`` for ``ablation_mode='no_calibration'``.
        common_gain_head / relative_gain_head: Deep-copies of the source's
            ``nn.Conv1d`` heads (no stateful conversion — kernel=1 means no
            state). ``relative_gain_head`` is ``None`` when
            ``use_relative_gain=False``.
        alpha_convblocks: ``nn.ModuleList`` of
            ``nn.Sequential(FunctionalStatefulCausalConv2d, BatchNorm2d, PReLU)``
            blocks (each a deep-copy + functional-conversion of the source).
        alpha_out: Deep-copy of the source's ``nn.Conv2d`` head (kernel=1,
            stateless).
        ablation_mode / use_calibration / use_relative_gain / mask_only_alpha:
            Algebraic flags (mirror the source BAFNet+ instance).
        calibration_max_common_log_gain / calibration_max_relative_log_gain:
            ``tanh``-scaled max log gains.
        n_fft: STFT size (used to derive ``freq_size = n_fft // 2 + 1``).
        phase_output_mode: ``'atan2'`` (default) or ``'complex'``. Forwarded to
            both backbone cores; the BAFNetPlusCore-level output stays atan2.
        state_frames_for_update: Bound on which leading input frames may
            update conv state across BOTH backbones AND the fusion convs.
            ``None`` = no bound (the entire input updates state — the right
            behaviour for a single-shot ``BAFNet+.forward``-equivalent run);
            ``chunk_size`` for streaming export.
    """

    def __init__(
        self,
        mapping_core: ExportableBackboneCore,
        masking_core: ExportableBackboneCore,
        *,
        calibration_encoder: Optional[nn.Module],
        common_gain_head: Optional[nn.Module],
        relative_gain_head: Optional[nn.Module],
        alpha_convblocks: nn.ModuleList,
        alpha_out: nn.Module,
        ablation_mode: str,
        use_calibration: bool,
        use_relative_gain: bool,
        mask_only_alpha: bool,
        calibration_max_common_log_gain: float,
        calibration_max_relative_log_gain: float,
        n_fft: int,
        phase_output_mode: str = "atan2",
    ) -> None:
        super().__init__()
        if mapping_core.infer_type != "mapping":
            raise ValueError(f"mapping_core.infer_type must be 'mapping', got {mapping_core.infer_type!r}")
        if masking_core.infer_type != "masking":
            raise ValueError(f"masking_core.infer_type must be 'masking', got {masking_core.infer_type!r}")
        if phase_output_mode not in ("atan2", "complex"):
            raise ValueError(f"phase_output_mode must be 'atan2' or 'complex', got {phase_output_mode!r}")
        if mapping_core.phase_output_mode != phase_output_mode:
            raise ValueError(
                "mapping_core.phase_output_mode != phase_output_mode "
                f"({mapping_core.phase_output_mode!r} vs {phase_output_mode!r})"
            )
        if masking_core.phase_output_mode != phase_output_mode:
            raise ValueError(
                "masking_core.phase_output_mode != phase_output_mode "
                f"({masking_core.phase_output_mode!r} vs {phase_output_mode!r})"
            )
        if ablation_mode not in BAFNetPlus.VALID_ABLATION_MODES:
            raise ValueError(f"ablation_mode must be one of {BAFNetPlus.VALID_ABLATION_MODES}, got {ablation_mode!r}")
        if use_calibration and calibration_encoder is None:
            raise ValueError("use_calibration=True but calibration_encoder is None")
        if use_calibration and common_gain_head is None:
            raise ValueError("use_calibration=True but common_gain_head is None")
        if use_calibration and use_relative_gain and relative_gain_head is None:
            raise ValueError("use_relative_gain=True but relative_gain_head is None")

        self.mapping_core = mapping_core
        self.masking_core = masking_core

        if calibration_encoder is not None:
            self.calibration_encoder = calibration_encoder
        else:
            self.calibration_encoder = None  # type: ignore[assignment]
        if common_gain_head is not None:
            self.common_gain_head = common_gain_head
        else:
            self.common_gain_head = None  # type: ignore[assignment]
        if relative_gain_head is not None:
            self.relative_gain_head = relative_gain_head
        else:
            self.relative_gain_head = None  # type: ignore[assignment]
        self.alpha_convblocks = alpha_convblocks
        self.alpha_out = alpha_out

        self.ablation_mode: str = ablation_mode
        self.use_calibration: bool = bool(use_calibration)
        self.use_relative_gain: bool = bool(use_relative_gain)
        self.mask_only_alpha: bool = bool(mask_only_alpha)
        self.calibration_max_common_log_gain: float = float(calibration_max_common_log_gain)
        self.calibration_max_relative_log_gain: float = float(calibration_max_relative_log_gain)
        self.n_fft: int = int(n_fft)
        self.phase_output_mode: str = phase_output_mode

        # Collect functional stateful modules in registration (DFS) order for state routing
        # + naming. Calibration first, then alpha — matches the frozen state-name order.
        self._calibration_functional_modules: List[Tuple[str, nn.Module]] = []
        self._alpha_functional_modules: List[Tuple[str, nn.Module]] = []
        if self.calibration_encoder is not None:
            for name, module in self.calibration_encoder.named_modules():
                if isinstance(module, FunctionalStatefulConv1d):
                    self._calibration_functional_modules.append((name, cast(nn.Module, module)))
        for name, module in self.alpha_convblocks.named_modules():
            if isinstance(module, FunctionalStatefulCausalConv2d):
                self._alpha_functional_modules.append((name, cast(nn.Module, module)))

        self.state_frames_for_update: Optional[int] = None

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "ExportableBAFNetPlusCore: %d mapping + %d masking + %d calibration + %d alpha = %d states",
                self.mapping_core.num_states,
                self.masking_core.num_states,
                self.num_calibration_states,
                self.num_alpha_states,
                self.num_states,
            )

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_bafnetplus(
        cls,
        model: BAFNetPlus,
        *,
        phase_output_mode: str = "atan2",
    ) -> "ExportableBAFNetPlusCore":
        """Build a functional-stateful exportable core from a :class:`BAFNetPlus`.

        Steps:

        1. Build the two backbone cores via
           :meth:`ExportableBackboneCore.from_backbone` on the source bafnet's
           ``mapping`` / ``masking`` branches (each constructor deep-copies +
           swaps TSBlocks to their reshape-free 4D variant + converts to
           stateful + then functional in place on its copy).
        2. Deep-copy the source bafnet's fusion modules (calibration encoder /
           common + relative gain heads / alpha conv stack / ``alpha_out``).
        3. ``convert_to_stateful`` → ``convert_stateful_to_functional`` the
           calibration encoder and alpha conv stack in place on the deep-copy
           (so the source bafnet stays non-streaming + usable as the bit-exact
           reference for ``prove_t_export`` / S5 :class:`BAFNetPlusCore`).
        4. The 1×1 gain heads + ``alpha_out`` are deep-copied without any
           stateful conversion — kernel=1 means no state.

        This is the documented S8 departure from S6's shared-reference contract.
        The S7 ``test_paired_paths_share_fusion_modules_byte_identical``
        assertion remains valid for S6's :class:`BAFNetPlusStreaming` (PT-side
        wrapper) but does NOT apply to S8's exportable core.

        Args:
            model: An instantiated :class:`BAFNetPlus` whose ``mapping`` /
                ``masking`` branches are still original ``Backbone`` instances.
            phase_output_mode: ``'atan2'`` (default, FP32) or ``'complex'``.
                Forwarded to both backbone cores.

        Returns:
            A ready :class:`ExportableBAFNetPlusCore`
            (``state_frames_for_update`` starts as ``None`` — set it via
            :meth:`set_state_frames_for_update` before streaming export).

        Raises:
            TypeError: If ``model`` is not a :class:`BAFNetPlus`.
        """
        if not isinstance(model, BAFNetPlus):
            raise TypeError(f"from_bafnetplus expects a BAFNetPlus, got {type(model).__name__}")

        mapping_core = ExportableBackboneCore.from_backbone(
            model.mapping,
            phase_output_mode=phase_output_mode,
        )
        masking_core = ExportableBackboneCore.from_backbone(
            model.masking,
            phase_output_mode=phase_output_mode,
        )

        if model.use_calibration:
            calibration_encoder = copy.deepcopy(model.calibration_encoder)
            convert_to_stateful(calibration_encoder, verbose=False, inplace=True)
            convert_stateful_to_functional(calibration_encoder)
            common_gain_head = copy.deepcopy(model.common_gain_head)
            if model.use_relative_gain:
                relative_gain_head = copy.deepcopy(model.relative_gain_head)
            else:
                relative_gain_head = None
        else:
            calibration_encoder = None
            common_gain_head = None
            relative_gain_head = None

        alpha_convblocks = copy.deepcopy(model.alpha_convblocks)
        convert_to_stateful(alpha_convblocks, verbose=False, inplace=True)
        convert_stateful_to_functional(alpha_convblocks)
        alpha_out = copy.deepcopy(model.alpha_out)

        n_fft = int(model.mapping.n_fft) if model.mapping.n_fft is not None else 0
        if n_fft == 0:
            raise ValueError("model.mapping.n_fft is not set")

        return cls(
            mapping_core=mapping_core,
            masking_core=masking_core,
            calibration_encoder=calibration_encoder,
            common_gain_head=common_gain_head,
            relative_gain_head=relative_gain_head,
            alpha_convblocks=alpha_convblocks,
            alpha_out=alpha_out,
            ablation_mode=model.ablation_mode,
            use_calibration=model.use_calibration,
            use_relative_gain=model.use_relative_gain,
            mask_only_alpha=model.mask_only_alpha,
            calibration_max_common_log_gain=model.calibration_max_common_log_gain,
            calibration_max_relative_log_gain=model.calibration_max_relative_log_gain,
            n_fft=n_fft,
            phase_output_mode=phase_output_mode,
        )

    # ------------------------------------------------------------- state counts
    @property
    def num_calibration_states(self) -> int:
        """Number of calibration functional stateful convs (0 for ``no_calibration``)."""
        return len(self._calibration_functional_modules)

    @property
    def num_alpha_states(self) -> int:
        """Number of alpha functional stateful convs (= ``conv_depth``)."""
        return len(self._alpha_functional_modules)

    @property
    def num_states(self) -> int:
        """Total number of state tensors across all branches + fusion convs."""
        return (
            self.mapping_core.num_states
            + self.masking_core.num_states
            + self.num_calibration_states
            + self.num_alpha_states
        )

    @property
    def total_lookahead(self) -> int:
        """Conservative monolithic-graph lookahead: max of the two branches'
        ``encoder_lookahead + decoder_lookahead``.

        Calibration + alpha contribute zero lookahead by construction (both
        use :class:`CausalConv1d` / :class:`CausalConv2d` from
        :mod:`src.models.backbone` — left-only time padding).
        """
        return max(self.mapping_core.total_lookahead, self.masking_core.total_lookahead)

    # ------------------------------------------------------------- state-frames
    def set_state_frames_for_update(self, state_frames: Optional[int]) -> None:
        """Bound how many leading input frames may update the conv states.

        Propagates to BOTH backbone cores AND is stored on ``self`` for the
        fusion path's functional convs to pick up at call time. The
        :class:`~src.models.streaming.context.StateFramesContext` thread-local
        is **not** consulted (that mechanism is for non-functional stateful
        layers; functional layers take ``state_frames`` as an explicit forward
        arg).

        Args:
            state_frames: ``None`` for the single-shot full-sequence parity run
                (the S8 ``(a)`` bit-exact gate); ``chunk_size`` for streaming
                export so the trailing ``L_enc + L_dec`` lookahead frames feed
                the conv outputs but do not corrupt the next chunk's recurrent
                state. The lookahead-protection contract extends from the
                backbones to the fusion convs even though ``L_alpha = 0`` makes
                it a no-op for fusion in the deployed 50 ms config.
        """
        self.state_frames_for_update = None if state_frames is None else int(state_frames)
        self.mapping_core.set_state_frames_for_update(state_frames)
        self.masking_core.set_state_frames_for_update(state_frames)

    # ------------------------------------------------------------------- naming
    def get_state_names(self) -> List[str]:
        """Return state names in the frozen S8 export order.

        Format: ``mapping/<sub>`` (all 92) then ``masking/<sub>`` (all 92) then
        ``calibration/state_<i>_<dotted_path_with_underscores>`` (one per
        ``FunctionalStatefulConv1d`` leaf) then ``alpha/state_<i>_<...>`` (one
        per ``FunctionalStatefulCausalConv2d`` leaf). Frozen as the ONNX state
        input name; the matching output is ``next_<that_name>``.
        """
        names: List[str] = []
        for n in self.mapping_core.get_state_names():
            names.append(f"mapping/{n}")
        for n in self.masking_core.get_state_names():
            names.append(f"masking/{n}")
        for i, (name, _) in enumerate(self._calibration_functional_modules):
            names.append(f"calibration/state_{i}_{name.replace('.', '_')}")
        for i, (name, _) in enumerate(self._alpha_functional_modules):
            names.append(f"alpha/state_{i}_{name.replace('.', '_')}")
        return names

    def input_names(self) -> List[str]:
        """Frozen ONNX input names:
        ``[bcs_mag, bcs_pha, acs_mag, acs_pha, *state_names]``."""
        return ["bcs_mag", "bcs_pha", "acs_mag", "acs_pha"] + self.get_state_names()

    def output_names(self) -> List[str]:
        """Frozen ONNX output names — depend on :attr:`phase_output_mode`.

        Atan2 mode (default — schema ``s8-bafnetplus-functional-fp32``):
        ``[est_mag, est_pha, est_com, *next_state_names]``. The fused
        ``est_com`` is the alpha-blended complex spectrogram and
        ``est_pha = atan2(est_imag+1e-8, est_real+1e-8)`` is computed in the
        graph via :func:`src.stft.complex_to_mag_pha`.

        Complex mode (S20 atan2-free — schema
        ``s20-bafnetplus-functional-fp32-complex``):
        ``[est_mag, phase_real, phase_imag, *next_state_names]``. The graph
        emits the FUSED complex spectrogram components directly (not the
        per-branch components — those are still consumed inside the fusion).
        ``est_mag = sqrt(real^2 + imag^2 + 1e-8)`` matches
        :func:`src.stft.complex_to_mag_pha`'s magnitude formula. The host
        is responsible for ``est_pha = atan2(phase_imag+1e-8, phase_real+1e-8)``
        — i.e., atan2 is moved OUT of the ONNX graph entirely (both per-branch
        and fused). This is the S20 fix for the S18(A) atan2 single-chunk
        drift dominating the ORT-vs-PT divergence on idx 1/3/4.
        """
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

        Backbones reuse :meth:`ExportableBackboneCore.init_states` (architecture-
        aware: time-stage 1D convs use ``batch_size * freq_enc`` batch dim,
        2D convs use ``batch_size`` with ``conv2d_freq`` per-encoder /
        per-decoder geometry — see the S4 module).

        Calibration ``FunctionalStatefulConv1d`` states are
        ``[B, in_channels, padding_size=4]`` per layer (kernel=5,
        padding=get_padding(5)=2 → :attr:`FunctionalStatefulConv1d.padding_size`
        = 4).

        Alpha ``FunctionalStatefulCausalConv2d`` states are
        ``[B, in_channels, time_padding=6, freq_size + 2 * freq_padding]`` per
        layer (kernel=(7, 7), padding=(3, 3) → ``time_padding`` = 6, freq pad
        symmetric = 3, so e.g. ``F = 201 → F_padded = 207``). The alpha input
        freq dim is the full ``freq_size = n_fft // 2 + 1`` (the alpha input is
        ``[B, 3 (or 1), T, F]`` built from the post-fusion spectrogram, not the
        encoder-strided ``freq_enc``).

        Args:
            batch_size: Logical batch size (typically 1 for streaming).
            freq_size: ``n_fft // 2 + 1`` if ``None``.
            time_frames: Forwarded to the backbone cores' ``time_frames``
                (used for the freq-stage 1D states; a no-op for BAFNet+ since
                its ``TSBlock.freq_stage`` is non-causal).
            device: State device (defaults to the model's parameter device).
            dtype: State dtype (defaults to the model's parameter dtype).

        Returns:
            List of zero-init state tensors in :meth:`get_state_names` order.
        """
        if freq_size is None:
            freq_size = self.n_fft // 2 + 1
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        states: List[Tensor] = []
        states.extend(
            self.mapping_core.init_states(
                batch_size=batch_size,
                freq_size=freq_size,
                time_frames=time_frames,
                device=device,
                dtype=dtype,
            )
        )
        states.extend(
            self.masking_core.init_states(
                batch_size=batch_size,
                freq_size=freq_size,
                time_frames=time_frames,
                device=device,
                dtype=dtype,
            )
        )
        for _, module in self._calibration_functional_modules:
            cal_conv = cast(FunctionalStatefulConv1d, module)
            states.append(cal_conv.init_state(batch_size=batch_size, device=device, dtype=dtype))
        for _, module in self._alpha_functional_modules:
            alpha_conv = cast(FunctionalStatefulCausalConv2d, module)
            states.append(alpha_conv.init_state(batch_size=batch_size, freq_size=freq_size, device=device, dtype=dtype))
        return states

    def get_state_shapes(
        self,
        batch_size: int = 1,
        freq_size: Optional[int] = None,
    ) -> List[Tuple[int, ...]]:
        """Return state shapes in :meth:`get_state_names` order (calls :meth:`init_states`)."""
        return [tuple(s.shape) for s in self.init_states(batch_size=batch_size, freq_size=freq_size)]

    # ----------------------------------------------------------------- metadata
    def metadata(self, *, chunk_size: int = 8, freq_size: Optional[int] = None) -> Dict[str, Any]:
        """Snapshot the functional core's I/O contract + frame geometry.

        Used by S8 export (sidecar JSON) and S9 / S11 binders. Mirrors S5's
        :meth:`BAFNetPlusCore.metadata` schema with the following S8 deltas:

        - ``schema_version='s8-bafnetplus-functional-fp32'``.
        - ``num_states`` now includes calibration + alpha (190 for the real 50
          ms ckpt; 110 for the synthetic ``num_tsblock=2`` test config).
        - ``state_names`` are extended with ``calibration/*`` (2) and
          ``alpha/*`` (4).
        - ``state_shapes`` populated for every state.
        - ``branches`` gains a ``fusion`` sub-dict with per-component
          ``num_states`` + ``state_shapes``.

        Args:
            chunk_size: Output chunk size; used to derive ``t_export``.
                Anchor for 50 ms BAFNet+ is ``chunk_size = 8``.
            freq_size: ``n_fft // 2 + 1`` if ``None``.
        """
        if freq_size is None:
            freq_size = self.n_fft // 2 + 1
        l_enc_m = self.mapping_core.encoder_lookahead
        l_dec_m = self.mapping_core.decoder_lookahead
        if self.masking_core.encoder_lookahead != l_enc_m or self.masking_core.decoder_lookahead != l_dec_m:
            logger.warning(
                "metadata: mapping/masking lookahead mismatch (enc=%d/%d, dec=%d/%d) — using the maximum",
                l_enc_m,
                self.masking_core.encoder_lookahead,
                l_dec_m,
                self.masking_core.decoder_lookahead,
            )
        l_enc = max(l_enc_m, self.masking_core.encoder_lookahead)
        l_dec = max(l_dec_m, self.masking_core.decoder_lookahead)
        alpha_lookahead = audit_alpha_time_lookahead(self.alpha_convblocks)
        t_export = int(chunk_size) + l_enc + l_dec + alpha_lookahead

        shapes = self.get_state_shapes(batch_size=1, freq_size=freq_size)
        n_map = self.mapping_core.num_states
        n_mask = self.masking_core.num_states
        n_cal = self.num_calibration_states
        n_alpha = self.num_alpha_states
        cal_shapes = [list(s) for s in shapes[n_map + n_mask : n_map + n_mask + n_cal]]
        alpha_shapes = [list(s) for s in shapes[n_map + n_mask + n_cal :]]

        return {
            "schema_version": "s8-bafnetplus-functional-fp32",
            "ablation_mode": self.ablation_mode,
            "use_calibration": self.use_calibration,
            "use_relative_gain": self.use_relative_gain,
            "mask_only_alpha": self.mask_only_alpha,
            "calibration_max_common_log_gain": self.calibration_max_common_log_gain,
            "calibration_max_relative_log_gain": self.calibration_max_relative_log_gain,
            "phase_output_mode": self.phase_output_mode,
            "n_fft": self.n_fft,
            "freq_size": int(freq_size),
            "chunk_size": int(chunk_size),
            "encoder_lookahead": int(l_enc),
            "decoder_lookahead": int(l_dec),
            "total_lookahead": int(l_enc + l_dec),
            "alpha_time_lookahead": int(alpha_lookahead),
            "t_export": int(t_export),
            "t_export_formula": "chunk_size + encoder_lookahead + decoder_lookahead + alpha_time_lookahead",
            "state_frames_for_update": self.state_frames_for_update,
            "num_states": self.num_states,
            "input_names": self.input_names(),
            "output_names": self.output_names(),
            "state_names": self.get_state_names(),
            "state_shapes": [list(s) for s in shapes],
            "state_order": "mapping/* masking/* calibration/* alpha/*",
            "branches": {
                "mapping": self.mapping_core.metadata(freq_size=freq_size),
                "masking": self.masking_core.metadata(freq_size=freq_size),
                "fusion": {
                    "calibration": {
                        "num_states": n_cal,
                        "state_shapes": cal_shapes,
                    },
                    "alpha": {
                        "num_states": n_alpha,
                        "state_shapes": alpha_shapes,
                    },
                },
            },
        }

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        bcs_mag: Tensor,
        bcs_pha: Tensor,
        acs_mag: Tensor,
        acs_pha: Tensor,
        *prev_states: Tensor,
    ) -> Tuple[Tensor, ...]:
        """Run the explicit-state BAFNet+ forward.

        Args:
            bcs_mag: BCS compressed magnitude ``[B, F, T]``.
            bcs_pha: BCS phase ``[B, F, T]``.
            acs_mag: ACS compressed magnitude ``[B, F, T]`` — must be strictly
                positive (the host's :func:`src.stft.mag_pha_stft` provides the
                ``sqrt(... + 1e-9)`` floor; the export driver's dummy inputs
                add a tiny epsilon for the trace).
            acs_pha: ACS phase ``[B, F, T]``.
            *prev_states: ``num_states`` prev-state tensors in
                :meth:`get_state_names` order.

        Returns:
            Atan2 mode: ``(est_mag [B, F, T], est_pha [B, F, T], est_com
            [B, F, T, 2], *next_states)`` — the fused complex spectrogram +
            its in-graph atan2 phase + sqrt magnitude.

            Complex mode (S20 atan2-free): ``(est_mag [B, F, T], phase_real
            [B, F, T], phase_imag [B, F, T], *next_states)`` — the fused
            complex spectrogram components emitted directly. atan2 is
            relocated to the host wrapper (FP32). Eliminates both per-branch
            and fused atan2 from the ONNX graph; the only remaining INT8-
            hostile op of the S17 cluster is ``softmax`` in the alpha fusion.

        Raises:
            ValueError: If ``len(prev_states) != self.num_states``.
        """
        total = self.num_states
        if len(prev_states) != total:
            raise ValueError(f"forward: expected {total} prev_states, got {len(prev_states)}")

        n_map = self.mapping_core.num_states
        n_mask = self.masking_core.num_states
        n_cal = self.num_calibration_states

        map_states = prev_states[:n_map]
        mask_states = prev_states[n_map : n_map + n_mask]
        cal_states = prev_states[n_map + n_mask : n_map + n_mask + n_cal]
        alpha_states = prev_states[n_map + n_mask + n_cal :]

        # ----- two backbones with explicit state I/O
        # S9.6: the masking branch is routed through ``forward_with_mask`` so
        # the raw mask (post-LearnableSigmoid, pre-multiplication) is returned
        # explicitly — 1-ULP equivalent to ``BAFNetPlus.forward``'s
        # ``self.masking(..., return_mask=True)`` consumption. Pre-S9.6 we
        # recovered ``acs_mask = acs_est_mag / acs_mag`` (a*b)/a, which drifts
        # by 1 ULP and propagates through the calibration / alpha softmax
        # blend; the explicit mask closes that drift at the source. Mapping
        # branch keeps the 3-output path (bcs_mask is never consumed).
        bcs_est_mag, bcs_est_com, _bcs_mask, map_next_states = self._run_branch(
            self.mapping_core, bcs_mag, bcs_pha, map_states, expose_mask=False
        )
        acs_est_mag, acs_est_com, acs_mask, mask_next_states = self._run_branch(
            self.masking_core, acs_mag, acs_pha, mask_states, expose_mask=True
        )
        # acs_mask is non-None here because expose_mask=True; the explicit
        # assertion would be redundant at trace time, so we just rely on the
        # tuple unpack above.

        # ----- calibration with functional state routing
        cal_iter = StateIterator(list(cal_states))
        alpha_iter = StateIterator(list(alpha_states))
        if self.use_calibration:
            calibration_feat = _build_calibration_features(bcs_est_mag, acs_est_mag, acs_mask)
            calibration_hidden = self._forward_calibration_encoder(calibration_feat, cal_iter)
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

        # ----- alpha fusion with functional state routing
        if self.mask_only_alpha:
            alpha = acs_mask.unsqueeze(1).transpose(2, 3)  # [B, 1, T, F]
        else:
            alpha = _build_alpha_features(bcs_com_fused, acs_com_fused, acs_mask)
        alpha = self._forward_alpha_convblocks(alpha, alpha_iter)
        alpha = self.alpha_out(alpha)
        alpha = alpha.transpose(2, 3)
        alpha = torch.softmax(alpha, dim=1)
        alpha_bcs = alpha[:, 0].unsqueeze(-1)
        alpha_acs = alpha[:, 1].unsqueeze(-1)
        est_com = bcs_com_fused * alpha_bcs + acs_com_fused * alpha_acs

        # Sanity: the state iterators must have consumed exactly the states we sliced.
        if cal_iter.consumed_count != n_cal:
            raise RuntimeError(f"calibration StateIterator consumed {cal_iter.consumed_count} of {n_cal} states")
        if alpha_iter.consumed_count != self.num_alpha_states:
            raise RuntimeError(
                f"alpha StateIterator consumed {alpha_iter.consumed_count} of {self.num_alpha_states} states"
            )

        all_next_states: List[Tensor] = (
            list(map_next_states) + list(mask_next_states) + list(cal_iter.next_states) + list(alpha_iter.next_states)
        )

        if self.phase_output_mode == "complex":
            # S20 atan2-free output: emit fused complex components directly.
            # est_mag uses the same sqrt formula complex_to_mag_pha would use
            # (sqrt(real^2 + imag^2 + 1e-8)) so consumers that need a magnitude
            # see no scale change.
            fused_real = est_com[..., 0]
            fused_imag = est_com[..., 1]
            est_mag = torch.sqrt(fused_real**2 + fused_imag**2 + 1e-8)
            return (est_mag, fused_real, fused_imag) + tuple(all_next_states)

        est_mag, est_pha = complex_to_mag_pha(est_com)
        return (est_mag, est_pha, est_com) + tuple(all_next_states)

    def _run_branch(
        self,
        core: ExportableBackboneCore,
        mag: Tensor,
        pha: Tensor,
        prev_states: Tuple[Tensor, ...],
        *,
        expose_mask: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], List[Tensor]]:
        """Run one branch core with the supplied prev-states; return
        ``(est_mag, est_com, raw_mask_or_None, next_states_list)``.

        Handles both ``phase_output_mode`` settings of the underlying core:

        - ``atan2``: backbone returns ``(est_mag, est_pha, est_com,
          *next_states)`` — we take ``est_mag`` and ``est_com`` directly.
        - ``complex`` (S20 atan2-free): backbone returns ``(est_mag,
          phase_real, phase_imag, *next_states)`` — we reconstruct ``est_com``
          via the algebraic identity
          ``mag_pha_to_complex(mag, atan2(im+eps, re+eps))
          = mag * (re+eps, im+eps) / sqrt((re+eps)^2 + (im+eps)^2)``,
          keeping ``atan2`` OUT of the per-branch graph compute. The fused
          output is also atan2-free in complex mode (see :meth:`forward`).

        ``expose_mask=True`` routes the call through
        :meth:`ExportableBackboneCore.forward_with_mask` so the raw
        ``mask_decoder`` output (the same tensor ``Backbone.forward(...,
        return_mask=True)`` returns, post-LearnableSigmoid, pre-multiplication)
        is returned alongside ``est_mag`` / ``est_com``. The outer
        :meth:`forward` uses this on the masking branch to skip the
        ``acs_mask = est_mag / mag`` recovery and its (a*b)/a 1-ULP drift.
        """
        if expose_mask:
            outs = core.forward_with_mask(mag, pha, *prev_states)
            est_mag = outs[0]
            if core.phase_output_mode == "complex":
                est_com = _phase_components_to_complex(est_mag, outs[1], outs[2])
            else:
                est_com = outs[2]
            mask = outs[3]
            next_states = list(outs[4:])
            return est_mag, est_com, mask, next_states

        outs = core(mag, pha, *prev_states)
        est_mag = outs[0]
        if core.phase_output_mode == "complex":
            est_com = _phase_components_to_complex(est_mag, outs[1], outs[2])
        else:
            est_com = outs[2]
        next_states = list(outs[3:])
        return est_mag, est_com, None, next_states

    def _forward_calibration_encoder(self, x: Tensor, state_iter: StateIterator) -> Tensor:
        """Run the calibration encoder with functional state routing.

        Structure: ``nn.Sequential( nn.Sequential( FunctionalStatefulConv1d,
        PReLU ) )^depth``. Walks the inner sequential and re-routes each
        ``FunctionalStatefulConv1d`` through ``state_iter`` (state-frames
        bound is propagated from :attr:`state_frames_for_update`).
        """
        cal_enc = cast(nn.Sequential, self.calibration_encoder)
        for block in cal_enc:
            inner = cast(nn.Sequential, block)
            for layer in inner:
                if isinstance(layer, FunctionalStatefulConv1d):
                    x = state_iter.run_layer(layer, x, state_frames=self.state_frames_for_update)
                else:
                    x = layer(x)
        return x

    def _forward_alpha_convblocks(self, x: Tensor, state_iter: StateIterator) -> Tensor:
        """Run the alpha conv stack with functional state routing.

        Structure: ``nn.ModuleList( nn.Sequential( FunctionalStatefulCausalConv2d,
        BatchNorm2d, PReLU ) )^depth``. Walks each block and re-routes the
        ``FunctionalStatefulCausalConv2d`` leaf through ``state_iter``.
        """
        for block in self.alpha_convblocks:
            inner = cast(nn.Sequential, block)
            for layer in inner:
                if isinstance(layer, FunctionalStatefulCausalConv2d):
                    x = state_iter.run_layer(layer, x, state_frames=self.state_frames_for_update)
                else:
                    x = layer(x)
        return x

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ablation_mode={self.ablation_mode!r}, "
            f"use_calibration={self.use_calibration}, use_relative_gain={self.use_relative_gain}, "
            f"mask_only_alpha={self.mask_only_alpha}, "
            f"phase_output_mode={self.phase_output_mode!r}, "
            f"num_states={self.num_states} "
            f"(mapping={self.mapping_core.num_states}, masking={self.masking_core.num_states}, "
            f"calibration={self.num_calibration_states}, alpha={self.num_alpha_states}), "
            f"total_lookahead={self.total_lookahead}, "
            f"state_frames_for_update={self.state_frames_for_update})"
        )


# ============================================================ S21 split core
#
# B2 graph-splitting design — see ``docs/wiki/projects/bafnetplus.md § S21``.
# Splits ``ExportableBAFNetPlusCore`` (S8/S20) into two graphs:
#
# * :class:`BAFNetPlusTrunkCore` — two backbone branches in
#   ``phase_output_mode='complex'`` (no in-graph atan2/sqrt/softmax). Emits the
#   per-branch raw phase components + the explicit ``acs_mask``. INT8-safe
#   (S18(A) confirmed ConvTranspose2d outputs at FP32 noise floor); designed
#   to be PTQ-quantized in a B2 deployable.
# * :class:`BAFNetPlusHeadCore` — calibration encoder + alpha softmax fusion +
#   final ``complex_to_mag_pha`` atan2. Stays FP32; carries the genuinely
#   INT8-hostile cluster (sqrt-based per-branch ``est_com`` reconstruction,
#   softmax, atan2) on the FP32 side.
#
# State partition (frozen 190-state order):
#   * Trunk: 92 ``mapping/*`` + 92 ``masking/*`` = 184 states.
#   * Head: 2 ``calibration/*`` + 4 ``alpha/*`` = 6 states.
#
# Boundary contract (trunk outputs → head inputs):
#   ``bcs_est_mag``, ``bcs_phase_real``, ``bcs_phase_imag``,
#   ``acs_est_mag``, ``acs_phase_real``, ``acs_phase_imag``, ``acs_mask``
#   — 7 tensors of shape ``[B, F, T]`` each.


class BAFNetPlusTrunkCore(nn.Module):
    """Trunk subgraph of the S21 split BAFNet+ export.

    Bundles the two ``ExportableBackboneCore`` branches in
    ``phase_output_mode='complex'`` and exposes a 7-tensor boundary suitable
    for chaining into a separate FP32 head. The trunk subgraph contains
    *only* ConvTranspose2d / Conv / BN / PReLU / LearnableSigmoid_2d /
    LayerNorm ops — the INT8-hostile cluster (atan2 / softmax / sqrt-based
    phase reconstruction) is deferred to :class:`BAFNetPlusHeadCore`.

    Why complex mode? The trunk's job is to feed a clean boundary to the
    FP32 head. Emitting ``(est_mag, phase_real, phase_imag)`` per branch
    keeps atan2 out of the trunk; the sqrt-based ``mag * (re, im) / sqrt(...)``
    reconstruction happens inside the head (FP32), not inside the trunk.

    I/O contract (frozen for the S21 split sidecar)
    -----------------------------------------------
    Inputs (in order):
        ``bcs_mag [B, F, T]``, ``bcs_pha [B, F, T]``,
        ``acs_mag [B, F, T]``, ``acs_pha [B, F, T]``,
        then state tensors in :meth:`get_state_names` order —
        ``mapping/state_<i>_<...>`` × 92 then ``masking/state_<i>_<...>`` × 92
        (the same per-branch names :class:`ExportableBackboneCore` emits,
        prefixed by branch).
    Outputs (in order):
        ``bcs_est_mag [B, F, T]``,
        ``bcs_phase_real [B, F, T]``, ``bcs_phase_imag [B, F, T]``,
        ``acs_est_mag [B, F, T]``,
        ``acs_phase_real [B, F, T]``, ``acs_phase_imag [B, F, T]``,
        ``acs_mask [B, F, T]``  (post-LearnableSigmoid, pre-multiplication —
        the same tensor :meth:`ExportableBackboneCore.forward_with_mask`
        appends),
        then ``next_<state_name>`` per state.

    Attributes:
        mapping_core: Functional :class:`ExportableBackboneCore`
            (``infer_type='mapping'``, ``phase_output_mode='complex'``).
        masking_core: Functional :class:`ExportableBackboneCore`
            (``infer_type='masking'``, ``phase_output_mode='complex'``).
    """

    def __init__(
        self,
        mapping_core: ExportableBackboneCore,
        masking_core: ExportableBackboneCore,
    ) -> None:
        super().__init__()
        if mapping_core.infer_type != "mapping":
            raise ValueError(f"mapping_core.infer_type must be 'mapping', got {mapping_core.infer_type!r}")
        if masking_core.infer_type != "masking":
            raise ValueError(f"masking_core.infer_type must be 'masking', got {masking_core.infer_type!r}")
        if mapping_core.phase_output_mode != "complex":
            raise ValueError(
                "BAFNetPlusTrunkCore requires mapping_core.phase_output_mode='complex', "
                f"got {mapping_core.phase_output_mode!r}"
            )
        if masking_core.phase_output_mode != "complex":
            raise ValueError(
                "BAFNetPlusTrunkCore requires masking_core.phase_output_mode='complex', "
                f"got {masking_core.phase_output_mode!r}"
            )
        self.mapping_core = mapping_core
        self.masking_core = masking_core

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_bafnetplus(cls, model: BAFNetPlus) -> "BAFNetPlusTrunkCore":
        """Build the trunk from an instantiated :class:`BAFNetPlus`.

        Each branch is deep-copied + stateful-converted + functional-converted
        in ``phase_output_mode='complex'`` (via
        :meth:`ExportableBackboneCore.from_backbone`, which always swaps every
        TSBlock to its reshape-free 4D variant). The source bafnet is
        unchanged.

        Args:
            model: An instantiated :class:`BAFNetPlus` whose ``mapping`` /
                ``masking`` branches are still original ``Backbone`` instances.

        Returns:
            A ready :class:`BAFNetPlusTrunkCore`.

        Raises:
            TypeError: If ``model`` is not a :class:`BAFNetPlus`.
        """
        if not isinstance(model, BAFNetPlus):
            raise TypeError(f"from_bafnetplus expects a BAFNetPlus, got {type(model).__name__}")
        mapping_core = ExportableBackboneCore.from_backbone(
            model.mapping,
            phase_output_mode="complex",
        )
        masking_core = ExportableBackboneCore.from_backbone(
            model.masking,
            phase_output_mode="complex",
        )
        return cls(mapping_core, masking_core)

    # ------------------------------------------------------------------ state ops
    @property
    def n_fft(self) -> int:
        """STFT size (mirrors mapping_core's ``n_fft``; both branches share this)."""
        return self.mapping_core.n_fft

    @property
    def num_states(self) -> int:
        """Total state count = mapping (92) + masking (92) = 184 for the 50 ms anchor."""
        return self.mapping_core.num_states + self.masking_core.num_states

    @property
    def total_lookahead(self) -> int:
        """Conservative monolithic-graph lookahead: max across the two branches'
        ``L_enc + L_dec``."""
        return max(self.mapping_core.total_lookahead, self.masking_core.total_lookahead)

    def set_state_frames_for_update(self, state_frames: Optional[int]) -> None:
        """Propagate the state-update window bound to both backbone cores."""
        self.mapping_core.set_state_frames_for_update(state_frames)
        self.masking_core.set_state_frames_for_update(state_frames)

    # ------------------------------------------------------------------- naming
    def get_state_names(self) -> List[str]:
        """State names in the frozen sidecar order: ``mapping/*`` (92) then ``masking/*`` (92)."""
        names: List[str] = []
        for n in self.mapping_core.get_state_names():
            names.append(f"mapping/{n}")
        for n in self.masking_core.get_state_names():
            names.append(f"masking/{n}")
        return names

    def input_names(self) -> List[str]:
        """Frozen ONNX input names: ``[bcs_mag, bcs_pha, acs_mag, acs_pha, *state_names]``."""
        return ["bcs_mag", "bcs_pha", "acs_mag", "acs_pha"] + self.get_state_names()

    def output_names(self) -> List[str]:
        """Frozen ONNX output names: 7 boundary tensors + ``next_<state>`` per state."""
        nexts = [f"next_{n}" for n in self.get_state_names()]
        return [
            "bcs_est_mag",
            "bcs_phase_real",
            "bcs_phase_imag",
            "acs_est_mag",
            "acs_phase_real",
            "acs_phase_imag",
            "acs_mask",
        ] + nexts

    # ---------------------------------------------------------------- init shapes
    def init_states(
        self,
        batch_size: int = 1,
        freq_size: Optional[int] = None,
        time_frames: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tensor]:
        """Zero-init every state tensor in :meth:`get_state_names` order."""
        if freq_size is None:
            freq_size = self.n_fft // 2 + 1
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        states: List[Tensor] = []
        states.extend(
            self.mapping_core.init_states(
                batch_size=batch_size,
                freq_size=freq_size,
                time_frames=time_frames,
                device=device,
                dtype=dtype,
            )
        )
        states.extend(
            self.masking_core.init_states(
                batch_size=batch_size,
                freq_size=freq_size,
                time_frames=time_frames,
                device=device,
                dtype=dtype,
            )
        )
        return states

    def get_state_shapes(
        self,
        batch_size: int = 1,
        freq_size: Optional[int] = None,
    ) -> List[Tuple[int, ...]]:
        """Return state shapes in :meth:`get_state_names` order."""
        return [tuple(s.shape) for s in self.init_states(batch_size=batch_size, freq_size=freq_size)]

    # ----------------------------------------------------------------- metadata
    def metadata(self, *, chunk_size: int = 8, freq_size: Optional[int] = None) -> Dict[str, Any]:
        """Snapshot the trunk subgraph's I/O contract + frame geometry.

        The trunk has no alpha conv stack — its ``T_export`` reduces to
        ``chunk_size + encoder_lookahead + decoder_lookahead``.
        """
        if freq_size is None:
            freq_size = self.n_fft // 2 + 1
        l_enc_m = self.mapping_core.encoder_lookahead
        l_dec_m = self.mapping_core.decoder_lookahead
        if self.masking_core.encoder_lookahead != l_enc_m or self.masking_core.decoder_lookahead != l_dec_m:
            logger.warning(
                "trunk metadata: mapping/masking lookahead mismatch (enc=%d/%d, dec=%d/%d) — using the maximum",
                l_enc_m,
                self.masking_core.encoder_lookahead,
                l_dec_m,
                self.masking_core.decoder_lookahead,
            )
        l_enc = max(l_enc_m, self.masking_core.encoder_lookahead)
        l_dec = max(l_dec_m, self.masking_core.decoder_lookahead)
        t_export = int(chunk_size) + l_enc + l_dec
        shapes = self.get_state_shapes(batch_size=1, freq_size=freq_size)
        return {
            "schema_version": "s21-bafnetplus-trunk-fp32",
            "phase_output_mode": "complex",
            "n_fft": self.n_fft,
            "freq_size": int(freq_size),
            "chunk_size": int(chunk_size),
            "encoder_lookahead": int(l_enc),
            "decoder_lookahead": int(l_dec),
            "total_lookahead": int(l_enc + l_dec),
            "t_export": int(t_export),
            "t_export_formula": "chunk_size + encoder_lookahead + decoder_lookahead",
            "num_states": self.num_states,
            "input_names": self.input_names(),
            "output_names": self.output_names(),
            "state_names": self.get_state_names(),
            "state_shapes": [list(s) for s in shapes],
            "state_order": "mapping/* masking/*",
            "branches": {
                "mapping": self.mapping_core.metadata(freq_size=freq_size),
                "masking": self.masking_core.metadata(freq_size=freq_size),
            },
        }

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        bcs_mag: Tensor,
        bcs_pha: Tensor,
        acs_mag: Tensor,
        acs_pha: Tensor,
        *prev_states: Tensor,
    ) -> Tuple[Tensor, ...]:
        """Run both backbone branches and return the 7-tensor boundary.

        Args:
            bcs_mag / bcs_pha / acs_mag / acs_pha: ``[B, F, T]`` spectrograms.
            *prev_states: ``num_states`` prev-state tensors in
                :meth:`get_state_names` order.

        Returns:
            ``(bcs_est_mag, bcs_phase_real, bcs_phase_imag,
              acs_est_mag, acs_phase_real, acs_phase_imag, acs_mask,
              *next_states)`` — same order as :meth:`output_names`.

        Raises:
            ValueError: If ``len(prev_states) != self.num_states``.
        """
        n_map = self.mapping_core.num_states
        n_mask = self.masking_core.num_states
        if len(prev_states) != n_map + n_mask:
            raise ValueError(
                f"BAFNetPlusTrunkCore.forward: expected {n_map + n_mask} prev_states "
                f"(mapping={n_map} + masking={n_mask}), got {len(prev_states)}"
            )
        map_states = prev_states[:n_map]
        mask_states = prev_states[n_map:]

        # Mapping branch (complex mode) returns ``(est_mag, phase_real, phase_imag, *next_states)``.
        map_outs = self.mapping_core(bcs_mag, bcs_pha, *map_states)
        bcs_est_mag = map_outs[0]
        bcs_phase_real = map_outs[1]
        bcs_phase_imag = map_outs[2]
        map_next_states = list(map_outs[3:])

        # Masking branch w/ explicit mask (complex mode) returns
        # ``(est_mag, phase_real, phase_imag, mask, *next_states)``.
        mask_outs = self.masking_core.forward_with_mask(acs_mag, acs_pha, *mask_states)
        acs_est_mag = mask_outs[0]
        acs_phase_real = mask_outs[1]
        acs_phase_imag = mask_outs[2]
        acs_mask = mask_outs[3]
        mask_next_states = list(mask_outs[4:])

        return (
            bcs_est_mag,
            bcs_phase_real,
            bcs_phase_imag,
            acs_est_mag,
            acs_phase_real,
            acs_phase_imag,
            acs_mask,
        ) + tuple(map_next_states) + tuple(mask_next_states)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_states={self.num_states} "
            f"(mapping={self.mapping_core.num_states}, masking={self.masking_core.num_states}), "
            f"total_lookahead={self.total_lookahead})"
        )


class BAFNetPlusHeadCore(nn.Module):
    """Head subgraph of the S21 split BAFNet+ export.

    Consumes the 7-tensor boundary emitted by :class:`BAFNetPlusTrunkCore`
    (per-branch ``est_mag`` + raw phase components + explicit ``acs_mask``),
    runs the calibration encoder + alpha softmax fusion, and returns the
    final atan2 triple ``(est_mag, est_pha, est_com)``. The head subgraph
    carries every INT8-hostile op identified in S18(A): per-branch sqrt-based
    ``est_com`` reconstruction (via :func:`_phase_components_to_complex`),
    softmax fusion, and the final ``complex_to_mag_pha`` atan2. Designed to
    stay FP32 in the deployable split asset; the trunk is the only graph
    that gets PTQ-quantized.

    I/O contract (frozen for the S21 split sidecar)
    -----------------------------------------------
    Inputs (in order):
        ``bcs_est_mag [B, F, T]``,
        ``bcs_phase_real [B, F, T]``, ``bcs_phase_imag [B, F, T]``,
        ``acs_est_mag [B, F, T]``,
        ``acs_phase_real [B, F, T]``, ``acs_phase_imag [B, F, T]``,
        ``acs_mask [B, F, T]``,
        then state tensors in :meth:`get_state_names` order —
        ``calibration/state_<i>_<...>`` × 2 then ``alpha/state_<i>_<...>`` × 4.
    Outputs (in order):
        ``est_mag [B, F, T]``, ``est_pha [B, F, T]``, ``est_com [B, F, T, 2]``
        (the canonical S8/S17 atan2 triple — matched to the existing host
        wrapper consumer), then ``next_<state_name>`` per state.

    Attributes:
        calibration_encoder: Deep-copied + functional-converted from the
            source BAFNet+; ``None`` for ``ablation_mode='no_calibration'``.
        common_gain_head / relative_gain_head: 1×1 ``nn.Conv1d`` heads
            (stateless deep-copies of the source).
        alpha_convblocks: Deep-copied + functional-converted alpha conv
            stack (``nn.ModuleList`` of ``nn.Sequential(FunctionalStatefulCausalConv2d,
            BatchNorm2d, PReLU)``).
        alpha_out: 1×1 ``nn.Conv2d`` output head (stateless deep-copy).
        ablation_mode / use_calibration / use_relative_gain / mask_only_alpha /
            calibration_max_common_log_gain /
            calibration_max_relative_log_gain: Mirror the source BAFNet+
            instance's algebra flags / tanh-scaling factors.
        n_fft: STFT size (used to derive ``freq_size`` for state init geometry).
    """

    def __init__(
        self,
        *,
        calibration_encoder: Optional[nn.Module],
        common_gain_head: Optional[nn.Module],
        relative_gain_head: Optional[nn.Module],
        alpha_convblocks: nn.ModuleList,
        alpha_out: nn.Module,
        ablation_mode: str,
        use_calibration: bool,
        use_relative_gain: bool,
        mask_only_alpha: bool,
        calibration_max_common_log_gain: float,
        calibration_max_relative_log_gain: float,
        n_fft: int,
    ) -> None:
        super().__init__()
        if ablation_mode not in BAFNetPlus.VALID_ABLATION_MODES:
            raise ValueError(f"ablation_mode must be one of {BAFNetPlus.VALID_ABLATION_MODES}, got {ablation_mode!r}")
        if use_calibration and calibration_encoder is None:
            raise ValueError("use_calibration=True but calibration_encoder is None")
        if use_calibration and common_gain_head is None:
            raise ValueError("use_calibration=True but common_gain_head is None")
        if use_calibration and use_relative_gain and relative_gain_head is None:
            raise ValueError("use_relative_gain=True but relative_gain_head is None")

        if calibration_encoder is not None:
            self.calibration_encoder = calibration_encoder
        else:
            self.calibration_encoder = None  # type: ignore[assignment]
        if common_gain_head is not None:
            self.common_gain_head = common_gain_head
        else:
            self.common_gain_head = None  # type: ignore[assignment]
        if relative_gain_head is not None:
            self.relative_gain_head = relative_gain_head
        else:
            self.relative_gain_head = None  # type: ignore[assignment]
        self.alpha_convblocks = alpha_convblocks
        self.alpha_out = alpha_out

        self.ablation_mode: str = ablation_mode
        self.use_calibration: bool = bool(use_calibration)
        self.use_relative_gain: bool = bool(use_relative_gain)
        self.mask_only_alpha: bool = bool(mask_only_alpha)
        self.calibration_max_common_log_gain: float = float(calibration_max_common_log_gain)
        self.calibration_max_relative_log_gain: float = float(calibration_max_relative_log_gain)
        self.n_fft: int = int(n_fft)

        # Collect functional stateful modules in DFS order. Calibration first, then alpha
        # — matches the frozen state-name order (calibration/* then alpha/*).
        self._calibration_functional_modules: List[Tuple[str, nn.Module]] = []
        self._alpha_functional_modules: List[Tuple[str, nn.Module]] = []
        if self.calibration_encoder is not None:
            for name, module in self.calibration_encoder.named_modules():
                if isinstance(module, FunctionalStatefulConv1d):
                    self._calibration_functional_modules.append((name, cast(nn.Module, module)))
        for name, module in self.alpha_convblocks.named_modules():
            if isinstance(module, FunctionalStatefulCausalConv2d):
                self._alpha_functional_modules.append((name, cast(nn.Module, module)))

        self.state_frames_for_update: Optional[int] = None

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "BAFNetPlusHeadCore: %d calibration + %d alpha = %d states",
                self.num_calibration_states,
                self.num_alpha_states,
                self.num_states,
            )

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_bafnetplus(cls, model: BAFNetPlus) -> "BAFNetPlusHeadCore":
        """Build the head from an instantiated :class:`BAFNetPlus`.

        Deep-copies the fusion path (calibration encoder / gain heads / alpha
        conv stack / ``alpha_out``) and ``convert_to_stateful`` →
        ``convert_stateful_to_functional`` the calibration encoder and alpha
        conv stack in place on the deep-copy. The 1×1 gain heads + ``alpha_out``
        are deep-copied without stateful conversion (kernel=1 → no state).
        The source bafnet is unchanged.

        Args:
            model: An instantiated :class:`BAFNetPlus`.

        Returns:
            A ready :class:`BAFNetPlusHeadCore`.

        Raises:
            TypeError: If ``model`` is not a :class:`BAFNetPlus`.
        """
        if not isinstance(model, BAFNetPlus):
            raise TypeError(f"from_bafnetplus expects a BAFNetPlus, got {type(model).__name__}")

        if model.use_calibration:
            calibration_encoder = copy.deepcopy(model.calibration_encoder)
            convert_to_stateful(calibration_encoder, verbose=False, inplace=True)
            convert_stateful_to_functional(calibration_encoder)
            common_gain_head = copy.deepcopy(model.common_gain_head)
            if model.use_relative_gain:
                relative_gain_head = copy.deepcopy(model.relative_gain_head)
            else:
                relative_gain_head = None
        else:
            calibration_encoder = None
            common_gain_head = None
            relative_gain_head = None

        alpha_convblocks = copy.deepcopy(model.alpha_convblocks)
        convert_to_stateful(alpha_convblocks, verbose=False, inplace=True)
        convert_stateful_to_functional(alpha_convblocks)
        alpha_out = copy.deepcopy(model.alpha_out)

        n_fft = int(model.mapping.n_fft) if model.mapping.n_fft is not None else 0
        if n_fft == 0:
            raise ValueError("model.mapping.n_fft is not set")

        return cls(
            calibration_encoder=calibration_encoder,
            common_gain_head=common_gain_head,
            relative_gain_head=relative_gain_head,
            alpha_convblocks=alpha_convblocks,
            alpha_out=alpha_out,
            ablation_mode=model.ablation_mode,
            use_calibration=model.use_calibration,
            use_relative_gain=model.use_relative_gain,
            mask_only_alpha=model.mask_only_alpha,
            calibration_max_common_log_gain=model.calibration_max_common_log_gain,
            calibration_max_relative_log_gain=model.calibration_max_relative_log_gain,
            n_fft=n_fft,
        )

    # ------------------------------------------------------------- state counts
    @property
    def num_calibration_states(self) -> int:
        """Number of calibration functional stateful convs (0 for ``no_calibration``)."""
        return len(self._calibration_functional_modules)

    @property
    def num_alpha_states(self) -> int:
        """Number of alpha functional stateful convs (= ``conv_depth``)."""
        return len(self._alpha_functional_modules)

    @property
    def num_states(self) -> int:
        """Total state count = calibration + alpha (= 6 for the canonical 50 ms anchor)."""
        return self.num_calibration_states + self.num_alpha_states

    @property
    def total_lookahead(self) -> int:
        """Head is causal: calibration uses ``CausalConv1d`` (no lookahead) + alpha uses
        ``CausalConv2d`` with ``bottom_time=0`` padding (no lookahead). The audited
        ``audit_alpha_time_lookahead`` value (= 0 for the deployed unified ckpt) is
        reported in :meth:`metadata` for completeness.
        """
        return audit_alpha_time_lookahead(self.alpha_convblocks)

    # ------------------------------------------------------------- state-frames
    def set_state_frames_for_update(self, state_frames: Optional[int]) -> None:
        """Bound how many leading input frames may update the head's conv states.

        Stored on ``self`` for the functional convs in calibration_encoder and
        alpha_convblocks to pick up at call time.
        """
        self.state_frames_for_update = None if state_frames is None else int(state_frames)

    # ------------------------------------------------------------------- naming
    def get_state_names(self) -> List[str]:
        """State names in the frozen sidecar order: ``calibration/*`` (2) then ``alpha/*`` (4)."""
        names: List[str] = []
        for i, (name, _) in enumerate(self._calibration_functional_modules):
            names.append(f"calibration/state_{i}_{name.replace('.', '_')}")
        for i, (name, _) in enumerate(self._alpha_functional_modules):
            names.append(f"alpha/state_{i}_{name.replace('.', '_')}")
        return names

    def input_names(self) -> List[str]:
        """Frozen ONNX input names: 7 boundary tensors + state names."""
        return [
            "bcs_est_mag",
            "bcs_phase_real",
            "bcs_phase_imag",
            "acs_est_mag",
            "acs_phase_real",
            "acs_phase_imag",
            "acs_mask",
        ] + self.get_state_names()

    def output_names(self) -> List[str]:
        """Frozen ONNX output names: ``[est_mag, est_pha, est_com, *next_state_names]``.

        Head stays in atan2 mode (no INT8 quantization pressure on the head — atan2 +
        softmax stay FP32 by construction, which is the whole point of the B2 split).
        """
        nexts = [f"next_{n}" for n in self.get_state_names()]
        return ["est_mag", "est_pha", "est_com"] + nexts

    # ---------------------------------------------------------------- init shapes
    def init_states(
        self,
        batch_size: int = 1,
        freq_size: Optional[int] = None,
        time_frames: int = 64,  # noqa: ARG002 — kept for API parity with trunk + ExportableBAFNetPlusCore
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tensor]:
        """Zero-init every state tensor in :meth:`get_state_names` order.

        Calibration ``FunctionalStatefulConv1d`` state shape: ``[B, in_channels, padding_size]``.
        Alpha ``FunctionalStatefulCausalConv2d`` state shape:
        ``[B, in_channels, time_padding, freq_size + 2 * freq_padding]``.
        """
        if freq_size is None:
            freq_size = self.n_fft // 2 + 1
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        states: List[Tensor] = []
        for _, module in self._calibration_functional_modules:
            cal_conv = cast(FunctionalStatefulConv1d, module)
            states.append(cal_conv.init_state(batch_size=batch_size, device=device, dtype=dtype))
        for _, module in self._alpha_functional_modules:
            alpha_conv = cast(FunctionalStatefulCausalConv2d, module)
            states.append(alpha_conv.init_state(batch_size=batch_size, freq_size=freq_size, device=device, dtype=dtype))
        return states

    def get_state_shapes(
        self,
        batch_size: int = 1,
        freq_size: Optional[int] = None,
    ) -> List[Tuple[int, ...]]:
        """Return state shapes in :meth:`get_state_names` order."""
        return [tuple(s.shape) for s in self.init_states(batch_size=batch_size, freq_size=freq_size)]

    # ----------------------------------------------------------------- metadata
    def metadata(self, *, chunk_size: int = 8, freq_size: Optional[int] = None) -> Dict[str, Any]:
        """Snapshot the head subgraph's I/O contract + frame geometry.

        Head is causal (no encoder/decoder lookahead) — ``t_export`` reduces to
        ``chunk_size + alpha_time_lookahead`` (= ``chunk_size`` for the canonical
        ``L_alpha = 0`` ckpt).
        """
        if freq_size is None:
            freq_size = self.n_fft // 2 + 1
        alpha_lookahead = audit_alpha_time_lookahead(self.alpha_convblocks)
        t_export = int(chunk_size) + alpha_lookahead
        shapes = self.get_state_shapes(batch_size=1, freq_size=freq_size)
        n_cal = self.num_calibration_states
        cal_shapes = [list(s) for s in shapes[:n_cal]]
        alpha_shapes = [list(s) for s in shapes[n_cal:]]
        return {
            "schema_version": "s21-bafnetplus-head-fp32",
            "phase_output_mode": "atan2",
            "ablation_mode": self.ablation_mode,
            "use_calibration": self.use_calibration,
            "use_relative_gain": self.use_relative_gain,
            "mask_only_alpha": self.mask_only_alpha,
            "calibration_max_common_log_gain": self.calibration_max_common_log_gain,
            "calibration_max_relative_log_gain": self.calibration_max_relative_log_gain,
            "n_fft": self.n_fft,
            "freq_size": int(freq_size),
            "chunk_size": int(chunk_size),
            "alpha_time_lookahead": int(alpha_lookahead),
            "total_lookahead": int(alpha_lookahead),
            "t_export": int(t_export),
            "t_export_formula": "chunk_size + alpha_time_lookahead",
            "state_frames_for_update": self.state_frames_for_update,
            "num_states": self.num_states,
            "input_names": self.input_names(),
            "output_names": self.output_names(),
            "state_names": self.get_state_names(),
            "state_shapes": [list(s) for s in shapes],
            "state_order": "calibration/* alpha/*",
            "fusion": {
                "calibration": {"num_states": n_cal, "state_shapes": cal_shapes},
                "alpha": {"num_states": self.num_alpha_states, "state_shapes": alpha_shapes},
            },
        }

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        bcs_est_mag: Tensor,
        bcs_phase_real: Tensor,
        bcs_phase_imag: Tensor,
        acs_est_mag: Tensor,
        acs_phase_real: Tensor,
        acs_phase_imag: Tensor,
        acs_mask: Tensor,
        *prev_states: Tensor,
    ) -> Tuple[Tensor, ...]:
        """Run the head: per-branch reconstruction + calibration + alpha fusion + final atan2.

        Args:
            bcs_est_mag / bcs_phase_real / bcs_phase_imag: Per-branch trunk outputs
                for the BCS mapping branch.
            acs_est_mag / acs_phase_real / acs_phase_imag: Per-branch trunk outputs
                for the ACS masking branch.
            acs_mask: Explicit raw mask from the masking branch
                (post-LearnableSigmoid, pre-multiplication).
            *prev_states: ``num_states`` prev-state tensors in
                :meth:`get_state_names` order.

        Returns:
            ``(est_mag, est_pha, est_com, *next_states)`` — the canonical S8/S17
            atan2 triple followed by the head's next-states.

        Raises:
            ValueError: If ``len(prev_states) != self.num_states``.
        """
        n_cal = self.num_calibration_states
        n_alpha = self.num_alpha_states
        if len(prev_states) != n_cal + n_alpha:
            raise ValueError(
                f"BAFNetPlusHeadCore.forward: expected {n_cal + n_alpha} prev_states "
                f"(calibration={n_cal} + alpha={n_alpha}), got {len(prev_states)}"
            )
        cal_states = prev_states[:n_cal]
        alpha_states = prev_states[n_cal:]

        # 1. Reconstruct per-branch est_com from the trunk's raw phase components.
        # This is where the sqrt-based reconstruction (the INT8-hostile op cluster
        # S20 D's INT8 variant choked on) lives — kept FP32 by construction since
        # this entire forward is the FP32 head subgraph.
        bcs_est_com = _phase_components_to_complex(bcs_est_mag, bcs_phase_real, bcs_phase_imag)
        acs_est_com = _phase_components_to_complex(acs_est_mag, acs_phase_real, acs_phase_imag)

        cal_iter = StateIterator(list(cal_states))
        alpha_iter = StateIterator(list(alpha_states))

        # 2. Calibration encoder + gain heads (frame-wise causal).
        if self.use_calibration:
            calibration_feat = _build_calibration_features(bcs_est_mag, acs_est_mag, acs_mask)
            calibration_hidden = self._forward_calibration_encoder(calibration_feat, cal_iter)
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

        # 3. Alpha softmax fusion.
        if self.mask_only_alpha:
            alpha = acs_mask.unsqueeze(1).transpose(2, 3)
        else:
            alpha = _build_alpha_features(bcs_com_fused, acs_com_fused, acs_mask)
        alpha = self._forward_alpha_convblocks(alpha, alpha_iter)
        alpha = self.alpha_out(alpha)
        alpha = alpha.transpose(2, 3)
        alpha = torch.softmax(alpha, dim=1)
        alpha_bcs = alpha[:, 0].unsqueeze(-1)
        alpha_acs = alpha[:, 1].unsqueeze(-1)
        est_com = bcs_com_fused * alpha_bcs + acs_com_fused * alpha_acs

        # 4. Final atan2 — head is FP32, so atan2 is fine here.
        est_mag, est_pha = complex_to_mag_pha(est_com)

        if cal_iter.consumed_count != n_cal:
            raise RuntimeError(f"calibration StateIterator consumed {cal_iter.consumed_count} of {n_cal} states")
        if alpha_iter.consumed_count != n_alpha:
            raise RuntimeError(f"alpha StateIterator consumed {alpha_iter.consumed_count} of {n_alpha} states")

        next_states = list(cal_iter.next_states) + list(alpha_iter.next_states)
        return (est_mag, est_pha, est_com) + tuple(next_states)

    def _forward_calibration_encoder(self, x: Tensor, state_iter: StateIterator) -> Tensor:
        """Run the calibration encoder with functional state routing.

        Structure: ``nn.Sequential( nn.Sequential( FunctionalStatefulConv1d, PReLU ) )^depth``.
        """
        cal_enc = cast(nn.Sequential, self.calibration_encoder)
        for block in cal_enc:
            inner = cast(nn.Sequential, block)
            for layer in inner:
                if isinstance(layer, FunctionalStatefulConv1d):
                    x = state_iter.run_layer(layer, x, state_frames=self.state_frames_for_update)
                else:
                    x = layer(x)
        return x

    def _forward_alpha_convblocks(self, x: Tensor, state_iter: StateIterator) -> Tensor:
        """Run the alpha conv stack with functional state routing.

        Structure: ``nn.ModuleList( nn.Sequential( FunctionalStatefulCausalConv2d,
        BatchNorm2d, PReLU ) )^depth``.
        """
        for block in self.alpha_convblocks:
            inner = cast(nn.Sequential, block)
            for layer in inner:
                if isinstance(layer, FunctionalStatefulCausalConv2d):
                    x = state_iter.run_layer(layer, x, state_frames=self.state_frames_for_update)
                else:
                    x = layer(x)
        return x

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ablation_mode={self.ablation_mode!r}, "
            f"use_calibration={self.use_calibration}, use_relative_gain={self.use_relative_gain}, "
            f"mask_only_alpha={self.mask_only_alpha}, "
            f"num_states={self.num_states} "
            f"(calibration={self.num_calibration_states}, alpha={self.num_alpha_states}), "
            f"state_frames_for_update={self.state_frames_for_update})"
        )


# ============================================================ algebra helpers
def _phase_components_to_complex(
    mag: Tensor,
    phase_real: Tensor,
    phase_imag: Tensor,
) -> Tensor:
    """Reconstruct a complex spectrogram from magnitude + raw phase components.

    Algebraic identity used here:
        ``mag_pha_to_complex(mag, atan2(im+eps, re+eps))``
        ``= mag * (cos, sin) of atan2(im+eps, re+eps)``
        ``= mag * (re+eps, im+eps) / sqrt((re+eps)^2 + (im+eps)^2)``.

    Eliminates the per-branch ``atan2`` op from the graph (S20 D track) while
    staying numerically equivalent to the atan2 path up to libm cos/sin/atan2
    implementation precision. The eps placement (``+1e-8`` on real and imag
    inputs) matches the eps used in :func:`src.stft.complex_to_mag_pha` /
    :func:`src.models.backbone.PhaseDecoder.forward` so the per-branch
    complex spectrogram is bit-equivalent (FP32, libm precision) to the
    atan2-mode reconstruction.

    Args:
        mag: ``[B, F, T]`` est_mag from the per-branch backbone.
        phase_real: ``[B, F, T]`` raw ``phase_conv_r`` output.
        phase_imag: ``[B, F, T]`` raw ``phase_conv_i`` output.

    Returns:
        ``[B, F, T, 2]`` complex spectrogram (real, imag stacked on dim=-1).
    """
    eps = 1e-8
    re = phase_real + eps
    im = phase_imag + eps
    # The +1e-16 floor inside the sqrt guards against the (re=im=-eps) corner
    # where the numerator vanishes exactly — its ULP impact on typical
    # phase-decoder magnitudes (re, im ~ O(0.1-1)) is below FP32 noise.
    norm = torch.sqrt(re**2 + im**2 + 1e-16)
    return torch.stack((mag * re / norm, mag * im / norm), dim=-1)


def _build_calibration_features(
    bcs_mag: Tensor,
    acs_mag: Tensor,
    acs_mask: Tensor,
) -> Tensor:
    """Frame-wise causal calibration features — mirrors
    :meth:`BAFNetPlus._build_calibration_features`."""
    eps = 1e-8
    bcs_log_energy = torch.log(bcs_mag.pow(2).mean(dim=1, keepdim=True) + eps)
    acs_log_energy = torch.log(acs_mag.pow(2).mean(dim=1, keepdim=True) + eps)
    log_energy_diff = bcs_log_energy - acs_log_energy
    acs_mask_mean = acs_mask.mean(dim=1, keepdim=True)
    acs_mask_var = acs_mask.var(dim=1, keepdim=True, unbiased=False)
    return torch.cat(
        [bcs_log_energy, acs_log_energy, log_energy_diff, acs_mask_mean, acs_mask_var],
        dim=1,
    )


def _build_alpha_features(
    bcs_com_cal: Tensor,
    acs_com_cal: Tensor,
    acs_mask: Tensor,
) -> Tensor:
    """TF-wise fusion features — mirrors
    :meth:`BAFNetPlus._build_alpha_features`."""
    eps = 1e-8
    bcs_mag_cal = torch.sqrt(bcs_com_cal[:, :, :, 0] ** 2 + bcs_com_cal[:, :, :, 1] ** 2 + eps)
    acs_mag_cal = torch.sqrt(acs_com_cal[:, :, :, 0] ** 2 + acs_com_cal[:, :, :, 1] ** 2 + eps)
    return torch.stack([bcs_mag_cal, acs_mag_cal, acs_mask], dim=1).transpose(2, 3)


# ============================================================ causality audits


def _is_inside_wrapper(root: nn.Module, child_path: str, wrapper_cls: type) -> bool:
    """Return True iff any ancestor of ``child_path`` under ``root`` is an
    instance of ``wrapper_cls``.

    Used by the audits to skip the inner ``nn.Conv1d`` / ``nn.Conv2d`` that
    :class:`CausalConv1d` / :class:`CausalConv2d` instantiate as ``self.conv``
    (with ``padding=0``; the asymmetric pad is applied by the wrapper).
    """
    if "." not in child_path:
        return False
    parts = child_path.split(".")
    for i in range(1, len(parts)):
        ancestor_path = ".".join(parts[:-i])
        try:
            ancestor = root.get_submodule(ancestor_path)
        except AttributeError:
            continue
        if isinstance(ancestor, wrapper_cls):
            return True
    return False


def audit_calibration_is_causal(calibration_encoder: nn.Module) -> Dict[str, Any]:
    """Walk a calibration encoder and assert every conv is left-only padded.

    BAFNet+ builds the calibration encoder as
    ``nn.Sequential( CausalConv1d + PReLU )^depth`` (see
    :meth:`BAFNetPlus.__init__`). Each :class:`CausalConv1d` is left-only
    padded by construction — :attr:`CausalConv1d.padding` is set to
    ``padding * 2`` and applied via ``F.pad(x, [self.padding, 0])``, so the
    right-side time-padding is identically zero. The audit walks
    ``named_modules()`` and skips the inner ``nn.Conv1d`` that
    :class:`CausalConv1d` wraps internally (it has ``padding=0`` and the
    asymmetric pad lives on the wrapper).

    Args:
        calibration_encoder: The BAFNet+ ``calibration_encoder`` ``nn.Module``.

    Returns:
        Dict with keys:
            ``causal`` (bool): True iff every Conv1d-like leaf is a
                :class:`CausalConv1d` (and no symmetric-padded ``nn.Conv1d`` leaks in).
            ``num_conv_layers`` (int): Number of top-level Conv1d-like leaves.
            ``layers`` (list): Per-layer ``(path, kind, left_pad, right_pad)``.
            ``total_left_pad`` (int): Sum of left padding (informational —
                this is the time-axis reach into the past, NOT added latency).
            ``total_right_pad`` (int): Sum of right padding (== **0** for a
                causal stack).
    """
    layers: List[Dict[str, Any]] = []
    num_conv = 0
    all_causal = True
    total_left = 0
    total_right = 0
    for path, module in calibration_encoder.named_modules():
        if isinstance(module, CausalConv1d):
            num_conv += 1
            left = int(module.padding)
            right = 0
            total_left += left
            total_right += right
            layers.append({"path": path, "kind": "CausalConv1d", "left_pad": left, "right_pad": right})
        elif isinstance(module, nn.Conv1d):
            # Skip the inner conv that CausalConv1d wraps as self.conv.
            if _is_inside_wrapper(calibration_encoder, path, CausalConv1d):
                continue
            # Bare nn.Conv1d (NOT wrapped in CausalConv1d) — symmetric padding by default.
            num_conv += 1
            left = right = int(module.padding[0]) if isinstance(module.padding, tuple) else int(module.padding)
            total_left += left
            total_right += right
            layers.append({"path": path, "kind": "nn.Conv1d", "left_pad": left, "right_pad": right})
            if right > 0:
                all_causal = False
    return {
        "causal": all_causal and total_right == 0,
        "num_conv_layers": num_conv,
        "layers": layers,
        "total_left_pad": total_left,
        "total_right_pad": total_right,
    }


def audit_alpha_time_lookahead(alpha_convblocks: nn.ModuleList) -> int:
    """Walk the alpha conv stack and sum the time-axis right-padding.

    BAFNet+ alpha uses :class:`CausalConv2d` from :mod:`src.models.backbone`
    (NOT a plain ``nn.Conv2d`` with symmetric padding — that's a common
    misread of the launch-prompt warning). Each :class:`CausalConv2d` has
    :attr:`padding` quartet ``(left_freq, right_freq, top_time, bottom_time)``
    with ``bottom_time = 0`` for left-only time padding — so the time-axis
    right-padding contribution is identically zero. The audit nonetheless
    inspects each leaf, returns the sum, and warns if a non-causal Conv2d
    leaks in.

    Args:
        alpha_convblocks: The BAFNet+ ``alpha_convblocks`` ``nn.ModuleList``
            (each entry is ``nn.Sequential(CausalConv2d, BN2d, PReLU)``).

    Returns:
        ``L_alpha`` (int) — the time-axis right-padding contribution summed
        over all :class:`CausalConv2d` / ``nn.Conv2d`` leaves under the
        alpha conv stack. **0** for the deployed unified ckpt.
    """
    total_right = 0
    for path, module in alpha_convblocks.named_modules():
        if isinstance(module, CausalConv2d):
            # CausalConv2d.padding = (left_freq, right_freq, top_time, bottom_time);
            # bottom_time (right time pad) is 0 by construction.
            bottom_time = int(module.padding[3])
            total_right += bottom_time
        elif isinstance(module, nn.Conv2d):
            # Skip the inner conv that CausalConv2d wraps as self.conv (padding=0 there).
            if _is_inside_wrapper(alpha_convblocks, path, CausalConv2d):
                continue
            # Bare nn.Conv2d (symmetric padding).
            if isinstance(module.padding, tuple):
                # padding=(pad_time, pad_freq) — symmetric on each axis.
                right_time = int(module.padding[0])
            else:
                right_time = int(module.padding)
            if right_time > 0:
                logger.warning(
                    "alpha conv leaf %s is a plain nn.Conv2d with time-axis right padding=%d "
                    "(non-causal) — adds %d frames of right-side lookahead",
                    path,
                    right_time,
                    right_time,
                )
            total_right += right_time
    return total_right


# ============================================================ T_export proof
def _wrapped_abs_diff(a: Tensor, b: Tensor) -> Tensor:
    """``|atan2(sin(a-b), cos(a-b))|`` — phase difference modulo 2π."""
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d)).abs()


def prove_t_export(
    core: BAFNetPlusCore,
    reference_model: BAFNetPlus,
    *,
    chunk_size: int = 8,
    candidate_geometries: Optional[List[int]] = None,
    total_frames: int = 64,
    seed: int = 1234,
    mag_tol: float = 1e-4,
    com_tol: float = 1e-3,
    pha_tol: float = 2e-3,
    default_t_export: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """One-chunk ``T_export`` proof harness for the decomposed BAFNet+ core.

    For each candidate ``T_export`` geometry, slices the first ``T_export``
    frames at ``t=0`` from a synthetic complete-spectrogram input, drives
    them through ``core`` (with the backbone cores' state_frames bounded to
    ``chunk_size``, zero init states), and compares the core's first
    ``chunk_size`` output frames against ``BAFNetPlus.forward()``'s first
    ``chunk_size`` output frames on the same input. The chosen ``T_export``
    is the smallest geometry that matches within ``(mag_tol, com_tol, pha_tol)``;
    if none pass, returns ``default_t_export`` (the conservative default,
    which is ``chunk_size + L_enc + L_dec + L_alpha`` from the audited core
    metadata when not specified).

    Why t=0 (the first chunk)? At ``t=0`` both
    ``BAFNetPlus.forward()`` (which processes the whole sequence with
    ``CausalConv*`` left-only padding) and ``BAFNetPlusCore`` (which
    processes the ``T_export``-frame window) see the SAME zero left-context
    for the backbones' state and the calibration / alpha causal-pad chains —
    so any mismatch in the first ``chunk_size`` outputs is attributable to
    the right-side lookahead alone. Steady-state (interior) chunks would
    require streaming state propagation across calibration / alpha, which
    is S8 territory.

    Args:
        core: A :class:`BAFNetPlusCore` (typically built via
            :meth:`BAFNetPlusCore.from_bafnetplus`).
        reference_model: The same :class:`BAFNetPlus` instance the ``core``
            was decomposed from. The harness calls
            ``reference_model((bcs_com, acs_com))`` for the full-sequence
            ground truth.
        chunk_size: Output chunk size (default 8 for the 50 ms anchor).
        candidate_geometries: Explicit list of ``T_export`` candidates. If
            ``None``, the harness uses ``[chunk + max(L_enc, L_dec),
            chunk + L_enc + L_dec]`` (plus ``chunk + L_enc + L_dec + L_alpha``
            if the alpha audit reports ``L_alpha > 0``), deduplicated and
            sorted ascending.
        total_frames: Length of the synthetic input spectrogram (must be
            ≥ the largest candidate + ``chunk_size``).
        seed: ``torch.manual_seed`` for reproducibility.
        mag_tol / com_tol / pha_tol: Per-output max-abs tolerances. The
            phase tolerance is wrapped-modulo-2π.
        default_t_export: Returned as ``chosen_t_export`` if no candidate
            passes. ``None`` falls back to ``chunk + L_enc + L_dec + L_alpha``
            from the core's metadata.
        verbose: If True, also prints per-candidate diffs to stdout.

    Returns:
        Dict with keys:
            ``candidates`` (list): Per-candidate report with ``t_export``,
                ``dmag``, ``dcom``, ``dpha_wrapped``, ``passes``.
            ``chosen_t_export`` (int): The smallest passing geometry (or
                ``default_t_export`` if none pass).
            ``ambiguous`` (bool): True iff none passed (chose the default).
            ``mag_tol`` / ``com_tol`` / ``pha_tol`` (float): Echoed tolerances.
            ``alpha_time_lookahead`` (int): The audited alpha-path lookahead.
            ``encoder_lookahead`` / ``decoder_lookahead`` (int): Per-branch.
    """
    # Derive default candidates if needed.
    l_enc = core.mapping_core.encoder_lookahead
    l_dec = core.mapping_core.decoder_lookahead
    l_alpha = audit_alpha_time_lookahead(core.alpha_convblocks)
    if candidate_geometries is None:
        cands = {chunk_size + max(l_enc, l_dec), chunk_size + l_enc + l_dec}
        if l_alpha > 0:
            cands.add(chunk_size + l_enc + l_dec + l_alpha)
        candidate_geometries = sorted(cands)
    if default_t_export is None:
        default_t_export = chunk_size + l_enc + l_dec + l_alpha

    if any(t > total_frames for t in candidate_geometries):
        raise ValueError(
            f"total_frames={total_frames} is too small for candidates={candidate_geometries}; "
            f"need total_frames >= max(candidates)"
        )

    # Generate the synthetic complete-spectrogram input (complex form for BAFNetPlus.forward).
    torch.manual_seed(seed)
    freq_size = core.n_fft // 2 + 1
    device = next(core.parameters()).device
    dtype = next(core.parameters()).dtype
    bcs_com = torch.randn(1, freq_size, total_frames, 2, device=device, dtype=dtype) * 0.3
    acs_com = torch.randn(1, freq_size, total_frames, 2, device=device, dtype=dtype) * 0.3

    reference_model.eval()
    with torch.no_grad():
        ref_mag, ref_pha, ref_com = reference_model((bcs_com, acs_com))

    # The same mag/pha boundary Backbone.forward uses internally — feed these to BAFNetPlusCore.
    bcs_mag_all, bcs_pha_all = complex_to_mag_pha(bcs_com, stack_dim=-1)
    acs_mag_all, acs_pha_all = complex_to_mag_pha(acs_com, stack_dim=-1)

    candidate_reports: List[Dict[str, Any]] = []
    chosen: Optional[int] = None
    for t_export in candidate_geometries:
        # Bound state updates to chunk_size for the backbone cores (the streaming-export
        # convention). Calibration / alpha are non-functional at S5 — no bound to apply.
        core.set_state_frames_for_update(chunk_size)
        try:
            with torch.no_grad():
                est_mag, est_pha, est_com = core(
                    bcs_mag_all[:, :, :t_export],
                    bcs_pha_all[:, :, :t_export],
                    acs_mag_all[:, :, :t_export],
                    acs_pha_all[:, :, :t_export],
                )
            est_mag_c = est_mag[:, :, :chunk_size]
            est_pha_c = est_pha[:, :, :chunk_size]
            est_com_c = est_com[:, :, :chunk_size, :]
            ref_mag_c = ref_mag[:, :, :chunk_size]
            ref_pha_c = ref_pha[:, :, :chunk_size]
            ref_com_c = ref_com[:, :, :chunk_size, :]
            dmag = (ref_mag_c - est_mag_c).abs().max().item()
            dcom = (ref_com_c - est_com_c).abs().max().item()
            dpha = _wrapped_abs_diff(ref_pha_c, est_pha_c).max().item()
            passes = (
                math.isfinite(dmag)
                and dmag < mag_tol
                and math.isfinite(dcom)
                and dcom < com_tol
                and math.isfinite(dpha)
                and dpha < pha_tol
            )
            report: Dict[str, Any] = {
                "t_export": int(t_export),
                "dmag": float(dmag),
                "dcom": float(dcom),
                "dpha_wrapped": float(dpha),
                "passes": bool(passes),
            }
        except Exception as exc:  # pragma: no cover  (defensive — should not occur on valid inputs)
            report = {
                "t_export": int(t_export),
                "error": repr(exc),
                "passes": False,
            }
        candidate_reports.append(report)
        if verbose:
            print(f"[prove_t_export] T_export={t_export}: {report}")
        if report["passes"] and chosen is None:
            chosen = int(t_export)

    ambiguous = chosen is None
    if chosen is None:
        chosen = int(default_t_export)
    # Reset to None so the caller doesn't inherit the bound state.
    core.set_state_frames_for_update(None)
    return {
        "candidates": candidate_reports,
        "chosen_t_export": int(chosen),
        "ambiguous": bool(ambiguous),
        "mag_tol": float(mag_tol),
        "com_tol": float(com_tol),
        "pha_tol": float(pha_tol),
        "alpha_time_lookahead": int(l_alpha),
        "encoder_lookahead": int(l_enc),
        "decoder_lookahead": int(l_dec),
    }


__all__ = [
    "BAFNetPlusCore",
    "ExportableBAFNetPlusCore",
    "BAFNetPlusTrunkCore",
    "BAFNetPlusHeadCore",
    "audit_calibration_is_causal",
    "audit_alpha_time_lookahead",
    "prove_t_export",
]
