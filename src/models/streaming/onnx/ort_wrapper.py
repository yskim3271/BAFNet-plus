"""ORT host wrapper for the BAFNet+ FP32 ONNX export (audio -> audio).

S9 of the streaming-ONNX rebuild (Stage 4 pt.2 - Stage-4 exit gate).
:class:`BAFNetPlusOrtStreaming` mirrors the public API of S6's
:class:`~src.models.streaming.bafnetplus_streaming.BAFNetPlusStreaming` exactly
- ``process_samples`` / ``process_audio`` / ``process_spectrogram`` / ``reset_state``
+ identical warm-up / return-shape semantics - but runs the S8 functional-stateful
:class:`~src.models.streaming.onnx.bafnetplus_core.ExportableBAFNetPlusCore`
exported FP32 ONNX graph via :mod:`onnxruntime` instead of PyTorch.

What this is / is not
---------------------
- It IS the deployment-shape reference: every chunk of audio runs through
  ONE ``sess.run(...)`` call on the functional graph (190 states propagated
  across chunks). Host-side STFT / iSTFT-OLA / FIFO logic stays in torch;
  the ORT call is the ONLY numpy boundary.
- It IS a drop-in API replacement for S6's :class:`BAFNetPlusStreaming` - the
  public method names, return shapes (warm-up ``None``-returns, then
  ``output_samples_per_chunk = 800`` matured samples), and ``reset_state``
  semantics match exactly. A downstream consumer (the S7 PESQ harness, the
  S11 Android binder) can swap PT <-> ORT without touching call sites.
- It IS CPU-only at S9 per the LP's "FP32 first" rule (``CPUExecutionProvider``).
  No PT shadow run during inference, no GPU/CUDA ORT path.
- It is **NOT** a re-export driver - it consumes an already-exported ONNX +
  sidecar JSON. Construction via :meth:`from_checkpoint` is a convenience
  that calls :func:`export_bafnetplus_to_onnx_from_checkpoint` once at the
  start, then loads the artifact.
- It does **NOT** modify :class:`ExportableBAFNetPlusCore` or the S8 IO
  contract - the 190-state graph is the frozen artifact this wrapper binds
  to.

Host buffers (mirror S6's :class:`BAFNetPlusStreaming` exactly)
---------------------------------------------------------------
(a) **Per-stream input sample buffer** ``_input_buffer_bcs`` /
    ``_input_buffer_acs`` - same fill / slide / sync semantics as S6.
(b) **Per-stream STFT-context buffer** ``_stft_context_bcs`` /
    ``_stft_context_acs``, each ``win_size // 2 = 200`` samples
    (``center=True`` emulation).
(c) **STFT-frame accumulator** ``_frame_buffer_*`` - a per-branch
    ``[1, F, T_buffered]`` magnitude+phase buffer. Each call appends one
    chunk's worth of STFT frames; once the buffer holds >= ``T_export = 14``
    frames the next ORT call fires on the first ``T_export`` frames and
    pops ``chunk_size = 8`` frames off the front. This is the S9 analogue
    of S6's per-branch ``feature_buffer`` - but it lives OUTSIDE the ORT
    graph because the ORT graph runs encoder + TS + decoders + fusion in
    one shot.
(d) **Shared iSTFT-OLA buffer** ``_ola_buffer`` / ``_ola_norm``, each
    ``win_size - hop_size = 300`` samples. Reuses
    :func:`src.stft.manual_istft_ola` for bit-equivalent cross-chunk OLA
    carry-over (same routine S6 / training use for ``center=False`` iSTFT).

Plus a small ``_input_mag_fifo: Deque[Tuple[bcs_mag_valid, acs_mag_valid]]``
mirroring S6's mask-recovery alignment FIFO. The S9 wrapper does NOT use
this FIFO for fusion - the fusion is done inside the ONNX graph - but the
``acs_mag_input`` slice is still needed for the in-graph mask-recovery
division (``acs_mask = acs_est_mag / acs_mag``) to stay well-defined. The
host STFT's ``sqrt(... + 1e-9)`` magnitude floor guarantees ``acs_mag > 0``
on every frame, so the division is safe.

Internal state propagation
--------------------------
``_current_states: List[np.ndarray]`` - a flat list of 190 entries in the
sidecar-frozen order (``mapping/*`` x92 -> ``masking/*`` x92 ->
``calibration/*`` x2 -> ``alpha/*`` x4). Initialised to zeros from the
sidecar's recorded ``state_shapes``. Per ORT call: feed the 4 spectrogram
inputs + the 190 prev-states; take 3 outputs (``est_mag``, ``est_pha``,
``est_com``) + 190 next-states; replace ``_current_states`` with the
next-states. The state-shape contract is verified at every step by
:func:`_assert_state_integrity`.

Public API (mirrors :class:`BAFNetPlusStreaming` exactly)
---------------------------------------------------------
- :meth:`process_samples(bcs_chunk, acs_chunk) -> Optional[Tensor]`:
  chunked paired input, returns ``None`` during the warm-up calls (the
  STFT-frame accumulator fills), else exactly ``output_samples_per_chunk =
  800`` mature audio samples (50 ms output step).
- :meth:`process_audio(bcs_audio, acs_audio) -> Tensor`: whole-utterance
  audio in, whole-utterance enhanced audio out (with flush + length trim).
- :meth:`process_spectrogram(noisy_bcs_com, noisy_acs_com) -> SpectrogramChunk`:
  spectrogram-in / spectrogram-out driver - feeds ``T_export``-frame slices
  to ORT, concatenates the matured triples along time, no internal flush.
- :meth:`forward(bcs_audio, acs_audio) == process_audio(bcs_audio, acs_audio)`.
- :meth:`reset_state`: clears all 4 host audio/STFT buffers, the OLA pair,
  the STFT-frame accumulator, the FIFO, and re-initialises
  ``_current_states`` to zeros.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from src.models.streaming.backbone_streaming import SpectrogramChunk
from src.models.streaming.onnx.export import export_bafnetplus_to_onnx_from_checkpoint
from src.stft import complex_to_mag_pha, mag_pha_to_complex, manual_istft_ola

logger = logging.getLogger(__name__)


class BAFNetPlusOrtStreaming:
    """Chunk-by-chunk ORT-driven streaming wrapper for the BAFNet+ FP32 ONNX export.

    Construct via :meth:`from_onnx` (consume an existing exported artifact +
    sidecar JSON) or :meth:`from_checkpoint` (one-shot: export the BAFNet+
    ONNX from a unified ckpt, then load it via :meth:`from_onnx`). The
    public API mirrors S6's :class:`BAFNetPlusStreaming` exactly so call
    sites can swap PT <-> ORT without changes.

    Attributes:
        session: The ``onnxruntime.InferenceSession`` (FP32, CPU-only).
        onnx_path / sidecar_path: Filesystem paths used at construction.
        chunk_size / encoder_lookahead / decoder_lookahead / total_lookahead /
            alpha_time_lookahead / t_export / total_frames_needed /
            samples_per_chunk / output_samples_per_chunk / latency_samples /
            latency_ms / n_fft / hop_size / win_size / compress_factor /
            sample_rate / freq_size / ola_tail_size: Streaming geometry
            (read from the sidecar at construction time; immutable
            thereafter).
        ablation_mode / use_calibration / use_relative_gain / mask_only_alpha /
            calibration_max_common_log_gain / calibration_max_relative_log_gain:
            Algebra flags - sourced from the sidecar's ``core`` section.
        state_names: The 190 (real ckpt) / 110 (synthetic) state-input names
            in the frozen sidecar order.
        state_shapes: Matching state shapes (each a tuple).
        input_names / output_names: The full ORT IO contract (4 + 190 inputs,
            3 + 190 outputs at the real ckpt scale).
        checkpoint_info: A copy of the sidecar's ``checkpoint`` block
            (``path``, ``md5``, ``model_lib``, ...) - used by the S9 PESQ
            triple gate to cross-check the wrapper was built from the same
            ckpt as the PT-streaming wrapper.
        _input_buffer_bcs / _input_buffer_acs: Per-stream raw audio buffer
            (torch tensor; FP32; CPU).
        _stft_context_bcs / _stft_context_acs: Per-stream ``win//2``-sample
            past-context buffer (torch tensor; FP32; CPU).
        _ola_buffer / _ola_norm: ``win - hop``-sample shared iSTFT-OLA
            carry-over (torch tensor; FP32; CPU).
        _frame_buffer_bcs_mag / _frame_buffer_bcs_pha /
            _frame_buffer_acs_mag / _frame_buffer_acs_pha:
            ``[1, F, T_buffered]`` STFT-frame accumulators (torch tensor;
            FP32). The S9 analogue of S6's per-branch encoder-feature buffer.
        _input_mag_fifo: Deque of ``(bcs_mag_valid, acs_mag_valid)`` for
            mask-recovery alignment (kept for API parity with S6; the ORT
            graph handles the mask recovery internally so this FIFO is
            not consumed at fusion time).
        _current_states: Flat list of 190 ``np.float32`` ndarrays - the
            streaming state across ORT chunks.
        _num_chunks_received: Counter for first-call detection in the
            STFT-frame accumulator (call 0 appends all ``total_frames_needed``
            frames; subsequent calls append ``chunk_size`` frames).
    """

    def __init__(
        self,
        session: Any,
        sidecar: Dict[str, Any],
        onnx_path: Path,
        sidecar_path: Path,
    ) -> None:
        """Initialise from a loaded ORT session + parsed sidecar JSON.

        Most users should use :meth:`from_onnx` / :meth:`from_checkpoint`
        instead of calling this constructor directly.

        Args:
            session: An :class:`onnxruntime.InferenceSession` for the
                BAFNet+ FP32 ONNX.
            sidecar: The parsed sidecar JSON dict (must carry
                ``schema_version='s8-bafnetplus-functional-fp32'``).
            onnx_path: Filesystem path to the loaded ``.onnx`` file.
            sidecar_path: Filesystem path to the loaded ``.onnx.json`` file.

        Raises:
            ValueError: If the sidecar schema version is unexpected, or the
                IO contract / geometry fields are inconsistent.
        """
        self.session = session
        self.onnx_path = Path(onnx_path)
        self.sidecar_path = Path(sidecar_path)
        self._sidecar = sidecar

        schema = sidecar.get("schema_version")
        # Accept the two unified FP32 schemas. The complex schema is the current
        # D3/c09 deployment contract; the atan2 schema remains supported for
        # FP32 host compatibility.
        _accepted_schemas = (
            "s8-bafnetplus-functional-fp32",
            "s20-bafnetplus-functional-fp32-complex",
        )
        if schema not in _accepted_schemas:
            raise ValueError(
                f"unexpected sidecar schema_version {schema!r}; expected one of {_accepted_schemas} "
                f"(sidecar at {sidecar_path})"
            )

        geom = sidecar["geometry"]
        stft = sidecar["stft"]
        core = sidecar["core"]
        io = sidecar["io"]

        self.chunk_size: int = int(geom["chunk_size"])
        self.encoder_lookahead: int = int(geom["encoder_lookahead"])
        self.decoder_lookahead: int = int(geom["decoder_lookahead"])
        self.alpha_time_lookahead: int = int(geom["alpha_time_lookahead"])
        self.total_lookahead: int = int(geom["total_lookahead"])
        self.t_export: int = int(geom["T_export"])
        self.freq_size: int = int(geom["freq_size"])

        self.n_fft: int = int(stft["n_fft"])
        self.hop_size: int = int(stft["hop_size"])
        self.win_size: int = int(stft["win_size"])
        self.compress_factor: float = float(stft["compress_factor"])
        self.sample_rate: int = int(stft["sample_rate"])

        # Derived audio geometry (mirrors S3 BackboneStreaming / S6 BAFNetPlusStreaming).
        self.input_lookahead_frames: int = self.encoder_lookahead
        self.total_frames_needed: int = self.chunk_size + self.input_lookahead_frames
        self.stft_future_samples: int = self.win_size // 2
        self.stft_center_delay_samples: int = self.stft_future_samples
        self.samples_per_chunk: int = (self.total_frames_needed - 1) * self.hop_size + self.stft_future_samples
        self.output_samples_per_chunk: int = self.chunk_size * self.hop_size
        self.latency_samples: int = self.total_lookahead * self.hop_size + self.stft_center_delay_samples
        self.latency_ms: float = self.latency_samples / self.sample_rate * 1000.0
        self.ola_tail_size: int = self.win_size - self.hop_size

        # Algebra flags + tanh-scaled gain max - mirror the bafnet instance the ONNX was exported from.
        self.ablation_mode: str = str(core["ablation_mode"])
        self.use_calibration: bool = bool(core["use_calibration"])
        self.use_relative_gain: bool = bool(core["use_relative_gain"])
        self.mask_only_alpha: bool = bool(core["mask_only_alpha"])
        self.calibration_max_common_log_gain: float = float(core["calibration_max_common_log_gain"])
        self.calibration_max_relative_log_gain: float = float(core["calibration_max_relative_log_gain"])
        self.phase_output_mode: str = str(core["phase_output_mode"])

        # IO contract.
        self.input_names: List[str] = list(io["input_names"])
        self.output_names: List[str] = list(io["output_names"])
        self.state_names: List[str] = list(io["state_names"])
        self.state_shapes: List[Tuple[int, ...]] = [tuple(s) for s in io["state_shapes"]]
        self.num_non_state_outputs: int = int(io["num_non_state_outputs"])
        if self.num_non_state_outputs != 3:
            raise ValueError(
                f"unexpected num_non_state_outputs={self.num_non_state_outputs}; "
                "wrapper assumes 3 non-state outputs in either mode "
                "(atan2: est_mag/est_pha/est_com; complex: est_mag/phase_real/phase_imag)"
            )
        self.num_states: int = len(self.state_names)
        if self.num_states != int(core["num_states"]):
            raise ValueError(
                f"sidecar inconsistency: core.num_states={core['num_states']} vs len(state_names)={self.num_states}"
            )

        # Cross-check the ORT session's IO matches the sidecar's IO names.
        sess_input_names = [t.name for t in self.session.get_inputs()]
        sess_output_names = [t.name for t in self.session.get_outputs()]
        if sess_input_names != self.input_names:
            raise ValueError(
                f"ORT session input names != sidecar input names\n  session: {sess_input_names[:5]}...\n"
                f"  sidecar: {self.input_names[:5]}..."
            )
        if sess_output_names != self.output_names:
            raise ValueError(
                f"ORT session output names != sidecar output names\n  session: {sess_output_names[:5]}...\n"
                f"  sidecar: {self.output_names[:5]}..."
            )

        # Checkpoint provenance - exposed for the S9 PESQ triple gate's cross-check.
        # The sidecar's `checkpoint` block is absent (or `None`) when the export was
        # synthesised without a ckpt (test fixtures); treat that as an empty dict.
        ckpt_block = sidecar.get("checkpoint") or {}
        self.checkpoint_info: Dict[str, Any] = dict(ckpt_block)

        self._streaming_config: Dict[str, Any] = {
            "schema_version": schema,
            "onnx_path": str(self.onnx_path),
            "sidecar_path": str(self.sidecar_path),
            "t_export_formula": str(geom.get("T_export_formula", "chunk_size + L_enc + L_dec + L_alpha")),
        }
        self._reset_host_buffers()

    # ------------------------------------------------------------------ buffers
    def _reset_host_buffers(self) -> None:
        """Reset every host-level buffer + re-initialise ``_current_states`` to zeros."""
        device = torch.device("cpu")
        self._input_buffer_bcs: Tensor = torch.tensor([], dtype=torch.float32, device=device)
        self._input_buffer_acs: Tensor = torch.tensor([], dtype=torch.float32, device=device)
        self._stft_context_bcs: Tensor = torch.zeros(self.win_size // 2, dtype=torch.float32, device=device)
        self._stft_context_acs: Tensor = torch.zeros(self.win_size // 2, dtype=torch.float32, device=device)
        self._ola_buffer: Tensor = torch.zeros(self.ola_tail_size, dtype=torch.float32, device=device)
        self._ola_norm: Tensor = torch.zeros(self.ola_tail_size, dtype=torch.float32, device=device)
        # STFT-frame accumulator (the S9 analogue of S6's per-branch feature buffer).
        self._frame_buffer_bcs_mag: Tensor = torch.zeros(1, self.freq_size, 0, dtype=torch.float32, device=device)
        self._frame_buffer_bcs_pha: Tensor = torch.zeros(1, self.freq_size, 0, dtype=torch.float32, device=device)
        self._frame_buffer_acs_mag: Tensor = torch.zeros(1, self.freq_size, 0, dtype=torch.float32, device=device)
        self._frame_buffer_acs_pha: Tensor = torch.zeros(1, self.freq_size, 0, dtype=torch.float32, device=device)
        self._input_mag_fifo: Deque[Tuple[Tensor, Tensor]] = deque()
        self._num_chunks_received: int = 0
        self._current_states: List[np.ndarray] = _init_states_from_sidecar(self.state_shapes)

    def reset_state(self) -> None:
        """Reset all streaming state for a new utterance.

        Clears every host-level buffer (input x 2, STFT context x 2, OLA
        pair, STFT-frame accumulator, mag FIFO) and re-initialises
        ``_current_states`` to zero ndarrays of the recorded sidecar
        ``state_shapes``.
        """
        self._reset_host_buffers()

    @property
    def streaming_config(self) -> Dict[str, Any]:
        """Geometry + provenance summary (for debug / cross-check; mirrors S6)."""
        return {
            **self._streaming_config,
            "ablation_mode": self.ablation_mode,
            "use_calibration": self.use_calibration,
            "use_relative_gain": self.use_relative_gain,
            "mask_only_alpha": self.mask_only_alpha,
            "chunk_size": self.chunk_size,
            "encoder_lookahead": self.encoder_lookahead,
            "decoder_lookahead": self.decoder_lookahead,
            "alpha_time_lookahead": self.alpha_time_lookahead,
            "total_lookahead": self.total_lookahead,
            "t_export": self.t_export,
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
            "freq_size": self.freq_size,
            "ola_tail_size": self.ola_tail_size,
            "num_states": self.num_states,
            "checkpoint_md5": self.checkpoint_info.get("md5"),
        }

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        sidecar_path: Optional[str] = None,
        *,
        chunk_size: Optional[int] = None,
        device: str = "cpu",
    ) -> "BAFNetPlusOrtStreaming":
        """Build a :class:`BAFNetPlusOrtStreaming` from an exported ONNX + sidecar.

        Loads the ONNX into an :class:`onnxruntime.InferenceSession` (FP32,
        ``CPUExecutionProvider``) and parses the sidecar JSON for the
        geometry / IO contract / algebra flags / checkpoint provenance.

        Args:
            onnx_path: Path to the ``.onnx`` file. Must be an
                ``ExportableBAFNetPlusCore`` export (sidecar
                ``schema_version='s8-bafnetplus-functional-fp32'``).
            sidecar_path: Path to the ``.onnx.json`` sidecar. Defaults to
                ``<onnx_path>.json``.
            chunk_size: Optional override for testing - the sidecar's value
                is authoritative; this argument exists only to surface
                mismatches at construction time. ``None`` = use sidecar.
            device: Execution device. S9 is CPU-only at the LP's
                "FP32 first" rule; non-``"cpu"`` values raise ``ValueError``.

        Returns:
            A ready-to-stream :class:`BAFNetPlusOrtStreaming` (state already
            reset to zeros).

        Raises:
            FileNotFoundError: If ``onnx_path`` or the inferred / supplied
                ``sidecar_path`` does not exist.
            ValueError: If the sidecar schema or chunk_size mismatches.
            ImportError: If ``onnxruntime`` is not installed.
        """
        if device != "cpu":
            raise ValueError(f"S9 ORT wrapper is CPU-only (per the LP FP32-first rule); got device={device!r}")

        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - environment check
            raise ImportError("onnxruntime is required for BAFNetPlusOrtStreaming") from e

        onnx_path_p = Path(onnx_path)
        if not onnx_path_p.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path_p}")
        sidecar_path_p = Path(sidecar_path) if sidecar_path is not None else Path(f"{onnx_path_p}.json")
        if not sidecar_path_p.exists():
            raise FileNotFoundError(f"Sidecar JSON not found: {sidecar_path_p} (for ONNX {onnx_path_p})")

        with sidecar_path_p.open("r") as f:
            sidecar = json.load(f)

        if chunk_size is not None:
            sidecar_chunk = int(sidecar.get("geometry", {}).get("chunk_size", -1))
            if chunk_size != sidecar_chunk:
                raise ValueError(
                    f"chunk_size override {chunk_size} != sidecar geometry.chunk_size {sidecar_chunk} "
                    f"(sidecar at {sidecar_path_p}); the sidecar's value is authoritative"
                )

        session = ort.InferenceSession(str(onnx_path_p), providers=["CPUExecutionProvider"])
        return cls(session=session, sidecar=sidecar, onnx_path=onnx_path_p, sidecar_path=sidecar_path_p)

    @classmethod
    def from_checkpoint(
        cls,
        chkpt_dir: str,
        chkpt_file: str = "best.th",
        chunk_size: int = 8,
        *,
        output_dir: Optional[str] = None,
        device: str = "cpu",
        verbose: bool = False,
    ) -> "BAFNetPlusOrtStreaming":
        """One-shot: export the BAFNet+ FP32 ONNX from a unified ckpt + load it.

        Calls :func:`export_bafnetplus_to_onnx_from_checkpoint` to export
        the BAFNet+ ONNX from the unified ckpt dir, then :meth:`from_onnx`
        on the result. Useful for tests that don't care which artifact
        path is used.

        Args:
            chkpt_dir: Unified BAFNet+ experiment directory (e.g.
                ``results/experiments/bafnetplus_50ms``).
            chkpt_file: Checkpoint filename (default ``"best.th"``).
            chunk_size: Streaming chunk size in STFT frames (50 ms anchor: 8).
            output_dir: Directory for the exported ``.onnx`` + sidecar.
                Defaults to a temp dir under :func:`tempfile.mkdtemp` so the
                test surface doesn't pollute ``results/onnx/``. The dir is
                NOT auto-cleaned (the caller / test runner manages it).
            device: Execution device for the ORT session (CPU-only at S9).
            verbose: Pass-through to the exporter (one-line export summary).

        Returns:
            A ready-to-stream :class:`BAFNetPlusOrtStreaming`.

        Raises:
            FileNotFoundError: If the unified Hydra config / ckpt or
                per-branch ``bm_*`` Hydra configs are absent locally.
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="bafnetplus_ort_streaming_")
        out_dir_p = Path(output_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        onnx_path = out_dir_p / f"bafnetplus_chunk{chunk_size}_fp32.onnx"
        export_bafnetplus_to_onnx_from_checkpoint(
            chkpt_dir,
            onnx_path,
            chunk_size=chunk_size,
            chkpt_file=chkpt_file,
            verbose=verbose,
        )
        return cls.from_onnx(str(onnx_path), device=device)

    # ----------------------------------------------------------------- host STFT
    def _stft(self, audio: Tensor) -> Tensor:
        """``center=False`` compressed-complex STFT - identical to S3 / S6's ``_stft``.

        Mirrors :func:`src.stft.mag_pha_stft` (``sqrt(...+1e-9)`` magnitude
        floor, ``atan2(...+1e-8)`` phase floor, ``mag**compress_factor``
        compression).

        Args:
            audio: ``[T]`` or ``[1, T]`` audio samples.

        Returns:
            Compressed complex spectrogram ``[1, F, T_fr, 2]``.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        hann_window = torch.hann_window(self.win_size, dtype=audio.dtype, device=audio.device)
        stft_spec = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        stft_spec = torch.view_as_real(stft_spec)
        mag = torch.sqrt(stft_spec.pow(2).sum(-1) + 1e-9)
        pha = torch.atan2(stft_spec[:, :, :, 1] + 1e-8, stft_spec[:, :, :, 0] + 1e-8)
        mag = torch.pow(mag, self.compress_factor)
        com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
        return com

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
            ``(spec [1, F, total_frames_needed, 2], next_stft_context)``.
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

    # ----------------------------------------------------------- ORT graph call
    def _run_ort(
        self,
        bcs_mag: Tensor,
        bcs_pha: Tensor,
        acs_mag: Tensor,
        acs_pha: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Feed one ``T_export``-frame chunk of spectrograms + the current states to ORT.

        Updates ``self._current_states`` with the next-states. Asserts every
        next-state shape equals the matching prev-state shape and that all
        next-states are finite (the structural integrity contract from
        :func:`verify_bafnetplus_core_multistep`).

        Args:
            bcs_mag / bcs_pha / acs_mag / acs_pha: Each ``[1, F, T_export]``
                FP32 torch tensors (on CPU). The host's STFT already ensures
                ``acs_mag > 0`` via the ``sqrt(...+1e-9)`` floor, so the
                in-graph mask-recovery division is safe.

        Returns:
            ``(est_mag [1, F, chunk_size], est_pha [1, F, chunk_size], est_com [1, F, chunk_size, 2])``
            - the ORT graph's first ``chunk_size`` matured output frames.
            (The graph returns ``[1, F, T_export]`` but only the first
            ``chunk_size`` are non-degraded; we trim here.)

        Raises:
            RuntimeError: If a next-state shape mismatches the matching
                prev-state shape or any next-state contains NaN/Inf.
        """
        for tensor, name in (
            (bcs_mag, "bcs_mag"),
            (bcs_pha, "bcs_pha"),
            (acs_mag, "acs_mag"),
            (acs_pha, "acs_pha"),
        ):
            if tensor.shape[2] != self.t_export:
                raise ValueError(
                    f"_run_ort: {name} has T={tensor.shape[2]} but T_export={self.t_export}; "
                    "the host caller must feed exactly T_export frames per ORT call"
                )

        ort_inputs: Dict[str, np.ndarray] = {
            "bcs_mag": bcs_mag.detach().cpu().numpy().astype(np.float32, copy=False),
            "bcs_pha": bcs_pha.detach().cpu().numpy().astype(np.float32, copy=False),
            "acs_mag": acs_mag.detach().cpu().numpy().astype(np.float32, copy=False),
            "acs_pha": acs_pha.detach().cpu().numpy().astype(np.float32, copy=False),
        }
        for name, arr in zip(self.state_names, self._current_states):
            ort_inputs[name] = arr

        outputs = self.session.run(None, ort_inputs)
        n_outs = self.num_non_state_outputs
        next_states = list(outputs[n_outs:])

        _assert_state_integrity(self._current_states, next_states, self.state_names, self.state_shapes)
        self._current_states = [np.ascontiguousarray(s, dtype=np.float32) for s in next_states]

        # Output[0] is always est_mag; output[1]/[2] depend on phase_output_mode.
        est_mag = torch.from_numpy(outputs[0][:, :, : self.chunk_size]).clone()

        if self.phase_output_mode == "complex":
            # S20 atan2-free graph: outputs[1]/[2] are phase_real / phase_imag,
            # the fused complex spectrogram components. Compute est_pha + est_com
            # on the host (FP32 atan2) — this is the whole point of the complex
            # graph variant: atan2 stays out of ORT, eliminating the dominant
            # S18(A) drift source. Eps placement matches src.stft.complex_to_mag_pha.
            phase_real = torch.from_numpy(outputs[1][:, :, : self.chunk_size]).clone()
            phase_imag = torch.from_numpy(outputs[2][:, :, : self.chunk_size]).clone()
            est_pha = torch.atan2(phase_imag + 1e-8, phase_real + 1e-8)
            est_com = mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)
        else:
            est_pha = torch.from_numpy(outputs[1][:, :, : self.chunk_size]).clone()
            est_com = torch.from_numpy(outputs[2][:, :, : self.chunk_size, :]).clone()

        return est_mag, est_pha, est_com

    # ----------------------------------------------------------- iSTFT-OLA
    def _manual_istft_ola(self, est_mag: Tensor, est_pha: Tensor) -> Tensor:
        """Cross-chunk iSTFT-OLA using :func:`manual_istft_ola` (same routine S6 uses).

        Maintains the wrapper's ``_ola_buffer`` / ``_ola_norm`` carry-over
        pair - each call returns ``output_samples_per_chunk = 800`` (50 ms)
        mature samples + updates the ``win - hop = 300``-sample tail.

        Args:
            est_mag: ``[1, F, chunk_size]`` compressed magnitude (post-fusion).
            est_pha: ``[1, F, chunk_size]`` phase (post-fusion).

        Returns:
            ``[output_samples_per_chunk]`` mature audio samples.
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

    # ----------------------------------------------------------- frame buffer
    def _append_chunk_frames(
        self,
        bcs_spec: Tensor,
        acs_spec: Tensor,
    ) -> None:
        """Append one chunk's STFT frames to the per-branch frame buffer.

        Call 0: append all ``total_frames_needed = 11`` frames (the wrapper
        sees fresh audio + zero past context; every frame is "new").
        Subsequent calls: append the last ``chunk_size = 8`` frames - the
        first ``L_enc = 3`` frames overlap with the previous call's tail
        (they're bit-identical: the same absolute audio samples + the
        same updated ``_stft_context`` produce identical STFTs).

        Args:
            bcs_spec / acs_spec: ``[1, F, total_frames_needed, 2]`` per-call
                STFT outputs.
        """
        bcs_mag, bcs_pha = complex_to_mag_pha(bcs_spec, stack_dim=-1)
        acs_mag, acs_pha = complex_to_mag_pha(acs_spec, stack_dim=-1)
        if self._num_chunks_received == 0:
            self._frame_buffer_bcs_mag = bcs_mag
            self._frame_buffer_bcs_pha = bcs_pha
            self._frame_buffer_acs_mag = acs_mag
            self._frame_buffer_acs_pha = acs_pha
        else:
            new_slice = slice(self.input_lookahead_frames, self.input_lookahead_frames + self.chunk_size)
            self._frame_buffer_bcs_mag = torch.cat([self._frame_buffer_bcs_mag, bcs_mag[:, :, new_slice]], dim=2)
            self._frame_buffer_bcs_pha = torch.cat([self._frame_buffer_bcs_pha, bcs_pha[:, :, new_slice]], dim=2)
            self._frame_buffer_acs_mag = torch.cat([self._frame_buffer_acs_mag, acs_mag[:, :, new_slice]], dim=2)
            self._frame_buffer_acs_pha = torch.cat([self._frame_buffer_acs_pha, acs_pha[:, :, new_slice]], dim=2)
        self._num_chunks_received += 1

    def _consume_one_export_chunk(self) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Take the first ``T_export`` frames, feed to ORT, pop ``chunk_size`` frames.

        Returns:
            ``(est_mag, est_pha, est_com)`` (each ``[1, F, chunk_size]`` /
            ``[1, F, chunk_size, 2]``) on success; ``None`` if the frame
            buffer doesn't yet hold ``T_export`` frames.
        """
        t_buffered = self._frame_buffer_bcs_mag.shape[2]
        if t_buffered < self.t_export:
            return None
        bcs_mag = self._frame_buffer_bcs_mag[:, :, : self.t_export]
        bcs_pha = self._frame_buffer_bcs_pha[:, :, : self.t_export]
        acs_mag = self._frame_buffer_acs_mag[:, :, : self.t_export]
        acs_pha = self._frame_buffer_acs_pha[:, :, : self.t_export]
        est_mag, est_pha, est_com = self._run_ort(bcs_mag, bcs_pha, acs_mag, acs_pha)
        # Pop chunk_size frames off the front.
        self._frame_buffer_bcs_mag = self._frame_buffer_bcs_mag[:, :, self.chunk_size :]
        self._frame_buffer_bcs_pha = self._frame_buffer_bcs_pha[:, :, self.chunk_size :]
        self._frame_buffer_acs_mag = self._frame_buffer_acs_mag[:, :, self.chunk_size :]
        self._frame_buffer_acs_pha = self._frame_buffer_acs_pha[:, :, self.chunk_size :]
        return est_mag, est_pha, est_com

    # ------------------------------------------------------------- public APIs
    @torch.inference_mode()
    def process_samples(self, bcs_chunk: Tensor, acs_chunk: Tensor) -> Optional[Tensor]:
        """Feed paired audio chunks; return one matured 800-sample (50 ms) output or ``None``.

        Mirrors :meth:`BAFNetPlusStreaming.process_samples` exactly -
        accumulates ``bcs_chunk`` and ``acs_chunk`` in the per-stream input
        buffers; once BOTH have ``>= samples_per_chunk`` samples, computes
        per-branch STFTs, appends to the STFT-frame accumulator, and if the
        accumulator holds ``>= T_export`` frames, runs ONE ORT call + the
        iSTFT-OLA and returns ``output_samples_per_chunk = 800`` matured
        samples. Returns ``None`` during the warm-up (typically the first 2
        calls for the 50 ms anchor: call 0 fills 11 frames, call 1 brings
        the accumulator to 19 frames >= 14, fires ORT).

        Args:
            bcs_chunk: ``[T]`` or ``[1, T]`` BCS audio (any length - buffered).
            acs_chunk: ``[T]`` or ``[1, T]`` ACS audio (any length - buffered).

        Returns:
            ``Tensor [output_samples_per_chunk]`` of mature fused-enhanced
            audio samples, or ``None`` while either input buffer / the
            STFT-frame accumulator is still filling.

        Raises:
            ValueError: If either chunk is 2-D with a batch dimension != 1.
        """
        if bcs_chunk.dim() == 2:
            if bcs_chunk.shape[0] != 1:
                raise ValueError("batch size must be 1 for streaming")
            bcs_chunk = bcs_chunk.squeeze(0)
        if acs_chunk.dim() == 2:
            if acs_chunk.shape[0] != 1:
                raise ValueError("batch size must be 1 for streaming")
            acs_chunk = acs_chunk.squeeze(0)
        bcs_chunk = bcs_chunk.to(torch.float32).cpu()
        acs_chunk = acs_chunk.to(torch.float32).cpu()

        self._input_buffer_bcs = torch.cat([self._input_buffer_bcs, bcs_chunk])
        self._input_buffer_acs = torch.cat([self._input_buffer_acs, acs_chunk])

        if len(self._input_buffer_bcs) < self.samples_per_chunk:
            return None
        if len(self._input_buffer_acs) < self.samples_per_chunk:
            return None

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

        bcs_mag_call, _ = complex_to_mag_pha(bcs_spec, stack_dim=-1)
        acs_mag_call, _ = complex_to_mag_pha(acs_spec, stack_dim=-1)
        valid = self.chunk_size
        self._input_mag_fifo.append((bcs_mag_call[:, :, :valid].clone(), acs_mag_call[:, :, :valid].clone()))

        self._append_chunk_frames(bcs_spec, acs_spec)

        # Slide input buffers regardless of whether ORT fires (matches S3/S6).
        self._input_buffer_bcs = self._input_buffer_bcs[self.output_samples_per_chunk :]
        self._input_buffer_acs = self._input_buffer_acs[self.output_samples_per_chunk :]

        ort_result = self._consume_one_export_chunk()
        if ort_result is None:
            return None

        # Pop the aligned input-mag chunk (popped per matured chunk for API parity with S6).
        if self._input_mag_fifo:
            self._input_mag_fifo.popleft()

        est_mag, est_pha, _est_com = ort_result
        return self._manual_istft_ola(est_mag, est_pha)

    @torch.inference_mode()
    def process_audio(self, bcs_audio: Tensor, acs_audio: Tensor) -> Tensor:
        """Stream a complete paired utterance and return the fused enhanced audio.

        Mirrors :meth:`BAFNetPlusStreaming.process_audio` exactly. Resets
        state, zero-pad-flushes both streams long enough to drain the
        lookahead pipeline + the OLA tail, loops :meth:`process_samples` in
        ``output_samples_per_chunk``-sized pieces, concatenates the matured
        outputs, and trims to ``len(bcs_audio)``.

        Args:
            bcs_audio: ``[T]`` or ``[1, T]`` BCS audio.
            acs_audio: ``[T]`` or ``[1, T]`` ACS audio (sample-aligned with BCS).

        Returns:
            ``Tensor [T']`` enhanced audio with ``T' = min(produced, len(bcs_audio))``.
        """
        if bcs_audio.dim() == 2:
            bcs_audio = bcs_audio.squeeze(0)
        if acs_audio.dim() == 2:
            acs_audio = acs_audio.squeeze(0)
        bcs_audio = bcs_audio.to(torch.float32).cpu()
        acs_audio = acs_audio.to(torch.float32).cpu()
        ref_length = min(len(bcs_audio), len(acs_audio))
        bcs_audio = bcs_audio[:ref_length]
        acs_audio = acs_audio[:ref_length]

        self.reset_state()

        flush_size = self.samples_per_chunk * (self.total_lookahead + 2)
        bcs_padded = torch.cat([bcs_audio, torch.zeros(flush_size, dtype=torch.float32)])
        acs_padded = torch.cat([acs_audio, torch.zeros(flush_size, dtype=torch.float32)])

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
            return torch.zeros(0, dtype=torch.float32)
        cat = torch.cat(outputs)
        if len(cat) > ref_length:
            cat = cat[:ref_length]
        return cat

    @torch.inference_mode()
    def process_spectrogram(self, noisy_bcs_com: Tensor, noisy_acs_com: Tensor) -> SpectrogramChunk:
        """Stream a complete paired spectrogram in chunks (spectrogram in -> out, no iSTFT).

        Mirrors :meth:`BAFNetPlusStreaming.process_spectrogram`. Resets, loops
        the time axis in ``chunk_size`` steps feeding ``noisy_*_com[:, :,
        t : t + T_export, :]`` slices, concatenates matured fused spectrogram
        triples along the time axis. No internal flush.

        Args:
            noisy_bcs_com: ``[B, F, T, 2]`` (``B == 1``) or ``[F, T, 2]`` BCS
                compressed complex spectrogram.
            noisy_acs_com: same shape for ACS.

        Returns:
            ``(est_mag, est_pha, est_com)`` for the streamed frames:
            ``est_mag`` / ``est_pha`` are ``[1, F, T_stream]`` and ``est_com``
            is ``[1, F, T_stream, 2]`` with ``T_stream < T``.

        Raises:
            ValueError: If either spectrogram has a batch dimension != 1,
                or the two have different shapes.
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
        for t in range(0, t_total - self.t_export + 1, self.chunk_size):
            bcs_chunk_com = noisy_bcs_com[:, :, t : t + self.t_export, :]
            acs_chunk_com = noisy_acs_com[:, :, t : t + self.t_export, :]
            bcs_mag, bcs_pha = complex_to_mag_pha(bcs_chunk_com, stack_dim=-1)
            acs_mag, acs_pha = complex_to_mag_pha(acs_chunk_com, stack_dim=-1)
            est_mag, est_pha, est_com = self._run_ort(bcs_mag, bcs_pha, acs_mag, acs_pha)
            mags.append(est_mag)
            phas.append(est_pha)
            coms.append(est_com)

        if not mags:
            empty = torch.zeros(1, self.freq_size, 0)
            return empty, empty.clone(), torch.zeros(1, self.freq_size, 0, 2)
        return torch.cat(mags, dim=2), torch.cat(phas, dim=2), torch.cat(coms, dim=2)

    def forward(self, bcs_audio: Tensor, acs_audio: Tensor) -> Tensor:
        """Equivalent to :meth:`process_audio`."""
        return self.process_audio(bcs_audio, acs_audio)

    # ------------------------------------------------------------- diagnostics
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  onnx_path={self.onnx_path}, ablation_mode={self.ablation_mode},\n"
            f"  chunk_size={self.chunk_size}, L_enc={self.encoder_lookahead}, "
            f"L_dec={self.decoder_lookahead}, L_alpha={self.alpha_time_lookahead},\n"
            f"  T_export={self.t_export}, samples_per_chunk={self.samples_per_chunk}, "
            f"output_samples_per_chunk={self.output_samples_per_chunk},\n"
            f"  latency_ms={self.latency_ms:.2f}, num_states={self.num_states},\n"
            f"  checkpoint.md5={self.checkpoint_info.get('md5')},\n"
            f")"
        )


# =============================================================== state helpers
def _init_states_from_sidecar(state_shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    """Build a flat list of zero ``np.float32`` ndarrays from the sidecar shapes."""
    return [np.zeros(shape, dtype=np.float32) for shape in state_shapes]


def _assert_state_integrity(
    prev_states: List[np.ndarray],
    next_states: List[np.ndarray],
    state_names: List[str],
    expected_shapes: List[Tuple[int, ...]],
) -> None:
    """Verify every next-state shape matches the prev-state shape + all states finite.

    Mirrors the structural gates that :func:`verify_bafnetplus_core_multistep`
    enforces - applied at every ``sess.run`` call so a regression surfaces
    immediately.
    """
    if not (len(prev_states) == len(next_states) == len(state_names) == len(expected_shapes)):
        raise RuntimeError(
            f"state count mismatch: prev={len(prev_states)} next={len(next_states)} "
            f"expected={len(state_names)} shapes={len(expected_shapes)}"
        )
    for i, (prev, nxt, name, expected) in enumerate(zip(prev_states, next_states, state_names, expected_shapes)):
        if tuple(prev.shape) != tuple(nxt.shape):
            raise RuntimeError(f"state {i} ({name!r}) shape mismatch: prev={tuple(prev.shape)} next={tuple(nxt.shape)}")
        if tuple(nxt.shape) != tuple(expected):
            raise RuntimeError(
                f"state {i} ({name!r}) next shape {tuple(nxt.shape)} != expected {tuple(expected)} from sidecar"
            )
        if not np.isfinite(nxt).all():
            raise RuntimeError(f"state {i} ({name!r}) contains non-finite values after ORT call")

__all__ = ["BAFNetPlusOrtStreaming"]
