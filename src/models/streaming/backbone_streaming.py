"""PyTorch streaming wrapper for a single BAFNet+ ``Backbone`` (spectrogram → spectrogram).

S3 of the streaming-ONNX rebuild (Stage 1 pt.2). ``BackboneStreaming`` runs a
single mapping/masking ``Backbone`` chunk by chunk so the chunked output equals
the full-sequence ``Backbone.forward`` output (within the float32 reordering
floor), using the S2 stateful convs (``convert_to_stateful`` + streaming mode) +
the ``StateFramesContext`` state guard + the lookahead calculator.

What this is / is not
---------------------
- It is **spectrogram → spectrogram**: it takes audio (or a spectrogram), runs
  the host STFT (``center=True`` emulated by ``center=False`` + a 200-sample
  past/future context buffer), and returns the *spectral* estimates
  ``(est_mag, est_pha, est_com)`` — exactly what ``Backbone.forward`` returns.
- It does **not** do iSTFT / overlap-add. There is no OLA buffer here (only the
  two host buffers below). The iSTFT OLA buffer that turns ``est_mag/est_pha``
  back into audio is added by S6's ``BAFNetPlusStreaming``.
- It does **not** do BAFNet+ fusion (mapping+masking calibration/alpha). That is
  S5+/S6 and is intentionally absent.

Two host buffers (cf. ``docs/wiki/concepts/lacosenet-backbone-streaming.md`` §4)
-------------------------------------------------------------------------------
(a) **STFT-context buffer** ``_stft_context`` — ``win_size // 2 = 200`` samples.
    Training uses ``center=True`` (reflect-pad ``n_fft//2`` on each side). The
    stream emulates that with ``center=False``: each chunk's STFT input is
    ``[ _stft_context (200 past samples) | chunk_samples ]`` and the chunk window
    already includes ``win_size//2`` *future* samples (the ``+ win_size//2`` term
    in ``samples_per_chunk``), so the first/last STFT frame of a chunk lands at
    the same position a ``center=True`` frame would. ``_stft_context`` is updated
    after each chunk from the input buffer (the 200 samples immediately preceding
    the next chunk). It starts as zeros, so the first ~2 STFT frames of a stream
    match ``center=True`` only when the utterance starts with ≥ ``win_size//2``
    leading silence (otherwise they differ; the documented gate trims them off —
    see "Parity gate" below).
(b) **Feature buffer** ``feature_buffer`` (``List[Dict]``) + ``_buffered_frames``.
    The encoder + TS-blocks run on every chunk over ``valid_frames = chunk_size``
    real frames (the ``L_enc`` lookahead frames are processed for context then
    discarded), appending ``chunk_size`` encoder-output frames to the buffer. The
    decoders wait until ``_buffered_frames >= chunk_size + L_dec``, then decode
    ``chunk_size + L_dec`` extended frames inside ``StateFramesContext(chunk_size)``
    and trim the output to the first ``chunk_size`` frames. If ``L_dec == 0`` the
    immediate-decode path is used instead (decode ``chunk_size`` frames directly,
    no buffering). ``L_enc`` is the *encoder*-side warm-up (``input_lookahead_frames``).

Chunk geometry (for the 50 ms BAFNet+ Backbone: chunk_size=8, L_enc=L_dec=3,
hop=100, win=n_fft=400 — but always derived at runtime, never hard-coded):
    ``input_lookahead_frames``       = ``L_enc``                          (= 3)
    ``total_lookahead``              = ``L_enc + L_dec``                  (= 6)
    ``total_frames_needed``          = ``chunk_size + L_enc``             (= 11)  STFT frames per encoder call
    ``samples_per_chunk``            = ``(total_frames_needed-1)*hop + win//2`` (= 1200)  input samples per chunk
    ``output_frames_per_chunk``      = ``chunk_size``                     (= 8)
    ``output_samples_per_chunk``     = ``chunk_size * hop``               (= 800 samples = 50 ms output step)
    ``latency_samples``              = ``total_lookahead*hop + win//2``   (= 800 = 50.0 ms @ 16 kHz)

Parity gate (S3 exit gate, model-level)
----------------------------------------
Feeding a complete utterance's spectrogram through ``process_spectrogram`` and
comparing to ``Backbone.forward(noisy_com)`` must match within float32 tolerance
*after the documented align/trim*:
  - streaming output frame ``k`` corresponds to reference frame ``k`` (no
    positional shift — the lookahead only delays *when* the first output appears);
  - the last ``total_lookahead`` produced frames are excluded (they used
    zero-padded lookahead because the input ran out), i.e. compare
    ``ref[..., :T_stream - total_lookahead]`` vs ``stream[..., :T_stream - total_lookahead]``.
This must hold for both ``infer_type='mapping'`` and ``infer_type='masking'``,
regardless of the weights (a freshly-initialised Backbone passes too). It mirrors
LaCoSENet ``src/verify_equivalence.py``.

Ported from LaCoSENet ``src/models/streaming/lacosenet.py``, dropping the
reshape-free / ``fold_bn`` / ``cpu_optimizations`` paths (FP32, no NPU) and the
iSTFT/OLA reconstruction (S6), and changing the return type from audio samples to
the spectral triple. BAFNet+ STFT numerical floors are preserved by reusing
``src.stft`` (``mag_pha_stft`` ``sqrt(...+1e-9)`` / ``atan2(...+1e-8)``,
``complex_to_mag_pha`` ``sqrt(...+1e-8)``) — no spectral math is re-implemented
here beyond inlining the same ``center=False`` STFT formula in ``_stft``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.checkpoint import load_checkpoint, load_model
from src.models.backbone import Backbone
from src.models.streaming.context import StateFramesContext
from src.models.streaming.converters import prepare_streaming_model, reset_streaming_state
from src.models.streaming.lookahead import compute_lookahead
from src.stft import complex_to_mag_pha, mag_pha_to_complex

logger = logging.getLogger(__name__)

# === FROZEN RETURN CONTRACT (S3) =============================================
# A "spectrogram chunk" is the matured-frame output of one streaming step:
#   est_mag : Tensor [1, F, chunk_size]      compressed magnitude  (F = n_fft//2 + 1)
#   est_pha : Tensor [1, F, chunk_size]      phase (radians)
#   est_com : Tensor [1, F, chunk_size, 2]   complex spectrogram (real, imag) =
#                                            mag_pha_to_complex(est_mag, est_pha)
# These are the SAME three tensors Backbone.forward returns, restricted to the
# chunk_size matured STFT frames — equivalent (once iSTFT/OLA'd downstream by S6)
# to a chunk_size * hop_size-sample (= 800 samples = 50 ms for the 50 ms variant)
# output step. process_samples() returns this triple, or None while the lookahead
# pipeline is still filling. process_audio() / process_spectrogram() return the
# concatenated full-utterance triple (each component cat'd along the time axis).
# =============================================================================
SpectrogramChunk = Tuple[Tensor, Tensor, Tensor]


class BackboneStreaming(nn.Module):
    """Chunk-by-chunk streaming wrapper around a single (stateful) ``Backbone``.

    Use :meth:`from_checkpoint` (load a ``bm_*`` experiment's Hydra config +
    ``best.th``) or :meth:`from_model` (an already-instantiated ``Backbone``) to
    build one — both convert the convs to stateful, enable streaming, and derive
    the chunk geometry from :func:`compute_lookahead`. ``__init__`` itself expects
    the ``model`` to already be prepared (stateful + streaming-enabled).

    Attributes:
        model: The prepared stateful ``Backbone`` (a registered submodule).
        infer_type: ``'masking'`` (``est_mag = mag * mask``) or ``'mapping'``
            (``est_mag = mask``), read from the backbone.
        chunk_size: STFT frames produced per output step.
        encoder_lookahead / decoder_lookahead: ``L_enc`` / ``L_dec`` (right-pad
            frame sums, from :func:`compute_lookahead`).
        total_lookahead: ``L_enc + L_dec``.
        total_frames_needed: STFT frames the encoder consumes per chunk
            (``chunk_size + L_enc``).
        samples_per_chunk: input audio samples buffered per chunk.
        output_samples_per_chunk: input-buffer advance per output step
            (``chunk_size * hop_size``).
        latency_samples / latency_ms: algorithmic latency.
        input_buffer / feature_buffer / _buffered_frames / _stft_context: the
            two host buffers (audio sample buffer + feature buffer) and the STFT
            past-context buffer. NOT ``nn.Module`` buffers — kept as plain tensors
            and device-synced on demand inside :meth:`process_samples`.
    """

    def __init__(
        self,
        model: Backbone,
        chunk_size: int = 8,
        encoder_lookahead: int = 0,
        decoder_lookahead: int = 0,
        hop_size: int = 100,
        n_fft: int = 400,
        win_size: int = 400,
        compress_factor: float = 0.3,
        sample_rate: int = 16000,
    ):
        """Initialize from an already-prepared (stateful, streaming-enabled) ``Backbone``.

        Args:
            model: A ``Backbone`` already passed through
                :func:`~src.models.streaming.converters.prepare_streaming_model`
                (convs are stateful, streaming mode on, in ``eval()``).
            chunk_size: STFT frames per output step (50 ms variant: 8).
            encoder_lookahead: ``L_enc`` (encoder right-pad frame sum).
            decoder_lookahead: ``L_dec`` (decoder right-pad frame sum).
            hop_size: STFT hop in samples.
            n_fft: FFT size.
            win_size: Window size (``win_size//2`` is the STFT past/future
                context size used to emulate ``center=True``).
            compress_factor: Magnitude compression exponent.
            sample_rate: Audio sample rate (informational — used for ``latency_ms``).
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.infer_type: str = getattr(model, "infer_type", "masking")

        # STFT parameters.
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.sample_rate = sample_rate
        # center=True is emulated by center=False + a win//2 past context buffer;
        # the chunk window already carries win//2 future samples (the +win//2 term
        # in samples_per_chunk). Both equal win//2 here.
        self.stft_future_samples = self.win_size // 2
        self.stft_center_delay_samples = self.stft_future_samples

        # Streaming geometry.
        self.chunk_size = chunk_size
        self.encoder_lookahead = encoder_lookahead
        self.decoder_lookahead = decoder_lookahead
        self.input_lookahead_frames = int(encoder_lookahead)
        self.total_lookahead = self.input_lookahead_frames + decoder_lookahead

        self.total_frames_needed = chunk_size + self.input_lookahead_frames
        # Full STFT input per chunk = [ past_context(win//2) | chunk_samples ];
        # chunk_samples already includes win//2 future samples for the right STFT
        # context, so center=False over that input yields total_frames_needed frames.
        self.samples_per_chunk = (self.total_frames_needed - 1) * hop_size + self.stft_future_samples

        # Output step is ALWAYS chunk_size frames. The input buffer slides by this
        # many samples per processed chunk; anything else drifts the alignment.
        self.output_frames_per_chunk = chunk_size
        self.output_samples_per_chunk = self.output_frames_per_chunk * hop_size

        # Algorithmic latency: lookahead frames + the STFT center delay.
        self.latency_samples = self.total_lookahead * hop_size + self.stft_center_delay_samples
        self.latency_ms = self.latency_samples / sample_rate * 1000.0

        self._streaming_config: Dict[str, Any] = {}
        self._reset_buffers()

    # ------------------------------------------------------------------ buffers
    def _reset_buffers(self) -> None:
        """Reset the audio buffer, feature buffer and STFT context (not conv state)."""
        self.input_buffer = torch.tensor([], dtype=torch.float32)
        self.feature_buffer: List[Dict[str, Any]] = []  # encoder outputs awaiting the decoder
        self._buffered_frames = 0
        self._stft_context = torch.zeros(self.win_size // 2, dtype=torch.float32)

    def reset_state(self) -> None:
        """Reset all streaming state for a new utterance.

        Clears ``input_buffer`` / ``feature_buffer`` / ``_buffered_frames`` /
        ``_stft_context`` (-> zeros of size ``win_size//2``) and calls
        :func:`~src.models.streaming.converters.reset_streaming_state` on the
        model (every stateful conv's ``_state`` -> ``None``).
        """
        self._reset_buffers()
        reset_streaming_state(self.model)

    @property
    def device(self) -> torch.device:
        """Device of the underlying model."""
        return next(self.model.parameters()).device

    @property
    def streaming_config(self) -> Dict[str, Any]:
        """Geometry + provenance summary (useful for export metadata / debugging)."""
        return {
            **self._streaming_config,
            "infer_type": self.infer_type,
            "chunk_size": self.chunk_size,
            "encoder_lookahead": self.encoder_lookahead,
            "decoder_lookahead": self.decoder_lookahead,
            "input_lookahead_frames": self.input_lookahead_frames,
            "total_lookahead": self.total_lookahead,
            "total_frames_needed": self.total_frames_needed,
            "samples_per_chunk": self.samples_per_chunk,
            "output_frames_per_chunk": self.output_frames_per_chunk,
            "output_samples_per_chunk": self.output_samples_per_chunk,
            "stft_center_delay_samples": self.stft_center_delay_samples,
            "latency_samples": self.latency_samples,
            "latency_ms": self.latency_ms,
            "n_fft": self.n_fft,
            "hop_size": self.hop_size,
            "win_size": self.win_size,
            "compress_factor": self.compress_factor,
            "sample_rate": self.sample_rate,
        }

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_model(
        cls,
        backbone: Backbone,
        chunk_size: int = 8,
        device: str = "cpu",
        verbose: bool = False,
    ) -> "BackboneStreaming":
        """Build a ``BackboneStreaming`` from an instantiated (non-stateful) ``Backbone``.

        Steps: derive ``L_enc`` / ``L_dec`` via :func:`compute_lookahead` on the
        *original* module (the stateful convs are not ``AsymmetricConv2d``
        instances, so lookahead must be read before conversion); read the STFT
        params off the backbone; convert + enable streaming via
        :func:`prepare_streaming_model` (operates on a deep copy, so ``backbone``
        is left untouched and can serve as the non-streaming reference).

        Args:
            backbone: An instantiated ``Backbone`` (its ``dense_encoder`` /
                ``sequence_block`` / ``mask_decoder`` / ``phase_decoder`` /
                ``n_fft`` / ``hop_size`` / ``win_size`` / ``compress_factor`` /
                ``infer_type`` are used).
            chunk_size: STFT frames per output step.
            device: Device for the prepared streaming model.
            verbose: Log a one-line geometry summary.

        Returns:
            A ready-to-stream ``BackboneStreaming`` (state already reset).

        Raises:
            TypeError: If ``backbone`` is not a ``Backbone``.
            ValueError: If ``backbone`` has unset STFT params (``n_fft`` etc.).
            AttributeError: If ``backbone`` lacks the expected submodules
                (re-raised from :func:`compute_lookahead`).
        """
        if not isinstance(backbone, Backbone):
            raise TypeError(f"from_model expects a Backbone, got {type(backbone).__name__}")
        if backbone.n_fft is None or backbone.hop_size is None or backbone.win_size is None:
            raise ValueError("Backbone is missing STFT params (n_fft / hop_size / win_size)")

        lookahead = compute_lookahead(backbone)  # MUST run before stateful conversion
        n_fft = int(backbone.n_fft)
        hop_size = int(backbone.hop_size)
        win_size = int(backbone.win_size)
        compress_factor = float(backbone.compress_factor)

        prepared, n_stateful = prepare_streaming_model(backbone, device=device, inplace=False, verbose=verbose)

        instance = cls(
            model=cast(Backbone, prepared),  # deepcopy of a Backbone with stateful conv leaves
            chunk_size=chunk_size,
            encoder_lookahead=lookahead.encoder_lookahead,
            decoder_lookahead=lookahead.decoder_lookahead,
            hop_size=hop_size,
            n_fft=n_fft,
            win_size=win_size,
            compress_factor=compress_factor,
        )
        instance._streaming_config = {
            "source": "from_model",
            "model_class": type(backbone).__name__,
            "stateful_layer_count": n_stateful,
            "T_export_planned": chunk_size + lookahead.total_lookahead,
            "encoder_breakdown": lookahead.encoder_breakdown,
            "decoder_breakdown": lookahead.decoder_breakdown,
        }
        instance.reset_state()

        if verbose:
            logger.info(
                "BackboneStreaming: infer_type=%s chunk_size=%d L_enc=%d L_dec=%d "
                "total_frames_needed=%d samples_per_chunk=%d output_samples_per_chunk=%d latency=%.1fms",
                instance.infer_type,
                chunk_size,
                lookahead.encoder_lookahead,
                lookahead.decoder_lookahead,
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
    ) -> "BackboneStreaming":
        """Build a ``BackboneStreaming`` from a single-``Backbone`` experiment directory.

        Reads ``<chkpt_dir>/.hydra/config.yaml#model`` (``model_lib`` /
        ``model_class`` / ``param``), instantiates the ``Backbone``, loads
        ``<chkpt_dir>/<chkpt_file>``'s ``model`` state dict, then delegates to
        :meth:`from_model`. Intended for the pre-joint single-Backbone checkpoints
        (e.g. ``results/experiments/bm_map_50ms`` / ``bm_mask_50ms``) — do NOT
        point this at the unified ``bafnetplus_50ms`` checkpoint (that is S6) and
        do NOT pass ``checkpoint_mapping`` / ``checkpoint_masking`` (the recorded
        paths are non-local).

        Args:
            chkpt_dir: Experiment directory containing ``.hydra/config.yaml`` and
                the checkpoint file.
            chkpt_file: Checkpoint filename (default ``"best.th"``).
            chunk_size: STFT frames per output step.
            device: Device for loading + streaming.
            verbose: Print/log a loading + geometry summary.

        Returns:
            A ready-to-stream ``BackboneStreaming``.

        Raises:
            FileNotFoundError: If the Hydra config or checkpoint file is missing.
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
        if verbose:
            infer_type = model_cfg.param.get("infer_type", "masking") if hasattr(model_cfg.param, "get") else "masking"
            print(
                f"Loading BackboneStreaming from {chkpt_dir_path} "
                f"({model_cfg.model_lib}.{model_cfg.model_class}, infer_type={infer_type}, chunk_size={chunk_size})"
            )

        backbone = load_model(model_cfg.model_lib, model_cfg.model_class, model_cfg.param, device)
        backbone = load_checkpoint(backbone, str(chkpt_dir_path), chkpt_file, device)

        instance = cls.from_model(backbone, chunk_size=chunk_size, device=device, verbose=verbose)
        instance._streaming_config["source"] = "from_checkpoint"
        instance._streaming_config["chkpt_dir"] = str(chkpt_dir_path)
        instance._streaming_config["chkpt_file"] = chkpt_file
        return instance

    # ----------------------------------------------------------------- host STFT
    def _stft(self, audio: Tensor) -> Tensor:
        """``center=False`` compressed-complex STFT (the ``center=True`` emulation's STFT).

        The caller (:meth:`process_samples`) prepends the ``win_size//2`` past
        context and ensures the input already carries ``win_size//2`` future
        samples, so this plain ``center=False`` STFT yields frames at the same
        positions a ``center=True`` STFT would. Mirrors ``src.stft.mag_pha_stft``
        (``sqrt(...+1e-9)`` magnitude floor, ``atan2(...+1e-8)`` phase floor,
        ``mag**compress_factor`` compression).

        Args:
            audio: ``[T]`` or ``[1, T]`` audio samples.

        Returns:
            Compressed complex spectrogram ``[1, F, T_fr, 2]``.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        hann_window = torch.hann_window(self.win_size).to(audio.device)
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

    # --------------------------------------------------------- model-level steps
    def _process_encoder(self, spectrogram: Tensor) -> Tuple[Tensor, Tensor, int]:
        """Run ``DenseEncoder`` + ``TS_BLOCK`` on one chunk's spectrogram.

        Args:
            spectrogram: ``[B, F, T, 2]`` compressed complex spectrogram, ``B == 1``;
                ``T`` is normally ``chunk_size + L_enc`` (fewer at the tail).

        Returns:
            ``(mag, ts_out, valid_frames)`` where ``mag`` is ``[B, F, T]``,
            ``ts_out`` is ``[B, C, T, F]`` and ``valid_frames = min(T, chunk_size)``
            is the number of leading frames that update conv state (the rest are
            lookahead-only future context).
        """
        _, _, T, _ = spectrogram.shape
        mag, pha = complex_to_mag_pha(spectrogram, stack_dim=-1)  # [B, F, T] each
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)  # [B, 2, T, F]
        valid_frames = min(T, self.chunk_size)
        with StateFramesContext(valid_frames):
            encoded = self.model.dense_encoder(x)
            ts_out = self.model.sequence_block(encoded)  # [B, C, T, F]
        return mag, ts_out, valid_frames

    def _decode(self, features: Tensor, mag_slice: Tensor) -> Tuple[Tensor, Tensor]:
        """Run ``MaskDecoder`` + ``PhaseDecoder`` on encoder-output features.

        State updates are bounded to ``chunk_size`` frames; the trailing ``L_dec``
        lookahead frames (when ``features`` is the extended ``chunk_size + L_dec``
        slice) feed the decoder DS-DDB right padding but not its conv state. The
        ``mask`` / ``est_pha`` shapes match ``Backbone.forward`` (``[B, F, T]``).

        Args:
            features: ``[B, C, T, F]`` encoder + TS-block output.
            mag_slice: ``[B, F, T]`` input magnitude aligned with ``features``
                (used only for ``infer_type == 'masking'``).

        Returns:
            ``(est_mag, est_pha)``, each ``[B, F, T]`` (untrimmed — the caller
            trims to ``chunk_size`` in the buffered path).
        """
        with StateFramesContext(self.chunk_size):
            mask = self.model.mask_decoder(features).squeeze(1).transpose(1, 2)  # [B, F, T]
            est_pha = self.model.phase_decoder(features).squeeze(1).transpose(1, 2)  # [B, F, T]
        if self.infer_type == "masking":
            est_mag = mag_slice * mask
        else:  # 'mapping'
            est_mag = mask
        return est_mag, est_pha

    def _process_decoder_immediate(self, ts_out: Tensor, mag: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode directly without feature buffering (used only when ``L_dec == 0``)."""
        features = ts_out[:, :, : self.chunk_size, :]
        mag_slice = mag[:, :, : self.chunk_size]
        return self._decode(features, mag_slice)

    def _process_decoder_buffered(
        self, ts_out: Tensor, mag: Tensor, valid_frames: int
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Decode with the feature buffer (``L_dec > 0``): wait, decode extended, trim.

        Appends ``valid_frames`` (= ``chunk_size``) encoder-output frames; once
        ``_buffered_frames >= chunk_size + L_dec``, decodes the first
        ``chunk_size + L_dec`` buffered frames, trims the output to ``chunk_size``,
        and pops ``chunk_size`` frames off the front of the buffer (the ``L_dec``
        lookahead frames carry over to the next decode).

        Returns:
            ``(est_mag, est_pha)`` (each ``[B, F, chunk_size]``) once enough frames
            are buffered, else ``None``.
        """
        self.feature_buffer.append(
            {
                "features": ts_out[:, :, :valid_frames, :],
                "mag": mag[:, :, :valid_frames],
                "frames": valid_frames,
            }
        )
        self._buffered_frames += valid_frames

        total_needed = self.chunk_size + self.decoder_lookahead
        if self._buffered_frames < total_needed:
            return None

        all_features = torch.cat([buf["features"] for buf in self.feature_buffer], dim=2)
        all_mag = torch.cat([buf["mag"] for buf in self.feature_buffer], dim=2)
        extended_features = all_features[:, :, :total_needed, :]
        extended_mag = all_mag[:, :, :total_needed]

        est_mag, est_pha = self._decode(extended_features, extended_mag)
        est_mag = est_mag[:, :, : self.chunk_size]
        est_pha = est_pha[:, :, : self.chunk_size]

        # Consume chunk_size frames from the front of the feature buffer.
        frames_to_remove = self.chunk_size
        removed = 0
        while removed < frames_to_remove and self.feature_buffer:
            buf = self.feature_buffer[0]
            if buf["frames"] <= (frames_to_remove - removed):
                removed += buf["frames"]
                self.feature_buffer.pop(0)
            else:
                keep = buf["frames"] - (frames_to_remove - removed)
                buf["features"] = buf["features"][:, :, -keep:, :]
                buf["mag"] = buf["mag"][:, :, -keep:]
                buf["frames"] = keep
                removed = frames_to_remove
        self._buffered_frames -= removed

        return est_mag, est_pha

    def process_spectrogram_buffered(self, spectrogram: Tensor) -> Optional[SpectrogramChunk]:
        """One model-level streaming step: encoder + TS-block + (conditional) decode.

        Args:
            spectrogram: ``[B, F, T, 2]`` (``B == 1``, or ``[F, T, 2]`` for a
                single stream). ``T`` is normally ``chunk_size + L_enc`` (the
                ``L_enc`` lookahead frames overlap the next chunk's leading frames).

        Returns:
            ``(est_mag, est_pha, est_com)`` for the matured ``chunk_size`` frames,
            or ``None`` while the decoder is still buffering (only with ``L_dec > 0``).
        """
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)
        if spectrogram.shape[0] != 1:
            raise ValueError(f"streaming requires batch size 1, got {spectrogram.shape[0]}")

        mag, ts_out, valid_frames = self._process_encoder(spectrogram)

        if self.decoder_lookahead == 0 and ts_out.shape[2] >= self.chunk_size:
            est_mag, est_pha = self._process_decoder_immediate(ts_out, mag)
        else:
            result = self._process_decoder_buffered(ts_out, mag, valid_frames)
            if result is None:
                return None
            est_mag, est_pha = result

        est_com = mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)
        return est_mag, est_pha, est_com

    # ------------------------------------------------------------- public APIs
    @torch.inference_mode()
    def process_samples(self, samples: Tensor) -> Optional[SpectrogramChunk]:
        """Feed incoming audio samples; return one matured spectrogram chunk or ``None``.

        Accumulates ``samples`` in the internal audio buffer, and once a full
        chunk (``samples_per_chunk`` samples) is available: STFTs
        ``[ _stft_context | chunk_samples ]`` with ``center=False``, updates
        ``_stft_context`` to the 200 samples preceding the next chunk, runs one
        :meth:`process_spectrogram_buffered` step, and advances the buffer by
        ``output_samples_per_chunk``. Does **not** reset state between calls
        (call :meth:`reset_state` — or use :meth:`process_audio` — for a new
        utterance).

        Args:
            samples: ``[T]`` or ``[1, T]`` audio (arbitrary length — buffered
                internally). Batch size must be 1.

        Returns:
            ``(est_mag, est_pha, est_com)`` (see the frozen contract at the top of
            this module) for the next ``chunk_size`` matured STFT frames, or
            ``None`` while the buffer / lookahead pipeline is still filling.

        Raises:
            ValueError: If ``samples`` is 2-D with a batch dimension != 1.
        """
        if samples.dim() == 2:
            if samples.shape[0] != 1:
                raise ValueError("batch size must be 1 for streaming")
            samples = samples.squeeze(0)
        samples = samples.to(self.device)

        if self._stft_context.device != samples.device:
            self._stft_context = self._stft_context.to(samples.device)

        self.input_buffer = torch.cat([self.input_buffer.to(self.device), samples])
        if len(self.input_buffer) < self.samples_per_chunk:
            return None

        chunk_samples = self.input_buffer[: self.samples_per_chunk]

        # center=True emulation: prepend the real past context (win//2 samples).
        context_size = self.win_size // 2
        chunk_with_context = torch.cat([self._stft_context.to(chunk_samples.device), chunk_samples])
        spectrogram = self._stft(chunk_with_context)

        # Save the win//2 samples immediately before the next chunk as the next
        # STFT context (the buffer still holds the un-advanced content here).
        advance = self.output_samples_per_chunk
        if advance >= context_size:
            self._stft_context = self.input_buffer[advance - context_size : advance].clone()
        else:
            need_from_prev = context_size - advance
            prev_part = self._stft_context[len(self._stft_context) - need_from_prev :]
            curr_part = self.input_buffer[:advance]
            self._stft_context = torch.cat([prev_part, curr_part]).clone()

        result = self.process_spectrogram_buffered(spectrogram)
        self.input_buffer = self.input_buffer[self.output_samples_per_chunk :]
        return result

    @torch.inference_mode()
    def process_audio(self, audio: Tensor) -> SpectrogramChunk:
        """Stream a complete utterance (audio in) and return its full spectrogram (out).

        Resets state, appends a zero flush long enough to drain the lookahead
        pipeline, feeds the audio through :meth:`process_samples` in
        ``output_samples_per_chunk``-sized pieces, concatenates the matured
        spectrogram chunks along the time axis, and trims to the reference frame
        count.

        Documented trim rule: ``ref_T = 1 + len(audio) // hop_size`` is exactly
        the number of STFT frames ``mag_pha_stft(audio, ..., center=True)``
        produces (``win_size == n_fft``), and streaming frame ``k`` corresponds to
        ``center=True`` frame ``k`` (no positional shift). So the output is
        trimmed to its first ``ref_T`` frames. (The caller still owns the
        additional ``total_lookahead``-frame tail trim before any parity check —
        those last frames used zero-padded lookahead because the input ran out;
        and the first ~2 frames match ``center=True`` only if the utterance starts
        with ≥ ``win_size//2`` leading silence.)

        Args:
            audio: ``[T]`` or ``[1, T]`` audio.

        Returns:
            ``(est_mag, est_pha, est_com)`` for the whole utterance: ``est_mag`` /
            ``est_pha`` are ``[1, F, T']`` and ``est_com`` is ``[1, F, T', 2]``
            with ``T' = min(produced, ref_T)``.
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        audio = audio.to(self.device)
        audio_length = len(audio)
        ref_t = 1 + audio_length // self.hop_size  # center=True STFT frame count (win == n_fft)

        self.reset_state()

        flush_size = self.samples_per_chunk * (self.total_lookahead + 2)
        padded = torch.cat([audio, torch.zeros(flush_size, device=audio.device)])

        mags: List[Tensor] = []
        phas: List[Tensor] = []
        coms: List[Tensor] = []
        for i in range(0, len(padded), self.output_samples_per_chunk):
            chunk = padded[i : i + self.output_samples_per_chunk]
            if len(chunk) == 0:
                break
            result = self.process_samples(chunk)
            if result is not None:
                est_mag, est_pha, est_com = result
                mags.append(est_mag)
                phas.append(est_pha)
                coms.append(est_com)

        f_bins = self.n_fft // 2 + 1
        if not mags:
            empty = torch.zeros(1, f_bins, 0, device=audio.device)
            return empty, empty.clone(), torch.zeros(1, f_bins, 0, 2, device=audio.device)

        est_mag = torch.cat(mags, dim=2)
        est_pha = torch.cat(phas, dim=2)
        est_com = torch.cat(coms, dim=2)
        keep_t = min(est_mag.shape[2], ref_t)
        return est_mag[:, :, :keep_t], est_pha[:, :, :keep_t], est_com[:, :, :keep_t, :]

    @torch.inference_mode()
    def process_spectrogram(self, noisy_com: Tensor) -> SpectrogramChunk:
        """Stream a complete utterance's spectrogram in chunks (spectrogram in → out).

        This is the host-STFT-free path used by the S3 model-level parity gate
        (mirrors LaCoSENet ``src/verify_equivalence.py``'s ``streaming_forward``):
        both this and ``Backbone.forward`` receive the *same* ``noisy_com``, so
        only chunk-boundary state handling can differ. Resets state, then loops
        the time axis in ``chunk_size`` steps, feeding each
        ``noisy_com[:, :, t : t + chunk_size + L_enc, :]`` slice (overlapping the
        next chunk by ``L_enc``) through :meth:`process_spectrogram_buffered`, and
        concatenates the matured outputs along the time axis. No internal flush:
        the last ``~chunk_size + total_lookahead`` reference frames are not drained
        (and the last ``total_lookahead`` produced frames used zero-padded
        lookahead) — the caller trims ``total_lookahead`` off ``T_stream`` before
        comparing.

        Args:
            noisy_com: ``[B, F, T, 2]`` (``B == 1``) or ``[F, T, 2]`` compressed
                complex spectrogram (e.g. ``mag_pha_stft(audio, center=True)[2]``).

        Returns:
            ``(est_mag, est_pha, est_com)`` for the streamed (matured) frames:
            ``est_mag`` / ``est_pha`` are ``[1, F, T_stream]`` and ``est_com`` is
            ``[1, F, T_stream, 2]`` with ``T_stream < T`` (the un-drained tail is
            omitted).

        Raises:
            ValueError: If ``noisy_com`` has a batch dimension != 1.
        """
        if noisy_com.dim() == 3:
            noisy_com = noisy_com.unsqueeze(0)
        if noisy_com.shape[0] != 1:
            raise ValueError(f"streaming requires batch size 1, got {noisy_com.shape[0]}")

        self.reset_state()
        t_total = noisy_com.shape[2]
        mags: List[Tensor] = []
        phas: List[Tensor] = []
        coms: List[Tensor] = []
        for t in range(0, t_total, self.chunk_size):
            t_end = min(t + self.chunk_size + self.encoder_lookahead, t_total)
            chunk = noisy_com[:, :, t:t_end, :]
            result = self.process_spectrogram_buffered(chunk)
            if result is not None:
                est_mag, est_pha, est_com = result
                mags.append(est_mag)
                phas.append(est_pha)
                coms.append(est_com)

        f_bins = self.n_fft // 2 + 1
        if not mags:
            empty = torch.zeros(1, f_bins, 0, device=noisy_com.device)
            return empty, empty.clone(), torch.zeros(1, f_bins, 0, 2, device=noisy_com.device)
        return torch.cat(mags, dim=2), torch.cat(phas, dim=2), torch.cat(coms, dim=2)

    def forward(self, audio: Tensor) -> SpectrogramChunk:
        """``nn.Module`` entry point — equivalent to :meth:`process_audio`."""
        return self.process_audio(audio)

    def __repr__(self) -> str:
        cfg = self._streaming_config
        return (
            f"{self.__class__.__name__}(\n"
            f"  source={cfg.get('source', 'direct')}, infer_type={self.infer_type},\n"
            f"  chunk_size={self.chunk_size}, encoder_lookahead={self.encoder_lookahead}, "
            f"decoder_lookahead={self.decoder_lookahead},\n"
            f"  total_frames_needed={self.total_frames_needed}, samples_per_chunk={self.samples_per_chunk}, "
            f"output_samples_per_chunk={self.output_samples_per_chunk},\n"
            f"  latency_ms={self.latency_ms:.2f},\n"
            f")"
        )


__all__ = ["BackboneStreaming", "SpectrogramChunk"]
