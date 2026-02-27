"""
LaCoSENet with Lookahead Buffering.

This module provides a streaming wrapper that addresses the asymmetric
convolution problem by buffering features and providing real lookahead context.
It supports models where encoder, decoder, or both have asymmetric padding.

Processing pipeline:
1. Input buffering + STFT future buffering:
   - Accumulate samples until a full processing window is available, including
     win_size/2 extra samples for real future context.
   - Use center=False STFT with manually prepended past context (win_size/2)
     and appended future context (win_size/2) to eliminate reflect padding artifacts.
2. Encoder + TSBlock:
   - Run the encoder path immediately once the processing window is ready.
   - StatefulConv provides past context via internal state buffers.
   - "Input lookahead" provides future context for asymmetric encoder padding.
3. Feature buffer (decoder lookahead):
   - Accumulate encoder features until enough frames are available for decoder
     processing with lookahead.
4. Decoder:
   - Process an extended time window (current chunk + lookahead + STFT frames)
     and produce only the current chunk output for iSTFT overlap-add.
5. StateFramesContext:
   - Prevent lookahead and STFT-induced extra frames from corrupting streaming
     states by limiting state updates to the current chunk frames.

This approach provides:
- Real past context from StatefulConv buffering
- Real future context from delayed processing (encoder/decoder lookahead)
- Minimal additional latency beyond the required lookahead frames

Example (decoder-only asymmetric):
    >>> streaming = LaCoSENet.from_checkpoint(
    ...     chkpt_dir="results/experiments/prk_1117_1",
    ...     chunk_size=64,
    ...     encoder_lookahead=0,   # Encoder is causal
    ...     decoder_lookahead=7,   # Decoder needs 7 frame lookahead
    ... )

Example (both encoder and decoder asymmetric):
    >>> streaming = LaCoSENet.from_checkpoint(
    ...     chkpt_dir="results/experiments/prk_1114_2",
    ...     chunk_size=64,
    ...     encoder_lookahead=7,   # Encoder needs 7 frame lookahead
    ...     decoder_lookahead=7,   # Decoder needs 7 frame lookahead
    ... )
    >>>
    >>> for chunk in audio_stream:
    ...     output = streaming.process_samples(chunk)
    ...     if output is not None:
    ...         play(output)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.stft import mag_pha_stft, manual_istft_ola

logger = logging.getLogger(__name__)


class LaCoSENet(nn.Module):
    """
    Streaming wrapper for Backbone with lookahead buffering.

    This wrapper enables true streaming inference for models with asymmetric
    convolutions in encoder, decoder, or both. It provides real future context
    through lookahead buffering instead of zero-padding.

    Supported model configurations:
    - **Fully Causal**: encoder_lookahead=0, decoder_lookahead=0
    - **Asymmetric Decoder Only**: encoder_lookahead=0, decoder_lookahead>0
    - **Asymmetric Encoder+Decoder**: encoder_lookahead>0, decoder_lookahead>0

    Attributes:
        encoder_lookahead: Frames needed for encoder (0 if encoder is causal)
        decoder_lookahead: Frames needed for decoder (0 if decoder is causal)
        total_lookahead: encoder_lookahead + decoder_lookahead
        latency_ms: Total latency in milliseconds
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_size: int = 64,
        encoder_lookahead: int = 0,
        decoder_lookahead: int = 7,
        hop_size: int = 100,
        n_fft: int = 400,
        win_size: int = 400,
        compress_factor: float = 0.3,
        sample_rate: int = 16000,
        rf_sequence_block: Optional[nn.ModuleList] = None,
        freq_size: int = 100,
        stft_center: bool = True,
        disable_state_guard: bool = False,
    ):
        """
        Initialize LaCoSENet.

        Note: Use `from_checkpoint()` for easier initialization.

        Args:
            model: Backbone model (should already have StatefulConv applied)
            chunk_size: Number of STFT frames per chunk
            encoder_lookahead: Frames to delay encoder processing (for asymmetric encoder)
            decoder_lookahead: Frames to delay decoder processing (for asymmetric decoder)
            hop_size: STFT hop size in samples
            n_fft: FFT size
            win_size: Window size
            compress_factor: Magnitude compression factor
            sample_rate: Audio sample rate in Hz
            rf_sequence_block: Reshape-free TSBlock ModuleList (if enabled)
            freq_size: Frequency bins (for state initialization)
            disable_state_guard: If True, disable StateFramesContext so all
                frames (including lookahead) update streaming state buffers.
                Used for ablation study of selective state update (C3).
        """
        super().__init__()

        # Ablation: disable selective state update
        self.disable_state_guard = disable_state_guard

        # Store model reference
        self.model = model
        self.model.eval()

        # STFT parameters
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.sample_rate = sample_rate
        self.stft_center = stft_center
        if stft_center:
            self.stft_future_samples = self.win_size // 2
            self.stft_center_delay_samples = self.stft_future_samples
        else:
            self.stft_future_samples = 0
            self.stft_center_delay_samples = 0

        # Streaming parameters
        self.chunk_size = chunk_size
        self.encoder_lookahead = encoder_lookahead
        self.decoder_lookahead = decoder_lookahead
        self.input_lookahead_frames = int(encoder_lookahead)
        self.total_lookahead = self.input_lookahead_frames + decoder_lookahead

        self.total_frames_needed = chunk_size + self.input_lookahead_frames
        if stft_center:
            self.samples_per_chunk = (self.total_frames_needed - 1) * hop_size + self.stft_future_samples
        else:
            self.samples_per_chunk = (self.total_frames_needed - 1) * hop_size + n_fft

        self.output_frames_per_chunk = chunk_size
        self.output_samples_per_chunk = self.output_frames_per_chunk * hop_size

        # OLA buffer parameters
        self.ola_tail_size = win_size - hop_size

        # Latency calculation
        self.latency_samples = self.total_lookahead * hop_size + self.stft_center_delay_samples
        self.latency_ms = self.latency_samples / sample_rate * 1000

        # Reshape-free TSBlock support
        self.rf_sequence_block = rf_sequence_block
        self.use_reshape_free = rf_sequence_block is not None
        self.freq_size = freq_size
        self._rf_states: Optional[List[List[Dict[str, Tensor]]]] = None

        # Initialize buffers
        self._reset_buffers()

        # Config storage
        self._streaming_config: Dict[str, Any] = {}

    def _reset_buffers(self) -> None:
        """Reset all internal buffers."""
        self.input_buffer = torch.tensor([], dtype=torch.float32)
        self.feature_buffer: List[Dict[str, Any]] = []
        self._buffered_frames = 0
        if self.stft_center:
            self._stft_context = torch.zeros(self.win_size // 2, dtype=torch.float32)
        else:
            self._stft_context = None

        self._ola_buffer = torch.zeros(self.ola_tail_size, dtype=torch.float32)
        self._ola_norm = torch.zeros(self.ola_tail_size, dtype=torch.float32)

        if self.use_reshape_free and self.rf_sequence_block is not None:
            self._rf_states = self._init_rf_states()

    def _init_rf_states(self) -> List[List[Dict[str, Tensor]]]:
        """Initialize reshape-free TSBlock states."""
        if self.rf_sequence_block is None:
            return []

        device = self.device
        dtype = next(self.model.parameters()).dtype

        all_states = []
        for rf_block in self.rf_sequence_block:
            block_states = rf_block.init_state(
                batch_size=1,
                freq_size=self.freq_size,
                device=device,
                dtype=dtype,
            )
            all_states.append(block_states)

        return all_states

    def _reset_rf_states(self) -> None:
        """Reset reshape-free TSBlock states."""
        if self.use_reshape_free:
            self._rf_states = self._init_rf_states()

    def reset_state(self) -> None:
        """Reset all streaming state for a new audio stream."""
        self._reset_buffers()

        if self.use_reshape_free:
            self._reset_rf_states()

        from src.models.streaming.converters.conv_converter import (
            reset_streaming_state,
        )
        reset_streaming_state(self.model)

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying model."""
        return next(self.model.parameters()).device

    @property
    def streaming_config(self) -> Dict[str, Any]:
        """Get streaming configuration information."""
        return {
            **self._streaming_config,
            "chunk_size_frames": self.chunk_size,
            "encoder_lookahead": self.encoder_lookahead,
            "decoder_lookahead": self.decoder_lookahead,
            "input_lookahead_frames": self.input_lookahead_frames,
            "total_lookahead": self.total_lookahead,
            "stft_center": self.stft_center,
            "stft_center_delay_samples": self.stft_center_delay_samples,
            "output_frames_per_chunk": self.output_frames_per_chunk,
            "samples_per_chunk": self.samples_per_chunk,
            "output_samples_per_chunk": self.output_samples_per_chunk,
            "latency_samples": self.latency_samples,
            "latency_ms": self.latency_ms,
            "hop_size": self.hop_size,
            "sample_rate": self.sample_rate,
            "use_reshape_free": self.use_reshape_free,
            "freq_size": self.freq_size,
        }

    @classmethod
    def from_checkpoint(
        cls,
        chkpt_dir: str,
        chkpt_file: str = "best.th",
        chunk_size: int = 64,
        encoder_lookahead: int = 0,
        decoder_lookahead: int = 7,
        use_reshape_free: bool = False,
        fold_bn: bool = False,
        device: Optional[str] = None,
        verbose: bool = True,
        disable_state_guard: bool = False,
    ) -> "LaCoSENet":
        """
        Create LaCoSENet from a checkpoint directory.

        Args:
            chkpt_dir: Path to checkpoint directory
            chkpt_file: Checkpoint file name (default: "best.th")
            chunk_size: Number of STFT frames per chunk
            encoder_lookahead: Frames for encoder lookahead
            decoder_lookahead: Frames for decoder lookahead
            use_reshape_free: Convert TSBlocks to reshape-free versions
            fold_bn: Apply BN folding for CPU inference
            device: Device to load model on
            verbose: Print loading information
            disable_state_guard: Disable selective state update (ablation)

        Returns:
            LaCoSENet instance
        """
        from src.models.streaming.utils import prepare_streaming_model

        if verbose:
            print(f"Loading LaCoSENet from: {chkpt_dir}")

        model, metadata = prepare_streaming_model(
            chkpt_dir=chkpt_dir,
            chkpt_file=chkpt_file,
            use_stateful_conv=True,
            use_reshape_free=use_reshape_free,
            fold_bn=fold_bn,
            device=device,
            verbose=verbose,
        )

        model_args = metadata["model_args"]
        model_params = model_args
        stateful_conv_count = metadata.get("stateful_conv_count", 0)
        rf_sequence_block = metadata.get("rf_sequence_block", None)
        rf_block_count = metadata.get("rf_block_count", 0)

        enc_padding = getattr(model_params, 'encoder_padding_ratio', [1.0, 0.0])
        dec_padding = getattr(model_params, 'decoder_padding_ratio', [1.0, 0.0])

        if verbose:
            print(f"  Encoder padding ratio: {enc_padding}")
            print(f"  Decoder padding ratio: {dec_padding}")

        hop_size = getattr(model_params, 'hop_size', 100)
        n_fft = getattr(model_params, 'n_fft', 400)
        win_size = getattr(model_params, 'win_size', 400)
        compress_factor = getattr(model_params, 'compress_factor', 0.3)
        stft_center = getattr(model_params, 'stft_center', True)

        stft_freq = n_fft // 2 + 1
        with torch.no_grad():
            dummy = torch.randn(1, 2, 4, stft_freq, device=next(model.parameters()).device)
            freq_size = model.dense_encoder(dummy).shape[3]

        instance = cls(
            model=model,
            chunk_size=chunk_size,
            encoder_lookahead=encoder_lookahead,
            decoder_lookahead=decoder_lookahead,
            hop_size=hop_size,
            n_fft=n_fft,
            win_size=win_size,
            compress_factor=compress_factor,
            rf_sequence_block=rf_sequence_block,
            freq_size=freq_size,
            stft_center=stft_center,
            disable_state_guard=disable_state_guard,
        )

        instance._streaming_config = {
            "chkpt_dir": chkpt_dir,
            "use_reshape_free": use_reshape_free,
            "encoder_padding_ratio": enc_padding,
            "decoder_padding_ratio": dec_padding,
            "stateful_conv_count": stateful_conv_count,
            "rf_block_count": rf_block_count,
            "model_class": "Backbone",
        }

        if verbose:
            print(f"  Chunk size: {chunk_size} frames")
            print(f"  Encoder lookahead: {encoder_lookahead} frames")
            print(f"  Decoder lookahead: {decoder_lookahead} frames")
            print(f"  Total latency: {instance.latency_ms:.1f}ms")
            if use_reshape_free:
                print(f"  Reshape-Free: {rf_block_count} TSBlocks converted")

        return instance

    def _stft(self, audio: Tensor) -> Tensor:
        """Compute STFT and return complex spectrogram."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        _, _, com = mag_pha_stft(
            audio,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            center=False,
        )
        return com

    def _process_encoder(
        self,
        spectrogram: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        """Process spectrogram through Encoder + TSBlock."""
        from src.models.backbone import complex_to_mag_pha
        from src.models.streaming.utils import StateFramesContext

        _, _, T, _ = spectrogram.shape

        mag, pha = complex_to_mag_pha(spectrogram, stack_dim=-1)
        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)  # [B, 2, T, F]

        valid_frames = min(T, self.chunk_size)

        with StateFramesContext(None if self.disable_state_guard else valid_frames):
            encoded = self.model.dense_encoder(x)

            if self.use_reshape_free and self.rf_sequence_block is not None:
                ts_out = self._process_rf_sequence_block(encoded)
            else:
                ts_out = self.model.sequence_block(encoded)

        return mag, ts_out, valid_frames

    def _process_rf_sequence_block(self, x: Tensor) -> Tensor:
        """Process through reshape-free TSBlocks with explicit state management."""
        if self.rf_sequence_block is None or self._rf_states is None:
            raise RuntimeError("Reshape-free sequence block not initialized")

        for i, rf_block in enumerate(self.rf_sequence_block):
            x, new_states = rf_block(x, self._rf_states[i])
            self._rf_states[i] = new_states

        return x

    def _can_process_immediately(self, ts_out: Tensor) -> bool:
        """Determine if decoder can process immediately without buffering."""
        if self.decoder_lookahead > 0:
            return False
        return ts_out.shape[2] >= self.chunk_size

    def _process_decoder_immediate(
        self,
        ts_out: Tensor,
        mag: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Process decoder immediately without feature buffer (decoder_lookahead=0)."""
        from src.models.streaming.utils import StateFramesContext

        features = ts_out[:, :, :self.chunk_size, :]
        chunk_mag = mag[:, :, :self.chunk_size]

        with StateFramesContext(None if self.disable_state_guard else self.chunk_size):
            mask = self.model.mask_decoder(features).squeeze(1).transpose(1, 2)
            est_pha = self.model.phase_decoder(features).squeeze(1).transpose(1, 2)

        est_mag = chunk_mag * mask

        return est_mag, est_pha

    def _process_decoder_buffered(
        self,
        ts_out: Tensor,
        mag: Tensor,
        valid_frames: int,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Process decoder with feature buffering (decoder_lookahead > 0)."""
        from src.models.streaming.utils import StateFramesContext

        self.feature_buffer.append({
            'features': ts_out[:, :, :valid_frames, :],
            'mag': mag[:, :, :valid_frames],
            'frames': valid_frames,
        })
        self._buffered_frames += valid_frames

        total_needed = self.chunk_size + self.decoder_lookahead
        if self._buffered_frames < total_needed:
            logger.debug(f"Buffering: {self._buffered_frames}/{total_needed}")
            return None

        all_features = torch.cat([buf['features'] for buf in self.feature_buffer], dim=2)
        all_mag = torch.cat([buf['mag'] for buf in self.feature_buffer], dim=2)

        extended_features = all_features[:, :, :total_needed, :]
        extended_mag = all_mag[:, :, :total_needed]

        with StateFramesContext(None if self.disable_state_guard else self.chunk_size):
            mask = self.model.mask_decoder(extended_features).squeeze(1).transpose(1, 2)
            est_pha = self.model.phase_decoder(extended_features).squeeze(1).transpose(1, 2)

        est_mag = extended_mag * mask

        est_mag = est_mag[:, :, :self.chunk_size]
        est_pha = est_pha[:, :, :self.chunk_size]

        frames_to_remove = self.chunk_size
        removed = 0

        while removed < frames_to_remove and self.feature_buffer:
            buf = self.feature_buffer[0]
            if buf['frames'] <= (frames_to_remove - removed):
                removed += buf['frames']
                self.feature_buffer.pop(0)
            else:
                keep_frames = buf['frames'] - (frames_to_remove - removed)
                buf['features'] = buf['features'][:, :, -keep_frames:, :]
                buf['mag'] = buf['mag'][:, :, -keep_frames:]
                buf['frames'] = keep_frames
                removed = frames_to_remove

        self._buffered_frames -= removed

        return est_mag, est_pha

    def process_spectrogram_buffered(
        self,
        spectrogram: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Process spectrogram using adaptive encoder-decoder approach."""
        mag, ts_out, valid_frames = self._process_encoder(spectrogram)

        if self._can_process_immediately(ts_out):
            return self._process_decoder_immediate(ts_out, mag)
        else:
            return self._process_decoder_buffered(ts_out, mag, valid_frames)

    def _manual_istft_ola(self, est_mag: Tensor, est_pha: Tensor) -> Tensor:
        """Perform manual iSTFT with cross-chunk OLA buffer."""
        if self._ola_buffer.device != est_mag.device:
            self._ola_buffer = self._ola_buffer.to(est_mag.device)
            self._ola_norm = self._ola_norm.to(est_mag.device)

        output, new_ola_buffer, new_ola_norm = manual_istft_ola(
            est_mag, est_pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            ola_buffer=self._ola_buffer,
            ola_norm=self._ola_norm,
        )

        self._ola_buffer = new_ola_buffer
        self._ola_norm = new_ola_norm

        return output[:self.output_samples_per_chunk]

    @torch.inference_mode()
    def process_samples(self, samples: Tensor) -> Optional[Tensor]:
        """
        Process incoming audio samples and return enhanced output if available.

        Args:
            samples: Input audio samples [T] or [B, T] (B must be 1)

        Returns:
            Enhanced audio samples [T] if output available, None otherwise
        """
        if samples.dim() == 2:
            if samples.shape[0] != 1:
                raise ValueError("Batch size must be 1 for streaming")
            samples = samples.squeeze(0)

        samples = samples.to(self.device)

        if self.stft_center and self._stft_context.device != samples.device:
            self._stft_context = self._stft_context.to(samples.device)

        self.input_buffer = torch.cat([self.input_buffer.to(self.device), samples])

        if len(self.input_buffer) < self.samples_per_chunk:
            return None

        chunk_samples = self.input_buffer[:self.samples_per_chunk]

        if self.stft_center:
            context_size = self.win_size // 2
            chunk_with_context = torch.cat([self._stft_context.to(chunk_samples.device), chunk_samples])
            spectrogram = self._stft(chunk_with_context)

            advance = self.output_samples_per_chunk
            if advance >= context_size:
                self._stft_context = self.input_buffer[advance - context_size:advance].clone()
            else:
                need_from_prev = context_size - advance
                prev_part = self._stft_context[len(self._stft_context) - need_from_prev:]
                curr_part = self.input_buffer[:advance]
                self._stft_context = torch.cat([prev_part, curr_part]).clone()
        else:
            spectrogram = self._stft(chunk_samples)

        result = self.process_spectrogram_buffered(spectrogram)

        if result is None:
            self.input_buffer = self.input_buffer[self.output_samples_per_chunk:]
            return None

        est_mag, est_pha = result

        valid_output = self._manual_istft_ola(est_mag, est_pha)

        self.input_buffer = self.input_buffer[self.output_samples_per_chunk:]

        return valid_output

    def process_audio(self, audio: Tensor) -> Tensor:
        """
        Process a complete audio signal in streaming fashion.

        Args:
            audio: Complete input audio [T] or [1, T]

        Returns:
            Enhanced audio [T'] where T' <= T
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        audio_length = len(audio)
        self.reset_state()
        outputs: List[Tensor] = []

        flush_size = self.samples_per_chunk * (self.total_lookahead + 2)
        padded = torch.cat([
            audio,
            torch.zeros(flush_size, device=audio.device),
        ])

        for i in range(0, len(padded), self.output_samples_per_chunk):
            chunk = padded[i:i + self.output_samples_per_chunk]
            if len(chunk) == 0:
                break
            result = self.process_samples(chunk)
            if result is not None and len(result) > 0:
                outputs.append(result)

        if not outputs:
            return torch.tensor([], device=audio.device)

        result = torch.cat(outputs)
        if len(result) > audio_length:
            result = result[:audio_length]
        return result

    def process_audio_fast(self, audio: Tensor) -> Tensor:
        """
        Process a complete audio signal using the 3-phase fast pipeline.

        Eliminates per-chunk STFT/iSTFT overhead by batching them.

        Args:
            audio: Complete input audio [T] or [1, T]

        Returns:
            Enhanced audio [T'] where T' <= T
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        audio_length = len(audio)
        device = self.device
        audio = audio.to(device)

        self.reset_state()

        # --- Phase 1: Batched STFT ---
        flush_size = self.samples_per_chunk * (self.total_lookahead + 2)
        padded = torch.cat([audio, torch.zeros(flush_size, device=device)])

        osp = self.output_samples_per_chunk
        if self.stft_center:
            context_size = self.win_size // 2
            pre_padded = torch.cat([
                torch.zeros(context_size, device=device),
                padded,
            ])
            stft_input_len = context_size + self.samples_per_chunk
        else:
            pre_padded = padded
            stft_input_len = self.samples_per_chunk

        n_calls = (len(pre_padded) - stft_input_len) // osp + 1

        batch_input = pre_padded.unfold(0, stft_input_len, osp)
        batch_input = batch_input[:n_calls]

        _, _, batch_com = mag_pha_stft(
            batch_input,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            center=False,
        )

        # --- Phase 2: Sequential model forward ---
        all_mag: List[Tensor] = []
        all_pha: List[Tensor] = []

        for j in range(n_calls):
            chunk_com = batch_com[j:j + 1]
            result = self.process_spectrogram_buffered(chunk_com)

            if result is not None:
                est_mag, est_pha = result
                all_mag.append(est_mag)
                all_pha.append(est_pha)

        if not all_mag:
            return torch.tensor([], device=device)

        # --- Phase 3: Batch iSTFT ---
        cat_mag = torch.cat(all_mag, dim=2)
        cat_pha = torch.cat(all_pha, dim=2)

        output, _, _ = manual_istft_ola(
            cat_mag, cat_pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            ola_buffer=None,
            ola_norm=None,
        )

        if len(output) > audio_length:
            output = output[:audio_length]

        return output

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for nn.Module compatibility."""
        return self.process_audio(x)

    def __repr__(self) -> str:
        config = self._streaming_config
        rf_info = ""
        if self.use_reshape_free:
            rf_count = config.get('rf_block_count', 0)
            rf_info = f"  use_reshape_free=True ({rf_count} blocks),\n"
        return (
            f"{self.__class__.__name__}(\n"
            f"{rf_info}"
            f"  chunk_size={self.chunk_size},\n"
            f"  encoder_lookahead={self.encoder_lookahead},\n"
            f"  decoder_lookahead={self.decoder_lookahead},\n"
            f"  total_lookahead={self.total_lookahead},\n"
            f"  latency_ms={self.latency_ms:.2f},\n"
            f")"
        )
