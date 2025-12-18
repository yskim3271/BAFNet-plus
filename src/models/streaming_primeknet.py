"""
Streaming wrapper for PrimeKnet.

This module provides a chunk-based streaming interface for PrimeKnet,
enabling real-time audio processing with fixed lookahead latency.

Example usage:
    >>> from src.models.primeknet import PrimeKnet
    >>> from src.models.streaming_primeknet import StreamingPrimeKnet
    >>>
    >>> # Load trained model
    >>> model = PrimeKnet(...)
    >>> model.load_state_dict(torch.load('best.th')['model'])
    >>>
    >>> # Create streaming wrapper
    >>> streaming = StreamingPrimeKnet(
    ...     model=model,
    ...     chunk_size=16,        # frames per chunk
    ...     lookahead_frames=8,   # future context frames
    ...     hop_size=100,
    ...     n_fft=400,
    ...     win_size=400,
    ...     compress_factor=0.3
    ... )
    >>>
    >>> # Process audio stream
    >>> for chunk in audio_stream:
    ...     output = streaming.process_samples(chunk)
    ...     if output is not None:
    ...         play(output)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.stft import mag_pha_stft, mag_pha_istft


class StreamingPrimeKnet(nn.Module):
    """
    Chunk-based streaming wrapper for PrimeKnet.

    This wrapper enables real-time processing of audio streams by:
    1. Buffering input samples until enough context is available
    2. Processing fixed-size chunks through the model
    3. Outputting enhanced audio with fixed latency

    The latency is determined by:
        latency = lookahead_frames * hop_size / sample_rate (seconds)

    Args:
        model: Trained PrimeKnet model (will be set to eval mode)
        chunk_size: Number of STFT frames to process per chunk
        lookahead_frames: Number of future frames needed (determines latency)
        hop_size: STFT hop size in samples (default: 100)
        n_fft: FFT size (default: 400)
        win_size: Window size (default: 400)
        compress_factor: Magnitude compression factor (default: 0.3)
        sample_rate: Audio sample rate in Hz (default: 16000)

    Attributes:
        latency_samples: Latency in samples
        latency_ms: Latency in milliseconds
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_size: int,
        lookahead_frames: int,
        hop_size: int = 100,
        n_fft: int = 400,
        win_size: int = 400,
        compress_factor: float = 0.3,
        sample_rate: int = 16000,
    ):
        super().__init__()

        # Store model (set to eval mode)
        self.model = model
        self.model.eval()

        # STFT parameters
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.sample_rate = sample_rate

        # Streaming parameters
        self.chunk_size = chunk_size
        self.lookahead_frames = lookahead_frames

        # Calculate required samples for one chunk
        # We need: chunk_size frames + lookahead frames worth of context
        # Total frames needed = chunk_size + lookahead_frames
        # But STFT with center=True adds padding, so we need:
        # samples = (total_frames - 1) * hop_size + win_size
        self.total_frames_needed = chunk_size + lookahead_frames
        self.samples_per_chunk = (self.total_frames_needed - 1) * hop_size + win_size

        # Samples to output per chunk (excluding lookahead)
        self.output_samples_per_chunk = chunk_size * hop_size

        # Latency calculation
        self.latency_samples = lookahead_frames * hop_size
        self.latency_ms = self.latency_samples / sample_rate * 1000

        # Initialize buffers
        self._reset_buffers()

    def _reset_buffers(self):
        """Reset internal buffers for new stream."""
        self.input_buffer = torch.tensor([], dtype=torch.float32)
        self._first_chunk = True

    def reset_state(self):
        """
        Reset the streaming state for a new audio stream.

        Call this method before processing a new audio file or
        when starting a fresh stream.
        """
        self._reset_buffers()

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying model."""
        return next(self.model.parameters()).device

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT and return complex spectrogram.

        Args:
            audio: Input audio [B, T] or [T]

        Returns:
            Complex spectrogram [B, F, T, 2]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        _, _, com = mag_pha_stft(
            audio,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            center=True
        )
        return com

    def _istft(self, mag: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse STFT.

        Args:
            mag: Magnitude spectrogram [B, F, T]
            pha: Phase spectrogram [B, F, T]

        Returns:
            Audio waveform [B, T]
        """
        return mag_pha_istft(
            mag, pha,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            compress_factor=self.compress_factor,
            center=True
        )

    def process_chunk(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single spectrogram chunk through the model.

        Args:
            spectrogram: Complex spectrogram [B, F, T, 2]

        Returns:
            Tuple of (est_mag, est_pha, est_com) for the chunk
        """
        with torch.no_grad():
            est_mag, est_pha, est_com = self.model(spectrogram)
        return est_mag, est_pha, est_com

    def process_samples(self, samples: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process incoming audio samples and return enhanced output if available.

        This method buffers input samples and processes them in chunks.
        Output is returned only when enough samples have been accumulated.

        Args:
            samples: Input audio samples [T] or [B, T] (B must be 1 for streaming)

        Returns:
            Enhanced audio samples [T] if a chunk was processed, None otherwise

        Note:
            The first output will be delayed by `lookahead_frames * hop_size` samples.
        """
        # Ensure samples are on the correct device and shape
        if samples.dim() == 2:
            if samples.shape[0] != 1:
                raise ValueError("Batch size must be 1 for streaming")
            samples = samples.squeeze(0)

        samples = samples.to(self.device)

        # Add to buffer
        self.input_buffer = torch.cat([self.input_buffer.to(self.device), samples])

        # Check if we have enough samples
        if len(self.input_buffer) < self.samples_per_chunk:
            return None

        # Extract chunk for processing
        chunk_samples = self.input_buffer[:self.samples_per_chunk]

        # Compute STFT
        spectrogram = self._stft(chunk_samples)  # [1, F, T, 2]

        # Process through model
        est_mag, est_pha, _ = self.process_chunk(spectrogram)

        # Compute iSTFT
        output_audio = self._istft(est_mag, est_pha)  # [1, T]
        output_audio = output_audio.squeeze(0)

        # Extract the valid output region (excluding lookahead)
        # For the first chunk, we skip initial transient
        if self._first_chunk:
            self._first_chunk = False
            # Skip the first half-window worth of samples (STFT center padding)
            start_idx = self.win_size // 2
        else:
            start_idx = 0

        # Output chunk_size worth of frames
        end_idx = start_idx + self.output_samples_per_chunk
        valid_output = output_audio[start_idx:end_idx]

        # Update buffer: remove processed samples
        # Keep remaining samples for next chunk processing
        self.input_buffer = self.input_buffer[self.output_samples_per_chunk:]

        return valid_output

    def process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Process a complete audio signal in streaming fashion.

        This is a convenience method that simulates streaming by
        processing the audio in chunks and concatenating the results.

        Args:
            audio: Complete input audio [T] or [1, T]

        Returns:
            Enhanced audio [T]

        Note:
            For actual real-time processing, use `process_samples()` in a loop.
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        self.reset_state()
        outputs = []

        # Process in small increments to simulate streaming
        # Use hop_size as the increment for fine-grained streaming
        increment = self.hop_size * 4  # Process 4 frames worth at a time

        for i in range(0, len(audio), increment):
            chunk = audio[i:i + increment]
            output = self.process_samples(chunk)
            if output is not None:
                outputs.append(output)

        # Flush remaining samples
        # Add padding to process final samples
        padding = torch.zeros(self.samples_per_chunk, device=audio.device)
        final_output = self.process_samples(padding)
        if final_output is not None:
            outputs.append(final_output)

        if outputs:
            return torch.cat(outputs)
        else:
            return torch.tensor([], device=audio.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for compatibility with nn.Module interface.

        For streaming, use `process_samples()` or `process_audio()` instead.

        Args:
            x: Input audio [B, T] or [T]

        Returns:
            Enhanced audio
        """
        return self.process_audio(x)

    def get_info(self) -> dict:
        """
        Get streaming configuration information.

        Returns:
            Dictionary with streaming parameters and latency info
        """
        return {
            "chunk_size_frames": self.chunk_size,
            "lookahead_frames": self.lookahead_frames,
            "total_frames_per_chunk": self.total_frames_needed,
            "samples_per_chunk": self.samples_per_chunk,
            "output_samples_per_chunk": self.output_samples_per_chunk,
            "latency_samples": self.latency_samples,
            "latency_ms": self.latency_ms,
            "hop_size": self.hop_size,
            "n_fft": self.n_fft,
            "win_size": self.win_size,
            "sample_rate": self.sample_rate,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  chunk_size={self.chunk_size},\n"
            f"  lookahead_frames={self.lookahead_frames},\n"
            f"  latency_ms={self.latency_ms:.2f},\n"
            f"  hop_size={self.hop_size},\n"
            f"  n_fft={self.n_fft}\n"
            f")"
        )


def calculate_receptive_field(model: nn.Module) -> dict:
    """
    Calculate the receptive field of a PrimeKnet model.

    This function analyzes the model architecture to determine
    the required context frames for streaming.

    Args:
        model: PrimeKnet model instance

    Returns:
        Dictionary with receptive field information:
        - encoder_rf: Encoder receptive field (frames)
        - tsblock_rf: TS_BLOCK receptive field (frames)
        - decoder_rf: Decoder receptive field (frames)
        - total_rf: Total receptive field (frames)
        - recommended_lookahead: Recommended lookahead frames
    """
    # Get model parameters
    dense_depth = getattr(model, 'dense_depth', 4)
    num_tsblock = getattr(model, 'num_tsblock', 4)

    # Encoder RF: DS_DDB with dilated convolutions
    # kernel_size=3, depth=4, dilations=[1, 2, 4, 8]
    # RF = sum of (kernel_size - 1) * dilation for each layer
    encoder_rf = 0
    for i in range(dense_depth):
        dilation = 2 ** i
        encoder_rf += (3 - 1) * dilation  # kernel_size = 3

    # TS_BLOCK RF: based on time_block_kernel
    # Default kernels: [3, 11, 23, 31]
    # Each block has time_block_num=2 layers
    tsblock_rf = 0
    time_block_kernel = getattr(model, 'time_block_kernel', [3, 11, 23, 31])
    time_block_num = 2  # default
    for kernel in time_block_kernel:
        # Each kernel contributes (kernel-1)/2 on each side
        # For causal, it's (kernel-1) on one side
        tsblock_rf += (kernel - 1) * time_block_num

    tsblock_rf *= num_tsblock

    # Decoder RF: same as encoder
    decoder_rf = encoder_rf

    total_rf = encoder_rf + tsblock_rf + decoder_rf

    # Recommended lookahead: half of total RF for symmetric padding
    # For asymmetric, use the right padding portion
    recommended_lookahead = total_rf // 2

    return {
        "encoder_rf": encoder_rf,
        "tsblock_rf": tsblock_rf,
        "decoder_rf": decoder_rf,
        "total_rf": total_rf,
        "recommended_lookahead": recommended_lookahead,
    }
