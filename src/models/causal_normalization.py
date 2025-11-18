"""
Causal Energy Normalization Modules for Streaming BAFNet

This module provides various strategies for causal magnitude/energy normalization
that can be used in both training and inference without train/test mismatch.

Author: Claude Code
Date: 2025-01-04
"""

import torch
import torch.nn as nn


class EMAEnergyNormalizer(nn.Module):
    """
    Exponential Moving Average (EMA) based energy normalizer.

    Uses EMA to track running statistics of magnitude mean in a causal manner.
    Works identically during training and inference, avoiding train/test mismatch.

    Args:
        momentum: EMA momentum (default: 0.99)
                  Higher = slower adaptation, more stable
                  Lower = faster adaptation, less stable
        learnable_bias: If True, adds learnable bias to compensate systematic errors
        epsilon: Small constant for numerical stability

    Attributes:
        running_mean: EMA of magnitude mean [1]
        bias: Learnable bias parameter (if enabled)

    Example:
        >>> normalizer = EMAEnergyNormalizer(momentum=0.99, learnable_bias=True)
        >>> mag = torch.randn(2, 201, 100).abs()
        >>> mag_norm, mag_mean = normalizer(mag)
        >>> # Streaming usage:
        >>> for frame in mag_frames:
        >>>     frame_norm, _ = normalizer(frame)
    """

    def __init__(self, momentum=0.99, learnable_bias=True, epsilon=1e-8):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon

        # Running mean buffer (not a parameter, but saved in state_dict)
        self.register_buffer('running_mean', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Learnable bias to compensate for systematic errors
        if learnable_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

    def forward(self, mag, return_mean=True):
        """
        Forward pass with causal EMA normalization.

        Args:
            mag: Magnitude spectrogram [B, F, T] or [B, F]
            return_mean: If True, returns both normalized mag and mean

        Returns:
            mag_norm: Normalized magnitude [B, F, T] or [B, F]
            mag_mean: EMA mean used for normalization [B, 1, T] or [B, 1] (if return_mean=True)
        """
        if mag.dim() == 2:
            # Single frame [B, F]
            return self._forward_frame(mag, return_mean)
        elif mag.dim() == 3:
            # Sequence [B, F, T]
            return self._forward_sequence(mag, return_mean)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {mag.dim()}D")

    def _forward_frame(self, mag, return_mean):
        """Process single frame."""
        B, F = mag.shape

        # Compute frame mean
        frame_mean = mag.mean(dim=1, keepdim=True)  # [B, 1]

        if self.training:
            # Update running mean
            batch_mean = frame_mean.mean()
            self.running_mean = self.momentum * self.running_mean + \
                                (1 - self.momentum) * batch_mean.detach()
            self.num_batches_tracked += 1

        # Apply bias if exists
        if self.bias is not None:
            frame_mean = frame_mean + self.bias

        # Normalize
        mag_norm = mag / (frame_mean + self.epsilon)

        if return_mean:
            return mag_norm, frame_mean
        return mag_norm

    def _forward_sequence(self, mag, return_mean):
        """Process sequence with per-frame EMA."""
        B, F, T = mag.shape

        # Process frame-by-frame to maintain causality
        mag_norm_list = []
        mag_mean_list = []

        # Initialize current_mean from running_mean (scalar buffer)
        # Expand to match batch size
        current_mean = self.running_mean.clone().expand(B).unsqueeze(1)  # [B, 1]

        for t in range(T):
            frame = mag[:, :, t]  # [B, F]
            frame_mean = frame.mean(dim=1, keepdim=True)  # [B, 1]

            # EMA update (causal: only uses current and past)
            current_mean = self.momentum * current_mean + \
                          (1 - self.momentum) * frame_mean

            # Apply bias
            mean_with_bias = current_mean
            if self.bias is not None:
                mean_with_bias = current_mean + self.bias

            # Normalize
            frame_norm = frame / (mean_with_bias + self.epsilon)

            mag_norm_list.append(frame_norm)
            mag_mean_list.append(mean_with_bias)

        # Stack results
        mag_norm = torch.stack(mag_norm_list, dim=2)  # [B, F, T]

        # Update running_mean buffer with batch average of last frame
        if self.training:
            batch_mean = current_mean.mean()
            # Detach to prevent gradient flow to buffer
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))
            self.num_batches_tracked += 1

        if return_mean:
            mag_mean = torch.stack(mag_mean_list, dim=2)  # [B, 1, T]
            return mag_norm, mag_mean
        return mag_norm

    def reset_state(self):
        """Reset running mean to initial state (useful for streaming)."""
        self.running_mean.fill_(1.0)
        self.num_batches_tracked.zero_()

    def extra_repr(self):
        return f'momentum={self.momentum}, learnable_bias={self.bias is not None}'


class SlidingWindowEnergyNormalizer(nn.Module):
    """
    Sliding window based energy normalizer.

    Computes magnitude mean over a fixed causal window.
    More responsive to energy changes than EMA, but requires buffering.

    Args:
        window_size: Number of frames to average (default: 20)
        learnable_bias: If True, adds learnable bias
        epsilon: Small constant for numerical stability

    Example:
        >>> normalizer = SlidingWindowEnergyNormalizer(window_size=20)
        >>> mag = torch.randn(2, 201, 100).abs()
        >>> mag_norm, mag_mean = normalizer(mag)
    """

    def __init__(self, window_size=20, learnable_bias=True, epsilon=1e-8):
        super().__init__()
        self.window_size = window_size
        self.epsilon = epsilon

        if learnable_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

        # Buffer for streaming (stores recent frames)
        self.register_buffer('buffer', None)
        self.register_buffer('buffer_ptr', torch.tensor(0, dtype=torch.long))

    def forward(self, mag, return_mean=True):
        """
        Forward pass with sliding window normalization.

        Args:
            mag: Magnitude spectrogram [B, F, T] or [B, F]
            return_mean: If True, returns both normalized mag and mean

        Returns:
            mag_norm: Normalized magnitude
            mag_mean: Window mean used for normalization (if return_mean=True)
        """
        if mag.dim() == 2:
            return self._forward_frame(mag, return_mean)
        elif mag.dim() == 3:
            return self._forward_sequence(mag, return_mean)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {mag.dim()}D")

    def _forward_sequence(self, mag, return_mean):
        """Process sequence with sliding window."""
        B, F, T = mag.shape

        mag_norm_list = []
        mag_mean_list = []

        for t in range(T):
            # Compute mean over window [max(0, t-window+1) : t+1]
            start = max(0, t - self.window_size + 1)
            window_mag = mag[:, :, start:t+1]
            window_mean = window_mag.mean(dim=[1, 2], keepdim=True)  # [B, 1, 1]

            # Apply bias
            if self.bias is not None:
                window_mean = window_mean + self.bias

            # Normalize current frame
            frame_norm = mag[:, :, t] / (window_mean.squeeze(-1) + self.epsilon)

            mag_norm_list.append(frame_norm)
            mag_mean_list.append(window_mean.squeeze(-1))

        mag_norm = torch.stack(mag_norm_list, dim=2)  # [B, F, T]

        if return_mean:
            mag_mean = torch.stack(mag_mean_list, dim=2)  # [B, 1, T]
            return mag_norm, mag_mean
        return mag_norm

    def _forward_frame(self, mag, return_mean):
        """Process single frame (streaming mode)."""
        B, F = mag.shape

        # Initialize buffer if needed
        if self.buffer is None:
            self.buffer = torch.zeros(B, F, self.window_size,
                                     dtype=mag.dtype, device=mag.device)

        # Add frame to buffer
        ptr = self.buffer_ptr.item()
        self.buffer[:, :, ptr] = mag
        # Use fill_ to update buffer (in-place operation)
        new_ptr = (ptr + 1) % self.window_size
        self.buffer_ptr.fill_(new_ptr)

        # Compute mean over valid buffer entries
        num_valid = min(ptr + 1, self.window_size)
        if ptr < self.window_size - 1:
            window_mean = self.buffer[:, :, :num_valid].mean(dim=[1, 2], keepdim=True)
        else:
            window_mean = self.buffer.mean(dim=[1, 2], keepdim=True)

        # Apply bias
        if self.bias is not None:
            window_mean = window_mean + self.bias

        # Normalize
        mag_norm = mag / (window_mean.squeeze(-1) + self.epsilon)

        if return_mean:
            return mag_norm, window_mean.squeeze(-1)
        return mag_norm

    def reset_state(self):
        """Reset buffer (useful for new utterance in streaming)."""
        if self.buffer is not None:
            self.buffer.zero_()
        self.buffer_ptr.zero_()

    def extra_repr(self):
        return f'window_size={self.window_size}, learnable_bias={self.bias is not None}'


class LearnableScaleNormalizer(nn.Module):
    """
    Learnable fixed scale normalization.

    Uses a single learnable scale factor instead of computing statistics.
    Fastest option, but less adaptive to input variations.

    Args:
        init_scale: Initial scale value (default: 1.0)
        per_freq: If True, learns different scale per frequency bin
        freq_bins: Number of frequency bins (required if per_freq=True)

    Example:
        >>> normalizer = LearnableScaleNormalizer(init_scale=2.0)
        >>> mag = torch.randn(2, 201, 100).abs()
        >>> mag_norm = normalizer(mag)
    """

    def __init__(self, init_scale=1.0, per_freq=False, freq_bins=None):
        super().__init__()
        self.per_freq = per_freq

        if per_freq:
            if freq_bins is None:
                raise ValueError("freq_bins must be specified when per_freq=True")
            self.scale = nn.Parameter(torch.full((1, freq_bins, 1), init_scale))
        else:
            self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, mag):
        """
        Forward pass with learnable scale normalization.

        Args:
            mag: Magnitude spectrogram [B, F, T] or [B, F]

        Returns:
            mag_norm: Normalized magnitude / scale
        """
        return mag / self.scale

    def extra_repr(self):
        return f'per_freq={self.per_freq}'


class HybridEnergyNormalizer(nn.Module):
    """
    Hybrid normalizer combining EMA with learnable compensation.

    Best of both worlds: adaptive to signal statistics (EMA) while learning
    to compensate for systematic biases (learnable scale/bias).

    Recommended for production use.

    Args:
        momentum: EMA momentum (default: 0.99)
        use_scale: If True, adds learnable scale factor
        use_bias: If True, adds learnable bias
        epsilon: Small constant for numerical stability

    Example:
        >>> normalizer = HybridEnergyNormalizer(momentum=0.99)
        >>> mag = torch.randn(2, 201, 100).abs()
        >>> mag_norm, stats = normalizer(mag)
    """

    def __init__(self, momentum=0.99, use_scale=True, use_bias=True, epsilon=1e-8):
        super().__init__()
        self.ema_normalizer = EMAEnergyNormalizer(momentum=momentum,
                                                   learnable_bias=False,
                                                   epsilon=epsilon)

        if use_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter('scale', None)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

    def forward(self, mag, return_stats=True):
        """
        Forward pass with hybrid normalization.

        Args:
            mag: Magnitude spectrogram [B, F, T] or [B, F]
            return_stats: If True, returns EMA mean and final scale

        Returns:
            mag_norm: Normalized magnitude
            stats: Dict with 'ema_mean' and 'final_scale' (if return_stats=True)
        """
        # EMA normalization
        mag_ema_norm, ema_mean = self.ema_normalizer(mag, return_mean=True)

        # Apply learnable scale/bias
        mag_norm = mag_ema_norm

        if self.scale is not None:
            mag_norm = mag_norm / self.scale

        if self.bias is not None:
            mag_norm = mag_norm + self.bias

        if return_stats:
            stats = {
                'ema_mean': ema_mean,
                'final_scale': self.scale if self.scale is not None else torch.tensor(1.0)
            }
            return mag_norm, stats
        return mag_norm

    def reset_state(self):
        """Reset EMA state."""
        self.ema_normalizer.reset_state()

    def extra_repr(self):
        return (f'momentum={self.ema_normalizer.momentum}, '
                f'use_scale={self.scale is not None}, '
                f'use_bias={self.bias is not None}')


# Factory function for easy instantiation
def create_energy_normalizer(norm_type='ema', **kwargs):
    """
    Factory function to create energy normalizers.

    Args:
        norm_type: Type of normalizer ('ema', 'sliding', 'learnable', 'hybrid')
        **kwargs: Arguments passed to the normalizer constructor

    Returns:
        Energy normalizer module

    Example:
        >>> norm = create_energy_normalizer('ema', momentum=0.99, learnable_bias=True)
        >>> norm = create_energy_normalizer('sliding', window_size=20)
        >>> norm = create_energy_normalizer('hybrid')
    """
    normalizers = {
        'ema': EMAEnergyNormalizer,
        'sliding': SlidingWindowEnergyNormalizer,
        'learnable': LearnableScaleNormalizer,
        'hybrid': HybridEnergyNormalizer,
    }

    if norm_type not in normalizers:
        raise ValueError(f"Unknown normalizer type: {norm_type}. "
                        f"Choose from {list(normalizers.keys())}")

    return normalizers[norm_type](**kwargs)
