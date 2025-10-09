"""Unit tests for STFT utilities."""

import torch
import pytest
from stft import mag_pha_to_complex


class TestSTFT:
    """Test STFT-related functions."""

    def test_mag_pha_to_complex_shape(self):
        """Test that mag_pha_to_complex produces correct output shape."""
        batch_size, freq_bins, time_frames = 2, 201, 100

        mag = torch.randn(batch_size, freq_bins, time_frames)
        pha = torch.randn(batch_size, freq_bins, time_frames)

        complex_spec = mag_pha_to_complex(mag, pha)

        # Output should have shape [B, F, T, 2] for real and imaginary parts
        assert complex_spec.shape == (batch_size, freq_bins, time_frames, 2)

    def test_mag_pha_to_complex_values(self):
        """Test that mag_pha_to_complex produces correct values."""
        mag = torch.tensor([[[1.0, 2.0]]])  # [1, 1, 2]
        pha = torch.tensor([[[0.0, 3.14159265]]])  # [1, 1, 2]

        complex_spec = mag_pha_to_complex(mag, pha)

        # Check real and imaginary parts
        # mag * cos(pha), mag * sin(pha)
        assert torch.allclose(complex_spec[0, 0, 0, 0], torch.tensor(1.0), atol=1e-5)  # real
        assert torch.allclose(complex_spec[0, 0, 0, 1], torch.tensor(0.0), atol=1e-5)  # imag
        assert torch.allclose(complex_spec[0, 0, 1, 0], torch.tensor(-2.0), atol=1e-5)  # real
        assert torch.allclose(complex_spec[0, 0, 1, 1], torch.tensor(0.0), atol=1e-4)  # imag (near 0)
