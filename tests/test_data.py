"""Unit tests for data processing."""

import torch
import numpy as np
import pytest
from data import tailor_dB_FS, norm_amplitude, is_clipped


class TestDataUtils:
    """Test data utility functions."""

    def test_tailor_dB_FS(self):
        """Test dB FS scaling."""
        y = torch.randn(16000)  # 1 second @ 16kHz
        target_dB = -25

        y_scaled, rms, scalar = tailor_dB_FS(y, target_dB_FS=target_dB)

        # Check that output has correct scale
        new_rms = torch.sqrt(torch.mean(y_scaled ** 2))
        expected_rms = 10 ** (target_dB / 20)

        assert torch.allclose(new_rms, torch.tensor(expected_rms), atol=1e-5)
        assert y_scaled.shape == y.shape

    def test_norm_amplitude(self):
        """Test amplitude normalization."""
        y = torch.tensor([1.0, -2.0, 0.5, -0.3])

        y_norm, scalar = norm_amplitude(y)

        # Maximum absolute value should be 1.0 or less
        assert torch.max(torch.abs(y_norm)) <= 1.0
        assert scalar == 2.0  # max abs value
        assert torch.allclose(y_norm, y / scalar)

    def test_is_clipped(self):
        """Test clipping detection."""
        y_clean = torch.tensor([0.5, -0.5, 0.3, -0.2])
        y_clipped = torch.tensor([1.0, -1.0, 0.5, -0.5])

        assert not is_clipped(y_clean, clipping_threshold=0.999)
        assert is_clipped(y_clipped, clipping_threshold=0.999)

    def test_tailor_dB_FS_preserves_shape(self):
        """Test that tailor_dB_FS preserves input shape."""
        shapes = [(16000,), (2, 16000), (4, 2, 16000)]

        for shape in shapes:
            y = torch.randn(shape)
            y_scaled, _, _ = tailor_dB_FS(y)
            assert y_scaled.shape == shape
