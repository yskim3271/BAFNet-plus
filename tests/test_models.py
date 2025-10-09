"""Unit tests for model architectures."""

import torch
import pytest
from models.primeknet import PrimeKnet
from models.primeknet_gru import PrimeKnet as PrimeKnetGRU


class TestPrimeKnet:
    """Test PrimeKnet model."""

    def test_primeknet_forward(self):
        """Test forward pass of PrimeKnet."""
        model = PrimeKnet(
            fft_len=400,
            dense_channel=64,
            sigmoid_beta=1.0,
            dense_depth=4,
            num_tsblock=2,
            time_dw_kernel_size=3,
            time_block_kernel=[3, 11, 23, 31],
            freq_block_kernel=[3, 11, 23, 31],
            time_block_num=2,
            freq_block_num=2,
            infer_type='masking',
            causal=False
        )
        model.eval()

        batch_size = 2
        freq_bins = 201  # (400 // 2) + 1
        time_frames = 100

        # Input: [B, F, T, 2] complex spectrogram
        noisy_com = torch.randn(batch_size, freq_bins, time_frames, 2)

        with torch.no_grad():
            mag_hat, pha_hat, com_hat = model(noisy_com)

        # Check output shapes
        assert mag_hat.shape == (batch_size, freq_bins, time_frames)
        assert pha_hat.shape == (batch_size, freq_bins, time_frames)
        assert com_hat.shape == (batch_size, freq_bins, time_frames, 2)

    def test_primeknet_gru_forward(self):
        """Test forward pass of PrimeKnet-GRU."""
        model = PrimeKnetGRU(
            fft_len=400,
            dense_channel=64,
            sigmoid_beta=1.0,
            dense_depth=4,
            num_tsblock=2,
            time_block_num=2,
            gru_layers=1,
            gru_hidden_size=64,
            freq_block_kernel=[3, 11, 23, 31],
            freq_block_num=2,
            infer_type='masking',
            causal=False
        )
        model.eval()

        batch_size = 2
        freq_bins = 201
        time_frames = 100

        noisy_com = torch.randn(batch_size, freq_bins, time_frames, 2)

        with torch.no_grad():
            mag_hat, pha_hat, com_hat = model(noisy_com)

        assert mag_hat.shape == (batch_size, freq_bins, time_frames)
        assert pha_hat.shape == (batch_size, freq_bins, time_frames)
        assert com_hat.shape == (batch_size, freq_bins, time_frames, 2)

    def test_model_parameters(self):
        """Test that model has reasonable number of parameters."""
        model = PrimeKnet(
            fft_len=400,
            dense_channel=64,
            sigmoid_beta=1.0,
            dense_depth=4,
            num_tsblock=4,
            causal=True
        )

        total_params = sum(p.numel() for p in model.parameters())

        # Should have between 1M and 10M parameters
        assert 1_000_000 < total_params < 10_000_000
