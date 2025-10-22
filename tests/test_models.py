"""Unit tests for model architectures."""

import torch
import pytest
from models.primeknet import PrimeKnet
from models.primeknet_gru import PrimeKnet as PrimeKnetGRU
from models.primeknet_lk import PrimeKnet as PrimeKnetLK


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


class TestPrimeKnetLookahead:
    """Test PrimeKnet with Lookahead Buffer."""

    def test_primeknet_lk_forward(self):
        """Test forward pass of PrimeKnet with lookahead."""
        # Test with different lookahead ratios
        lookahead_ratios = [0.0, 0.2, 0.3, 0.5]

        for lr in lookahead_ratios:
            model = PrimeKnetLK(
                win_len=400,
                hop_len=100,
                fft_len=400,
                dense_channel=64,
                sigmoid_beta=1.0,
                compress_factor=0.3,
                dense_depth=3,  # Updated default
                num_tsblock=2,
                freq_block_kernel=[3, 11, 23, 31],
                time_block_num=2,
                freq_block_num=2,
                infer_type='masking',
                lookahead_ratio=lr
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

    def test_lookahead_info(self):
        """Test that lookahead info is correctly calculated."""
        model = PrimeKnetLK(
            win_len=400,
            hop_len=100,
            fft_len=400,
            dense_channel=64,
            sigmoid_beta=1.0,
            compress_factor=0.3,
            lookahead_ratio=0.3
        )

        info = model.get_lookahead_info()

        assert info['lookahead_ratio'] == 0.3
        assert info['lookahead_frames'] > 0
        assert info['lookahead_ms'] > 0
        assert info['past_frames'] > 0
        assert info['total_rf_frames'] == info['lookahead_frames'] + info['past_frames']

    def test_lookahead_ratio_bounds(self):
        """Test that lookahead ratio is bounded correctly."""
        # Should work with ratio = 0.0
        model = PrimeKnetLK(
            win_len=400,
            hop_len=100,
            fft_len=400,
            dense_channel=64,
            sigmoid_beta=1.0,
            compress_factor=0.3,
            lookahead_ratio=0.0
        )
        assert model.lookahead_ratio == 0.0

        # Should work with ratio = 0.5
        model = PrimeKnetLK(
            win_len=400,
            hop_len=100,
            fft_len=400,
            dense_channel=64,
            sigmoid_beta=1.0,
            compress_factor=0.3,
            lookahead_ratio=0.5
        )
        assert model.lookahead_ratio == 0.5

        # Should fail with ratio > 0.5
        with pytest.raises(AssertionError):
            model = PrimeKnetLK(
                win_len=400,
                hop_len=100,
                fft_len=400,
                dense_channel=64,
                sigmoid_beta=1.0,
                compress_factor=0.3,
                lookahead_ratio=0.6
            )

        # Should fail with negative ratio
        with pytest.raises(AssertionError):
            model = PrimeKnetLK(
                win_len=400,
                hop_len=100,
                fft_len=400,
                dense_channel=64,
                sigmoid_beta=1.0,
                compress_factor=0.3,
                lookahead_ratio=-0.1
            )
