"""
PrimeKnetV2: Streaming-friendly variant with Channel-wise LayerNorm.

This module replaces InstanceNorm2d with ChannelWiseLayerNorm2d for online
streaming compatibility. InstanceNorm2d computes statistics across T×F dimensions,
requiring the full sequence, while ChannelWiseLayerNorm2d normalizes across channels
at each (t, f) position independently.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.stft import mag_pha_to_complex, complex_to_mag_pha

# Import shared components from primeknet
from src.models.primeknet import (
    get_padding,
    get_padding_2d,
    CausalConv1d,
    CausalConv2d,
    AsymmetricConv2d,
    SimpleGate,
    LayerNorm1d,
    Group_Prime_Kernel_FFN,
    LearnableSigmoid_2d,
)


class ChannelWiseLayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 4D tensor [B, C, T, F].

    Normalizes across channel dimension at each (t, f) position.
    Streaming-friendly: no dependency on time axis.

    Args:
        num_channels: Number of channels (C dimension)
        eps: Small constant for numerical stability

    Shape:
        - Input: [B, C, T, F]
        - Output: [B, C, T, F]
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: [B, C, T, F]
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, T, F]
        var = x.var(dim=1, keepdim=True, unbiased=False)  # [B, 1, T, F]
        x = (x - mean) / torch.sqrt(var + self.eps)
        # weight, bias: [C] -> [1, C, 1, 1]
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class StreamingSqueeze1d(nn.Module):
    """Streaming-friendly squeeze operator for [B, C, T] -> [B, C, 1].

    Modes:
        - "global": mean over the full chunk time axis (equivalent to AdaptiveAvgPool1d(1)).
        - "ema": exponential moving average of chunk means (stateful in eval if enabled).
        - "local": mean over the last W frames of the chunk (causal summary).
    """

    def __init__(
        self,
        channels: int,
        mode: str = "global",
        ema_alpha: float = 0.99,
        local_window: int = 32,
        stateful_eval: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.channels = channels
        self.mode = mode
        self.ema_alpha = float(ema_alpha)
        self.local_window = int(local_window)
        self.stateful_eval = bool(stateful_eval)
        self.eps = float(eps)

        # Keep a per-item running mean: shape [B, C, 1] where B is the runtime batch size.
        # We initialize lazily to avoid hard-coding B (e.g., B*F inside TS_BLOCK time stage).
        self.register_buffer("_ema_mean", torch.zeros(0), persistent=False)

    def reset_state(self):
        """Reset internal EMA state (intended for streaming inference between utterances)."""
        self._ema_mean = torch.zeros(0, device=self._ema_mean.device)

    def _chunk_mean(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, C, 1]
        return x.mean(dim=2, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if x.dim() != 3:
            raise ValueError(f"StreamingSqueeze1d expects [B, C, T], got {tuple(x.shape)}")

        if self.mode == "global":
            return self._chunk_mean(x)

        if self.mode == "local":
            w = max(1, self.local_window)
            x_tail = x[..., -w:]
            return x_tail.mean(dim=2, keepdim=True)

        if self.mode == "ema":
            m_chunk = self._chunk_mean(x)

            # During training, avoid cross-mini-batch state carry to keep behavior well-defined.
            if self.training or not self.stateful_eval:
                return m_chunk

            # Eval + stateful: update and use EMA mean.
            with torch.no_grad():
                if self._ema_mean.numel() == 0 or self._ema_mean.shape != m_chunk.shape:
                    self._ema_mean = m_chunk.detach()
                else:
                    a = self.ema_alpha
                    self._ema_mean.mul_(a).add_(m_chunk.detach(), alpha=(1.0 - a))
            return self._ema_mean

        raise ValueError(f"Unsupported squeeze mode: {self.mode!r} (expected 'global'|'ema'|'local')")


class Channel_Attention_Block_V2(nn.Module):
    """Channel Attention Block with selectable streaming-friendly squeeze.

    This is a V2 replacement for primeknet.Channel_Attention_Block to allow switching
    the squeeze strategy via hyperparameters while keeping the rest of the block intact.
    """

    def __init__(
        self,
        in_channels: int = 64,
        dw_kernel_size: int = 3,
        causal: bool = True,
        sca_mode: str = "global",
        sca_ema_alpha: float = 0.99,
        sca_local_window: int = 32,
        sca_stateful_eval: bool = True,
    ):
        super().__init__()

        dw_channel = in_channels * 2

        conv_fn = CausalConv1d if causal else nn.Conv1d

        self.norm = LayerNorm1d(in_channels)
        self.pwconv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.dwconv = conv_fn(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=dw_kernel_size,
            padding=get_padding(dw_kernel_size),
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.sg = SimpleGate()

        # SCA: squeeze over time -> project -> multiply
        sca_channels = dw_channel // 2
        self.sca_squeeze = StreamingSqueeze1d(
            channels=sca_channels,
            mode=sca_mode,
            ema_alpha=sca_ema_alpha,
            local_window=sca_local_window,
            stateful_eval=sca_stateful_eval,
        )
        self.sca_proj = nn.Conv1d(
            in_channels=sca_channels,
            out_channels=sca_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.pwconv2 = nn.Conv1d(
            in_channels=sca_channels,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)

    def reset_state(self):
        """Reset streaming state of internal squeeze module."""
        self.sca_squeeze.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = self.sg(x)

        s = self.sca_squeeze(x)  # [B, C, 1]
        g = self.sca_proj(s)     # [B, C, 1]
        x = x * g

        x = self.pwconv2(x)
        x = skip + x * self.beta
        return x


class TS_BLOCK_V2(nn.Module):
    """TS_BLOCK variant that uses Channel_Attention_Block_V2 for selectable SCA behavior."""

    def __init__(
        self,
        dense_channel: int = 64,
        time_block_num: int = 2,
        freq_block_num: int = 2,
        time_dw_kernel_size: int = 3,
        time_block_kernel: List[int] = [3, 11, 23, 31],
        freq_block_kernel: List[int] = [3, 11, 23, 31],
        causal: bool = True,
        sca_mode: str = "global",
        sca_ema_alpha: float = 0.99,
        sca_local_window: int = 32,
        sca_stateful_eval: bool = True,
    ):
        super().__init__()
        self.dense_channel = dense_channel

        time_stage = nn.ModuleList([])
        freq_stage = nn.ModuleList([])

        for _ in range(time_block_num):
            time_stage.append(
                nn.Sequential(
                    Channel_Attention_Block_V2(
                        in_channels=dense_channel,
                        dw_kernel_size=time_dw_kernel_size,
                        causal=causal,
                        sca_mode=sca_mode,
                        sca_ema_alpha=sca_ema_alpha,
                        sca_local_window=sca_local_window,
                        sca_stateful_eval=sca_stateful_eval,
                    ),
                    Group_Prime_Kernel_FFN(in_channel=dense_channel, kernel_list=time_block_kernel, causal=causal),
                )
            )

        # Frequency stage remains non-causal on the time axis (it operates along frequency length).
        for _ in range(freq_block_num):
            freq_stage.append(
                nn.Sequential(
                    Channel_Attention_Block_V2(
                        in_channels=dense_channel,
                        dw_kernel_size=3,
                        causal=False,
                        sca_mode=sca_mode,
                        sca_ema_alpha=sca_ema_alpha,
                        sca_local_window=sca_local_window,
                        sca_stateful_eval=sca_stateful_eval,
                    ),
                    Group_Prime_Kernel_FFN(in_channel=dense_channel, kernel_list=freq_block_kernel, causal=False),
                )
            )

        self.time_stage = nn.Sequential(*time_stage)
        self.freq_stage = nn.Sequential(*freq_stage)

        self.beta_t = nn.Parameter(torch.zeros((1, dense_channel, 1)), requires_grad=True)
        self.beta_f = nn.Parameter(torch.zeros((1, dense_channel, 1)), requires_grad=True)

    def reset_state(self):
        """Reset streaming state for all Channel_Attention_Block_V2 modules."""
        for m in self.modules():
            if isinstance(m, Channel_Attention_Block_V2):
                m.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        B, C, T, Freq = x.size()
        x = x.permute(0, 3, 1, 2).reshape(B * Freq, C, T)

        x = self.time_stage(x) + x * self.beta_t
        x = x.view(B, Freq, C, T).permute(0, 3, 2, 1).reshape(B * T, C, Freq)

        x = self.freq_stage(x) + x * self.beta_f
        x = x.view(B, T, C, Freq).permute(0, 2, 1, 3)
        return x


class DS_DDB_V2(nn.Module):
    """
    Dense Dilated Depthwise Block V2 with ChannelWiseLayerNorm2d.

    Streaming-friendly version using channel-wise normalization instead of
    InstanceNorm2d.

    Args:
        dense_channel: Number of dense channels
        kernel_size: Convolution kernel size (time, freq)
        depth: Number of dilated layers (default: 4)
        causal: Deprecated parameter, kept for compatibility but ignored
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding on time axis
    """
    def __init__(self, dense_channel, kernel_size=(3, 3), depth=4, causal=False, padding_ratio=(0.5, 0.5)):
        super().__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.dense_block = nn.ModuleList([])

        # Validate padding_ratio
        left_ratio, right_ratio = padding_ratio
        assert abs(left_ratio + right_ratio - 1.0) < 1e-6, \
            f"padding_ratio must sum to 1.0, got {left_ratio + right_ratio}"
        assert 0.0 <= left_ratio <= 1.0 and 0.0 <= right_ratio <= 1.0, \
            f"padding_ratio values must be in [0, 1], got ({left_ratio}, {right_ratio})"

        self.padding_ratio = padding_ratio

        # Always use AsymmetricConv2d with specified padding_ratio
        conv_fn = lambda in_ch, out_ch, ks, dil, pad, groups, bias: AsymmetricConv2d(
            in_ch, out_ch, ks, padding=pad, padding_ratio=self.padding_ratio,
            dilation=dil, groups=groups, bias=bias
        )

        for i in range(depth):
            dil = 2 ** i
            padding = get_padding_2d(kernel_size, (dil, 1))
            dense_conv = nn.Sequential(
                conv_fn(dense_channel*(i+1), dense_channel*(i+1), kernel_size,
                       dil=(dil, 1), pad=padding, groups=dense_channel*(i+1), bias=True),
                nn.Conv2d(in_channels=dense_channel*(i+1), out_channels=dense_channel,
                         kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                ChannelWiseLayerNorm2d(dense_channel),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoderV2(nn.Module):
    """
    Dense Encoder V2 with ChannelWiseLayerNorm2d.

    Streaming-friendly version using channel-wise normalization.

    Args:
        dense_channel: Number of dense channels
        in_channel: Number of input channels
        depth: Depth of DS_DDB_V2 (default: 4)
        causal: Deprecated parameter, kept for compatibility but ignored
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding
    """
    def __init__(self, dense_channel, in_channel, depth=4, causal=False, padding_ratio=(0.5, 0.5)):
        super().__init__()
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            ChannelWiseLayerNorm2d(dense_channel),
            nn.PReLU(dense_channel))
        self.dense_block = DS_DDB_V2(dense_channel, depth=depth, causal=causal, padding_ratio=padding_ratio)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            ChannelWiseLayerNorm2d(dense_channel),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MaskDecoderV2(nn.Module):
    """
    Mask Decoder V2 with ChannelWiseLayerNorm2d.

    Streaming-friendly version using channel-wise normalization.

    Args:
        dense_channel: Number of dense channels
        n_fft: FFT size
        sigmoid_beta: Beta parameter for learnable sigmoid
        out_channel: Number of output channels (default: 1)
        depth: Depth of DS_DDB_V2 (default: 4)
        causal: Deprecated parameter, kept for compatibility but ignored
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding
    """
    def __init__(self,
                 dense_channel,
                 n_fft,
                 sigmoid_beta,
                 out_channel=1,
                 depth=4,
                 causal=False,
                 padding_ratio=(0.5, 0.5)):
        super().__init__()
        self.n_fft = n_fft
        self.dense_block = DS_DDB_V2(dense_channel, depth=depth, causal=causal, padding_ratio=padding_ratio)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            ChannelWiseLayerNorm2d(out_channel),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(n_fft//2+1, beta=sigmoid_beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.squeeze(1).transpose(1, 2)
        x = self.lsigmoid(x).transpose(1,2).unsqueeze(1)
        return x


class PhaseDecoderV2(nn.Module):
    """
    Phase Decoder V2 with ChannelWiseLayerNorm2d.

    Streaming-friendly version using channel-wise normalization.

    Args:
        dense_channel: Number of dense channels
        out_channel: Number of output channels (default: 1)
        depth: Depth of DS_DDB_V2 (default: 4)
        causal: Deprecated parameter, kept for compatibility but ignored
        padding_ratio: (left_ratio, right_ratio) for asymmetric padding
    """
    def __init__(self,
                 dense_channel,
                 out_channel=1,
                 depth=4,
                 causal=False,
                 padding_ratio=(0.5, 0.5)):
        super().__init__()
        self.dense_block = DS_DDB_V2(dense_channel, depth=depth, causal=causal, padding_ratio=padding_ratio)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            ChannelWiseLayerNorm2d(dense_channel),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        # Add epsilon to prevent gradient explosion in atan2 backward
        x = torch.atan2(x_i + 1e-8, x_r + 1e-8)
        return x


class PrimeKnetV2(nn.Module):
    """
    PrimeKnetV2: Streaming-friendly variant with Channel-wise LayerNorm.

    This version replaces all InstanceNorm2d layers with ChannelWiseLayerNorm2d,
    enabling frame-by-frame streaming inference without requiring statistics
    from the entire sequence.

    Key differences from PrimeKnet:
    - Uses ChannelWiseLayerNorm2d instead of InstanceNorm2d
    - Normalizes across channels at each (t,f) position
    - No temporal dependency in normalization layers

    Args:
        encoder_padding_ratio: (left_ratio, right_ratio) for encoder asymmetric padding
            - (1.0, 0.0): Fully causal (minimum latency)
            - (0.5, 0.5): Symmetric
        decoder_padding_ratio: (left_ratio, right_ratio) for decoder asymmetric padding
        causal: Controls TS_BLOCK causality (True for causal time processing)

    Example (streaming configuration):
        model = PrimeKnetV2(
            ...,
            encoder_padding_ratio=(1.0, 0.0),  # Fully causal
            decoder_padding_ratio=(1.0, 0.0),  # Fully causal
            causal=True,                        # TS_BLOCK causal
        )
    """
    def __init__(self,
                 win_len,
                 hop_len,
                 fft_len,
                 dense_channel,
                 sigmoid_beta,
                 compress_factor,
                 dense_depth=4,
                 num_tsblock=4,
                 time_dw_kernel_size=3,
                 time_block_kernel=[3, 11, 23, 31],
                 freq_block_kernel=[3, 11, 23, 31],
                 time_block_num=2,
                 freq_block_num=2,
                 infer_type='masking',
                 causal=False,
                 encoder_padding_ratio=(0.5, 0.5),
                 decoder_padding_ratio=(0.5, 0.5),
                 sca_mode: str = "global",
                 sca_ema_alpha: float = 0.99,
                 sca_local_window: int = 32,
                 sca_stateful_eval: bool = True,
                 ):
        super().__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.dense_channel = dense_channel
        self.dense_depth = dense_depth
        self.sigmoid_beta = sigmoid_beta
        self.compress_factor = compress_factor
        self.num_tsblock = num_tsblock
        self.causal = causal
        self.infer_type = infer_type
        self.sca_mode = sca_mode
        self.sca_ema_alpha = float(sca_ema_alpha)
        self.sca_local_window = int(sca_local_window)
        self.sca_stateful_eval = bool(sca_stateful_eval)

        # Validate encoder_padding_ratio
        enc_left, enc_right = encoder_padding_ratio
        assert abs(enc_left + enc_right - 1.0) < 1e-6, \
            f"encoder_padding_ratio must sum to 1.0, got {enc_left + enc_right}"
        assert 0.0 <= enc_left <= 1.0 and 0.0 <= enc_right <= 1.0, \
            f"encoder_padding_ratio values must be in [0, 1], got ({enc_left}, {enc_right})"

        # Validate decoder_padding_ratio
        dec_left, dec_right = decoder_padding_ratio
        assert abs(dec_left + dec_right - 1.0) < 1e-6, \
            f"decoder_padding_ratio must sum to 1.0, got {dec_left + dec_right}"
        assert 0.0 <= dec_left <= 1.0 and 0.0 <= dec_right <= 1.0, \
            f"decoder_padding_ratio values must be in [0, 1], got ({dec_left}, {dec_right})"

        self.encoder_padding_ratio = encoder_padding_ratio
        self.decoder_padding_ratio = decoder_padding_ratio
        assert infer_type in ['masking', 'mapping'], 'infer_type must be either masking or mapping'

        # Use V2 versions with ChannelWiseLayerNorm2d
        self.dense_encoder = DenseEncoderV2(dense_channel, in_channel=2, depth=dense_depth,
                                            causal=causal, padding_ratio=encoder_padding_ratio)
        # TS block: use V2 variant to make SCA (channel attention squeeze) selectable.
        self.sequence_block = nn.Sequential(
            *[
                TS_BLOCK_V2(
                    dense_channel,
                    time_block_num,
                    freq_block_num,
                    time_dw_kernel_size,
                    time_block_kernel,
                    freq_block_kernel,
                    causal=causal,
                    sca_mode=sca_mode,
                    sca_ema_alpha=sca_ema_alpha,
                    sca_local_window=sca_local_window,
                    sca_stateful_eval=sca_stateful_eval,
                )
                for _ in range(num_tsblock)
            ]
        )
        self.mask_decoder = MaskDecoderV2(dense_channel, fft_len, sigmoid_beta, out_channel=1,
                                          depth=dense_depth, causal=causal, padding_ratio=decoder_padding_ratio)
        self.phase_decoder = PhaseDecoderV2(dense_channel, out_channel=1, depth=dense_depth,
                                            causal=causal, padding_ratio=decoder_padding_ratio)

    def reset_streaming_state(self):
        """Reset stateful components for streaming inference (e.g., EMA squeeze)."""
        for m in self.modules():
            if isinstance(m, TS_BLOCK_V2):
                m.reset_state()

    def forward(self, noisy_com):
        # Input shape: [B, F, T, 2]
        mag, pha = complex_to_mag_pha(noisy_com, stack_dim=-1)  # [B, F, T] each

        x = torch.stack((mag, pha), dim=1).permute(0, 1, 3, 2)  # [B, 2, T, F]
        x = self.dense_encoder(x)
        x = self.sequence_block(x)

        # mask_decoder output: [B, 1, T, F] -> squeeze/transpose -> [B, F, T]
        mask = self.mask_decoder(x).squeeze(1).transpose(1, 2)
        if self.infer_type == 'masking':
            est_mag = mag * mask  # [B, F, T]
        elif self.infer_type == 'mapping':
            est_mag = mask

        est_pha = self.phase_decoder(x).squeeze(1).transpose(1, 2)  # [B, F, T]
        est_com = mag_pha_to_complex(est_mag, est_pha, stack_dim=-1)

        return est_mag, est_pha, est_com
