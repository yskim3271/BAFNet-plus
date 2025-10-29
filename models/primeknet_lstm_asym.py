"""PrimeKnet-LSTM with Asymmetric Padding for Latency-Performance Trade-off.

This module extends PrimeKnet-LSTM with flexible asymmetric padding control
for encoder and decoder, enabling exploration of latency-performance Pareto frontier.

Key Features:
- AsymmetricConv2d: Flexible L:R padding ratio for time axis
- Separate encoder/decoder padding control
- Uniform ratio across all layers within each module
- Maintains LSTM unidirectionality for streaming compatibility

Reference: docs/lookahead_design.md
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from stft import mag_pha_to_complex


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(
    kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)
) -> Tuple[int, int]:
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


class AsymmetricConv2d(nn.Module):
    """Conv2d with asymmetric time-axis padding.

    Applies flexible L:R padding ratio on time axis while keeping
    frequency axis symmetric. Uniform ratio is applied across all
    dilated convolution layers for consistent RF distribution.

    Args:
        *args: Positional arguments for nn.Conv2d
        padding_ratio: (left_ratio, right_ratio) for time axis
            - (1.0, 0.0): Fully causal (left-only)
            - (0.75, 0.25): 75% past, 25% future
            - (0.5, 0.5): Symmetric
        **kwargs: Keyword arguments for nn.Conv2d

    Example:
        >>> # Causal convolution
        >>> conv = AsymmetricConv2d(64, 64, kernel_size=(3,3),
        ...                          padding=(1,1), padding_ratio=(1.0, 0.0))
        >>> # Asymmetric 3:1 ratio
        >>> conv = AsymmetricConv2d(64, 64, kernel_size=(3,3),
        ...                          padding=(1,1), padding_ratio=(0.75, 0.25))
    """
    def __init__(self, *args, padding_ratio=(1.0, 0.0), **kwargs):
        super().__init__()

        # Extract padding argument
        padding = kwargs.get('padding', (0, 0))
        if isinstance(padding, int):
            padding = (padding, padding)

        # Note: padding[0] from get_padding_2d is single-side padding
        # Total padding for both sides = padding[0] * 2 (same as CausalConv2d logic)
        time_pad_total = padding[0] * 2
        freq_pad = padding[1]

        # Split time padding according to ratio
        left_ratio, right_ratio = padding_ratio
        total_ratio = left_ratio + right_ratio

        if total_ratio == 0:
            time_pad_left = time_pad_right = 0
        else:
            # Round to nearest integer for balanced distribution
            time_pad_left = round(time_pad_total * left_ratio / total_ratio)
            time_pad_right = time_pad_total - time_pad_left

        # F.pad order: (left, right, top, bottom)
        # Frequency: symmetric, Time: asymmetric
        self.padding = (freq_pad, freq_pad, time_pad_left, time_pad_right)

        # Remove padding from conv (we'll manually pad)
        kwargs['padding'] = 0
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, self.padding))


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
        ctx.eps = eps
        B, C, T = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        eps = ctx.eps

        B, C, T = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None


class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super(LayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LSTM_Group_Feature_Network(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,
        hidden_size: int = 64,
        layer_num: int = 2,
    ) -> None:
        super().__init__()

        self.proj_conv1 = nn.Conv1d(in_channel, hidden_size, kernel_size=1)
        self.proj_conv2 = nn.Conv1d(hidden_size, in_channel, kernel_size=1)
        self.norm = LayerNorm1d(in_channel)
        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1)), requires_grad=True)

        # Always unidirectional for streaming compatibility
        self.lstm = nn.LSTM(hidden_size, hidden_size, layer_num, batch_first=True, bidirectional=False)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.norm(x)
        x = self.proj_conv1(x)
        x = x.permute(0, 2, 1)
        x = self.lstm(x)[0]
        x = x.permute(0, 2, 1)
        x = self.proj_conv2(x) * self.beta + skip

        return x


class Group_Prime_Kernel_FFN(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,
        kernel_list: List[int] = [3, 11, 23, 31],
    ) -> None:
        super().__init__()

        mid_channel = in_channel * len(kernel_list)

        self.expand_ratio = len(kernel_list)
        self.kernel_list = kernel_list
        self.norm = LayerNorm1d(in_channel)
        self.proj_conv1 = nn.Conv1d(in_channel, mid_channel, kernel_size=1)
        self.proj_conv2 = nn.Conv1d(mid_channel, in_channel, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1)), requires_grad=True)

        for kernel_size in kernel_list:
            setattr(self, f"attn_{kernel_size}",
                nn.Sequential(
                    nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, groups=in_channel, padding=get_padding(kernel_size)),
                    nn.Conv1d(in_channel, in_channel, kernel_size=1)))
            setattr(self, f"conv_{kernel_size}",
                nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, padding=get_padding(kernel_size), groups=in_channel))

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.norm(x)
        x = self.proj_conv1(x)

        x_chunks = list(torch.chunk(x, self.expand_ratio, dim=1))
        for i in range(self.expand_ratio):
            x_chunks[i] = getattr(self, f"attn_{self.kernel_list[i]}")(x_chunks[i]) * getattr(self, f"conv_{self.kernel_list[i]}")(x_chunks[i])

        x = torch.cat(x_chunks, dim=1)
        x = self.proj_conv2(x) * self.beta + skip
        return x


class Channel_Attention_Block(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,
        dw_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.norm = LayerNorm1d(in_channel)
        self.gate = nn.Sequential(
            nn.Conv1d(in_channel, in_channel*2, kernel_size=1, padding=0, groups=1),
            nn.Conv1d(in_channel*2, in_channel*2, kernel_size=dw_kernel_size, padding=get_padding(dw_kernel_size), groups=in_channel*2),
            SimpleGate(),
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channel, in_channel, kernel_size=1, padding=0, groups=1),
        )
        self.pwconv = nn.Conv1d(in_channel, in_channel, kernel_size=1, padding=0, groups=1)
        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.norm(x)
        x = self.gate(x)
        x = x * self.channel_attn(x)
        x = self.pwconv(x)
        x = skip + x * self.beta
        return x


class Two_Stage_Block(nn.Module):
    def __init__(
        self,
        dense_channel: int = 64,
        time_block_num: int = 2,
        lstm_layers: int = 2,
        lstm_hidden_size: int = 64,
        freq_block_num: int = 2,
        freq_block_kernel: List[int] = [3, 11, 23, 31],
    ) -> None:
        super().__init__()
        self.dense_channel = dense_channel

        time_stage = nn.ModuleList([])
        freq_stage = nn.ModuleList([])

        # Time stage: dw_kernel_size=1 hardcoded (RF=1)
        for i in range(time_block_num):
            time_stage.append(
                nn.Sequential(
                    Channel_Attention_Block(dense_channel, dw_kernel_size=1),
                    LSTM_Group_Feature_Network(dense_channel, lstm_hidden_size, lstm_layers),
                )
            )
        # Frequency stage: standard kernel size
        for i in range(freq_block_num):
            freq_stage.append(
                nn.Sequential(
                    Channel_Attention_Block(dense_channel, dw_kernel_size=3),
                    Group_Prime_Kernel_FFN(dense_channel, freq_block_kernel),
                )
            )
        self.time_stage = nn.Sequential(*time_stage)
        self.freq_stage = nn.Sequential(*freq_stage)

        self.beta_t = nn.Parameter(torch.zeros((1, dense_channel, 1)), requires_grad=True)
        self.beta_f = nn.Parameter(torch.zeros((1, dense_channel, 1)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:

        B, C, T, F = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B * F, C, T)

        x = self.time_stage(x) + x * self.beta_t
        x = x.view(B, F, C, T).permute(0, 3, 2, 1).contiguous().view(B * T, C, F)

        x = self.freq_stage(x) + x * self.beta_f
        x = x.view(B, T, C, F).permute(0, 2, 1, 3)
        return x


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features: int, beta: float = 1) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x: Tensor) -> Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class DS_DDB(nn.Module):
    """Dense Dilated Dense Block with asymmetric padding control.

    Args:
        dense_channel: Number of channels
        kernel_size: Convolution kernel size (time, freq)
        depth: Number of dilated layers
        padding_ratio: (left_ratio, right_ratio) for time axis padding
            - (1.0, 0.0): Fully causal
            - (0.75, 0.25): 75% past, 25% future
            - (0.5, 0.5): Symmetric
    """
    def __init__(
        self,
        dense_channel: int,
        kernel_size: Tuple[int, int] = (3, 3),
        depth: int = 4,
        padding_ratio: Tuple[float, float] = (1.0, 0.0),
    ) -> None:
        super().__init__()

        self.dense_block = nn.ModuleList([])

        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                AsymmetricConv2d(
                    dense_channel*(i+1), dense_channel*(i+1),
                    kernel_size=kernel_size,
                    dilation=(dil, 1),
                    padding=get_padding_2d(kernel_size, (dil, 1)),
                    groups=dense_channel*(i+1),
                    padding_ratio=padding_ratio
                ),
                nn.Conv2d(dense_channel*(i+1), dense_channel, kernel_size=1, padding=0),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        for block in self.dense_block:
            x = block(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    """Dense encoder with asymmetric padding control.

    Args:
        dense_channel: Number of channels
        in_channel: Input channels (2 for mag+pha)
        depth: Number of dilated layers in DS_DDB
        padding_ratio: (left_ratio, right_ratio) for encoder
    """
    def __init__(
        self,
        dense_channel: int,
        in_channel: int,
        depth: int = 4,
        padding_ratio: Tuple[float, float] = (1.0, 0.0),
    ) -> None:
        super().__init__()
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, kernel_size=1),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))
        self.dense_block = DS_DDB(dense_channel, depth=depth, padding_ratio=padding_ratio)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MaskDecoder(nn.Module):
    """Mask decoder with asymmetric padding control.

    Args:
        dense_channel: Number of channels
        n_fft: FFT size
        sigmoid_beta: Beta parameter for learnable sigmoid
        out_channel: Output channels (1 for mask)
        depth: Number of dilated layers in DS_DDB
        padding_ratio: (left_ratio, right_ratio) for decoder
    """
    def __init__(
        self,
        dense_channel: int,
        n_fft: int,
        sigmoid_beta: float,
        out_channel: int = 1,
        depth: int = 4,
        padding_ratio: Tuple[float, float] = (1.0, 0.0),
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.dense_block = DS_DDB(dense_channel, depth=depth, padding_ratio=padding_ratio)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, kernel_size=1),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=1)
        )
        self.lsigmoid = LearnableSigmoid_2d(n_fft//2+1, beta=sigmoid_beta)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    """Phase decoder with asymmetric padding control.

    Args:
        dense_channel: Number of channels
        out_channel: Output channels (1 for phase)
        depth: Number of dilated layers in DS_DDB
        padding_ratio: (left_ratio, right_ratio) for decoder
    """
    def __init__(
        self,
        dense_channel: int,
        out_channel: int = 1,
        depth: int = 4,
        padding_ratio: Tuple[float, float] = (1.0, 0.0),
    ) -> None:
        super().__init__()
        self.dense_block = DS_DDB(dense_channel, depth=depth, padding_ratio=padding_ratio)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, kernel_size=1)
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class PrimeKnet(nn.Module):
    """PrimeKnet-LSTM with separate encoder/decoder padding ratio control.

    Enables exploration of latency-performance trade-off through:
    - Option A: Both encoder/decoder use same ratio (high latency)
    - Option B: Encoder asymmetric, decoder causal (low latency)

    Args:
        fft_len: FFT size
        dense_channel: Number of channels
        sigmoid_beta: Beta for learnable sigmoid
        dense_depth: Depth of DS_DDB blocks
        num_tsblock: Number of two-stage blocks
        time_block_num: Number of time blocks per stage
        lstm_layers: Number of LSTM layers
        lstm_hidden_size: LSTM hidden size
        freq_block_kernel: Kernel sizes for frequency blocks
        freq_block_num: Number of frequency blocks per stage
        infer_type: 'masking' or 'mapping'
        encoder_padding_ratio: (left_ratio, right_ratio) for encoder
            e.g., (0.75, 0.25) for 75% past, 25% future
        decoder_padding_ratio: (left_ratio, right_ratio) for decoder
            e.g., (1.0, 0.0) for causal decoder

    Example:
        >>> # Option B: Encoder asymmetric + Decoder causal (40ms latency)
        >>> model = PrimeKnet(..., encoder_padding_ratio=(0.75, 0.25),
        ...                   decoder_padding_ratio=(1.0, 0.0))
        >>>
        >>> # Option A: Both asymmetric (80ms latency)
        >>> model = PrimeKnet(..., encoder_padding_ratio=(0.75, 0.25),
        ...                   decoder_padding_ratio=(0.75, 0.25))
    """
    def __init__(
        self,
        fft_len: int,
        dense_channel: int,
        sigmoid_beta: float,
        dense_depth: int = 4,
        num_tsblock: int = 4,
        time_block_num: int = 2,
        lstm_layers: int = 2,
        lstm_hidden_size: int = 64,
        freq_block_kernel: List[int] = [3, 11, 23, 31],
        freq_block_num: int = 2,
        infer_type: str = 'masking',
        encoder_padding_ratio: Tuple[float, float] = (1.0, 0.0),
        decoder_padding_ratio: Tuple[float, float] = (1.0, 0.0),
    ) -> None:
        super().__init__()
        assert infer_type in ['masking', 'mapping'], 'infer_type must be either masking or mapping'

        self.dense_encoder = DenseEncoder(
            dense_channel, in_channel=2, depth=dense_depth,
            padding_ratio=encoder_padding_ratio
        )
        self.sequence_block = nn.Sequential(
            *[Two_Stage_Block(
                dense_channel=dense_channel,
                time_block_num=time_block_num,
                lstm_layers=lstm_layers,
                lstm_hidden_size=lstm_hidden_size,
                freq_block_num=freq_block_num,
                freq_block_kernel=freq_block_kernel,
            ) for _ in range(num_tsblock)]
        )
        self.mask_decoder = MaskDecoder(
            dense_channel, fft_len, sigmoid_beta, out_channel=1,
            depth=dense_depth, padding_ratio=decoder_padding_ratio
        )
        self.phase_decoder = PhaseDecoder(
            dense_channel, out_channel=1, depth=dense_depth,
            padding_ratio=decoder_padding_ratio
        )

    def forward(self, noisy_com: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Input shape: [B, F, T, 2]

        real = noisy_com[:, :, :, 0]
        imag = noisy_com[:, :, :, 1]

        mag = torch.sqrt(real**2 + imag**2)
        pha = torch.atan2(imag, real)

        mag = mag.unsqueeze(1).permute(0, 1, 3, 2) # [B, 1, T, F]
        pha = pha.unsqueeze(1).permute(0, 1, 3, 2) # [B, 1, T, F]

        x = torch.cat((mag, pha), dim=1) # [B, 2, T, F]

        x = self.dense_encoder(x)

        x = self.sequence_block(x)

        denoised_mag = (mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        denoised_com = mag_pha_to_complex(denoised_mag, denoised_pha)

        return denoised_mag, denoised_pha, denoised_com
