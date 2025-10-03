"""PrimeKnet model and building blocks (PyTorch).

This module implements the PrimeKnet architecture components for speech/noise
spectrogram processing. It contains causal/non-causal convolution wrappers,
normalization utilities, feature blocks, and the end-to-end model.

"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        'PrimeKnet Mamba variant requires the mamba-ssm package. '
        'Install it with `pip install mamba-ssm` in the active environment.'
    ) from exc

from stft import mag_pha_to_complex

# ==================================================================================
# Causal Real-Time Model: Receptive Field and Streaming Parameters Calculation
#
# Acknowledgment: The initial analysis has been refined based on the insight that
# the `TS_BLOCK_Stream` processes time and frequency axes independently.
# The following calculation for the time-axis receptive field considers *only*
# the modules that operate along the time dimension.
#
# ----------------------------------------------------------------------------------
# 1. Receptive Field Calculation (in Spectrogram Frames)
#
# The total receptive field along the time axis is a sum of contributions from
# the Encoder, the time-processing blocks within LKFCAnet, and the Decoder.
#
# (1) RF of DS_DDB_Stream (in Encoder & Decoder):
#     - This module contains 4 layers of `CausalConv2d` with a time-axis kernel of 3
#       and dilations of [1, 2, 4, 8].
#     - Formula for a stack of causal dilated convolutions: 1 + sum((k-1)*d)
#     - RF_ds_ddb = 1 + (3-1)*1 + (3-1)*2 + (3-1)*4 + (3-1)*8 = 31 frames.
#
# (2) RF of LKFCAnet (Time-processing part ONLY):
#     - LKFCAnet consists of `num_tsblock` (default=4) `TS_BLOCK_Stream` modules.
#     - Each `TS_BLOCK_Stream` contains a `time` block and a `freq` block. For the
#       time-axis receptive field, we ONLY consider the `time` block.
#     - The `time` block contains 2 sequential `LKFCA_Block_Stream` modules.
#     - Thus, `num_tsblock * 2` (i.e., 8) `LKFCA_Block_Stream` modules are applied
#       sequentially along the time axis.
#
#     - RF of a single LKFCA_Block_Stream:
#         - This block applies a `dwconv` (kernel=3) and a `GCGFNStream` sequentially.
#         - `GCGFNStream` uses parallel convolutions, so its RF is determined by the
#           largest kernel in `kernel_list`.
#         - RF_lkfca_block = RF_dwconv + RF_gcgfn - 1
#                         = 3 + max(kernel_list) - 1 = max(kernel_list) + 2 frames.
#         - With default kernel_list=[3, 11, 23, 31], RF_lkfca_block = 31 + 2 = 33 frames.
#
#     - Total RF for the time-part of LKFCAnet (stack of 8 blocks):
#         - RF_lkfcanet_time = 1 + (num_tsblock * 2) * (RF_lkfca_block - 1)
#         - RF_lkfcanet_time = 1 + 8 * (33 - 1) = 257 frames.
#
# (3) Total Model Receptive Field (Time-axis):
#     - The total RF is the sum of receptive fields of the sequential components.
#     - Formula: RF_total = RF_encoder + (RF_lkfcanet_time - 1) + (RF_decoder - 1)
#     - RF_total = RF_ds_ddb + (RF_lkfcanet_time - 1) + (RF_ds_ddb - 1)
#     - RF_total = 31 + (257 - 1) + (31 - 1) = 31 + 256 + 30 = 317 frames.
#
# ----------------------------------------------------------------------------------
# 2. Receptive Field Calculation (in Audio Samples)
#
#     - Formula: RF_samples = (RF_frames - 1) * hop_len + win_len
#     - Using default parameters (win_len=400, hop_len=100):
#     - RF_samples = (317 - 1) * 100 + 400 = 31600 + 400 = 32,000 samples.
#
# At a 16kHz sampling rate, this is a 2-second receptive field (32000 / 16000).
#
# ----------------------------------------------------------------------------------
# 3. Streaming Implementation Guidelines (Unchanged)
#
# - Window Size (Chunk Size): Min input size is 32,000 samples.
# - Hop Size (Processing Step): e.g., 100 samples for minimum latency.
# - Latency: The algorithmic latency is at least 2 seconds.
# ==================================================================================

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(
    kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)
) -> Tuple[int, int]:
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )

class CausalConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.padding = kwargs.get('padding', 0) * 2
        kwargs['padding'] = 0
        self.conv = nn.Conv1d(*args, **kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, [self.padding, 0]))

class CausalConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        padding = kwargs.get('padding', 0)
        self.padding = (padding[1], padding[1], padding[0]*2, 0)
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


class Mamba_Group_Feature_Network(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,
        hidden_size: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        causal: bool = False,
    ) -> None:
        super().__init__()

        self.proj_conv1 = nn.Conv1d(in_channel, hidden_size, kernel_size=1)
        self.proj_conv2 = nn.Conv1d(hidden_size, in_channel, kernel_size=1)
        self.norm = LayerNorm1d(in_channel)
        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1)), requires_grad=True)

        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False
        )

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.norm(x)
        x = self.proj_conv1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.mamba(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.proj_conv2(x) * self.beta + skip

        return x


class Group_Prime_Kernel_FFN(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,
        kernel_list: List[int] = [3, 11, 23, 31],
        causal: bool = False,
    ) -> None:
        super().__init__()
        
        mid_channel = in_channel * len(kernel_list)
        conv_fn = CausalConv1d if causal else nn.Conv1d

        self.expand_ratio = len(kernel_list)
        self.kernel_list = kernel_list
        self.norm = LayerNorm1d(in_channel)
        self.proj_conv1 = nn.Conv1d(in_channel, mid_channel, kernel_size=1)
        self.proj_conv2 = nn.Conv1d(mid_channel, in_channel, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1)), requires_grad=True)

        for kernel_size in kernel_list:
            setattr(self, f"attn_{kernel_size}", 
                nn.Sequential(
                    conv_fn(in_channel, in_channel, kernel_size=kernel_size, groups=in_channel, padding=get_padding(kernel_size)),
                    nn.Conv1d(in_channel, in_channel, kernel_size=1)))
            setattr(self, f"conv_{kernel_size}", 
                conv_fn(in_channel, in_channel, kernel_size=kernel_size, padding=get_padding(kernel_size), groups=in_channel))
    
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
        causal: bool = False,
    ) -> None:
        super().__init__()

        conv_fn = CausalConv1d if causal else nn.Conv1d

        self.norm = LayerNorm1d(in_channel)
        self.gate = nn.Sequential(
            nn.Conv1d(in_channel, in_channel*2, kernel_size=1, padding=0, groups=1),
            conv_fn(in_channel*2, in_channel*2, kernel_size=dw_kernel_size, padding=get_padding(dw_kernel_size), groups=in_channel*2),
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
        mamba_hidden_size: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        freq_block_num: int = 2,
        freq_block_kernel: List[int] = [3, 11, 23, 31],
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.dense_channel = dense_channel
        self.causal = causal

        time_stage = nn.ModuleList([])
        freq_stage = nn.ModuleList([])


        for _ in range(time_block_num):
            time_stage.append(
                nn.Sequential(
                    Channel_Attention_Block(dense_channel, dw_kernel_size=1, causal=causal),
                    Mamba_Group_Feature_Network(
                        dense_channel,
                        hidden_size=mamba_hidden_size,
                        d_state=mamba_d_state,
                        d_conv=mamba_d_conv,
                        expand=mamba_expand,
                        causal=causal,
                    ),
                )
            )
        for i in range(freq_block_num):
            freq_stage.append(
                nn.Sequential(
                    Channel_Attention_Block(dense_channel, dw_kernel_size=3, causal=causal),
                    Group_Prime_Kernel_FFN(dense_channel, freq_block_kernel, causal=causal),
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
    def __init__(
        self,
        dense_channel: int,
        kernel_size: Tuple[int, int] = (3, 3),
        depth: int = 4,
        causal: bool = False,
    ) -> None:
        super().__init__()
        
        conv_fn = CausalConv2d if causal else nn.Conv2d

        self.dense_block = nn.ModuleList([])

        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                conv_fn(dense_channel*(i+1), dense_channel*(i+1), kernel_size=kernel_size, dilation=(dil, 1), 
                        padding=get_padding_2d(kernel_size, (dil, 1)), groups=dense_channel*(i+1)),
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
    def __init__(
        self,
        dense_channel: int,
        in_channel: int,
        depth: int = 4,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, kernel_size=1),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))
        self.dense_block = DS_DDB(dense_channel, depth=depth, causal=causal)
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
    def __init__(
        self,
        dense_channel: int,
        n_fft: int,
        sigmoid_beta: float,
        out_channel: int = 1,
        depth: int = 4,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.dense_block = DS_DDB(dense_channel, depth=depth, causal=causal)
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
    def __init__(
        self,
        dense_channel: int,
        out_channel: int = 1,
        depth: int = 4,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.dense_block = DS_DDB(dense_channel, depth=depth, causal=causal)
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
    def __init__(
        self,
        fft_len: int,
        dense_channel: int,
        sigmoid_beta: float,
        dense_depth: int = 4,
        num_tsblock: int = 4,
        time_block_num: int = 2,
        mamba_hidden_size: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        freq_block_kernel: List[int] = [3, 11, 23, 31],
        freq_block_num: int = 2,
        infer_type: str = 'masking',
        causal: bool = False,
    ) -> None:
        super().__init__()
        assert infer_type in ['masking', 'mapping'], 'infer_type must be either masking or mapping'

        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2, depth=dense_depth, causal=causal)
        self.sequence_block = nn.Sequential(
            *[Two_Stage_Block(
                dense_channel=dense_channel,
                time_block_num=time_block_num,
                mamba_hidden_size=mamba_hidden_size,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                freq_block_num=freq_block_num,
                freq_block_kernel=freq_block_kernel,
                causal=causal
            ) for _ in range(num_tsblock)]
        )
        self.mask_decoder = MaskDecoder(dense_channel, fft_len, sigmoid_beta, out_channel=1, depth=dense_depth, causal=causal)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1, depth=dense_depth, causal=causal)

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