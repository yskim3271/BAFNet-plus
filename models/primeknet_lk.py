"""
PrimeKnet with Simple Lookahead Buffer
This model implements a lookahead buffer by redistributing the receptive field
between past and future frames based on a lookahead ratio parameter.

Lookahead Ratio (L):
- L=0.0: Pure causal (all past frames)
- L=0.3: 70% past, 30% future frames
- L=0.5: 50% past, 50% future frames (maximum lookahead)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stft import mag_pha_to_complex

# ==================================================================================
# Lookahead Causal Model: Receptive Field with Future Context
#
# This model redistributes the receptive field between past and future frames
# based on a lookahead_ratio parameter L (0 <= L <= 0.5).
#
# Example with Total RF = 317 frames:
# - L=0.0: 317 past frames, 0 future frames (pure causal)
# - L=0.3: 222 past frames, 95 future frames
# - L=0.5: 158 past frames, 159 future frames
#
# Trade-off:
# - Higher L: Better performance but higher latency
# - Lower L: Lower latency but potentially reduced performance
#
# Output length is reduced by the number of lookahead frames since
# the last frames don't have sufficient future context.
# ==================================================================================


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2),
            int((kernel_size[1]*dilation[1] - dilation[1])/2))


class LookaheadConv2d(nn.Module):
    """2D Convolution with lookahead buffer support.

    Only applies lookahead to time dimension, frequency dimension remains unchanged.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1,
                 dilation=(1, 1), groups=1, bias=True, lookahead_ratio=0.0):
        super().__init__()
        time_total_padding = padding[0] * 2
        freq_padding = padding[1]

        # Calculate time dimension padding distribution
        time_right_padding = int(time_total_padding * lookahead_ratio)
        time_left_padding = time_total_padding - time_right_padding

        # Padding format: (left, right, top, bottom) for F.pad
        self.padding = (freq_padding, freq_padding, time_left_padding, time_right_padding)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=0, stride=stride, dilation=dilation,
                              groups=groups, bias=bias)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        B, C, T = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        B, C, T = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(dim=0), None


class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Channel_Attention_Block(nn.Module):
    """Channel attention block with gating and squeeze-excitation.

    Follows primeknet_lstm structure. Uses standard nn.Conv1d since:
    - Time RF is fixed at 1 (dw_kernel_size=1 for time stage)
    - Frequency axis has no causal meaning
    No lookahead in this block - lookahead is only applied at DS_DDB level.
    """
    def __init__(self, in_channel=64, dw_kernel_size=3):
        super().__init__()

        self.norm = LayerNorm1d(in_channel)
        self.gate = nn.Sequential(
            nn.Conv1d(in_channel, in_channel*2, kernel_size=1, padding=0, groups=1),
            nn.Conv1d(in_channel*2, in_channel*2, kernel_size=dw_kernel_size,
                   padding=get_padding(dw_kernel_size), groups=in_channel*2),
            SimpleGate(),
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channel, in_channel, kernel_size=1, padding=0, groups=1),
        )
        self.pwconv = nn.Conv1d(in_channel, in_channel, kernel_size=1, padding=0, groups=1)
        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1)), requires_grad=True)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.gate(x)
        x = x * self.channel_attn(x)
        x = self.pwconv(x)
        x = skip + x * self.beta
        return x


class Group_Prime_Kernel_FFN(nn.Module):
    """Group convolution with prime kernels.

    Follows primeknet_lstm structure. Uses standard nn.Conv1d since:
    - Time RF is fixed at 1 (kernel_list=[1] for time stage)
    - Frequency axis has no causal meaning
    No lookahead in this block - lookahead is only applied at DS_DDB level.
    """
    def __init__(self, in_channel=64, kernel_list=[3, 11, 23, 31]):
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
                    nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size,
                           groups=in_channel, padding=get_padding(kernel_size)),
                    nn.Conv1d(in_channel, in_channel, kernel_size=1)))
            setattr(self, f"conv_{kernel_size}",
                nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size,
                       padding=get_padding(kernel_size), groups=in_channel))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.proj_conv1(x)

        x_chunks = list(torch.chunk(x, self.expand_ratio, dim=1))
        for i in range(self.expand_ratio):
            x_chunks[i] = getattr(self, f"attn_{self.kernel_list[i]}")(x_chunks[i]) * \
                         getattr(self, f"conv_{self.kernel_list[i]}")(x_chunks[i])

        x = torch.cat(x_chunks, dim=1)
        x = self.proj_conv2(x) * self.beta + skip
        return x


class Two_Stage_Block(nn.Module):
    """Two-stage processing block following primeknet_lstm structure.

    Time stage: Channel_Attention_Block + Group_Prime_Kernel_FFN (minimal kernels)
    Freq stage: Channel_Attention_Block + Group_Prime_Kernel_FFN (standard kernels)

    Note: Lookahead is NOT applied in this block. Lookahead is only applied
    at the DS_DDB (encoder/decoder) level where the actual RF expansion happens.
    Time RF is fixed at 1, and frequency axis has no causal meaning.
    """
    def __init__(self,
                 dense_channel=64,
                 time_block_num=2,
                 freq_block_num=2,
                 freq_block_kernel=[3, 11, 23, 31]):
        super().__init__()
        self.dense_channel = dense_channel

        time_stage = nn.ModuleList([])
        freq_stage = nn.ModuleList([])

        # Time stage: minimal kernel (dw_kernel_size=1 as hardcoded)
        for _ in range(time_block_num):
            time_stage.append(
                nn.Sequential(
                    Channel_Attention_Block(dense_channel, dw_kernel_size=1),
                    Group_Prime_Kernel_FFN(dense_channel, kernel_list=[1]),
                )
            )

        # Frequency stage: standard kernel (dw_kernel_size=3)
        for _ in range(freq_block_num):
            freq_stage.append(
                nn.Sequential(
                    Channel_Attention_Block(dense_channel, dw_kernel_size=3),
                    Group_Prime_Kernel_FFN(dense_channel, kernel_list=freq_block_kernel),
                )
            )

        self.time_stage = nn.Sequential(*time_stage)
        self.freq_stage = nn.Sequential(*freq_stage)

        self.beta_t = nn.Parameter(torch.zeros((1, dense_channel, 1)), requires_grad=True)
        self.beta_f = nn.Parameter(torch.zeros((1, dense_channel, 1)), requires_grad=True)

    def forward(self, x):
        B, C, T, F = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B * F, C, T)

        x = self.time_stage(x) + x * self.beta_t
        x = x.view(B, F, C, T).permute(0, 3, 2, 1).contiguous().view(B * T, C, F)

        x = self.freq_stage(x) + x * self.beta_f
        x = x.view(B, T, C, F).permute(0, 2, 1, 3)
        return x


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DS_DDB(nn.Module):
    def __init__(self, dense_channel, kernel_size=(3, 3), depth=4, lookahead_ratio=0.0):
        super().__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.lookahead_ratio = lookahead_ratio
        self.dense_block = nn.ModuleList([])

        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                LookaheadConv2d(dense_channel*(i+1), dense_channel*(i+1), kernel_size,
                               dilation=(dil, 1),
                               padding=get_padding_2d(kernel_size, (dil, 1)),
                               groups=dense_channel*(i+1), bias=True,
                               lookahead_ratio=lookahead_ratio),
                nn.Conv2d(in_channels=dense_channel*(i+1), out_channels=dense_channel,
                         kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, dense_channel, in_channel, depth=4, lookahead_ratio=0.0):
        super().__init__()
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))
        self.dense_block = DS_DDB(dense_channel, depth=depth, lookahead_ratio=lookahead_ratio)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self,
                 dense_channel,
                 n_fft,
                 sigmoid_beta,
                 out_channel=1,
                 depth=4,
                 lookahead_ratio=0.0):
        super().__init__()
        self.n_fft = n_fft
        self.dense_block = DS_DDB(dense_channel, depth=depth, lookahead_ratio=lookahead_ratio)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(n_fft//2+1, beta=sigmoid_beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self,
                 dense_channel,
                 out_channel=1,
                 depth=4,
                 lookahead_ratio=0.0):
        super().__init__()
        self.dense_block = DS_DDB(dense_channel, depth=depth, lookahead_ratio=lookahead_ratio)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class PrimeKnet(nn.Module):
    def __init__(self,
                 win_len,
                 hop_len,
                 fft_len,
                 dense_channel,
                 sigmoid_beta,
                 compress_factor,
                 dense_depth=3,
                 num_tsblock=4,
                 freq_block_kernel=[3, 11, 23, 31],
                 time_block_num=2,
                 freq_block_num=2,
                 infer_type='masking',
                 lookahead_ratio=0.0):
        """
        PrimeKnet with Lookahead Buffer

        Fixed parameters (hardcoded for all experiments):
            time_dw_kernel_size: 1 (minimal time kernel)
            time_block_kernel: [1] (no GCGFN expansion in time)

        Args:
            lookahead_ratio: Fraction of receptive field used for future frames (0.0 to 0.5)
                            - 0.0: Pure causal (no lookahead)
                            - 0.3: 30% lookahead, 70% past context
                            - 0.5: 50% lookahead, 50% past context (maximum)
        """
        super().__init__()
        assert 0.0 <= lookahead_ratio <= 0.5, "lookahead_ratio must be between 0.0 and 0.5"
        assert infer_type in ['masking', 'mapping'], 'infer_type must be either masking or mapping'

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.dense_channel = dense_channel
        self.dense_depth = dense_depth
        self.sigmoid_beta = sigmoid_beta
        self.compress_factor = compress_factor
        self.num_tsblock = num_tsblock
        self.lookahead_ratio = lookahead_ratio
        self.infer_type = infer_type

        # Fixed parameters for all experiments
        self.time_dw_kernel_size = 1
        self.time_block_kernel = [1]

        # Calculate receptive field for logging purposes
        self._calculate_receptive_field()

        # Initialize components with lookahead
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2, depth=dense_depth,
                                         lookahead_ratio=lookahead_ratio)

        # Use Two_Stage_Block following primeknet_lstm structure
        # Note: Two_Stage_Block doesn't use lookahead or causal convs (RF=1 in time)
        # Lookahead is only applied at DS_DDB level in encoder/decoder
        self.sequence_block = nn.Sequential(
            *[Two_Stage_Block(
                dense_channel=dense_channel,
                time_block_num=time_block_num,
                freq_block_num=freq_block_num,
                freq_block_kernel=freq_block_kernel,
            ) for _ in range(num_tsblock)]
        )

        self.mask_decoder = MaskDecoder(dense_channel, fft_len, sigmoid_beta, out_channel=1,
                                       depth=dense_depth, lookahead_ratio=lookahead_ratio)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1, depth=dense_depth,
                                         lookahead_ratio=lookahead_ratio)

    def _calculate_receptive_field(self):
        """Calculate and log receptive field information

        Receptive field calculation with fixed parameters:
        - dense_depth=3: DS_DDB with 3 layers, dilation=[1,2,4]
        - time_dw_kernel_size=1, time_block_kernel=[1]

        RF calculation:
        (1) Encoder/Decoder DS_DDB (time axis):
            RF_ds_ddb = 1 + (k_ds - 1) * sum(dilations)
                      = 1 + (3 - 1) * (1 + 2 + 4)
                      = 1 + 2 * 7 = 15 frames

        (2) LKFCAnet (time path):
            RF_block = k_dw + K_g - 1 = 1 + 1 - 1 = 1
            RF_lkfcanet_time = 1 + (N_ts * N_tb) * (RF_block - 1)
                             = 1 + (4 * 2) * (1 - 1) = 1 frame

        (3) Total RF:
            RF_total = RF_ds_ddb + (RF_lkfcanet_time - 1) + (RF_ds_ddb - 1)
                     = 15 + 0 + 14 = 29 frames
        """
        # Encoder/Decoder DS_DDB: depth=3, dilations=[1,2,4]
        rf_encoder = 1 + (3 - 1) * (1 + 2 + 4)  # 15 frames
        rf_decoder = rf_encoder  # Same as encoder

        # LKFCAnet: minimal time kernels (k_dw=1, K_g=1)
        rf_block = 1  # k_dw + K_g - 1 = 1 + 1 - 1
        rf_lkfca = 1 + (self.num_tsblock * 2) * (rf_block - 1)  # time_block_num=2

        self.total_rf_frames = rf_encoder + (rf_lkfca - 1) + (rf_decoder - 1)
        self.lookahead_frames = int(self.total_rf_frames * self.lookahead_ratio)
        self.past_frames = self.total_rf_frames - self.lookahead_frames

        # Calculate latency in ms (assuming 16kHz sampling rate)
        self.lookahead_ms = (self.lookahead_frames * self.hop_len) / 16.0

        print(f"[PrimeKnet Lookahead] Initialized with:")
        print(f"  - Lookahead ratio: {self.lookahead_ratio:.1%}")
        print(f"  - Total RF: {self.total_rf_frames} frames")
        print(f"  - Past frames: {self.past_frames}")
        print(f"  - Future frames: {self.lookahead_frames}")
        print(f"  - Algorithmic latency: {self.lookahead_ms:.1f} ms")

    def forward(self, noisy_com):
        # Input shape: [B, F, T, 2]

        real = noisy_com[:, :, :, 0]
        imag = noisy_com[:, :, :, 1]

        mag = torch.sqrt(real**2 + imag**2)
        pha = torch.atan2(imag, real)

        mag = mag.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, T, F]
        pha = pha.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, T, F]

        x = torch.cat((mag, pha), dim=1)  # [B, 2, T, F]

        # Process through network
        x = self.dense_encoder(x)
        x = self.sequence_block(x)

        # Decode magnitude and phase
        denoised_mag = (mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)

        # Note: Output length is slightly reduced due to lookahead
        # In streaming mode, this would be handled by buffering

        denoised_com = mag_pha_to_complex(denoised_mag, denoised_pha)

        return denoised_mag, denoised_pha, denoised_com

    def get_lookahead_info(self):
        """Return lookahead configuration information"""
        return {
            'lookahead_ratio': self.lookahead_ratio,
            'lookahead_frames': self.lookahead_frames,
            'lookahead_ms': self.lookahead_ms,
            'past_frames': self.past_frames,
            'total_rf_frames': self.total_rf_frames
        }