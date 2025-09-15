import torch
import torch.nn as nn
import torch.nn.functional as F
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


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.padding = padding * 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0,
                              stride=stride,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

    def forward(self, x):
        x = F.pad(x, [self.padding, 0])
        return self.conv(x)

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dilation=(1, 1), groups=1, bias=True):
        super(CausalConv2d, self).__init__()
        time_padding = padding[0] * 2
        freq_padding = padding[1]
        self.padding = (freq_padding, freq_padding, time_padding, 0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=0,
                              stride=stride,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

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
        return gx, (grad_output * y).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None

class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class GCGFN(nn.Module):
    def __init__(self, in_channel=64, kernel_list=[3, 11, 23, 31], causal=False):
        super().__init__()
        self.in_channel = in_channel
        self.expand_ratio = len(kernel_list)
        self.mid_channel = self.in_channel * self.expand_ratio
        self.kernel_list = kernel_list
        self.causal = causal

        if causal:
            conv_fn = CausalConv1d
        else:
            conv_fn = nn.Conv1d

        self.proj_first = nn.Sequential(
            nn.Conv1d(self.in_channel, self.mid_channel, kernel_size=1))
        self.proj_last = nn.Sequential(
            nn.Conv1d(self.mid_channel, self.in_channel, kernel_size=1))
        self.norm = LayerNorm1d(self.in_channel)
        self.scale = nn.Parameter(torch.zeros((1, self.in_channel, 1)), requires_grad=True)

        for kernel_size in self.kernel_list:
            setattr(self, f"attn_{kernel_size}", nn.Sequential(
                conv_fn(self.in_channel, self.in_channel, kernel_size=kernel_size, groups=self.in_channel, padding=get_padding(kernel_size)),
                nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1)
            ))
            setattr(self, f"conv_{kernel_size}", conv_fn(self.in_channel, self.in_channel, kernel_size=kernel_size, padding=get_padding(kernel_size), groups=self.in_channel))
    
    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        
        x_chunks = list(torch.chunk(x, self.expand_ratio, dim=1))
        for i in range(self.expand_ratio):
            x_chunks[i] = getattr(self, f"attn_{self.kernel_list[i]}")(x_chunks[i]) * getattr(self, f"conv_{self.kernel_list[i]}")(x_chunks[i])

        x = torch.cat(x_chunks, dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x

class LKFCA_Block(nn.Module):
    def __init__(self, in_channels=64, kernel_list=[3, 11, 23, 31], causal=False):
        super().__init__()

        dw_channel = in_channels * 2

        if causal:
            conv_fn = CausalConv1d
        else:
            conv_fn = nn.Conv1d

        self.pwconv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.dwconv = conv_fn(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)
        self.pwconv2 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.sg = SimpleGate()
        self.norm1 = LayerNorm1d(in_channels)
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.GCGFN = GCGFN(in_channels, kernel_list, causal=causal)


    def forward(self, x):

        inp2 = x
        x = self.norm1(inp2)
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.pwconv2(x)

        inp3 = inp2 + x * self.beta
        x = self.GCGFN(inp3)

        return x

class TS_BLOCK(nn.Module):
    def __init__(self, 
                 dense_channel=64, 
                 time_block_num=2, 
                 freq_block_num=2, 
                 time_block_kernel=[3, 11, 23, 31], 
                 freq_block_kernel=[3, 11, 23, 31],
                 causal=False
                 ):
        super().__init__()
        self.dense_channel = dense_channel
        self.causal = causal
        self.time = nn.Sequential(
            *[LKFCA_Block(dense_channel, time_block_kernel, causal=causal) for _ in range(time_block_num)],
        )
        self.freq = nn.Sequential(
            *[LKFCA_Block(dense_channel, freq_block_kernel, causal=causal) for _ in range(freq_block_num)],
        )
        self.beta = nn.Parameter(torch.zeros((1, 1, dense_channel, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, dense_channel, 1)), requires_grad=True)
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b*f, c, t) 

        x = self.time(x) + x * self.beta
        x = x.view(b, f, c, t).permute(0, 3, 2, 1).contiguous().view(b*t, c, f)

        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, c, f).permute(0, 2, 1, 3)
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
    def __init__(self, dense_channel, kernel_size=(3, 3), depth=4, causal=False):
        super().__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.causal = causal
        self.dense_block = nn.ModuleList([])
        if causal:
            conv_fn = CausalConv2d
        else:
            conv_fn = nn.Conv2d
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                conv_fn(dense_channel*(i+1), dense_channel*(i+1), kernel_size, dilation=(dil, 1), 
                        padding=get_padding_2d(kernel_size, (dil, 1)), groups=dense_channel*(i+1), bias=True),
                nn.Conv2d(in_channels=dense_channel*(i+1), out_channels=dense_channel, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
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
    def __init__(self, dense_channel, in_channel, depth=4, causal=False):
        super().__init__()
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))
        self.dense_block = DS_DDB(dense_channel, depth=depth, causal=causal)
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
                 causal=False):
        super().__init__()
        self.n_fft = n_fft
        self.dense_block = DS_DDB(dense_channel, depth=depth, causal=causal)
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
                 causal=False):
        super().__init__()
        self.dense_block = DS_DDB(dense_channel, depth=depth, causal=causal)
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
                 dense_depth=4,
                 num_tsblock=4,
                 time_block_kernel=[3, 11, 23, 31],
                 freq_block_kernel=[3, 11, 23, 31],
                 time_block_num=2,
                 freq_block_num=2,
                 infer_type='masking',
                 causal=False
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
        assert infer_type in ['masking', 'mapping'], 'infer_type must be either masking or mapping'

        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2, depth=dense_depth, causal=causal)
        self.LKFCAnet = nn.ModuleList([])
        for i in range(num_tsblock):
            self.LKFCAnet.append(TS_BLOCK(dense_channel, time_block_num, freq_block_num, time_block_kernel, freq_block_kernel, causal=causal))
        self.mask_decoder = MaskDecoder(dense_channel, fft_len, sigmoid_beta, out_channel=1, depth=dense_depth, causal=causal)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1, depth=dense_depth, causal=causal)

    def forward(self, noisy_com):
        # Input shape: [B, F, T, 2]

        real = noisy_com[:, :, :, 0]
        imag = noisy_com[:, :, :, 1]

        mag = torch.sqrt(real**2 + imag**2)
        pha = torch.atan2(imag, real)

        mag = mag.unsqueeze(1).permute(0, 1, 3, 2) # [B, 1, T, F]
        pha = pha.unsqueeze(1).permute(0, 1, 3, 2) # [B, 1, T, F]

        x = torch.cat((mag, pha), dim=1) # [B, 2, T, F]

        x = self.dense_encoder(x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

        for i in range(self.num_tsblock):
            x = self.LKFCAnet[i](x)
        
        denoised_mag = (mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        
        denoised_com = mag_pha_to_complex(denoised_mag, denoised_pha)

        return denoised_mag, denoised_pha, denoised_com