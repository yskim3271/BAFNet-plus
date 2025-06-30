import torch
import torch.nn.functional as F
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
from torchvision.ops.deform_conv import DeformConv2d

from stft import mag_pha_stft, mag_pha_istft, pad_stft_input

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))

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


class LKFCA_Block(nn.Module):
    def __init__(self, in_channels, DW_Expand=2, FFN_Expand=1, drop_out_rate=0., type='sca', kernel_size=31):
        super().__init__()

        dw_channel = in_channels * DW_Expand

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        self.norm1 = LayerNorm1d(in_channels)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.GCGFN = GCGFN(in_channels)


    def forward(self, x):

        inp2 = x

        x = self.norm1(inp2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        inp3 = inp2 + x * self.beta
        x = self.GCGFN(inp3)

        return x


class GCGFN(nn.Module):
    def __init__(self, n_feats, fnn_expend=4):
        super().__init__()
        i_feats = fnn_expend * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm1d(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention (replaced with 1D convolutions)
        self.LKA9 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA7 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA5 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA3 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.X3 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4)
        self.X5 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4)
        self.X7 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4)
        self.X9 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, kernel_size=1))

        self.proj_last = nn.Sequential(
            nn.Conv1d(i_feats, n_feats, kernel_size=1))


    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a_1, a_2, a_3, a_4 = torch.chunk(x, 4, dim=1)
        x = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3),
                       self.LKA9(a_4) * self.X9(a_4)], dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, kernel_size=3, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv1d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv1d(hidden_features * 2, hidden_features * 2, kernel_size=kernel_size, stride=1, padding=get_padding(kernel_size),
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv1d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GPFCA(nn.Module):
    def __init__(self, in_channels, num_blocks=2):
        super().__init__()

        self.naf_blocks = nn.ModuleList([LKFCA_Block(in_channels) for _ in range(num_blocks)])
        self.Rearrange1 = Rearrange('b n c -> b c n')
        self.Rearrange2 = Rearrange('b c n -> b n c')

    def forward(self, x):
        x = self.Rearrange1(x)
        for block in self.naf_blocks:
            x = block(x)
        x = self.Rearrange2(x)
        return x


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DWConv2d_BN(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1, 1)
    ):
        super().__init__()

        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=3,
                                                      stride= 1,padding= 1,bias= False,groups=in_ch),
                                            nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                      kernel_size=1,
                                                      stride=1, padding=0, bias=False)
                                            )
        self.dcn=DeformConv2d(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    groups=in_ch
                    )
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        offset = self.offset_generator(x)

        if self.offset_clamp:
            offset = torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])
        x = self.dcn(x, offset)

        x = self.pwconv(x)
        x = self.act(x)
        return x


class DS_DDB(nn.Module):
    def __init__(self, dense_channel, kernel_size=(3, 3), depth=4):
        super(DS_DDB, self).__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(dense_channel*(i+1), dense_channel*(i+1), kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, dilation=(dil, 1)), groups=dense_channel*(i+1), bias=True),
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
    def __init__(self, dense_channel, in_channel):
        super(DenseEncoder, self).__init__()
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

        self.dense_block = DS_DDB(dense_channel, depth=4) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, 
                 dense_channel,
                 n_fft,
                 sigmoid_beta,
                 out_channel=1):
        super(MaskDecoder, self).__init__()
        self.n_fft = n_fft
        self.dense_block = DS_DDB(dense_channel, depth=4)
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
                 out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DS_DDB(dense_channel, depth=4)
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

class TS_BLOCK(nn.Module):
    def __init__(self, dense_channel):
        super(TS_BLOCK, self).__init__()
        self.dense_channel = dense_channel
        self.time = GPFCA(dense_channel)
        self.freq = GPFCA(dense_channel)
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, dense_channel)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, 1, dense_channel)), requires_grad=True)
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        
        x = self.time(x) + x * self.beta
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)

        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x



class PrimeKnet_masking(nn.Module):
    def __init__(self, 
                 win_len, 
                 hop_len, 
                 fft_len, 
                 dense_channel, 
                 sigmoid_beta, 
                 compress_factor,
                 num_tsblock=4
                 ):
        super(PrimeKnet_masking, self).__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.dense_channel = dense_channel
        self.sigmoid_beta = sigmoid_beta
        self.compress_factor = compress_factor
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)
        self.LKFCAnet = nn.ModuleList([])
        for i in range(num_tsblock):
            self.LKFCAnet.append(TS_BLOCK(dense_channel))
        self.mask_decoder = MaskDecoder(dense_channel, fft_len, sigmoid_beta, out_channel=1)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1)

    def forward(self, inputs):

        in_len = inputs.size(-1)
        padded_inputs = pad_stft_input(inputs, 
                                       n_fft=self.fft_len,
                                       hop_size=self.hop_len
                                       ).squeeze(1)

        mag, pha, com = mag_pha_stft(padded_inputs, 
                                     n_fft=self.fft_len,
                                     hop_size=self.hop_len,
                                     win_size=self.win_len,
                                     compress_factor=self.compress_factor,
                                     center=False
                                     )

        mag = mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        pha = pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        x = torch.cat((mag, pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tsblock):
            x = self.LKFCAnet[i](x)
        
        denoised_mag = (mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        
        output_wav = mag_pha_istft(denoised_mag, denoised_pha,
                                   n_fft=self.fft_len,
                                   hop_size=self.hop_len,
                                   win_size=self.win_len,
                                   compress_factor=self.compress_factor,
                                   center=True
                                   )
        
        output_wav = output_wav.unsqueeze(1)
        output_wav = output_wav[..., :in_len]

        return output_wav
    
class PrimeKnet_mapping(nn.Module):
    def __init__(self, 
                 win_len, 
                 hop_len, 
                 fft_len, 
                 dense_channel, 
                 sigmoid_beta, 
                 compress_factor,
                 num_tsblock=4
                 ):
        super(PrimeKnet_mapping, self).__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.dense_channel = dense_channel
        self.sigmoid_beta = sigmoid_beta
        self.compress_factor = compress_factor
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)
        self.LKFCAnet = nn.ModuleList([])
        for i in range(num_tsblock):
            self.LKFCAnet.append(TS_BLOCK(dense_channel))
        self.mask_decoder = MaskDecoder(dense_channel, fft_len, sigmoid_beta, out_channel=1)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1)

    def forward(self, inputs):

        in_len = inputs.size(-1)
        padded_inputs = pad_stft_input(inputs, 
                                       n_fft=self.fft_len,
                                       hop_size=self.hop_len
                                       ).squeeze(1)

        mag, pha, com = mag_pha_stft(padded_inputs, 
                                     n_fft=self.fft_len,
                                     hop_size=self.hop_len,
                                     win_size=self.win_len,
                                     compress_factor=self.compress_factor,
                                     center=False
                                     )

        mag = mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        pha = pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        x = torch.cat((mag, pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tsblock):
            x = self.LKFCAnet[i](x)
        
        mag = self.mask_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        
        output_wav = mag_pha_istft(mag, pha,
                                   n_fft=self.fft_len,
                                   hop_size=self.hop_len,
                                   win_size=self.win_len,
                                   compress_factor=self.compress_factor,
                                   center=True
                                   )
        
        output_wav = output_wav.unsqueeze(1)
        output_wav = output_wav[..., :in_len]

        return output_wav



