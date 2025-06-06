"""
dccrn: Deep complex convolution recurrent network
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models.conformer import _FeedForwardModule, _ConvolutionModule
from models.stft import mag_pha_stft, mag_pha_istft, pad_stft_input

class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

class DenseEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 depth=4,
                 channels=64
                 ):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=depth, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=out_channel, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x

class Conmer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual
        x = self._apply_convolution(x)
        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        
        x = self.final_layer_norm(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conmer = Conmer(
            input_dim=num_channel,
            ffn_dim=num_channel // 4,
            depthwise_conv_kernel_size=15,
            dropout=0.2,
            use_group_norm=True,
        )
        self.freq_conmer = Conmer(
            input_dim=num_channel,
            ffn_dim=num_channel // 4,
            depthwise_conv_kernel_size=15,
            dropout=0.2,
            use_group_norm=True,
        )

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conmer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conmer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()

        self.pad1 = nn.ConstantPad2d((0, 0, 1, 1), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        # (batch_size, in_channels, F_in, T_in)
        x = self.pad1(x)
        # (batch_size, in_channels, F_in + 2, T_in)

        out = self.conv(x)
        batch_size, n_channels_times_r, H_conv, W_conv = out.shape

        # (batch_size, self.out_channels, self.r, H_conv, W_conv)
        out = out.view(batch_size, self.out_channels, self.r, H_conv, W_conv)

        # (batch_size, self.out_channels, H_conv, self.r, W_conv) 
        out = out.permute(0, 1, 3, 2, 4)

        # (batch_size, self.out_channels, H_conv * self.r, W_conv) 
        out = out.contiguous().view(batch_size, self.out_channels, H_conv * self.r, W_conv)
        # (batch_size, self.out_channels, (F_in + 2 - k_f + 1) * self.r, (T_in - k_t + 1))

        return out


class DenseDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, depth=4, channels=64):
        super(DenseDecoder, self).__init__()
        self.conv_1 = nn.Sequential(
            SPConvTranspose2d(in_channel, out_channel, (3, 1), 2),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
        )
        self.dilated_dense = DilatedDenseNet(depth=depth, in_channels=out_channel)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1, beta=2):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.mask_conv = nn.Sequential(
            nn.InstanceNorm2d(num_channel, affine=True),
            nn.PReLU(num_channel),
            nn.Conv2d(num_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid2d(num_features, beta=beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.squeeze(1)
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, num_channel=64, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.phase_conv = nn.Sequential(
            nn.InstanceNorm2d(num_channel, affine=True),
            nn.PReLU(num_channel)
        )
        self.phase_conv_r = nn.Conv2d(num_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(num_channel, out_channel, (1, 1))
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.squeeze(1)
        return x

class masking(nn.Module):
    def __init__(
            self,
            win_len=400,
            hop_len=100,
            fft_len=400,
            win_type='hann',
            hidden=[16, 32, 64],
            TSCB_numb=4,
    ):
        '''

            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(masking, self).__init__()

        # for fft
        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.win_type = win_type
        self.hidden = [3] + hidden
        self.TSCBNumb = TSCB_numb


        for i in range(len(self.hidden) - 1):
            setattr(self, f"dense_encoder_{i}", 
                    DenseEncoder(in_channel=self.hidden[i], out_channel=self.hidden[i+1], depth=4, channels=64))
        for i in range(len(self.hidden) - 1, 0, -1):
            setattr(self, f"dense_decoder_{i}", 
                    DenseDecoder(in_channel=self.hidden[i], out_channel=self.hidden[i-1], depth=4, channels=64))
                
        self.mask_decoder = MaskDecoder(num_features=fft_len//2, num_channel=self.hidden[0], out_channel=1)
        self.phase_decoder = PhaseDecoder(num_channel=self.hidden[0], out_channel=1)
            
        for i in range(self.TSCBNumb):
            setattr(self, f"tscb_{i}", TSCB(num_channel=self.hidden[-1]))


    def forward(self, inputs, lens=False):
        
        in_len = inputs.size(-1)
        padded_inputs = pad_stft_input(inputs, self.fft_len, self.hop_len).squeeze(1)       
        
        mag, pha, com = mag_pha_stft(padded_inputs, self.fft_len, self.hop_len, self.win_len, compress_factor=0.3, center=False)
        # print(f"mag shape: {mag.shape}")
        # print(f"pha shape: {pha.shape}")
        # print(f"com shape: {com.shape}")
        
        real = com[:, :, :, 0]
        imag = com[:, :, :, 1]

        real = real[:, 1:, :]
        imag = imag[:, 1:, :]

        # print(f"real shape: {real.shape}")
        # print(f"imag shape: {imag.shape}")

        mag = mag[:, 1:, :]
        pha = pha[:, 1:, :]

        input = torch.stack([mag, real, imag], dim=1)
        # print(f"input shape: {input.shape}")


        skips = []
        for i in range(len(self.hidden) - 1):
            input = getattr(self, f"dense_encoder_{i}")(input)
            # print(f"dense_encoder_{i} output shape: {input.shape}")
            skips.append(input)
        
        for i in range(self.TSCBNumb):
            input = getattr(self, f"tscb_{i}")(input)
            # print(f"tscb_{i} output shape: {input.shape}")
        
        for i in range(len(self.hidden) - 1, 0, -1):
            # print(f"skips[{i-1}] shape: {skips[i-1].shape}")    
            input = input + skips[i-1]
            input = getattr(self, f"dense_decoder_{i}")(input)
            # print(f"dense_decoder_{i} output shape: {input.shape}")
        
        mask = self.mask_decoder(input)
        phase = self.phase_decoder(input)
        # print(f"mask shape: {mask.shape}")
        # print(f"phase shape: {phase.shape}")

        mags = mag * mask
        mags = F.pad(mags, [0, 0, 1, 0])
        phase = F.pad(phase, [0, 0, 1, 0])

        # print(f"mag shape: {mag.shape}")
        # print(f"phase shape: {phase.shape}")

        output_wav = mag_pha_istft(mags, phase, self.fft_len, self.hop_len, self.win_len, compress_factor=0.3)

        output_wav = output_wav.unsqueeze(1)

        output_wav = output_wav[..., :in_len]

        if lens == True:
            return mask, output_wav
        else:
            return output_wav
