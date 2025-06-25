"""
dccrn: Deep complex convolution recurrent network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models.conformer import _FeedForwardModule, _ConvolutionModule, ConformerLayer
from stft import mag_pha_stft, mag_pha_istft, pad_stft_input

class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


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

class Conformer(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 ffn_dim: int, 
                 depthwise_conv_kernel_size: int, 
                 num_attention_heads: int,
                 dropout: float = 0.0, 
                 use_group_norm: bool = False) -> None:
        super().__init__()
        self.conformer_layer = ConformerLayer(
            input_dim=input_dim,
            ffn_dim=ffn_dim,
            num_attention_heads=num_attention_heads,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conformer_layer(input, None)  # key_padding_mask is not used in this case

class TSCB(nn.Module):
    def __init__(self, 
                 input_dim= 64,
                 ffn_dim= 16,
                 depthwise_conv_kernel_size=15,
                 dropout=0.2,
                 use_group_norm=True,
                 use_conformer=False,
                 num_attention_heads=4
                 ):
        super(TSCB, self).__init__()
        self.use_conformer = use_conformer
        
        if use_conformer:
            self.time_conmer = ConformerLayer(
                input_dim=input_dim,
                ffn_dim=ffn_dim,
                num_attention_heads=num_attention_heads,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
                use_group_norm=use_group_norm,
            )
            self.freq_conmer = ConformerLayer(
                input_dim=input_dim,
                ffn_dim=ffn_dim,
                num_attention_heads=num_attention_heads,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
                use_group_norm=use_group_norm,
            )
        else:
            self.time_conmer = Conmer(
                input_dim=input_dim,
                ffn_dim=ffn_dim,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
                use_group_norm=use_group_norm,
            )
            self.freq_conmer = Conmer(
                input_dim=input_dim,
                ffn_dim=ffn_dim,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
                use_group_norm=use_group_norm,
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
                 in_channels,
                 out_channels,
                 dense_depth=4,
                 dense_channels=64
                 ):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dense_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.InstanceNorm2d(dense_channels, affine=True),
            nn.PReLU(dense_channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=dense_depth, in_channels=dense_channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=dense_channels, out_channels=out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class DenseDecoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 dense_depth=4, 
                 dense_channels=64):
        super(DenseDecoder, self).__init__()
        self.conv_1 = nn.Sequential(
            SPConvTranspose2d(in_channels, out_channels, (3, 1), 2),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(out_channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=dense_depth, in_channels=out_channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, 
                 num_features, 
                 dense_channels=64, 
                 out_channels=1, 
                 dense_depth=4,
                 beta=2):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=dense_depth, in_channels=dense_channels)
        self.mask_conv = nn.Sequential(
            nn.InstanceNorm2d(dense_channels, affine=True),
            nn.PReLU(dense_channels),
            nn.Conv2d(dense_channels, out_channels, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid2d(num_features, beta=beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.squeeze(1)
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, 
                 dense_channels=64, 
                 out_channels=1, 
                 dense_depth=4):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=dense_depth, in_channels=dense_channels)
        self.phase_conv = nn.Sequential(
            nn.InstanceNorm2d(dense_channels, affine=True),
            nn.PReLU(dense_channels)
        )
        self.phase_conv_r = nn.Conv2d(dense_channels, out_channels, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channels, out_channels, (1, 1))
        
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
            hidden=[16, 32, 64],
            dense_channels=64,
            dense_depth=4,
            TSCB_numb=4,
            ffn_dim=16,
            depthwise_conv_kernel_size=15,
            dropout=0.2,
            use_group_norm=True,
            use_conformer=False,
            num_attention_heads=4
    ):
        '''

            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(masking, self).__init__()

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.hidden = [4] + hidden
        self.dense_channels = dense_channels
        self.dense_depth = dense_depth
        self.TSCB_numb = TSCB_numb
        self.ffn_dim = ffn_dim
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.dropout = dropout
        self.use_group_norm = use_group_norm
        self.use_conformer = use_conformer
        self.num_attention_heads = num_attention_heads

        for i in range(len(self.hidden) - 1):
            setattr(self, f"dense_encoder_{i}", 
                    DenseEncoder(in_channels=self.hidden[i], 
                                 out_channels=self.hidden[i+1], 
                                 dense_channels=self.dense_channels,
                                 dense_depth=self.dense_depth))
            
        for i in range(len(self.hidden) - 1, 0, -1):
            setattr(self, f"dense_decoder_{i}", 
                    DenseDecoder(in_channels=self.hidden[i], 
                                 out_channels=self.hidden[i-1], 
                                 dense_channels=self.dense_channels,
                                 dense_depth=self.dense_depth))
                
        self.mask_decoder = MaskDecoder(num_features=fft_len//2, 
                                        dense_channels=self.hidden[0], 
                                        out_channels=1, 
                                        dense_depth=self.dense_depth)
        
        self.phase_decoder = PhaseDecoder(dense_channels=self.hidden[0], 
                                          out_channels=1, 
                                          dense_depth=self.dense_depth)
            
        for i in range(self.TSCB_numb):
            setattr(self, f"tscb_{i}", 
                    TSCB(input_dim=self.hidden[-1], 
                         ffn_dim=self.ffn_dim, 
                         depthwise_conv_kernel_size=self.depthwise_conv_kernel_size, 
                         dropout=self.dropout, 
                         use_group_norm=self.use_group_norm,
                         use_conformer=self.use_conformer,
                         num_attention_heads=self.num_attention_heads))


    def forward(self, inputs, lens=False):
        
        in_len = inputs.size(-1)
        padded_inputs = pad_stft_input(inputs, self.fft_len, self.hop_len).squeeze(1)       
        
        mag, pha, com = mag_pha_stft(padded_inputs, self.fft_len, self.hop_len, self.win_len, compress_factor=0.3, center=False)
        
        real = com[:, :, :, 0]
        imag = com[:, :, :, 1]

        real = real[:, 1:, :]
        imag = imag[:, 1:, :]

        mag = mag[:, 1:, :]
        pha = pha[:, 1:, :]

        input = torch.stack([mag, pha, real, imag], dim=1)

        skips = []
        for i in range(len(self.hidden) - 1):
            input = getattr(self, f"dense_encoder_{i}")(input)
            skips.append(input)
        
        for i in range(self.TSCB_numb):
            input = getattr(self, f"tscb_{i}")(input)
        
        for i in range(len(self.hidden) - 1, 0, -1):
            input = input + skips[i-1]
            input = getattr(self, f"dense_decoder_{i}")(input)
        
        mask = self.mask_decoder(input)
        pha = self.phase_decoder(input)

        mag = mag * mask
        mag = F.pad(mag, [0, 0, 1, 0])
        pha = F.pad(pha, [0, 0, 1, 0])
        mask = F.pad(mask, [0, 0, 1, 0])

        output_wav = mag_pha_istft(mag, pha, self.fft_len, self.hop_len, self.win_len, compress_factor=0.3)

        output_wav = output_wav.unsqueeze(1)
        output_wav = output_wav[..., :in_len]

        if lens == True:
            return mask, output_wav
        else:
            return output_wav


