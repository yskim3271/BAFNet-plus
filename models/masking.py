import torch
import math
from torch import nn
from torch.nn import functional as F
from torchaudio.models.conformer import _FeedForwardModule, _ConvolutionModule
from models.module_dccrn import ConviSTFT, ConvSTFT

class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = (1, 1),
        stride: int = (1, 1),
        padding: int = (0, 0),
        groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if groups > 1:
            self.groups = groups
        else:
            self.groups = 1       
        
        self.real_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)
        self.imag_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)
        
        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        real, imag = torch.chunk(inputs, 2, dim=1)
        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)
        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)
        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], dim=1)
        return out

class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = (1, 1),
        stride: int = (1, 1),
        padding: int = (0, 0),
        groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if groups > 1:
            self.groups = groups
        else:
            self.groups = 1
        
        self.real_conv = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            padding=self.padding)
            
        self.imag_conv = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            padding=self.padding)
        
        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        real, imag = torch.chunk(inputs, 2, dim=1)
        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)
        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)
        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], dim=1)
        return out

class Conmer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = True,
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
        self.convolution_first = convolution_first

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

class EncoderLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = (1, 1),
                 stride: int = (1, 1),
                 padding: int = (0, 0),
                 dropout: float = 0.0,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout

        self.layers = nn.Sequential(
            ComplexConv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GLU(dim=1),
            ComplexConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(7, 1),
                stride=(1, 1),
                padding=(3, 0),
            ),
            nn.BatchNorm2d(out_channels * 2),
            nn.SiLU(),
            ComplexConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.Dropout(dropout),
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.layers(input)
        return out
        
class DecoderLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 activation: bool = True,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
        self.layers = nn.Sequential(
            ComplexConv2d(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.GLU(dim=1),
            ComplexConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels * 2),
            nn.SiLU() if activation else nn.Identity(),
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.layers(input)
        return out

class masking(nn.Module):
    def __init__(self,
                 window_length=400,
                 hop_length=100,
                 fft_len=512,
                 hidden=[64, 128, 256, 512],
                 kernel_size=(1, 4),
                 stride=(1, 2),
                 depthwise_conv_kernel_size=7,
                 seq_module_depth=4,
                 dropout=0.1,
                 ):

        super().__init__()

        self.window_length = window_length
        self.hop_length = hop_length
        self.fft_len = fft_len
        
        self.hidden = [1] + hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.dropout = dropout
        
        self.stft = ConvSTFT(
            win_len=window_length,
            win_inc=hop_length,
            fft_len=fft_len,
            win_type="hann",
            feature_type='complex',
        )
        self.istft = ConviSTFT(
            win_len=window_length,
            win_inc=hop_length,
            fft_len=fft_len,
            win_type="hann",
            feature_type='complex',
        )
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        for index in range(len(self.hidden) - 1):
            self.encoder.append(
                EncoderLayer(
                    in_channels=self.hidden[index],
                    out_channels=self.hidden[index + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout=dropout,
                )
            )
            self.decoder.insert(0, 
                DecoderLayer(
                    in_channels=self.hidden[index + 1],
                    out_channels=self.hidden[index],
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=True if index > 0 else False,
                )
            )        
        seq_dim = self.hidden[-1] * 2
        self.seq_modules = nn.ModuleList()
        
        for index in range(seq_module_depth):
            self.seq_modules.append(
                Conmer(
                    input_dim=seq_dim,
                    ffn_dim=seq_dim,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                )
            )
        
            
    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        # print(f'input length: {length}')
        length = math.ceil(length / self.hop_length) * self.hop_length 
        
        frame_len = math.ceil((length + self.window_length - 2*self.hop_length) / self.hop_length) + 1
        
        # print(f'frame_len: {frame_len}')
                        
        for idx in range(len(self.hidden)):
            
            frame_len = math.ceil((frame_len - self.kernel_size[1]) / self.stride[1]) + 1
            frame_len = max(frame_len, 1)
            # print(f'frame_len: {frame_len}')
        
        for idx in range(len(self.hidden)):
            frame_len = (frame_len - 1) * self.stride[1] + self.kernel_size[1]
            # print(f'frame_len: {frame_len}')
        
        length = frame_len * self.hop_length - (self.window_length - self.hop_length)
        # print(f'length: {length}')

        return length
        
            
    def forward(self, input):
        length = input.shape[-1]
        
        valid_length = self.valid_length(length)
        
        r_pad = int((valid_length - length) / 2)
        l_pad = valid_length - length - r_pad
        
        if r_pad or l_pad:
            input = F.pad(input, (l_pad, r_pad))
        
        # print(f'input length: {input.shape[-1]}')
        
        spec = self.stft(input)
        
        # print(f'spec: {spec.shape}')
        
        real = spec[:, :self.fft_len // 2 + 1]
        imag = spec[:, self.fft_len // 2 + 1:]
        
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan2(imag, real)
        
        real = real[:, 1:]
        imag = imag[:, 1:]
                
        cspec = torch.stack([real, imag], 1)
                        
        out = cspec
                
        skips = []
        # print(f'input: {out.shape}')
        for encode in self.encoder:
            out = encode(out)
            # print(f'encode: {out.shape}')
            skips.append(out)

        
        b, c, t, f = out.shape
        out = out.view(b, c, t * f)
                
        out = out.permute(2, 0, 1)
        
        for seq_module in self.seq_modules:
            out = seq_module(out)
        
        out = out.permute(1, 2, 0)

        out = out.view(b, c, t, f)

        for decode in self.decoder:
            skip = skips.pop(-1)
            # print(f'skip: {skip.shape}')
            # print(f'out: {out.shape}')
            out = out + skip[..., :out.shape[-1]]
            out = decode(out)
        
        # print(f'out: {out.shape}')
        
        mask_real = out[:, 0, :, :]
        mask_imag = out[:, 1, :, :]
        
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])
        
        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        mask_phase = torch.atan2(mask_imag, mask_real)
        
        mask_mags = torch.tanh(mask_mags)
        
        est_mag = mask_mags * spec_mags
        est_phase = mask_phase + spec_phase
        
        # print(f'est_spec: {est_spec.shape}')
        
        out_wav = self.istft(est_mag, est_phase)
        
        # print(f'out_wav: {out_wav.shape}')
        if l_pad:
            out_wav = out_wav[..., l_pad:]
        if r_pad:
            out_wav = out_wav[..., :-r_pad]
        
        # print(f'out_wav: {out_wav.shape}')
        
        return out_wav
        
        
        
        

