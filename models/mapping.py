import torch
import math
from torch import nn
from torch.nn import functional as F
from torchaudio.models.conformer import _FeedForwardModule, _ConvolutionModule, ConformerLayer


class SPConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose1d, self).__init__()

        pad_l = (kernel_size - 1) // 2
        pad_r = kernel_size - 1 - pad_l

        self.pad = nn.ConstantPad1d((pad_l, pad_r), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv1d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=1)
        self.r = r

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, T = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, T))
        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, -1))
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


class mapping(nn.Module):
    def __init__(self,
                 hidden=[64, 128, 256, 256, 256],
                 kernel_size=8,
                 kernel_size_2=9,
                 stride=[2, 2, 4, 4, 4],
                 depthwise_conv_kernel_size=15,
                 seq_module_depth=4,
                 dropout=0.1,
                 normalize=True,
                 concat=False,
                 spconv=False,
                 use_conformer=False,
                 num_attention_heads=4,
                 ):

        super().__init__()

        assert len(hidden) == len(stride), "hidden and stride must have the same length"

        self.hidden = [1] + hidden
        self.kernel_size = kernel_size
        self.kernel_size_2 = kernel_size_2

        pad_l = (kernel_size - 1) // 2
        pad_r = kernel_size - 1 - pad_l

        self.stride = stride
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.dropout = dropout
        self.normalize = normalize
        self.concat = concat
        self.spconv = spconv
        self.use_conformer = use_conformer

        if self.spconv:
            convtr = SPConvTranspose1d
        else:
            convtr = nn.ConvTranspose1d

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip = nn.ModuleList()
        
        for index in range(len(self.hidden) - 1):
            encode = []
            encode += [
                nn.Conv1d(self.hidden[index], self.hidden[index + 1] *2, kernel_size=kernel_size, stride=stride[index]),
                nn.GLU(1),
                nn.ConstantPad1d((pad_l, pad_r), value=0.0),
                nn.Conv1d(self.hidden[index + 1], self.hidden[index + 1], kernel_size=self.kernel_size_2, stride=1, groups=self.hidden[index + 1]),
                nn.BatchNorm1d(self.hidden[index + 1]),
                nn.SiLU(),
                nn.Conv1d(self.hidden[index + 1], self.hidden[index + 1], 1),
                nn.Dropout(dropout),
            ]
            self.encoder.append(nn.Sequential(*encode))
            
            decode = []
            decode += [
                nn.Conv1d(self.hidden[index + 1] * 2 if self.concat else self.hidden[index + 1], self.hidden[index + 1]* 2, 1), 
                nn.GLU(1),
                convtr(self.hidden[index + 1], self.hidden[index], kernel_size, self.stride[index]),
                nn.BatchNorm1d(self.hidden[index]),
            ]
            if index > 0:
                decode.append(nn.SiLU())
            
            self.decoder.insert(0, nn.Sequential(*decode))
        
        seq_dim = self.hidden[-1]
        
        self.seq_modules = nn.ModuleList()
        
        
        for index in range(seq_module_depth):
            self.seq_modules.append(
                Conmer(
                    input_dim=seq_dim,
                    ffn_dim=seq_dim,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                ) if not use_conformer else
                Conformer(
                    input_dim=seq_dim,
                    ffn_dim=seq_dim,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    use_group_norm=False
                )
            )

    def valid_length(self, length):
        for idx in range(len(self.encoder)):
            length = math.ceil((length - self.kernel_size) / self.stride[idx]) + 1
            length = max(length, 1)
        for idx in range(len(self.encoder)):
            length = (length - 1) * self.stride[idx] + self.kernel_size
        length = int(math.ceil(length))
        return int(length)

    def forward(self, mix):

        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (1e-3 + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
            
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
                
        x = x.permute(2, 0, 1)
                
        for seq_module in self.seq_modules:
            x = seq_module(x)
        
        x = x.permute(1, 2, 0)
        
        for decode in self.decoder:
            skip = skips.pop(-1)
            if self.concat:
                x = torch.cat((x, skip[..., :x.shape[-1]]), dim=1)
            else:
                x = x + skip[..., :x.shape[-1]]
            x = decode(x)

        x = x[..., :length]

        output_wav = std * x
        
        return output_wav
