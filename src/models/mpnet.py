import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout
from src.stft import mag_pha_stft, mag_pha_istft, pad_stft_input


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    

class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super(FFN, self).__init__()
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = Linear(d_model*2*2, d_model)
        else:
            self.linear = Linear(d_model*2, d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = Dropout(dropout)
        
        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 r=1
                 ):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out
    

class DenseBlock(nn.Module):
    def __init__(self, 
                 dense_channel, 
                 depth=4
                 ):
        super(DenseBlock, self).__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                nn.Conv2d(self.dense_channel*(i+1), 
                          self.dense_channel, 
                          kernel_size=(2, 3), 
                          dilation=(dilation, 1)
                          ),
                nn.InstanceNorm2d(self.dense_channel, affine=True),
                nn.PReLU(self.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self,
                 dense_channel,
                 in_channel
                 ):
        super(DenseEncoder, self).__init__()
        
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

        self.dense_block = DenseBlock(dense_channel, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self,
                 fft_len,
                 dense_channel,
                 sigmoid_beta,
                 out_channel=1
                 ):
        super(MaskDecoder, self).__init__()
        
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
            nn.Conv2d(dense_channel, out_channel, (1, 2))
        )
        self.lsigmoid = LearnableSigmoid2d(fft_len//2+1, beta=sigmoid_beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self,
                 dense_channel,
                 out_channel=1
                 ):
        super(PhaseDecoder, self).__init__()
        
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        return x


class TSTransformerBlock(nn.Module):
    def __init__(self, 
                 dense_channel,
                 numb_attention_heads=4
                 ):
        super(TSTransformerBlock, self).__init__()
        self.time_transformer = TransformerBlock(d_model=dense_channel, n_heads=numb_attention_heads)
        self.freq_transformer = TransformerBlock(d_model=dense_channel, n_heads=numb_attention_heads)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MPNet(nn.Module):
    def __init__(self, 
                 win_len,
                 hop_len,
                 fft_len,
                 compress_factor=1.0,
                 dense_channel=64,
                 sigmoid_beta=2.0,
                 numb_attention_heads=4,
                 num_tsblocks=4,
                 infer_type='masking'
                 ):
        super(MPNet, self).__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.compress_factor = compress_factor
        self.dense_channel = dense_channel
        self.sigmoid_beta = sigmoid_beta
        self.numb_attention_heads = numb_attention_heads
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(dense_channel=dense_channel, in_channel=2)
        self.infer_type = infer_type
        assert infer_type in ['masking', 'mapping'], 'infer_type must be either masking or mapping'

        self.TSTransformer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.TSTransformer.append(TSTransformerBlock(dense_channel, numb_attention_heads=numb_attention_heads))

        self.mask_decoder = MaskDecoder(fft_len=fft_len, 
                                        dense_channel=dense_channel, 
                                        sigmoid_beta=sigmoid_beta, 
                                        out_channel=1)
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

        x = torch.stack((mag, pha), dim=-1).permute(0, 3, 2, 1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)
        
        if self.infer_type == 'masking':
            mag = (mag * self.mask_decoder(x))
        elif self.infer_type == 'mapping':
            mag = self.mask_decoder(x)
            
        pha = self.phase_decoder(x)

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