import importlib
import torch
import torch.nn as nn
from src.stft import mag_pha_stft, mag_pha_istft, pad_stft_input, complex_to_mag_pha

class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class BAFNet(torch.nn.Module):
    def __init__(self,
                 win_len=400,
                 hop_len=100,
                 fft_len=400,
                 conv_depth=4, 
                 conv_channels=16, 
                 conv_kernel_size=7,
                 learnable_sigmoid=True,
                 args_mapping=None, 
                 args_masking=None, 
                 checkpoint_mapping=None,
                 checkpoint_masking=None,
                 ):
        super(BAFNet, self).__init__()

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        
        self.conv_depth = conv_depth
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_kernel_size // 2
        self.learnable_sigmoid = learnable_sigmoid

        module_mapping = importlib.import_module("models." + args_mapping.model_lib)
        module_masking = importlib.import_module("models." + args_masking.model_lib)

        mapping_class = getattr(module_mapping, args_mapping.model_class)
        masking_class = getattr(module_masking, args_masking.model_class)

        self.mapping = mapping_class(**args_mapping.param)
        self.masking = masking_class(**args_masking.param)
        
        for i in range(self.conv_depth):
            setattr(self, f"convblock_{i}", nn.Sequential(
                nn.Conv2d(in_channels=self.conv_channels if i != 0 else 1, 
                          out_channels=self.conv_channels if i != self.conv_depth - 1 else 1, 
                          kernel_size=self.conv_kernel_size, 
                          stride=1, 
                          padding=self.conv_padding),
                nn.BatchNorm2d(self.conv_channels if i != self.conv_depth - 1 else 1),
                nn.PReLU() if i != self.conv_depth - 1 else nn.Identity()
            ))
        
        if self.learnable_sigmoid:
            self.sigmoid = LearnableSigmoid2d(self.fft_len // 2 + 1)
        else:
            self.sigmoid = nn.Sigmoid()
        
        
        self.init_modules()

        if checkpoint_mapping is not None:
            self.mapping.load_state_dict(torch.load(checkpoint_mapping)['model'])
        if checkpoint_masking is not None:
            self.masking.load_state_dict(torch.load(checkpoint_masking)['model'])
        
    def init_modules(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, tm, am, lens=False):
                        
        in_len = tm.size(-1)
        
        tm_wav = self.mapping(tm)

        mask, am_wav = self.masking(am, lens=True)

        tm_padded = pad_stft_input(tm_wav.squeeze(1), self.fft_len, self.hop_len)
        am_padded = pad_stft_input(am_wav.squeeze(1), self.fft_len, self.hop_len)

        tm_mag, _, tm_com = mag_pha_stft(tm_padded, self.fft_len, self.hop_len, self.win_len, center=False, stack_dim=1)
        am_mag, _, am_com = mag_pha_stft(am_padded, self.fft_len, self.hop_len, self.win_len, center=False, stack_dim=1)

        tm_real, tm_imag = tm_com[:, 0, :, :], tm_com[:, 1, :, :]
        am_real, am_imag = am_com[:, 0, :, :], am_com[:, 1, :, :]

        tm_mag_mean = tm_mag.mean(dim=[1, 2], keepdim=True)
        am_mag_mean = am_mag.mean(dim=[1, 2], keepdim=True)

        tm_real, tm_imag = tm_real / (tm_mag_mean + 1e-8), tm_imag / (tm_mag_mean + 1e-8)
        am_real, am_imag = am_real / (am_mag_mean + 1e-8), am_imag / (am_mag_mean + 1e-8)

        alpha = mask.unsqueeze(1)
        for i in range(self.conv_depth):
            alpha = getattr(self, f"convblock_{i}")(alpha)
        alpha = alpha.squeeze(1)
        alpha = self.sigmoid(alpha)

        est_real = tm_real * alpha + am_real * (1 - alpha)
        est_imag = tm_imag * alpha + am_imag * (1 - alpha)

        est_real = est_real * (tm_mag_mean + am_mag_mean) / 2.0
        est_imag = est_imag * (tm_mag_mean + am_mag_mean) / 2.0

        est_com = torch.stack([est_real, est_imag], dim=1)

        est_mag, est_pha = complex_to_mag_pha(est_com, stack_dim=1)

        est_wav = mag_pha_istft(est_mag, est_pha, self.fft_len, self.hop_len, self.win_len, compress_factor=1)

        est_wav = est_wav[..., :in_len]
        est_wav = est_wav.unsqueeze(1)
        
        if lens == True:
            return mask, alpha, tm_wav, am_wav, est_wav
        
        return est_wav
        
        
        
        
        
        

        
        
        
        
        