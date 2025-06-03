import importlib
import torch
import torch.nn as nn
from models.stft import ConvSTFT, ConviSTFT

def normal_energy(spec: torch.Tensor, eps: float = 1e-8):
    
    real = spec[:, :spec.shape[1]//2, :]  # [B, F, T]
    imag = spec[:, spec.shape[1]//2:, :]  # [B, F, T]
    power = real**2 + imag**2            # [B, F, T]
    energy = power.mean(dim=[1,2], keepdim=True).sqrt()  
    spec_norm = spec / (energy + eps)

    return spec_norm, energy

class BAFNet(torch.nn.Module):
    def __init__(self,
                 win_len=400,
                 win_inc=100,
                 fft_len=512,
                 win_type='hann',
                 depth=4, 
                 channels=16, 
                 kernel_size=7,
                 args_mapping=None, 
                 args_masking=None, 
                 checkpoint_mapping=None,
                 checkpoint_masking=None,
                 ):
        super(BAFNet, self).__init__()

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type
        
        self.depth = depth
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        module_mapping = importlib.import_module("models." + args_mapping.model_lib)
        module_masking = importlib.import_module("models." + args_masking.model_lib)

        mapping_class = getattr(module_mapping, args_mapping.model_class)
        masking_class = getattr(module_masking, args_masking.model_class)

        self.mapping = mapping_class(**args_mapping.param)
        self.masking = masking_class(**args_masking.param)

                
        self.stft = ConvSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=True
        )
        self.istft = ConviSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=True
        )
        
        estimator = []
        channels = 1
        for i in range(self.depth - 1):
            estimator += [
                nn.Conv2d(in_channels=channels, 
                          out_channels=self.channels, 
                          kernel_size=self.kernel_size, 
                          stride=1, 
                          padding=self.padding),
                nn.BatchNorm2d(self.channels),
                nn.PReLU(),
            ]
            channels = self.channels
        estimator += [
            nn.Conv2d(in_channels=channels, 
                      out_channels=1, 
                      kernel_size=self.kernel_size, 
                      stride=1, 
                      padding=self.padding),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        ]
        
        self.ratio_estimator = nn.Sequential(*estimator)
        self.init_estimator()

        if checkpoint_mapping is not None:
            self.mapping.load_state_dict(torch.load(checkpoint_mapping)['model'])
        if checkpoint_masking is not None:
            self.masking.load_state_dict(torch.load(checkpoint_masking)['model'])
        
    def init_estimator(self):
        for m in self.ratio_estimator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.ratio_estimator.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, tm, am, lens=False):
                        
        in_len = tm.size(-1)
        
        tm_wav = self.mapping(tm)
        
        mask_mags, am_spec, am_wav = self.masking(am, lens=True)

        tm_spec = self.stft(tm_wav)

        tm_spec, tm_energy = normal_energy(tm_spec)
        am_spec, am_energy = normal_energy(am_spec)

        mask_mags = mask_mags.unsqueeze(1)
        
        alpha = self.ratio_estimator(mask_mags)
        alpha = alpha.squeeze(1)
        
        alpha = alpha.repeat(1, 2, 1)  # [B, 2*F, T]
        
        est_spec = tm_spec * alpha + am_spec * (1 - alpha)
        
        rms_target = (tm_energy + am_energy) / 2.0
        
        est_spec = rms_target * est_spec
        
        est_wav = self.istft(est_spec)
                
        out_len = est_wav.size(-1)
        
        if out_len > in_len:
            leftover = out_len - in_len 
            est_wav = est_wav[..., leftover//2:-(leftover//2)]
            am_wav = am_wav[..., leftover//2:-(leftover//2)]
        
        if lens == True:
            return mask_mags, alpha, tm_wav, am_wav, est_wav
        
        return est_wav
        
        
        
        
        
        

        
        
        
        
        