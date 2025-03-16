import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy)
    )
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to("cuda")


class Discriminator(nn.Module):
    def __init__(self,
                 ndf=16, 
                 in_channel=2):
        super().__init__()
                
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
        )
        
    def stft_mag(self, x, eps=1e-9):
        x_spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, onesided=True)
        x_mag = torch.sqrt(x_spec[..., 0] ** 2 + x_spec[..., 1] ** 2 + eps)
        return x_mag

    def forward(self, x, y):
        xy = torch.stack([x, y], dim=1)
        return self.layers(xy)

def cal_stft_mag(x, n_fft=400, hop_length=100, eps=1e-9):
    x = x.squeeze(1)
    x_spec = torch.stft(x, 
                        n_fft=n_fft, 
                        hop_length=hop_length,
                        win_length=n_fft,
                        window=torch.hamming_window(n_fft).to(x.device),
                        onesided=True,
                        return_complex=True)
    x_mag = torch.sqrt(x_spec.real ** 2 + x_spec.imag ** 2 + eps)
    return x_mag

def cal_generator_loss(est, clean, discriminator):
    est_mag = cal_stft_mag(est)
    clean_mag = cal_stft_mag(clean)
        
    predict_fake_metric = discriminator(clean_mag, est_mag)
    gen_loss_GAN = F.mse_loss(predict_fake_metric, torch.ones_like(predict_fake_metric).to(est.device))
    return gen_loss_GAN

def cal_discriminator_loss(est, clean, discriminator):
    est_audio_list = list(est.detach().cpu().numpy())
    clean_audio_list = list(clean.detach().cpu().numpy())
    pesq_score = batch_pesq(clean_audio_list, est_audio_list)
    if pesq_score is None:
        return None

    clean_mag = cal_stft_mag(clean)
    est_mag = cal_stft_mag(est)

    predict_enhance_metric = discriminator(clean_mag, est_mag)
    predict_max_metric = discriminator(clean_mag, clean_mag)
    discriminator_loss_metric = F.mse_loss(
        predict_max_metric, torch.ones_like(predict_max_metric.to(est.device))
    ) + F.mse_loss(predict_enhance_metric, pesq_score)
    return discriminator_loss_metric
    