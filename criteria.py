import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from pesq import pesq
from stft import mag_pha_stft
from torch_pesq import PesqLoss

class L1_Loss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(L1_Loss, self).__init__()
        self.name = "L1_Loss"
        self.weight = weight

    def forward(self, x, y):
        return F.l1_loss(x, y) * self.weight

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss, gd_loss, iaf_loss

class SpectralLoss(torch.nn.Module):
    def __init__(self, 
                 fft_size=400, 
                 hop_size=100, 
                 win_length=400,
                 compress_factor=1.0,
                 weight_lmag=1.0,
                 weight_conv=1.0,
                 weight_comp=1.0,
                 weight_pha=1.0
                 ):
        super(SpectralLoss, self).__init__()
        self.name = "Spectral_Loss"
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.compress_factor = compress_factor
        self.weight_lmag = weight_lmag
        self.weight_conv = weight_conv
        self.weight_comp = weight_comp
        self.weight_pha = weight_pha

    def forward(self, x, y):
        
        x, y = x.squeeze(1), y.squeeze(1)
        
        x_mag, x_pha, x_com = mag_pha_stft(x, self.fft_size, self.hop_size, self.win_length, compress_factor=self.compress_factor, center=True)
        y_mag, y_pha, y_com = mag_pha_stft(y, self.fft_size, self.hop_size, self.win_length, compress_factor=self.compress_factor, center=True)

        loss_lmag = F.l1_loss(torch.log(x_mag + 1e-8), torch.log(y_mag + 1e-8))
        loss_conv = torch.norm(x_mag - y_mag, p="fro") / torch.norm(y_mag, p="fro")
        loss_comp = F.mse_loss(x_com, y_com)

        loss_ip, loss_gd, loss_iaf = phase_losses(x_pha, y_pha)
        loss_pha = loss_ip + loss_gd + loss_iaf

        return loss_lmag * self.weight_lmag + loss_conv * self.weight_conv + loss_comp * self.weight_comp + loss_pha * self.weight_pha

class MultiResolutionSpectralLoss(torch.nn.Module):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 compress_factor=1.0,
                 weight_lmag=1.0,
                 weight_conv=1.0,
                 weight_comp=1.0,
                 weight_pha=1.0
    ):
        """Initialize Multi Resolution Spectral Loss.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
        """
        super(MultiResolutionSpectralLoss, self).__init__()
        self.name = "MultiResolutionSpectral_Loss"
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.losses += [SpectralLoss(fft_size=fs, hop_size=ss, win_length=wl, compress_factor=compress_factor,
                                         weight_lmag=weight_lmag, weight_conv=weight_conv, weight_comp=weight_comp, weight_pha=weight_pha)]

    def forward(self, x, y):
        total_loss = 0.0
        for f in self.losses:
            total_loss += f(x, y)

        return total_loss


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy, workers=10):
    pesq_score = Parallel(n_jobs=workers)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)

class MetricGAN_Loss(torch.nn.Module):
    def __init__(self, 
                 discriminator, 
                 fft_size, 
                 hop_size, 
                 win_length,
                 compress_factor=1.0, 
                 weight_gen=1.0,
                 weight_disc=1.0, 
                 pesq_workers=10,
                 use_torch_pesq=False
                 ):
        super().__init__()
        self.name = "MetricGAN_Loss"
        self.discriminator = discriminator
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.compress_factor = compress_factor
        self.weight_gen = weight_gen
        self.weight_disc = weight_disc
        self.pesq_workers = pesq_workers
        self.use_torch_pesq = use_torch_pesq
        self.torch_pesq = None
        if self.use_torch_pesq:
            self.torch_pesq = PesqLoss(factor=1.0, sample_rate=16000)

    def calculate_disc_loss(self, x, y):
        batch_size = x.shape[0]

        if self.use_torch_pesq:
            self.torch_pesq.to(x.device)
            pesq_score = self.torch_pesq.mos(ref=y.squeeze(1), deg=x.squeeze(1))
            pesq_score = (pesq_score - 1.08) / 4.999
            pesq_score = pesq_score.type(torch.FloatTensor)
        else:
            x_list = list(x.squeeze(1).detach().cpu().numpy())
            y_list = list(y.squeeze(1).detach().cpu().numpy())
            pesq_score = batch_pesq(y_list, x_list, workers=self.pesq_workers)
        
        if pesq_score is not None:
            x_mag, _, _ = mag_pha_stft(x.squeeze(1), self.fft_size, self.hop_size, self.win_length, compress_factor=self.compress_factor, center=True)
            y_mag, _, _ = mag_pha_stft(y.squeeze(1), self.fft_size, self.hop_size, self.win_length, compress_factor=self.compress_factor, center=True)
            x_mag = x_mag.unsqueeze(1)
            y_mag = y_mag.unsqueeze(1)

            predict_enhance_metric = self.discriminator(y_mag, x_mag)
            predict_max_metric = self.discriminator(y_mag, y_mag)
            discriminator_loss = F.mse_loss(
                predict_max_metric.flatten(), torch.ones(batch_size).to(x.device)
            ) + F.mse_loss(
                predict_enhance_metric.flatten(), pesq_score.to(x.device))
            discriminator_loss = discriminator_loss * self.weight_disc
        else:
            discriminator_loss = None

        return discriminator_loss

    def forward(self, x, y):
        batch_size = x.shape[0]
    
        x_mag, _, _ = mag_pha_stft(x.squeeze(1), self.fft_size, self.hop_size, self.win_length, compress_factor=self.compress_factor, center=True)
        y_mag, _, _ = mag_pha_stft(y.squeeze(1), self.fft_size, self.hop_size, self.win_length, compress_factor=self.compress_factor, center=True)

        predict_fake_metric = self.discriminator(y_mag.unsqueeze(1), x_mag.unsqueeze(1))

        generator_loss = F.mse_loss(
            predict_fake_metric.flatten(), torch.ones(batch_size).to(x.device)
        ).float()

        return generator_loss * self.weight_gen


class CompositeLoss(torch.nn.Module):
    def __init__(self, args, discriminator=None):
        super(CompositeLoss, self).__init__()
        
        self.loss_dict = {}
        self.discriminator = discriminator

        allowed_keys = {"l1_loss", "spectral_loss", "multispectral_loss", "metricgan_loss"}
        unknown_keys = set(args.keys()) - allowed_keys
        if unknown_keys:
            raise ValueError(f"Unknown loss keys in args: {unknown_keys}.")
        
        l1_loss_cfg = args.get("l1_loss")
        if l1_loss_cfg is not None:
            self.loss_dict['L1'] = L1_Loss(
                **l1_loss_cfg
            )

        spectral_loss_cfg = args.get("spectral_loss")
        if spectral_loss_cfg is not None:
            self.loss_dict['Spectral'] = SpectralLoss(
                **spectral_loss_cfg
            )

        multispectral_loss_cfg = args.get("multispectral_loss")
        if multispectral_loss_cfg is not None:
            self.loss_dict['MultiResolutionSpectral'] = MultiResolutionSpectralLoss(
                **multispectral_loss_cfg
            )
        
        metricganloss_cfg = args.get("metricgan_loss")
        if metricganloss_cfg is not None:
            self.loss_dict['MetricGAN'] = MetricGAN_Loss(
                self.discriminator, 
                **metricganloss_cfg
            )
            
    def forward(self, x, y):
        loss_all = 0
        loss_dict = {}
        
        for loss_name, loss_fn in self.loss_dict.items():
            loss = loss_fn(x, y)
            loss_all += loss
            loss_dict[loss_name] = loss
        
        return loss_all, loss_dict
    
    def forward_disc_loss(self, x, y):
        return self.loss_dict['MetricGAN'].calculate_disc_loss(x, y)
