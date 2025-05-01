import torch
import torch.nn.functional as F
from torch_pesq import PesqLoss

def masking_and_split(preds, target, mask):
    B, C, T = preds.shape
    lengths = mask.sum(dim=[1, 2]).int()  # (B,)

    preds_list = []
    target_list = []

    for i in range(B):
        L = lengths[i]
        preds_i = preds[i, :, :L]
        target_i = target[i, :, :L]
        preds_list.append(preds_i)
        target_list.append(target_i)

    return preds_list, target_list

def l2_loss(preds, target, mask=None):
    if mask is not None:
        return F.mse_loss(preds * mask, target * mask)
    return F.mse_loss(preds, target)

def l1_loss(preds, target, mask=None):
    if mask is not None:
        return F.l1_loss(preds * mask, target * mask)
    return F.l1_loss(preds, target)

def si_snr_loss(preds, target, mask=None):
    if mask is not None:
        preds = preds * mask
        target = target * mask
    
    if preds.dim() == 3:
        preds = preds.squeeze(1)
        target = target.squeeze(1)
    
    # # mean centering
    # preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    # target = target - torch.mean(target, dim=-1, keepdim=True)
    
    target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
    preds_norm = preds / (torch.norm(preds, dim=-1, keepdim=True) + 1e-8)
    
    s_target = torch.sum(target_norm * preds_norm, dim=-1, keepdim=True) * target_norm
    
    e_noise = preds - s_target
    
    signal_power = torch.sum(s_target ** 2, dim=-1)
    noise_power = torch.sum(e_noise ** 2, dim=-1)
    si_snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    return -torch.mean(si_snr)

def si_sdr_loss(preds, target, mask=None):
    if mask is not None:
        preds = preds * mask
        target = target * mask
    
    if preds.dim() == 3:
        preds = preds.squeeze(1)
        target = target.squeeze(1)
    
    # preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    # target = target - torch.mean(target, dim=-1, keepdim=True)
    
    alpha = torch.sum(target * preds, dim=-1, keepdim=True) / (torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8)
    
    s_target = alpha * target
    
    e_noise = preds - s_target
    
    signal_power = torch.sum(s_target ** 2, dim=-1)
    noise_power = torch.sum(e_noise ** 2, dim=-1)
    si_sdr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    return -torch.mean(si_sdr)

def squeeze_to_2d(x):
    """Squeeze tensor to 2D.
    Args:
        x (Tensor): Input tensor (B, ..., T).
    Returns:
        Tensor: Squeezed tensor (B, T).
    """
    return x.view(x.size(0), -1)

def stft(x, fft_size, hop_size, win_length, window, onesided=False, center=True):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x = squeeze_to_2d(x)
    window = window.to(x.device)
    x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, 
                        return_complex=True, onesided=onesided, center=center)
    real = x_stft.real
    imag = x_stft.imag
    return torch.sqrt(real ** 2 + imag ** 2 + 1e-9).transpose(2, 1)

class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, hop_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.name = "STFTLoss"
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.hop_size, self.win_length, self.window, onesided=False, center = True)
        y_mag = stft(y, self.fft_size, self.hop_size, self.win_length, self.window, onesided=False, center = True)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1, device='cpu'):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.name = "MultiResolutionSTFTLoss"
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window).to(device)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        
    def get_stftm(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        stftm = torch.abs(stft_R) + torch.abs(stft_I)
        return stftm

    def forward(self, x, y, mask=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution    spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        if mask is not None:
            x, y = masking_and_split(x, y, mask)
            x = [x_i.unsqueeze(1) for x_i in x]
            y = [y_i.unsqueeze(1) for y_i in y]
            for x_i, y_i in zip(x, y):
                for f in self.stft_losses:
                    sc_l, mag_l = f(x_i, y_i)
                    sc_loss += sc_l
                    mag_loss += mag_l
            sc_loss /= len(x)
            mag_loss /= len(x)
        else:
            for f in self.stft_losses:
                sc_l, mag_l = f(x, y)
                sc_loss += sc_l
                mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        loss = self.factor_sc * sc_loss + self.factor_mag * mag_loss
        return loss
    

class CompositeLoss(torch.nn.Module):
    def __init__(self, args):
        super(CompositeLoss, self).__init__()
        
        self.loss_dict = {}
        self.loss_weight = {}
        
        if 'l1_loss' in args:
            self.loss_dict['l1_loss'] = l1_loss
            self.loss_weight['l1_loss'] = args.l1_loss
        
        if 'l2_loss' in args:
            self.loss_dict['l2_loss'] = l2_loss
            self.loss_weight['l2_loss'] = args.l2_loss
            
        if 'multistftloss' in args:
            self.loss_weight['multistft_loss'] = args.multistftloss.weight
            del args.multistftloss.weight
            
            self.loss_dict['multistft_loss'] = MultiResolutionSTFTLoss(
                **args.multistftloss
            )

        if 'sisnrloss' in args:
            self.loss_dict['sisnr_loss'] = si_snr_loss         
            self.loss_weight['sisnr_loss'] = args.sisnrloss
            
        if 'sisdrloss' in args:
            self.loss_dict['sisdr_loss'] = si_sdr_loss
            self.loss_weight['sisdr_loss'] = args.sisdrloss

            
    def forward(self, x, y, mask=None):
        loss_all = 0
        loss_dict = {}

        for loss_name, loss_fn in self.loss_dict.items():
            loss = loss_fn(x, y, mask)
            loss_all += self.loss_weight[loss_name] * loss
            loss_dict[loss_name] = loss
        
        return loss_all, loss_dict
