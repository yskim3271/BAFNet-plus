import torch
import torch.nn.functional as F
import math

def pad_stft_input(y, n_fft, hop_size):
    
    in_len = y.size(-1)
    num_frames = math.ceil(in_len / hop_size) + 1
    total_length = (num_frames - 1) * hop_size + n_fft
    pad_left = n_fft // 2
    pad_right = total_length - in_len - pad_left
    padded_y = F.pad(y, (pad_left, 0), mode='reflect')
    padded_y = F.pad(padded_y, (0, pad_right), mode='constant', value=0.0)
    return padded_y

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, stack_dim=-1):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=stack_dim)

    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav

def complex_to_mag_pha(com, stack_dim=-1):
    real, imag = com.chunk(2, dim=stack_dim)
    mag = torch.sqrt(real**2 + imag**2).squeeze(stack_dim)
    pha = torch.atan2(imag, real).squeeze(stack_dim)
    return mag, pha
