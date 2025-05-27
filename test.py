import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mapping import mapping, SPConvTranspose1d
# from models.masking import TSCNet
from models.dccrn import dccrn
from models.stft import ConviSTFT, ConvSTFT
from models.disc_hifigan import HiFiGAN_Discriminator
from models.disc_cmgan import CMGAN_Discriminator
from criteria import HiFiGAN_Loss, CompositeLoss
from omegaconf import OmegaConf

import time
from datasets import load_dataset

# def test_mapping():
#     # Define the model
#     """  hidden: [64, 128, 256, 256, 256]
#   kernel_size: 8
#   stride: [2, 2, 4, 4, 4]
#   depthwise_conv_kernel_size: 31
#   seq_module_depth: 4
#   dropout: 0.1
#   normalize: true

#     """
    
#     model = mapping(
#         hidden=[64, 128, 256, 256, 256],
#         kernel_size=9,
#         stride=[2, 2, 4, 4, 4],
#         depthwise_conv_kernel_size=31,
#         seq_module_depth=4,
#         dropout=0.1,
#         normalize=True,
#     )

#     # Generate random input
#     input1 = torch.randn(1, 1, 32000)
#     input2 = torch.randn(1, 1, 32001)
#     input3 = torch.randn(1, 1, 32002)

#     # Forward pass
#     output = model(input1)
#     output = model(input2)
#     output = model(input3)

#     # Print the shape of the output
#     # print(f"Output shape: {output.shape}")

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


# def stft():
#     y = torch.randn(1, 1534)
#     window_length = 400
#     hop_length = 100
#     fft_len = 512
    
#     stft = ConvSTFT(
#         win_len=window_length,
#         win_inc=hop_length,
#         fft_len=fft_len,
#         win_type='hann',
#         feature_type='complex',
#     )
    
#     istft = ConviSTFT(
#         win_len=window_length,
#         win_inc=hop_length,
#         fft_len=fft_len,
#         win_type='hann',
#         feature_type='complex',
#     )
#     x = stft(y)
#     print(f'stft: {x.shape}')
#     y_hat = istft(x)
    
#     print(f'istft: {y_hat.shape}')
    
def test_dccrn():
    dccrn_model = dccrn(
    )
    input1 = torch.randn(1, 8340)

    output = dccrn_model(input1)

    print(f"Output shape: {output.shape}")


def test_spconv_transpose_1d():
    spconv_transpose_1d = SPConvTranspose1d(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        r=2,
    )
    input1 = torch.randn(1, 64, 512)

    output = spconv_transpose_1d(input1)

    print(f"Output shape: {output.shape}")

def test_hifigan_discriminator():
    discriminator = HiFiGAN_Discriminator()
    discriminator = discriminator.to("cuda")
    x = torch.randn(1, 1, 16000).to("cuda")
    y = torch.randn(1, 1, 16000).to("cuda")

    outputs_msd, outputs_mpd = discriminator(x, y)

    print(outputs_msd[0].shape)
    print(outputs_mpd[0].shape)

    print(outputs_msd[1].shape)
    print(outputs_mpd[1].shape)

    print(outputs_msd[2].shape)
    print(outputs_mpd[2].shape)
    
def test_cmgan_discriminator():


    discriminator = CMGAN_Discriminator(ndf=16)
    discriminator = discriminator.to("cuda")
    x = torch.randn(1, 1, 48000).to("cuda")
    y = torch.randn(1, 1, 48000).to("cuda")

    window = torch.hann_window(512)

    x_mag = stft(x, 512, 256, 512, window, onesided=False, center = True).unsqueeze(1)
    y_mag = stft(y, 512, 256, 512, window, onesided=False, center = True).unsqueeze(1)


    discriminator_loss = discriminator(x_mag, y_mag)

    print(discriminator_loss)
    

def test_dataset():
    dataset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")

    print(dataset[0])    

if __name__ == "__main__":
    # test_mapping()
    # test_masking()
    # stft()
    # test_dccrn()
    # test_tscnet()
    # test_cmgan_discriminator()
    test_hifigan_discriminator()
    # test_dataset()
    