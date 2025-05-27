import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mapping import mapping, SPConvTranspose1d
# from models.masking import TSCNet
from models.dccrn import dccrn
from models.stft import ConviSTFT, ConvSTFT
from models.discriminator import Discriminator
from criteria import GAN_Loss, CompositeLoss
from omegaconf import OmegaConf
from seconformer import Seconformer

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

def test_tscnet():
    model = TSCNet()
    input1 = torch.randn(2, 1, 32000)
    output = model(input1)
    

def stft():
    y = torch.randn(1, 1534)
    window_length = 400
    hop_length = 100
    fft_len = 512
    
    stft = ConvSTFT(
        win_len=window_length,
        win_inc=hop_length,
        fft_len=fft_len,
        win_type='hann',
        feature_type='complex',
    )
    
    istft = ConviSTFT(
        win_len=window_length,
        win_inc=hop_length,
        fft_len=fft_len,
        win_type='hann',
        feature_type='complex',
    )
    x = stft(y)
    print(f'stft: {x.shape}')
    y_hat = istft(x)
    
    print(f'istft: {y_hat.shape}')
    
def test_dccrn():
    dccrn_model = dccrn(
    )
    input1 = torch.randn(1, 8340)

    output = dccrn_model(input1)

    print(f"Output shape: {output.shape}")

def test_dilated_dense_net():
    dilated_dense_net = DilatedDenseNet(
    )
    input1 = torch.randn(1, 64, 512, 100)

    output = dilated_dense_net(input1)

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

def test_discriminator():
    discriminator = Discriminator(ndf=16)
    discriminator = discriminator.to("cuda")
    x = torch.randn(16, 16000).to("cuda")
    y = torch.randn(16, 16000).to("cuda")

    x = F.pad()

    args = OmegaConf.create({
        "l1_loss": 1.0,
        "ganloss": {
            "fft_size": 512,
            "hop_size": 256,
            "win_length": 512,
            "window": "hann_window",
            "factor_disc": 1.0,
            "factor_gen": 0.1,
        }
    })

    loss = CompositeLoss(args, discriminator=discriminator)
    print(loss.forward_disc_loss(x, y))
    # print(loss(x, y))
    
def test_dataset():
    dataset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")

    print(dataset[0])    

if __name__ == "__main__":
    # test_mapping()
    # test_masking()
    # stft()
    # test_dccrn()
    # test_tscnet()
    # test_discriminator()
    test_dataset()
    