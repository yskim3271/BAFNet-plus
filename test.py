import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mapping import mapping
from models.masking import masking
from models.dccrn import dccrn
from models.stft import ConviSTFT, ConvSTFT
import time

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

def test_masking():
    # Define the model
    model = masking(
    )

    # Generate random input
    input1 = torch.randn(1, 15400)
    # input2 = torch.randn(1, 32001)
    # input3 = torch.randn(1, 32002)

    # Forward pass
    output = model(input1)
    # output = model(input2)
    # output = model(input3)

    # Print the shape of the output
    print(f"Output shape: {output.shape}")
    
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
    
    

if __name__ == "__main__":
    # test_mapping()
    test_masking()
    # stft()
    # test_dccrn()