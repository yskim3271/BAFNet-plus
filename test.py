import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mapping import mapping
import time


def test_mapping():
    # Define the model
    """  hidden: [64, 128, 256, 256, 256]
  kernel_size: 8
  stride: [2, 2, 4, 4, 4]
  depthwise_conv_kernel_size: 31
  seq_module_depth: 4
  dropout: 0.1
  normalize: true

    """
    
    model = mapping(
        hidden=[64, 128, 256, 256, 256],
        kernel_size=9,
        stride=[2, 2, 4, 4, 4],
        depthwise_conv_kernel_size=31,
        seq_module_depth=4,
        dropout=0.1,
        normalize=True,
    )

    # Generate random input
    input1 = torch.randn(1, 1, 32000)
    input2 = torch.randn(1, 1, 32001)
    input3 = torch.randn(1, 1, 32002)

    # Forward pass
    output = model(input1)
    output = model(input2)
    output = model(input3)

    # Print the shape of the output
    # print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_mapping()