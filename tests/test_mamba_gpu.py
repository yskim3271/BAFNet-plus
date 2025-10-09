"""GPU test for PrimeKnet-Mamba implementation."""

import sys
import os
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("GPU Environment Check")
print("=" * 80)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Get compute capability
    cc_major, cc_minor = torch.cuda.get_device_capability()
    print(f"Compute Capability: {cc_major}.{cc_minor}")
    print()
else:
    print("ERROR: CUDA not available. Cannot test Mamba on GPU.")
    sys.exit(1)

print("=" * 80)
print("Testing Mamba-SSM on GPU")
print("=" * 80)
print()

# Test 1: Basic Mamba
print("[1/4] Testing basic Mamba module...")
try:
    from mamba_ssm import Mamba

    device = torch.device("cuda:0")
    batch, length, dim = 2, 64, 16

    x = torch.randn(batch, length, dim).to(device)
    print(f"  Input shape: {x.shape}, device: {x.device}")

    model = Mamba(
        d_model=dim,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    print(f"  Model on device: {next(model.parameters()).device}")

    with torch.no_grad():
        y = model(x)

    print(f"  Output shape: {y.shape}, device: {y.device}")
    print("  ✅ Basic Mamba test PASSED")
    print()
except Exception as e:
    print(f"  ❌ Basic Mamba test FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 2: Mamba_Group_Feature_Network
print("[2/4] Testing Mamba_Group_Feature_Network...")
try:
    from models.primeknet_mamba import Mamba_Group_Feature_Network

    device = torch.device("cuda:0")

    model = Mamba_Group_Feature_Network(
        in_channel=64,
        hidden_size=64,
        d_state=16,
        d_conv=4,
        expand=2,
        causal=False
    ).to(device)

    # Input shape: (B, C, T)
    batch = 2
    channels = 64
    time = 100

    x = torch.randn(batch, channels, time).to(device)
    print(f"  Input shape: {x.shape}, device: {x.device}")

    with torch.no_grad():
        y = model(x)

    print(f"  Output shape: {y.shape}, device: {y.device}")
    assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"
    print("  ✅ Mamba_Group_Feature_Network test PASSED")
    print()
except Exception as e:
    print(f"  ❌ Mamba_Group_Feature_Network test FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 3: PrimeKnet-Mamba instantiation
print("[3/4] Testing PrimeKnet-Mamba instantiation...")
try:
    from models.primeknet_mamba import PrimeKnet

    device = torch.device("cuda:0")

    model = PrimeKnet(
        fft_len=400,
        dense_channel=64,
        sigmoid_beta=1.0,
        dense_depth=4,
        num_tsblock=2,
        time_block_num=1,
        mamba_hidden_size=64,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        freq_block_num=1,
        freq_block_kernel=[3, 11, 23, 31],
        infer_type='masking',
        causal=False
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Model on device: {next(model.parameters()).device}")
    print("  ✅ PrimeKnet-Mamba instantiation PASSED")
    print()
except Exception as e:
    print(f"  ❌ PrimeKnet-Mamba instantiation FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test 4: PrimeKnet-Mamba forward pass
print("[4/4] Testing PrimeKnet-Mamba forward pass...")
try:
    from models.primeknet_mamba import PrimeKnet

    device = torch.device("cuda:0")

    model = PrimeKnet(
        fft_len=400,
        dense_channel=64,
        sigmoid_beta=1.0,
        dense_depth=4,
        num_tsblock=2,
        time_block_num=1,
        mamba_hidden_size=64,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        freq_block_num=1,
        freq_block_kernel=[3, 11, 23, 31],
        infer_type='masking',
        causal=False
    ).to(device)

    model.eval()

    # Create test input
    batch_size = 2
    freq_bins = 201  # (400 // 2) + 1
    time_frames = 100

    noisy_com = torch.randn(batch_size, freq_bins, time_frames, 2).to(device)

    print(f"  Input shape: {noisy_com.shape}, device: {noisy_com.device}")

    with torch.no_grad():
        mag_hat, pha_hat, com_hat = model(noisy_com)

    print(f"  Output shapes:")
    print(f"    mag_hat: {mag_hat.shape}, device: {mag_hat.device}")
    print(f"    pha_hat: {pha_hat.shape}, device: {pha_hat.device}")
    print(f"    com_hat: {com_hat.shape}, device: {com_hat.device}")

    # Verify shapes
    assert mag_hat.shape == (batch_size, freq_bins, time_frames), \
        f"mag_hat shape mismatch: expected {(batch_size, freq_bins, time_frames)}, got {mag_hat.shape}"
    assert pha_hat.shape == (batch_size, freq_bins, time_frames), \
        f"pha_hat shape mismatch: expected {(batch_size, freq_bins, time_frames)}, got {pha_hat.shape}"
    assert com_hat.shape == (batch_size, freq_bins, time_frames, 2), \
        f"com_hat shape mismatch: expected {(batch_size, freq_bins, time_frames, 2)}, got {com_hat.shape}"

    print("  ✅ PrimeKnet-Mamba forward pass PASSED")
    print()
except Exception as e:
    print(f"  ❌ PrimeKnet-Mamba forward pass FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

print("=" * 80)
print("FINAL RESULT")
print("=" * 80)
print("✅ ALL TESTS PASSED!")
print("PrimeKnet-Mamba is working correctly on GPU.")
print()
print("The model can now be used for training with:")
print("  CUDA_VISIBLE_DEVICES=0 python train.py +model=primeknet_mamba_masking +dset=taps")
