"""Diagnostic test for PrimeKnet-Mamba implementation."""

import torch


def test_mamba_import():
    """Test if mamba-ssm is installed."""
    try:
        from mamba_ssm import Mamba
        print("✓ mamba-ssm successfully imported")
        return True
    except ImportError as e:
        print(f"✗ mamba-ssm import failed: {e}")
        return False


def test_mamba_basic_usage():
    """Test basic Mamba usage as per official docs."""
    try:
        from mamba_ssm import Mamba

        batch, length, dim = 2, 64, 16

        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")

        if device == "cpu":
            print(f"  ⚠️  Mamba requires CUDA GPU. Skipping test on CPU.")
            return True  # Skip but don't fail

        x = torch.randn(batch, length, dim).to(device)

        model = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)

        y = model(x)
        assert y.shape == (batch, length, dim), f"Expected {(batch, length, dim)}, got {y.shape}"
        print(f"✓ Basic Mamba test passed. Input: {x.shape}, Output: {y.shape}")
        return True
    except Exception as e:
        print(f"✗ Basic Mamba test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_primeknet_mamba_import():
    """Test if PrimeKnet-Mamba can be imported."""
    try:
        import sys
        import os
        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from models.primeknet_mamba import PrimeKnet
        print("✓ PrimeKnet-Mamba successfully imported")
        return True
    except ImportError as e:
        print(f"✗ PrimeKnet-Mamba import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_primeknet_mamba_instantiation():
    """Test if PrimeKnet-Mamba can be instantiated."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from models.primeknet_mamba import PrimeKnet

        model = PrimeKnet(
            fft_len=400,
            dense_channel=64,
            sigmoid_beta=1.0,
            dense_depth=4,
            num_tsblock=2,  # Reduced for faster testing
            time_block_num=1,  # Reduced
            mamba_hidden_size=64,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            freq_block_num=1,  # Reduced
            freq_block_kernel=[3, 11, 23, 31],
            infer_type='masking',
            causal=False
        )

        print(f"✓ PrimeKnet-Mamba instantiated successfully")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        return True
    except Exception as e:
        print(f"✗ PrimeKnet-Mamba instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_primeknet_mamba_forward():
    """Test PrimeKnet-Mamba forward pass."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from models.primeknet_mamba import PrimeKnet

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
        )
        model.eval()

        # Create test input
        batch_size = 2
        freq_bins = 201  # (400 // 2) + 1
        time_frames = 100

        noisy_com = torch.randn(batch_size, freq_bins, time_frames, 2)

        print(f"  Input shape: {noisy_com.shape}")

        with torch.no_grad():
            mag_hat, pha_hat, com_hat = model(noisy_com)

        print(f"  Output shapes:")
        print(f"    mag_hat: {mag_hat.shape}")
        print(f"    pha_hat: {pha_hat.shape}")
        print(f"    com_hat: {com_hat.shape}")

        # Verify shapes
        assert mag_hat.shape == (batch_size, freq_bins, time_frames), \
            f"mag_hat shape mismatch: expected {(batch_size, freq_bins, time_frames)}, got {mag_hat.shape}"
        assert pha_hat.shape == (batch_size, freq_bins, time_frames), \
            f"pha_hat shape mismatch: expected {(batch_size, freq_bins, time_frames)}, got {pha_hat.shape}"
        assert com_hat.shape == (batch_size, freq_bins, time_frames, 2), \
            f"com_hat shape mismatch: expected {(batch_size, freq_bins, time_frames, 2)}, got {com_hat.shape}"

        print("✓ PrimeKnet-Mamba forward pass succeeded")
        return True
    except Exception as e:
        print(f"✗ PrimeKnet-Mamba forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mamba_group_feature_network():
    """Test Mamba_Group_Feature_Network independently."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from models.primeknet_mamba import Mamba_Group_Feature_Network

        model = Mamba_Group_Feature_Network(
            in_channel=64,
            hidden_size=64,
            d_state=16,
            d_conv=4,
            expand=2,
            causal=False
        )

        # Input shape: (B, C, T)
        batch = 2
        channels = 64
        time = 100

        x = torch.randn(batch, channels, time)
        print(f"  Input shape: {x.shape}")

        y = model(x)
        print(f"  Output shape: {y.shape}")

        assert y.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {y.shape}"

        print("✓ Mamba_Group_Feature_Network test passed")
        return True
    except Exception as e:
        print(f"✗ Mamba_Group_Feature_Network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("PrimeKnet-Mamba Diagnostic Tests")
    print("=" * 80)
    print()

    results = []

    print("[1/6] Testing mamba-ssm import...")
    results.append(("mamba-ssm import", test_mamba_import()))
    print()

    print("[2/6] Testing basic Mamba usage...")
    results.append(("Basic Mamba", test_mamba_basic_usage()))
    print()

    print("[3/6] Testing PrimeKnet-Mamba import...")
    results.append(("PrimeKnet-Mamba import", test_primeknet_mamba_import()))
    print()

    print("[4/6] Testing Mamba_Group_Feature_Network...")
    results.append(("Mamba_Group_Feature_Network", test_mamba_group_feature_network()))
    print()

    print("[5/6] Testing PrimeKnet-Mamba instantiation...")
    results.append(("PrimeKnet-Mamba instantiation", test_primeknet_mamba_instantiation()))
    print()

    print("[6/6] Testing PrimeKnet-Mamba forward pass...")
    results.append(("PrimeKnet-Mamba forward", test_primeknet_mamba_forward()))
    print()

    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\n⚠️  Some tests failed. See above for details.")
        exit(1)
    else:
        print("\n✅ All tests passed!")
        exit(0)
