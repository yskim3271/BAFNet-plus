#!/usr/bin/env python3
"""
Merge LoRA weights into base model for efficient deployment.

This script takes a checkpoint with LoRA weights and merges them into the
base model weights, eliminating the LoRA overhead during inference.

Usage:
    python scripts/merge_lora_checkpoint.py \
        --checkpoint outputs/bafnet_lora_r8/best.th \
        --output outputs/bafnet_lora_r8/best_merged.th
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
from src.models.lora import merge_lora_weights
from src.utils import load_model


def main():
    """
    Merge LoRA weights into base model for efficient deployment.
    """
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base model checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint with LoRA weights",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save merged checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for loading model (default: cpu)",
    )
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Get model config from checkpoint
    if "args" not in ckpt:
        print("Error: Checkpoint does not contain args. Cannot determine model config.")
        sys.exit(1)

    model_config = ckpt["args"].model
    model_lib = model_config.model_lib
    model_class = model_config.model_class

    # Load model (without LoRA first)
    print(f"Creating {model_lib}.{model_class}")
    model = load_model(model_lib, model_class, model_config.param, args.device)

    # Load state dict (which includes LoRA weights)
    print("Loading model state dict...")
    model.load_state_dict(ckpt["model"])

    # Count parameters before merge
    total_params_before = sum(p.numel() for p in model.parameters())
    lora_params_before = 0
    for module in model.modules():
        if hasattr(module, "lora"):
            lora_params_before += sum(p.numel() for p in module.lora.parameters())

    print(f"\nBefore merge:")
    print(f"  Total parameters: {total_params_before:,}")
    print(f"  LoRA parameters: {lora_params_before:,}")

    # Merge LoRA weights into base layers
    print("\nMerging LoRA weights into base model...")
    merge_lora_weights(model)

    # Verify merge (LoRA weights should be zeroed)
    lora_params_after = 0
    for module in model.modules():
        if hasattr(module, "lora"):
            lora_params_after += sum(
                p.abs().sum().item() for p in module.lora.parameters()
            )

    print(f"\nAfter merge:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  LoRA param sum (should be ~0): {lora_params_after:.6f}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCreated output directory: {output_dir}")

    # Save merged model
    print(f"\nSaving merged model to {args.output}")
    torch.save(
        {
            "model": model.state_dict(),
            "args": ckpt.get("args", None),
            "history": ckpt.get("history", []),
            "best_loss": ckpt.get("best_loss", 0.0),
        },
        args.output,
    )

    print("\n✅ LoRA weights merged successfully!")
    print("   Inference will be as fast as original model (no LoRA overhead)")
    print(f"\n   Merged checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
