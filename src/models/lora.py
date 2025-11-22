"""
LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.

This module provides LoRA adapters for Linear, Conv1d, and Conv2d layers,
enabling parameter-efficient fine-tuning of pretrained models.

Reference:
    LoRA: Low-Rank Adaptation of Large Language Models
    https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.

    Adds trainable low-rank decomposition to a frozen linear layer:
        y = W_frozen @ x + (B @ A) @ x

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of decomposition (r)
        alpha: Scaling factor for LoRA updates
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout (applied to LoRA path)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        Compute LoRA update: (B @ A) @ x

        Args:
            x: Input tensor [..., in_features]

        Returns:
            LoRA update [..., out_features]
        """
        # x: [..., in_features]
        # A: [rank, in_features]
        # B: [out_features, rank]

        # Step 1: x @ A^T -> [..., rank]
        h = x @ self.lora_A.T

        # Step 2: dropout
        h = self.dropout(h)

        # Step 3: h @ B^T -> [..., out_features]
        output = h @ self.lora_B.T

        # Step 4: scale
        output = output * self.scaling

        return output


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Wraps a frozen linear layer and adds trainable LoRA parameters.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Frozen base layer
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # LoRA adapter
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x):
        # Base output (frozen)
        base_out = self.base_layer(x)

        # LoRA update (trainable)
        lora_out = self.lora(x)

        # Combined
        return base_out + lora_out

    def merge_weights(self):
        """
        Merge LoRA weights into base layer for inference.

        After training, we can merge LoRA into the base weight:
            W_new = W_base + B @ A

        This removes the overhead of two separate computations.
        """
        with torch.no_grad():
            # Compute LoRA weight: B @ A
            lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling

            # Add to base weight
            self.base_layer.weight.data += lora_weight

            # Zero out LoRA to avoid double-counting
            self.lora.lora_A.zero_()
            self.lora.lora_B.zero_()


class Conv1dWithLoRA(nn.Module):
    """
    Conv1d layer with LoRA adaptation.

    Treats 1D convolution as a linear transformation with spatial structure.
    """

    def __init__(
        self,
        base_layer: nn.Conv1d,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Frozen base layer
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # LoRA parameters
        out_channels = base_layer.out_channels
        in_channels = base_layer.in_channels
        kernel_size = base_layer.kernel_size[0]
        groups = base_layer.groups

        # Effective dimensions (accounting for groups)
        in_features = (in_channels // groups) * kernel_size
        out_features = out_channels // groups

        # Create LoRA adapter per group
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        self.groups = groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # Base convolution (frozen)
        base_out = self.base_layer(x)

        # For groups=1 case, implement LoRA via unfold
        if self.groups == 1:
            # x: [B, C_in, T]
            B, C_in, T = x.shape

            # Unfold to extract sliding windows
            # Add height dimension for unfold: [B, C_in, T, 1]
            x_4d = x.unsqueeze(-1)

            x_unfold = torch.nn.functional.unfold(
                x_4d,
                kernel_size=(self.kernel_size, 1),
                padding=(self.base_layer.padding[0], 0),
                stride=(self.base_layer.stride[0], 1),
            )  # [B, C_in*K*1, T_out]

            # Remove the height dimension: [B, C_in*K, T_out]
            x_unfold = x_unfold.reshape(B, C_in * self.kernel_size, -1)

            # Reshape for LoRA: [B*T_out, C_in*K]
            T_out = x_unfold.shape[2]
            x_unfold = x_unfold.permute(0, 2, 1).reshape(-1, C_in * self.kernel_size)

            # Apply LoRA
            lora_out = self.lora(x_unfold)  # [B*T_out, C_out]

            # Reshape back: [B, C_out, T_out]
            lora_out = lora_out.reshape(B, T_out, self.out_channels)
            lora_out = lora_out.permute(0, 2, 1)

            return base_out + lora_out
        else:
            # Grouped convolution: return base only for simplicity
            # Full implementation would apply LoRA per group
            return base_out

    def merge_weights(self):
        """Merge LoRA weights into base conv layer."""
        with torch.no_grad():
            # Compute LoRA weight: B @ A
            # Shape: [out_features, in_features] where in_features = C_in * K
            lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling

            # Reshape to conv1d weight format: [out_channels, in_channels, kernel_size]
            lora_weight = lora_weight.reshape(
                self.out_channels, self.in_channels, self.kernel_size
            )

            # Add to base weight
            self.base_layer.weight.data += lora_weight

            # Zero out LoRA
            self.lora.lora_A.zero_()
            self.lora.lora_B.zero_()


class Conv2dWithLoRA(nn.Module):
    """
    Conv2d layer with LoRA adaptation.

    Treats 2D convolution as a linear transformation using im2col.
    """

    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Frozen base layer
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Decompose conv2d weight: [C_out, C_in, K_h, K_w]
        out_channels = base_layer.out_channels
        in_channels = base_layer.in_channels
        kernel_h, kernel_w = base_layer.kernel_size

        # Treat as linear: [C_out, C_in * K_h * K_w]
        in_features = in_channels * kernel_h * kernel_w
        out_features = out_channels

        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        self.kernel_size = (kernel_h, kernel_w)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # Base convolution
        base_out = self.base_layer(x)

        # LoRA via im2col (unfold)
        # x: [B, C_in, H, W]
        B, C_in, H, W = x.shape

        # Extract patches
        x_unfold = torch.nn.functional.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.base_layer.padding,
            stride=self.base_layer.stride,
            dilation=self.base_layer.dilation,
        )  # [B, C_in*K_h*K_w, L] where L = H_out * W_out

        # Transpose for LoRA: [B, L, C_in*K_h*K_w]
        x_unfold = x_unfold.transpose(1, 2)

        # Flatten batch and spatial: [B*L, C_in*K_h*K_w]
        B, L, D = x_unfold.shape
        x_flat = x_unfold.reshape(B * L, D)

        # Apply LoRA: [B*L, C_out]
        lora_flat = self.lora(x_flat)

        # Reshape back: [B, L, C_out] -> [B, C_out, L]
        lora_out = lora_flat.reshape(B, L, self.out_channels).transpose(1, 2)

        # Fold back to spatial dimensions
        H_out, W_out = base_out.shape[2], base_out.shape[3]
        lora_out = lora_out.reshape(B, self.out_channels, H_out, W_out)

        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into base conv layer."""
        with torch.no_grad():
            # Compute LoRA weight: B @ A
            lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling

            # Reshape to conv2d weight format: [out_channels, in_channels, K_h, K_w]
            lora_weight = lora_weight.reshape(
                self.out_channels, self.in_channels, *self.kernel_size
            )

            # Add to base weight
            self.base_layer.weight.data += lora_weight

            # Zero out LoRA
            self.lora.lora_A.zero_()
            self.lora.lora_B.zero_()


def inject_lora_into_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
) -> int:
    """
    Inject LoRA adapters into target modules of a pretrained model.

    Args:
        model: Pretrained model to adapt
        target_modules: List of module types to adapt (e.g., ['Conv1d', 'Linear'])
                       If None, adapts all Linear and Conv layers
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout for LoRA path

    Returns:
        Number of LoRA parameters added

    Example:
        >>> model = PrimeKnet(...)
        >>> model.load_state_dict(torch.load('pretrained.th'))
        >>> num_lora_params = inject_lora_into_model(model, rank=8)
        >>> print(f"Added {num_lora_params} LoRA parameters")
    """
    if target_modules is None:
        target_modules = ["Linear", "Conv1d", "Conv2d"]

    lora_param_count = 0

    # Collect modules to replace (to avoid modifying dict during iteration)
    modules_to_replace = []

    for name, module in model.named_modules():
        module_type = type(module).__name__

        if module_type in target_modules:
            modules_to_replace.append((name, module))

    # Replace modules
    for name, module in modules_to_replace:
        # Get parent module and attribute name
        if "." in name:
            *parent_path, attr_name = name.split(".")
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
        else:
            parent = model
            attr_name = name

        # Create LoRA wrapper
        if isinstance(module, nn.Linear):
            lora_module = LinearWithLoRA(module, rank, alpha, dropout)
        elif isinstance(module, nn.Conv1d):
            lora_module = Conv1dWithLoRA(module, rank, alpha, dropout)
        elif isinstance(module, nn.Conv2d):
            lora_module = Conv2dWithLoRA(module, rank, alpha, dropout)
        else:
            continue

        # Replace module
        setattr(parent, attr_name, lora_module)

        # Count parameters
        lora_param_count += sum(p.numel() for p in lora_module.lora.parameters())

    return lora_param_count


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.

    Useful for creating optimizer that only updates LoRA weights.

    Args:
        model: Model with injected LoRA adapters

    Returns:
        List of LoRA parameters
    """
    lora_params = []

    for module in model.modules():
        if isinstance(module, (LinearWithLoRA, Conv1dWithLoRA, Conv2dWithLoRA)):
            lora_params.extend(module.lora.parameters())

    return lora_params


def merge_lora_weights(model: nn.Module):
    """
    Merge all LoRA weights into base model for efficient inference.

    After training, call this to eliminate LoRA overhead.

    Args:
        model: Model with injected LoRA adapters
    """
    for module in model.modules():
        if hasattr(module, "merge_weights"):
            module.merge_weights()
