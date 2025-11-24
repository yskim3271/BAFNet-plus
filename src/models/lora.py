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
    Conv1d layer with LoRA adaptation using layer-wise decomposition.

    Uses layer-wise low-rank decomposition for memory efficiency:
    W_new = W_frozen + alpha * (B @ A)

    Reference: LoRA-C (https://arxiv.org/abs/2410.16954)
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

        # Check if grouped convolution
        self.groups = base_layer.groups
        self.is_grouped = self.groups > 1

        # Layer-wise LoRA parameters
        out_channels = base_layer.out_channels
        in_channels = base_layer.in_channels
        # Handle kernel_size (tuple or int)
        kernel_size = base_layer.kernel_size[0] if isinstance(base_layer.kernel_size, tuple) else base_layer.kernel_size

        if not self.is_grouped:
            # Layer-wise decomposition: A and B matrices
            # A: [rank, in_channels, kernel_size]
            # B: [out_channels, rank, 1]
            self.lora_A = nn.Parameter(torch.zeros(rank, in_channels, kernel_size))
            self.lora_B = nn.Parameter(torch.zeros(out_channels, rank, 1))

            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank

            # Dropout (applied to input, not LoRA bottleneck due to weight merge approach)
            # Note: Unlike LinearWithLoRA, we cannot apply dropout between A and B
            # because weights are merged before forward pass for efficiency.
            self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

            # Initialize
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            # Skip LoRA for grouped convolution
            self.lora_A = None
            self.lora_B = None
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.dropout = None

    def forward(self, x):
        # Skip LoRA for grouped convolution
        if self.is_grouped:
            return self.base_layer(x)

        # Compute LoRA weight update
        # A: [rank, in_channels, kernel_size]
        # B: [out_channels, rank, 1]

        # Reshape A for matmul: [rank, in_channels * kernel_size]
        A_flat = self.lora_A.view(self.rank, -1)

        # Reshape B for matmul: [out_channels, rank]
        B_flat = self.lora_B.squeeze(-1)

        # Compute LoRA weight: B @ A
        lora_weight = (B_flat @ A_flat).view(
            self.base_layer.weight.shape
        ) * self.scaling  # [out_channels, in_channels, kernel_size]

        # Combined weight
        combined_weight = self.base_layer.weight + lora_weight

        # Apply dropout to input if enabled
        if self.dropout is not None:
            x = self.dropout(x)

        # Single convolution with combined weight
        return torch.nn.functional.conv1d(
            x, combined_weight, self.base_layer.bias,
            self.base_layer.stride, self.base_layer.padding,
            self.base_layer.dilation, self.base_layer.groups
        )

    def merge_weights(self):
        """Merge LoRA weights into base conv layer."""
        if self.is_grouped:
            return  # Skip for grouped convolution

        with torch.no_grad():
            # Compute LoRA weight: B @ A
            A_flat = self.lora_A.view(self.rank, -1)
            B_flat = self.lora_B.squeeze(-1)

            lora_weight = (B_flat @ A_flat).view(
                self.base_layer.weight.shape
            ) * self.scaling

            # Add to base weight
            self.base_layer.weight.data += lora_weight

            # Zero out LoRA
            self.lora_A.zero_()
            self.lora_B.zero_()


class Conv2dWithLoRA(nn.Module):
    """
    Conv2d layer with LoRA adaptation using layer-wise decomposition.

    Uses layer-wise low-rank decomposition for memory efficiency:
    W_new = W_frozen + alpha * (B @ A)

    Reference: LoRA-C (https://arxiv.org/abs/2410.16954)
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

        # Check if grouped convolution
        self.groups = base_layer.groups
        self.is_grouped = self.groups > 1

        # Layer-wise LoRA parameters
        out_channels = base_layer.out_channels
        in_channels = base_layer.in_channels
        # Handle kernel_size (tuple or int)
        if isinstance(base_layer.kernel_size, tuple):
            kernel_h, kernel_w = base_layer.kernel_size
        else:
            kernel_h = kernel_w = base_layer.kernel_size

        if not self.is_grouped:
            # Layer-wise decomposition: A and B matrices
            # A: [rank, in_channels, kernel_h, kernel_w]
            # B: [out_channels, rank, 1, 1]
            self.lora_A = nn.Parameter(torch.zeros(rank, in_channels, kernel_h, kernel_w))
            self.lora_B = nn.Parameter(torch.zeros(out_channels, rank, 1, 1))

            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank

            # Dropout (applied to input, not LoRA bottleneck due to weight merge approach)
            # Note: Unlike LinearWithLoRA, we cannot apply dropout between A and B
            # because weights are merged before forward pass for efficiency.
            self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

            # Initialize
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            # Skip LoRA for grouped convolution
            self.lora_A = None
            self.lora_B = None
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.dropout = None

    def forward(self, x):
        # Skip LoRA for grouped convolution
        if self.is_grouped:
            return self.base_layer(x)

        # Compute LoRA weight update
        # A: [rank, in_channels, kernel_h, kernel_w]
        # B: [out_channels, rank, 1, 1]

        # Reshape A for matmul: [rank, in_channels * kernel_h * kernel_w]
        A_flat = self.lora_A.view(self.rank, -1)

        # Reshape B for matmul: [out_channels, rank]
        B_flat = self.lora_B.squeeze(-1).squeeze(-1)

        # Compute LoRA weight: B @ A
        lora_weight = (B_flat @ A_flat).view(
            self.base_layer.weight.shape
        ) * self.scaling  # [out_channels, in_channels, kernel_h, kernel_w]

        # Combined weight
        combined_weight = self.base_layer.weight + lora_weight

        # Apply dropout to input if enabled
        if self.dropout is not None:
            x = self.dropout(x)

        # Single convolution with combined weight
        return torch.nn.functional.conv2d(
            x, combined_weight, self.base_layer.bias,
            self.base_layer.stride, self.base_layer.padding,
            self.base_layer.dilation, self.base_layer.groups
        )

    def merge_weights(self):
        """Merge LoRA weights into base conv layer."""
        if self.is_grouped:
            return  # Skip for grouped convolution

        with torch.no_grad():
            # Compute LoRA weight: B @ A
            A_flat = self.lora_A.view(self.rank, -1)
            B_flat = self.lora_B.squeeze(-1).squeeze(-1)

            lora_weight = (B_flat @ A_flat).view(
                self.base_layer.weight.shape
            ) * self.scaling

            # Add to base weight
            self.base_layer.weight.data += lora_weight

            # Zero out LoRA
            self.lora_A.zero_()
            self.lora_B.zero_()


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
        if isinstance(lora_module, LinearWithLoRA):
            lora_param_count += sum(p.numel() for p in lora_module.lora.parameters())
        elif isinstance(lora_module, (Conv1dWithLoRA, Conv2dWithLoRA)):
            # Skip grouped convolutions (lora_A/lora_B are None)
            if lora_module.lora_A is not None and lora_module.lora_B is not None:
                lora_param_count += lora_module.lora_A.numel() + lora_module.lora_B.numel()

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
        if isinstance(module, LinearWithLoRA):
            # LinearWithLoRA uses self.lora sub-module
            lora_params.extend(module.lora.parameters())
        elif isinstance(module, (Conv1dWithLoRA, Conv2dWithLoRA)):
            # Conv LoRA uses self.lora_A and self.lora_B directly
            # Skip grouped convolutions (lora_A/lora_B are None)
            if module.lora_A is not None and module.lora_B is not None:
                lora_params.append(module.lora_A)
                lora_params.append(module.lora_B)

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


def unfreeze_critical_scalar_parameters(model: nn.Module, verbose: bool = True) -> int:
    """
    Unfreeze critical scalar parameters (scale, beta, slope, etc.)

    These parameters are essential for residual connections and gating mechanisms.
    In models like PrimeKnet, they control residual paths, attention scaling, etc.
    If frozen, the model's residual/gating mechanisms will be disabled.

    Typical parameters:
    - Group_Prime_Kernel_FFN.scale: Projection scaling
    - Channel_Attention_Block.beta: Skip connection weight
    - TS_BLOCK.beta_t, beta_f: Time/Freq stage weights
    - LearnableSigmoid_2d.slope: Sigmoid slope control

    Args:
        model: Model with potential scalar parameters
        verbose: Print unfrozen parameters

    Returns:
        Number of scalar parameters unfrozen
    """
    count = 0
    critical_keywords = ['scale', 'beta', 'slope', 'gamma', 'alpha']

    for name, param in model.named_parameters():
        # Check if parameter name contains critical keywords
        if any(keyword in name.lower() for keyword in critical_keywords):
            # Additional check: should be small (scalar or small tensor)
            # Heuristic: true scalars/gating params have < 1000 elements
            if param.numel() < 1000:
                param.requires_grad = True
                count += 1
                if verbose:
                    print(f"  [Critical] Unfroze {name}: shape={list(param.shape)}, numel={param.numel()}")

    return count


def unfreeze_normalization_layers(model: nn.Module, verbose: bool = True) -> dict:
    """
    Unfreeze normalization layers (InstanceNorm, LayerNorm, BatchNorm, custom LayerNorm1d).

    Important for domain adaptation in transfer learning.
    Handles both standard PyTorch norms and custom implementations (e.g., PrimeKnet's LayerNorm1d).

    Args:
        model: Model with normalization layers
        verbose: Print unfrozen modules

    Returns:
        dict with counts of each norm type
    """
    # Try to import custom LayerNorm1d from primeknet
    LayerNorm1d = None
    try:
        from src.models.primeknet import LayerNorm1d as CustomLayerNorm1d
        LayerNorm1d = CustomLayerNorm1d
    except ImportError:
        pass

    stats = {
        'instance_norm': 0,
        'layer_norm': 0,
        'batch_norm': 0,
        'custom_ln1d': 0
    }

    for name, module in model.named_modules():
        unfrozen = False
        module_type = None

        if isinstance(module, nn.InstanceNorm2d):
            for param in module.parameters():
                param.requires_grad = True
            stats['instance_norm'] += 1
            module_type = 'InstanceNorm2d'
            unfrozen = True

        elif isinstance(module, (nn.LayerNorm,)):
            for param in module.parameters():
                param.requires_grad = True
            stats['layer_norm'] += 1
            module_type = 'LayerNorm'
            unfrozen = True

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for param in module.parameters():
                param.requires_grad = True
            stats['batch_norm'] += 1
            module_type = 'BatchNorm'
            unfrozen = True

        elif LayerNorm1d is not None and isinstance(module, LayerNorm1d):
            for param in module.parameters():
                param.requires_grad = True
            stats['custom_ln1d'] += 1
            module_type = 'Custom LayerNorm1d'
            unfrozen = True

        if unfrozen and verbose:
            num_params = sum(p.numel() for p in module.parameters())
            print(f"  [Norm] Unfroze {module_type} at '{name}': {num_params} params")

    return stats


def get_depthwise_parameters(model: nn.Module) -> list:
    """
    Get all depthwise convolution parameters from a model.

    Depthwise convolution: groups == in_channels (each channel has its own kernel).
    Useful for Track 2 strategy where depthwise layers get full fine-tuning
    with differential learning rate.

    **CRITICAL**: When using these parameters in optimizer:
        - Set weight_decay=0.0 to preserve pretrained spatial patterns
        - Use low learning rate (e.g., 1e-5, ~10x lower than LoRA LR)
        - Using weight_decay > 0 will cause catastrophic forgetting!

    Args:
        model: Model with potential depthwise convolutions

    Returns:
        List of depthwise parameters

    Example:
        >>> dw_params = get_depthwise_parameters(model)
        >>> optimizer = torch.optim.AdamW([
        ...     {'params': lora_params, 'lr': 1e-4, 'weight_decay': 0.01},
        ...     {'params': dw_params, 'lr': 1e-5, 'weight_decay': 0.0},  # No decay!
        ... ])
    """
    dw_params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            # Depthwise: groups == in_channels and groups > 1
            if module.groups == module.in_channels and module.groups > 1:
                dw_params.extend(module.parameters())
    return dw_params


def get_conv_transpose_parameters(model: nn.Module, verbose: bool = True) -> list:
    """
    Get all ConvTranspose parameters.

    ConvTranspose (upsampling) layers are not supported by standard LoRA,
    so we handle them with full fine-tuning (typically with low LR).

    Args:
        model: Model with potential ConvTranspose layers
        verbose: Print found modules

    Returns:
        List of ConvTranspose parameters
    """
    params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
            module_params = list(module.parameters())
            params.extend(module_params)
            if verbose:
                num_params = sum(p.numel() for p in module_params)
                print(f"  [ConvTranspose] Found {module.__class__.__name__} at '{name}': {num_params} params")
    return params


def get_normalization_parameters(model: nn.Module) -> list:
    """
    Get all normalization layer parameters.

    Includes: InstanceNorm, LayerNorm, BatchNorm, and custom LayerNorm1d.

    Returns:
        List of normalization parameters
    """
    # Try to import custom LayerNorm1d
    LayerNorm1d = None
    try:
        from src.models.primeknet import LayerNorm1d as CustomLayerNorm1d
        LayerNorm1d = CustomLayerNorm1d
    except ImportError:
        pass

    norm_params = []
    for module in model.modules():
        if isinstance(module, (nn.InstanceNorm2d, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            norm_params.extend(module.parameters())
        elif LayerNorm1d is not None and isinstance(module, LayerNorm1d):
            norm_params.extend(module.parameters())

    return norm_params


def get_critical_scalar_parameters(model: nn.Module) -> list:
    """
    Get critical scalar parameters (scale, beta, slope, etc.).

    Returns:
        List of critical scalar parameters
    """
    scalar_params = []
    critical_keywords = ['scale', 'beta', 'slope', 'gamma', 'alpha']

    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in critical_keywords):
            if param.numel() < 1000:  # Heuristic for scalars
                scalar_params.append(param)

    return scalar_params


def inject_lora_with_strategy(
    model: nn.Module,
    strategy: str = 'track1',
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    verbose: bool = True
) -> dict:
    """
    PrimeKnet-aware LoRA injection with strategy-based parameter handling.

    Handles:
    - Pointwise vs Depthwise convolutions
    - Critical scalar parameters (scale, beta, slope) - ALWAYS unfrozen
    - Custom normalization layers (LayerNorm1d, InstanceNorm2d) - ALWAYS unfrozen
    - ConvTranspose2d layers - Full fine-tuning with low LR

    Strategies:
        'track1': PW-LoRA + DW-Freeze
            - Pointwise: LoRA adaptation
            - Depthwise: Frozen (preserves pre-trained spatial patterns)
            - Best for: Baseline, safe fine-tuning, quick experiments

        'track2': PW-LoRA + DW-Full + Differential LR
            - Pointwise: LoRA adaptation
            - Depthwise: Full fine-tuning (with very low LR to protect pre-trained)
            - Best for: Maximum performance, production use
            - Note: Requires differential LR in optimizer (dw_lr ~10x lower)

    Args:
        model: Pre-trained model to adapt
        strategy: 'track1' or 'track2'
        rank: LoRA rank for pointwise convolutions
        alpha: LoRA alpha for pointwise (typically 2 * rank)
        dropout: Dropout for LoRA path
        target_modules: Module types to consider (default: ['Linear', 'Conv1d', 'Conv2d'])
        verbose: Print detailed statistics

    Returns:
        dict with parameter statistics:
            - 'lora_params': Number of LoRA parameters
            - 'dw_params': Number of depthwise parameters (Track 2 only)
            - 'conv_transpose_params': Number of ConvTranspose parameters
            - 'scalar_params': Number of critical scalar parameters
            - 'norm_params': Number of normalization parameters
            - 'total_params': Total model parameters
            - 'trainable_params': Total trainable parameters
            - 'trainable_ratio': Percentage of trainable parameters
    """
    if target_modules is None:
        target_modules = ['Linear', 'Conv1d', 'Conv2d']

    if verbose:
        print(f"\n{'='*70}")
        print(f"[LoRA Strategy] Applying {strategy.upper()}")
        print(f"{'='*70}\n")

    # ========================================================================
    # Step 1: Apply standard LoRA injection (handles pointwise + grouped)
    # ========================================================================
    if verbose:
        print(f"[Step 1] Injecting LoRA (rank={rank}, alpha={alpha})...")

    lora_param_count = inject_lora_into_model(
        model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )

    if verbose:
        print(f"  Added {lora_param_count:,} LoRA parameters\n")

    # ========================================================================
    # Step 2: Strategy-specific handling of Depthwise convolutions
    # ========================================================================
    if verbose:
        print(f"[Step 2] Applying {strategy} strategy for depthwise convolutions...")

    dw_param_count = 0

    if strategy == 'track1':
        # Track 1: Depthwise stays frozen (already handled by inject_lora_into_model)
        if verbose:
            print("  Depthwise convolutions: FROZEN (preserves pre-trained spatial patterns)")

    elif strategy == 'track2':
        # Track 2: Unfreeze depthwise for full fine-tuning
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # Check if depthwise
                if module.groups == module.in_channels and module.groups > 1:
                    for param in module.parameters():
                        param.requires_grad = True
                    dw_param_count += sum(p.numel() for p in module.parameters())

        if verbose:
            print(f"  Depthwise convolutions: UNFROZEN for full fine-tuning")
            print(f"  Depthwise parameters: {dw_param_count:,}")
            print(f"\n  ⚠️  CRITICAL: Optimizer Configuration for Track2")
            print(f"      - Use differential LR: dw_lr ~10x lower than lora_lr (e.g., 1e-5 vs 1e-4)")
            print(f"      - Set weight_decay=0.0 for depthwise params to PRESERVE pretrained features!")
            print(f"      - Using weight_decay > 0 will cause catastrophic forgetting (filters → 0)")
            print(f"      Example:")
            print(f"        {{'params': dw_params, 'lr': 1e-5, 'weight_decay': 0.0}}")

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'track1' or 'track2'")

    if verbose:
        print()

    # ========================================================================
    # Step 3: Handle ConvTranspose2d (upsampling layers)
    # ========================================================================
    if verbose:
        print("[Step 3] Handling ConvTranspose layers...")

    conv_transpose_param_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
            for param in module.parameters():
                param.requires_grad = True
            module_params = sum(p.numel() for p in module.parameters())
            conv_transpose_param_count += module_params
            if verbose:
                print(f"  [ConvTranspose] Unfroze {module.__class__.__name__} at '{name}': {module_params} params")

    if conv_transpose_param_count == 0 and verbose:
        print("  No ConvTranspose layers found")

    if verbose:
        print()

    # ========================================================================
    # Step 4: Unfreeze Critical Scalar Parameters (CRITICAL!)
    # ========================================================================
    if verbose:
        print("[Step 4] Unfreezing critical scalar parameters (scale/beta/slope)...")

    scalar_param_count = unfreeze_critical_scalar_parameters(model, verbose=verbose)

    if verbose:
        print(f"  Total critical scalar parameters unfrozen: {scalar_param_count}\n")

    # ========================================================================
    # Step 5: Unfreeze Normalization Layers
    # ========================================================================
    if verbose:
        print("[Step 5] Unfreezing normalization layers...")

    norm_stats = unfreeze_normalization_layers(model, verbose=verbose)

    # Count normalization parameters
    norm_param_count = 0
    for param in get_normalization_parameters(model):
        if param.requires_grad:
            norm_param_count += param.numel()

    if verbose:
        print(f"  Total normalization modules: {sum(norm_stats.values())}")
        print(f"  Total normalization parameters: {norm_param_count:,}\n")

    # ========================================================================
    # Step 6: Final Statistics
    # ========================================================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = 100 * trainable_params / total_params

    stats = {
        'lora_params': lora_param_count,
        'dw_params': dw_param_count,
        'conv_transpose_params': conv_transpose_param_count,
        'scalar_params': scalar_param_count,
        'norm_params': norm_param_count,
        'norm_stats': norm_stats,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_ratio
    }

    if verbose:
        print(f"{'='*70}")
        print(f"[Summary]")
        print(f"{'='*70}")
        print(f"Strategy: {strategy.upper()}")
        print(f"\nParameter Breakdown:")
        print(f"  LoRA parameters:           {lora_param_count:>12,}")
        if strategy == 'track2':
            print(f"  Depthwise parameters:      {dw_param_count:>12,} (Full FT)")
        else:
            print(f"  Depthwise parameters:      {dw_param_count:>12,} (Frozen)")
        print(f"  ConvTranspose parameters:  {conv_transpose_param_count:>12,}")
        print(f"  Critical scalars:          {scalar_param_count:>12}")
        print(f"  Normalization parameters:  {norm_param_count:>12,}")
        print(f"    - InstanceNorm2d:        {norm_stats['instance_norm']:>12} modules")
        print(f"    - LayerNorm:             {norm_stats['layer_norm']:>12} modules")
        print(f"    - BatchNorm:             {norm_stats['batch_norm']:>12} modules")
        print(f"    - Custom LayerNorm1d:    {norm_stats['custom_ln1d']:>12} modules")
        print(f"\n  Total parameters:          {total_params:>12,}")
        print(f"  Trainable parameters:      {trainable_params:>12,} ({trainable_ratio:.2f}%)")
        print(f"{'='*70}\n")

    return stats
