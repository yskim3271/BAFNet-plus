"""Model and checkpoint I/O utilities.

Handles dynamic model loading by module/class name, checkpoint deserialization,
model state snapshot/restore helpers, and a small attribute-access wrapper for
nested config dictionaries extracted from checkpoints.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Any, Dict

import torch


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def load_model(model_lib: str, model_class_name: str, model_params: Dict[str, Any], device: str = 'cuda'):
    """
    Load model dynamically from models directory.

    Args:
        model_lib: Model library name (e.g., "backbone", "bafnet")
        model_class_name: Model class name (e.g., "Backbone")
        model_params: Model parameters dictionary
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Model instance loaded on specified device

    Example:
        >>> model = load_model("backbone", "Backbone", params, "cuda")
    """
    module = importlib.import_module(f"src.models.{model_lib}")
    model_class = getattr(module, model_class_name)
    model = model_class(**model_params)
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, chkpt_dir: str, chkpt_file: str, device: str = 'cuda'):
    """
    Load model checkpoint.

    Args:
        model: Model instance to load checkpoint into
        chkpt_dir: Directory containing checkpoint file
        chkpt_file: Checkpoint filename (e.g., "best.th", "checkpoint.th")
        device: Device for loading checkpoint

    Returns:
        Model with loaded checkpoint

    Example:
        >>> model = load_checkpoint(model, "outputs/exp", "best.th", "cuda")
    """
    chkpt_path = os.path.join(chkpt_dir, chkpt_file)
    chkpt = torch.load(chkpt_path, map_location=device, weights_only=False)
    model.load_state_dict(chkpt['model'])
    return model


class ConfigDict:
    """
    Convert nested dictionary to object with attribute access.

    Useful for converting checkpoint args to object-like structure.

    Example:
        >>> config = ConfigDict({'model': {'name': 'test', 'params': {'lr': 0.01}}})
        >>> config.model.name  # 'test'
        >>> config.model.params.lr  # 0.01
        >>> config.to_dict()  # {'model': {'name': 'test', 'params': {'lr': 0.01}}}
    """

    def __init__(self, d: Dict[str, Any]):
        for key, value in d.items():
            setattr(self, key, value if not isinstance(value, dict) else ConfigDict(value))

    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfigDict back to regular dict for ** unpacking"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_model_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load model configuration from checkpoint file.

    This function extracts the model configuration (args.model) from a saved checkpoint,
    allowing BAFNet to automatically configure mapping/masking submodels without
    requiring manual parameter specification.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., "outputs/exp/best.th")

    Returns:
        Dictionary containing model configuration with keys:
            - model_lib: Model library name (e.g., "backbone")
            - model_class: Model class name (e.g., "Backbone")
            - param: Dictionary of model parameters

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint doesn't contain 'args' or 'args.model'

    Example:
        >>> config = load_model_config_from_checkpoint("outputs/backbone_exp/best.th")
        >>> print(config['model_lib'])  # "backbone"
        >>> print(config['param']['dense_channel'])  # 64
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'args' not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'args' field: {checkpoint_path}")

    args = checkpoint['args']

    if not hasattr(args, 'model'):
        raise KeyError(f"Checkpoint args does not contain 'model' field: {checkpoint_path}")

    model_config = args.model

    config = {
        'model_lib': model_config.model_lib,
        'model_class': model_config.model_class,
        'param': dict(model_config.param)
    }

    return config
