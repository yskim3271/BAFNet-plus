"""State registry for explicit-state ONNX export.

Walks a model built from functional stateful convolutions, assigns a unique name
to each conv's state tensor, and provides ordered initialisation / dict
round-tripping so the export driver can wire every state as graph I/O.

Ported from LaCoSENet ``src/models/onnx_export/state_registry.py``; adjusted for
BAFNet+ module paths (functional layers live under
``src.models.streaming.onnx.functional_stateful``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class StateInfo:
    """Metadata for a single state tensor.

    Attributes:
        name: Unique name used for ONNX I/O.
        module_path: Dotted path to the owning module in the model hierarchy.
        shape: Expected shape (including the batch dim).
        dtype: Tensor dtype.
        init_fn: Callable ``(device, dtype) -> Tensor`` producing a fresh state.
    """

    name: str
    module_path: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    init_fn: Any


class StateRegistry:
    """Ordered registry of state tensors for a functional-stateful model.

    Collects stateful layers, names their states, and provides initialisation,
    indexing and list <-> dict conversion utilities.
    """

    def __init__(self) -> None:
        self._states: List[StateInfo] = []
        self._name_to_idx: Dict[str, int] = {}

    def register(
        self,
        name: str,
        module_path: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        init_fn: Any = None,
    ) -> int:
        """Register a state tensor.

        Args:
            name: Unique state name.
            module_path: Dotted path to the owning module.
            shape: Expected shape (including batch dim).
            dtype: Tensor dtype.
            init_fn: Optional ``(device, dtype) -> Tensor`` initialiser; defaults
                to a zero tensor of ``shape``.

        Returns:
            The registration index.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self._name_to_idx:
            raise ValueError(f"State '{name}' already registered")
        idx = len(self._states)
        self._states.append(
            StateInfo(
                name=name,
                module_path=module_path,
                shape=shape,
                dtype=dtype,
                init_fn=init_fn or (lambda dev, dt: torch.zeros(shape, device=dev, dtype=dt)),
            )
        )
        self._name_to_idx[name] = idx
        return idx

    def get_by_name(self, name: str) -> StateInfo:
        """Return the :class:`StateInfo` registered under ``name``."""
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Unknown state: {name}")
        return self._states[idx]

    def get_by_index(self, idx: int) -> StateInfo:
        """Return the :class:`StateInfo` at registration index ``idx``."""
        return self._states[idx]

    @property
    def num_states(self) -> int:
        """Number of registered states."""
        return len(self._states)

    @property
    def state_names(self) -> List[str]:
        """State names in registration order."""
        return [s.name for s in self._states]

    def init_all_states(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tensor]:
        """Initialise every state (zeros), substituting ``batch_size`` for dim 0.

        Args:
            batch_size: Batch size for the leading dimension.
            device: Device for the tensors.
            dtype: Dtype override (falls back to each state's recorded dtype).

        Returns:
            State tensors in registration order.
        """
        states: List[Tensor] = []
        for info in self._states:
            shape = (batch_size,) + tuple(info.shape[1:])
            states.append(torch.zeros(shape, device=device, dtype=dtype or info.dtype))
        return states

    def to_dict(self, states: List[Tensor]) -> Dict[str, Tensor]:
        """Map an ordered state list to a ``{name: tensor}`` dict."""
        if len(states) != len(self._states):
            raise ValueError(f"Expected {len(self._states)} states, got {len(states)}")
        return {info.name: states[i] for i, info in enumerate(self._states)}

    def from_dict(self, state_dict: Dict[str, Tensor]) -> List[Tensor]:
        """Map a ``{name: tensor}`` dict to an ordered state list."""
        states: List[Tensor] = []
        for info in self._states:
            if info.name not in state_dict:
                raise KeyError(f"Missing state: {info.name}")
            states.append(state_dict[info.name])
        return states

    def summary(self) -> str:
        """Return a human-readable listing of all registered states."""
        lines = [f"StateRegistry: {self.num_states} states"]
        total = 0
        for i, info in enumerate(self._states):
            elems = 1
            for d in info.shape:
                elems *= d
            total += elems
            lines.append(f"  [{i}] {info.name}: {info.shape} ({info.dtype})")
        lines.append(f"  Total elements: {total:,}")
        return "\n".join(lines)


def collect_states_from_model(
    model: torch.nn.Module,
    batch_size: int = 1,
    freq_size: int = 129,
) -> Tuple[StateRegistry, List[Tensor]]:
    """Collect every functional-stateful conv state from ``model``.

    Walks ``model.named_modules()`` for functional stateful convs and registers
    one state per module, named ``state_<dotted_path_with_underscores>``.

    Args:
        model: Model built from functional stateful layers.
        batch_size: Batch size for state initialisation.
        freq_size: Frequency size used by the 2D conv ``init_state`` calls (the
            conv input width before frequency padding). Pass the actual encoder
            output width for the real backbone.

    Returns:
        ``(registry, initial_states)`` where ``initial_states`` is the registry's
        zero-initialised list in registration order.
    """
    from src.models.streaming.onnx.functional_stateful import (
        FunctionalStatefulCausalConv2d,
        FunctionalStatefulConv1d,
        FunctionalStatefulConv2d,
    )

    registry = StateRegistry()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for name, module in model.named_modules():
        if isinstance(module, FunctionalStatefulConv1d):
            state = module.init_state(batch_size, device, dtype)
            registry.register(
                name=f"state_{name.replace('.', '_')}",
                module_path=name,
                shape=tuple(state.shape),
                dtype=dtype,
            )
        elif isinstance(module, (FunctionalStatefulConv2d, FunctionalStatefulCausalConv2d)):
            state = module.init_state(batch_size, freq_size, device, dtype)
            registry.register(
                name=f"state_{name.replace('.', '_')}",
                module_path=name,
                shape=tuple(state.shape),
                dtype=dtype,
            )

    initial_states = registry.init_all_states(batch_size, device, dtype)
    return registry, initial_states


__all__ = [
    "StateInfo",
    "StateRegistry",
    "collect_states_from_model",
]
