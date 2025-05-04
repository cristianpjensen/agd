import torch.nn as nn
from typing import Dict, Any

from .adapter import Adapter, ModelWithAdapters


class AdapterConfig:
    def __init__(self, adapter: type[Adapter], kwargs: dict[str, Any]):
        self.adapter = adapter
        self.kwargs = kwargs

    def construct_adapter(self, block: nn.Module) -> Adapter:
        return self.adapter(block, **self.kwargs)


def inject_adapters(
    model: nn.Module,
    adapters: Dict[str, AdapterConfig],
) -> ModelWithAdapters:
    """Injects adapters into a model.

    Args:
        model: The model to inject the adapters into.
        adapters: A dictionary mapping target paths to AdapterConfig, which contains the adapter
            class and its keyword arguments (next to the block).
    """

    for target, adapter_config in adapters.items():
        # Get target block and its parent
        atoms = target.split(".")
        block = model.get_submodule(target)
        parent_block = model.get_submodule(".".join(atoms[:-1]))

        setattr(parent_block, atoms[-1], adapter_config.construct_adapter(block))
    
    return ModelWithAdapters(model)
