import torch.nn as nn


class ModelWithAdapters(nn.Module):
    """
    Handles adapters in the model, such as freezing the base model and setting keyword arguments for
    the adapters---external conditioning cannot be done another way, because the model is not "aware"
    there are adapters and we cannot change the internal `forward` of the model.

    Example:
    ```python
    # Input data, where `external_cond` is the external conditioning for the adapter, which the block
    # does not receive in the original model
    external_cond = torch.randn(4, 1280)
    x = torch.randn(4, 3, 256, 256)

    # `model` contains `Adapter` modules and we want to train the adapters with a frozen base model
    model = ModelWithAdapters(model)
    model.train_adapters()

    # Forward pass where we first have to pass external conditioning to the adapter
    model.set_adapter_kwargs(adapter_cond=external_cond)
    y = model(x)
    ```
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def named_adapter_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, Adapter):
                yield name, module

    def adapter_modules(self):
        for _, module in self.named_adapter_modules():
            yield module

    def named_adapter_parameters(self):
        for name, module in self.named_adapter_modules():
            for param_name, param in module.named_parameters():
                if param_name.startswith("block."):
                    continue

                yield f"{name}.{param_name}", param

    def adapter_parameters(self):
        for _, param in self.named_adapter_parameters():
            yield param

    def adapter_state_dicts(self):
        state_dicts = {}
        for name, module in self.named_adapter_modules():
            state_dicts[name] = { k: v for k, v in module.state_dict().items() if not k.startswith("block.") }
        
        return state_dicts

    def load_adapter_state_dicts(self, state_dicts: dict):
        if not set(state_dicts.keys()) == set(dict(self.named_adapter_modules()).keys()):
            raise ValueError("State dicts do not match the adapters in the model")

        for name, state_dict in state_dicts.items():
            incompatible = self.model.get_submodule(name).load_state_dict(state_dict, strict=False)
            missing_keys = [k for k in incompatible.missing_keys if not k.startswith("block.")]

            if len(incompatible.unexpected_keys) > 0:
                raise ValueError(f"Unexpected keys in adapter {name}: {incompatible.unexpected_keys}")

            if len(missing_keys) > 0:
                raise ValueError(f"Missing keys in adapter {name}: {missing_keys}")

    def set_adapter_kwargs(self, **kwargs):
        for adapter in self.adapter_modules():
            adapter.set_kwargs(**kwargs)

    def update_adapter_kwargs(self, **kwargs):
        for adapter in self.adapter_modules():
            adapter.update_kwargs(**kwargs)

    def freeze_base_model(self):
        """Freeze the base model and enable gradient calculation on adapter parameters."""

        self.model.requires_grad_(False)
        for param in self.adapter_parameters():
            param.requires_grad_(True)

    def __getattr__(self, name):
        if name == "model":
            return super().__getattr__(name)

        return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class Adapter(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.kwargs = {}

    def __getattr__(self, name):
        if name in ["block", "kwargs"]:
            return super().__getattr__(name)

        # If it does not already exist in the adapter, try to get it from the block
        if hasattr(self.block, name):
            return self.block.__getattr__(name)

        return super().__getattr__(name)

    def set_kwargs(self, **kwargs):
        self.kwargs = kwargs

    def update_kwargs(self, **kwargs):
        self.kwargs.update(kwargs)

    def forward(self):
        raise NotImplementedError


class ExampleResidualLinearCondAdapter(Adapter):
    """Example of an adapter that makes use of external conditioning."""

    def __init__(self, block: nn.Module, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__(block)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(cond_dim, out_dim)

    def forward(self, *args, **kwargs):
        if self.kwargs is None:
            raise ValueError(f"kwargs not set in {self.__class__.__name__}")

        if "adapter_cond" not in self.kwargs:
            raise ValueError(f"adapter_cond not in kwargs in {self.__class__.__name__}")

        adapter_cond = self.kwargs["adapter_cond"]
        block_out = self.block(*args, **kwargs)
        block_out[0] = block_out[0] + self.linear1(args[0]) + self.linear2(adapter_cond)
        return block_out
