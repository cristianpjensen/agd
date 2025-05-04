from typing import Optional

import torch
import os

from agd.dit.DiT.models import DiT, DiT_XL_2
from agd.dit.DiT.download import find_model

from agd.models import ADAPTER_MODELS
from agd.adapters import AdapterConfig, ModelWithAdapters
from agd.adapters.inject_adapters import inject_adapters
from agd.utils.training_loop import get_config
from agd.utils import CatchCalls


def get_pretrained_dit(image_size: int) -> DiT:
    assert image_size in [256, 512], "only 256 and 512 image sizes are supported by DiT"

    model = DiT_XL_2(input_size=image_size // 8, num_classes=1000)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.y_embedder.dropout_prob = 0

    return model


def get_untrained_adapter_dit(
    image_size: int,
    hidden_dim=256,
    dropout=0.0,
    use_timestep=False,
    use_class_label=False,
    zero_init=True,
    arch="additive",
    cfg_mult=1,
) -> ModelWithAdapters:
    model = get_pretrained_dit(image_size)

    y_embedder = lambda y: model.y_embedder(y, False)
    y_emb_size = model.y_embedder.embedding_table.embedding_dim
    t_embedder = lambda t: model.t_embedder(t)
    t_emb_size = model.t_embedder.mlp[-1].out_features

    adapters = {}
    for i, block in enumerate(model.blocks):
        adapters[f"blocks.{i}.attn"] = AdapterConfig(
            adapter=ADAPTER_MODELS[arch],
            kwargs=dict(
                input_dim=block.adaLN_modulation[1].in_features,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_class_label=use_class_label,
                class_embedder=y_embedder if use_class_label else None,
                class_dim=y_emb_size if use_class_label else None,
                use_timestep=use_timestep,
                timestep_embedder=t_embedder if use_timestep else None,
                timestep_dim=t_emb_size if use_timestep else None,
                zero_init=zero_init,
                cfg_mult=cfg_mult,
            )
        )

    return inject_adapters(model, adapters)


def get_trained_adapter_dit(device: torch.device, dir: os.PathLike, ckpt: Optional[int]=None, use_adapter=True):
    config = get_config(dir)

    if not use_adapter:
        return get_pretrained_dit(config["image_size"]).to(device)

    model = get_untrained_adapter_dit(
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        image_size=config["image_size"],
        use_timestep=config["use_timestep"],
        use_class_label=config["use_class_label"],
        zero_init=config["zero_init"],
        arch=config["arch"],
        cfg_mult=config["cfg_mult"],
    ).to(device)

    assert ckpt is not None

    sd = torch.load(
        os.path.join(dir, "checkpoints", f"{ckpt:07d}.pt"),
        map_location=device,
        weights_only=False,
    )["model"]
    model.load_adapter_state_dicts(sd)
    
    if not config["use_timestep"]:
        return model

    def update_timestep(x, t, y):
        model.update_adapter_kwargs(timestep=t)

    model.forward = CatchCalls(model.forward, update_timestep)
    return model
