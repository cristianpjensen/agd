from typing import Optional

import torch
from diffusers import (
    UNet2DConditionModel,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    SD3Transformer2DModel,
    ControlNetModel,
)
from typing import Optional
import os

from agd.utils import CatchCalls
from agd.utils.training_loop import get_config
from agd.adapters.inject_adapters import inject_adapters
from agd.models import ADAPTER_MODELS
from agd.adapters import AdapterConfig, ModelWithAdapters


SUPPORTED_MODELS = [
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-3-medium-diffusers",
]


def get_pipeline(model_name: str, denoiser=None, controlnet=None):
    match model_name:
        case "stabilityai/stable-diffusion-2-1":
            if controlnet is None:
                return StableDiffusionPipeline.from_pretrained(model_name)
            else:
                return StableDiffusionControlNetPipeline.from_pretrained(model_name, controlnet=controlnet)

        case "stabilityai/stable-diffusion-xl-base-1.0":
            if controlnet is None:
                return StableDiffusionXLPipeline.from_pretrained(model_name)
            else:
                return StableDiffusionXLControlNetPipeline.from_pretrained(model_name, controlnet=controlnet)

        case "stabilityai/stable-diffusion-3-medium-diffusers":
            return StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)

        case _:
            raise ValueError(f"unsupported model: {model_name}")


def get_trained_adapter_pipeline(
    device: torch.device,
    result_dir: os.PathLike,
    ckpt: Optional[int]=None,
    use_adapter=True,
    controlnet: Optional[ControlNetModel]=None,
):
    config = get_config(result_dir)

    if not use_adapter:
        return get_pipeline(config["base_model"], controlnet=controlnet).to(device)

    assert ckpt is not None

    denoiser = get_untrained_adapter(
        model_name=config["base_model"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        use_timestep=config["use_timestep"],
        use_prompt=config["use_prompt"],
        zero_init=config["zero_init"],
        arch=config["arch"],
    ).to(device)

    pipe = get_pipeline(config["base_model"], controlnet=controlnet).to(device)

    # Move unet to cpu to avoid memory issues
    if hasattr(pipe, "unet"):
        pipe.unet.cpu()
    elif hasattr(pipe, "transformer"):
        pipe.transformer.cpu()
    else:
        raise ValueError("pipeline has no unet or transformer")

    sd = torch.load(
        os.path.join(result_dir, "checkpoints", f"{ckpt:07d}.pt"),
        map_location=denoiser.device,
        weights_only=False,
    )["model"]
    denoiser.load_adapter_state_dicts(sd)

    def update_timestep(sample, timestep, *args, **kwargs):
        t = torch.tensor([timestep] * sample.shape[0], device=sample.device)
        denoiser.update_adapter_kwargs(timestep=t)

    pipe.unet = CatchCalls(denoiser, update_timestep)

    return pipe


def get_untrained_adapter(
    model_name: str,
    hidden_dim=320,
    dropout=0.0,
    use_timestep=True,
    use_prompt=True,
    zero_init=False,
    arch="attention",
) -> ModelWithAdapters:
    kwargs = dict(
        model_name=model_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_timestep=use_timestep,
        use_prompt=use_prompt,
        zero_init=zero_init,
        arch=arch,
    )

    match model_name:
        case "stabilityai/stable-diffusion-2-1" | "stabilityai/stable-diffusion-xl-base-1.0":
            return get_untrained_adapter_unet(**kwargs)
        case "stabilityai/stable-diffusion-3-medium-diffusers":
            return get_untrained_adapter_transformer(**kwargs)
        case _:
            raise ValueError(f"unsupported model: {model_name}")


def get_untrained_adapter_transformer(
    model_name: str,
    hidden_dim=320,
    dropout=0.0,
    use_timestep=True,
    use_prompt=True,
    zero_init=False,
    arch="attention",
) -> ModelWithAdapters:
    denoiser = SD3Transformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.float16)

    input_dim = denoiser.config.num_attention_heads * denoiser.config.attention_head_dim
    t_embedder = lambda t: denoiser.time_text_embed.timestep_embedder(denoiser.time_text_embed.time_proj(t).to(dtype=denoiser.dtype))
    prompt_dim = 768

    common_kwargs = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_prompt=use_prompt,
        use_timestep=use_timestep,
        timestep_embedder=t_embedder,
        timestep_dim=input_dim,
        prompt_dim=prompt_dim,
        zero_init=zero_init,
    )

    adapters = {}

    for i in range(len(denoiser.transformer_blocks)):
        adapters[f"transformer_blocks.{i}.attn"] = AdapterConfig(
            adapter=ADAPTER_MODELS[arch],
            kwargs=common_kwargs,
        )

    return inject_adapters(denoiser, adapters)


def get_untrained_adapter_unet(
    model_name: str,
    hidden_dim=320,
    dropout=0.0,
    use_timestep=True,
    use_prompt=True,
    zero_init=False,
    arch="attention",
) -> ModelWithAdapters:
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

    channels = unet.config.block_out_channels
    layers_per_block = unet.config.layers_per_block
    transformer_layers = unet.config.transformer_layers_per_block

    if isinstance(transformer_layers, int):
        transformer_layers = [transformer_layers] * len(channels)

    down_block_types = unet.config.down_block_types
    up_block_types = unet.config.up_block_types
    prompt_dim = unet.config.cross_attention_dim

    t_embedder = lambda t: unet.time_embedding(unet.time_proj(t))
    t_emb_size = unet.time_embedding.linear_2.out_features

    common_kwargs = dict(
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_prompt=use_prompt,
        use_timestep=use_timestep,
        timestep_embedder=t_embedder,
        timestep_dim=t_emb_size,
        prompt_dim=prompt_dim,
        zero_init=zero_init,
    )

    adapters = {}

    # Down blocks
    for i, block_type in enumerate(down_block_types):
        if block_type != "CrossAttnDownBlock2D":
            continue

        for j in range(layers_per_block):
            for k in range(transformer_layers[i]):
                adapters[f"down_blocks.{i}.attentions.{j}.transformer_blocks.{k}.attn2"] = AdapterConfig(
                    adapter=ADAPTER_MODELS[arch],
                    kwargs={ "input_dim": channels[i], **common_kwargs },
                )

    # Mid block
    for k in range(transformer_layers[-1]):
        adapters[f"mid_block.attentions.0.transformer_blocks.{k}.attn2"] = AdapterConfig(
            adapter=ADAPTER_MODELS[arch],
            kwargs={ "input_dim": channels[-1], **common_kwargs },
        )

    # Up blocks
    for i, block_type in enumerate(up_block_types):
        if block_type != "CrossAttnUpBlock2D":
            continue

        for j in range(layers_per_block):
            for k in range(transformer_layers[::-1][i]):
                adapters[f"up_blocks.{i}.attentions.{j}.transformer_blocks.{k}.attn2"] = AdapterConfig(
                    adapter=ADAPTER_MODELS[arch],
                    kwargs={ "input_dim": channels[::-1][i], **common_kwargs },
                )

    return inject_adapters(unet, adapters)
