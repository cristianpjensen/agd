from typing import Tuple

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline
from accelerate.utils import set_seed
from tqdm import tqdm
import argparse
import random
import os

from .adapter import get_pipeline
from agd.utils import (
    to_cpu,
    remove_none,
    remove_uncond_dim,
    default_disable_tqdm,
)
from agd.utils.training_loop import save_config
from .adapter import SUPPORTED_MODELS


@default_disable_tqdm
def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # Create output directory and save config
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(vars(args), args.output_dir)

    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()] * args.num_per_prompt

    # Load model
    pipe = get_pipeline(args.base_model).to(device)
    if hasattr(pipe, "unet"):
        wrapper = SaveCFGTrajectoryUnetWrapper(pipe.unet, args.inference_steps)
        pipe.unet = wrapper
    elif hasattr(pipe, "transformer"):
        wrapper = SaveCFGTrajectoryUnetWrapper(pipe.transformer, args.inference_steps)
        pipe.transformer = wrapper

    # Save step -> timestep mapping
    pipe.scheduler.set_timesteps(args.inference_steps)
    torch.save(pipe.scheduler.timesteps, os.path.join(args.output_dir, "timesteps.pt"))

    # Save shape of latent space
    torch.save(
        [wrapper.denoiser.config.in_channels, wrapper.denoiser.config.sample_size, wrapper.denoiser.config.sample_size],
        os.path.join(args.output_dir, "latent_shape.pt"),
    )

    n_sampled = 0
    for prompt in tqdm(prompts, disable=False):
        guidance_scale = random.uniform(args.min_guidance_scale, args.max_guidance_scale)
        wrapper.guidance_scale = guidance_scale

        pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=args.inference_steps,
        )

        # Save trajectory and conditioning variables
        sample_dir = os.path.join(args.output_dir, str(n_sampled).zfill(6))
        os.mkdir(sample_dir)

        encoder_hidden_states = encode_prompt(pipe, prompt, device)
        torch.save(
            {
                "last_hidden_state": encoder_hidden_states.cpu(),
                "cfg_scale": guidance_scale,
                "additional_model_kwargs": wrapper.additional_kwargs,
            },
            os.path.join(sample_dir, "conditioning.pt"),
        )

        # Save and reset
        trajectory = wrapper.trajectories.numpy()
        fp = np.memmap(
            os.path.join(os.path.join(sample_dir, "trajectory.npy")),
            dtype=np.float16,
            mode="w+",
            shape=trajectory.shape,
        )
        fp[:] = trajectory[:]
        fp.flush()

        wrapper.reset()
        n_sampled += 1


def encode_prompt(pipe: StableDiffusionPipeline, prompt: str, device: torch.device, batch_size=1) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(pipe, (StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline)):
        prompt_embeds, _, _, _ = pipe.encode_prompt(prompt, prompt)
    elif isinstance(pipe, (StableDiffusionPipeline, StableDiffusionControlNetPipeline)):
        text_input_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
    
        prompt_embeds = pipe.text_encoder(text_input_ids.to(device))
        prompt_embeds = prompt_embeds.last_hidden_state
    else:
        raise ValueError(f"pipe class `{pipe.__class__.__name__}` not recognized")
        
    return prompt_embeds.repeat(batch_size, 1, 1)


class SaveCFGTrajectoryUnetWrapper:
    def __init__(self, denoiser, num_steps: int):
        self.denoiser = denoiser
        self.channels = denoiser.config.in_channels
        self.sample_size = denoiser.config.sample_size

        self.guidance_scale = -1
        self.num_steps = num_steps
        self.reset()

    def __call__(self, *args, **kwargs):
        output = self.denoiser(*args, **kwargs)

        # Save model input
        if len(args) > 0:
            latent_input = args[0]
        elif "sample" in kwargs:
            latent_input = kwargs["sample"]
        elif "hidden_states" in kwargs:
            latent_input = kwargs["hidden_states"]
        else:
            raise ValueError("No latent input found")

        self.trajectories[self.step, 0] = remove_uncond_dim(latent_input).cpu()

        # Save model CFG output
        noise_pred = output[0] if isinstance(output, tuple) else output.sample

        # Remove learned variance if present
        if noise_pred.shape[1] == 2 * self.channels:
            noise_pred = noise_pred[:, :self.channels]

        # Do classifier-free guidance and save output
        eps_uncond, eps_cond = noise_pred.chunk(2)
        eps_cfg = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        self.trajectories[self.step, 1] = eps_cfg.cpu()

        # Save additional arguments that need to be passed to the model during training
        if self.additional_kwargs is None:
            self.additional_kwargs = kwargs
            # Remove information that is already saved or unnecessary
            for key in [
                "sample",
                "timestep",
                "class_labels",
                "return_dict",
            ]:
                self.additional_kwargs.pop(key, None)

            self.additional_kwargs = remove_none(self.additional_kwargs)
            self.additional_kwargs = remove_uncond_dim(self.additional_kwargs)
            self.additional_kwargs = to_cpu(self.additional_kwargs)

        # Increase step
        self.step += 1

        return output

    def reset(self):
        self.additional_kwargs = None
        self.trajectories = torch.zeros(self.num_steps, 2, self.channels, self.sample_size, self.sample_size)
        self.step = 0

    def __getattr__(self, name):
        return getattr(self.denoiser, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, choices=SUPPORTED_MODELS, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--prompt-file", type=str, required=True, help="File with new line separated prompts.")
    parser.add_argument("--num-per-prompt", type=int, default=1, help="Number of samples per prompt.")
    parser.add_argument("--inference-steps", type=int, default=999)
    parser.add_argument("--min-guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-guidance-scale", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
