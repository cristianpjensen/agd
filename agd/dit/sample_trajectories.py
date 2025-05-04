import numpy as np
import torch
from accelerate.utils import set_seed
from tqdm import tqdm
import argparse
import random
import os

from agd.dit.DiT.diffusion import create_diffusion

from agd.dit.adapter import get_pretrained_dit
from agd.utils.training_loop import save_config


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # Create output directory and save config
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(vars(args), args.output_dir)

    # Load model
    model = get_pretrained_dit(args.image_size).to(device)
    diffusion = create_diffusion(str(args.inference_steps))
    latent_size = args.image_size // 8
    save_wrapper = SaveCFGTrajectoryUnetWrapper(model.forward_with_cfg, args.inference_steps, latent_size)
    model.forward_with_cfg = save_wrapper

    # Save shape of latent space and step -> timestep mapping
    torch.save([4, latent_size, latent_size], os.path.join(args.output_dir, "latent_shape.pt"))
    torch.save(list(range(diffusion.num_timesteps))[::-1], os.path.join(args.output_dir, "timesteps.pt"))

    class_labels = list(range(1000)) * args.num_per_class

    n_sampled = 0
    for class_label in tqdm(class_labels):
        guidance_scale = random.uniform(args.min_guidance_scale, args.max_guidance_scale)
        save_wrapper.guidance_scale = guidance_scale

        y = torch.tensor([class_label], device=device)
        y_null = torch.tensor([1000], device=device)
        y = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=guidance_scale)

        z = torch.randn(1, 4, latent_size, latent_size, device=device)
        z = torch.cat([z, z], dim=0)

        diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )

        # Save trajectory
        sample_dir = os.path.join(args.output_dir, str(n_sampled).zfill(6))
        os.mkdir(sample_dir)

        # Save conditioning variables
        torch.save({ "class_label": class_label, "cfg_scale": guidance_scale }, os.path.join(sample_dir, "conditioning.pt"))

        # Save and reset
        trajectory = save_wrapper.trajectories.numpy()
        fp = np.memmap(
            os.path.join(os.path.join(sample_dir, "trajectory.npy")),
            dtype=np.float16,
            mode="w+",
            shape=trajectory.shape,
        )
        fp[:] = trajectory[:]
        fp.flush()

        save_wrapper.reset()
        n_sampled += 1


class SaveCFGTrajectoryUnetWrapper:
    def __init__(self, fn: callable, num_steps: int, sample_size: int):
        self.fn = fn
        self.sample_size = sample_size

        self.guidance_scale = -1
        self.num_steps = num_steps
        self.reset()

    def __call__(self, *args, **kwargs):
        output = self.fn(*args, **kwargs)

        # Save model input
        self.trajectories[self.step, 0] = args[0][0]

        # Save model CFG output
        eps, var = output[0].chunk(2, dim=0)
        self.trajectories[self.step, 1] = eps.cpu()
        self.trajectories[self.step, 2] = var.cpu()

        # Increase step
        self.step += 1

        return output

    def reset(self):
        # 1: Latent input, 2: Noise output, 3: Log variance
        self.trajectories = torch.zeros(self.num_steps, 3, 4, self.sample_size, self.sample_size)
        self.guidance_scale = -1
        self.step = 0

    def __getattr__(self, name):
        return getattr(self.unet, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-per-class", type=int, default=1, help="Number of samples per label.")
    parser.add_argument("--inference-steps", type=int, default=1000)
    parser.add_argument("--min-guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
