from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import tensorflow.compat.v1 as tf
from diffusers import AutoencoderKL
import argparse
import yaml
import os
from tqdm import tqdm

from agd.dit.DiT.diffusion import create_diffusion

from agd.utils import get_checkpoint
from agd.utils.training_loop import get_config
from agd.dit.adapter import get_trained_adapter_dit
from agd.eval.evaluator import Evaluator


def main(args):
    train_config = get_config(args.result_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # Construct pipeline
    model = get_trained_adapter_dit(device, args.result_dir, args.ckpt, args.use_adapter)
    model.eval()

    # Compute metrics
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    ref_acts = evaluator.read_activations(args.ref)
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref, ref_acts)

    samples_iterable = get_samples(device, model, train_config["image_size"], args.num_samples, args.batch_size, args.num_inference_steps, args.guidance_scale, args.use_adapter)
    sample_acts = evaluator.compute_activations(samples_iterable)
    sample_stats, sample_stats_spatial = evaluator.read_statistics(None, sample_acts)

    precision, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
    metrics = {
        "inception_score": float(evaluator.compute_inception_score(sample_acts[0])),
        "fid": float(sample_stats.frechet_distance(ref_stats)),
        "sfid": float(sample_stats_spatial.frechet_distance(ref_stats_spatial)),
        "precision": float(precision),
        "recall": float(recall),
    }
    relevant_args = {
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_samples": args.num_samples,
        "checkpoint": args.ckpt,
        "use_adapter": args.use_adapter,
    }

    # Append to metrics.yaml
    with open(os.path.join(args.result_dir, "metrics.yaml"), "a") as f:
        yaml.safe_dump([{ **relevant_args, **metrics }], f)


@torch.no_grad()
def get_samples(
    device: torch.device,
    model: nn.Module,
    image_size: int,
    num_samples: int,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    use_adapter: bool,
) -> Iterable[np.ndarray]:
    diffusion = create_diffusion(str(num_inference_steps))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    latent_size = image_size // 8

    for i in tqdm(range(0, num_samples, batch_size), desc="sampling", disable=False):
        bs = min(batch_size, num_samples - i)
        y = torch.randint(0, 1000, (bs,), device=device)
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)

        if use_adapter:
            model_kwargs = dict(y=y)
            model.set_adapter_kwargs(class_label=y, guidance_scale=torch.tensor([guidance_scale] * bs, device=device))
            sample_fn = model.forward
        elif guidance_scale > 1:
            z = torch.cat([z, z], dim=0)
            y_null = torch.tensor([1000] * bs, device=device)
            y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y, cfg_scale=guidance_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        latents = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        latents = latents[:bs]
        
        samples = vae.decode(latents / 0.18215).sample
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = (samples * 255).astype(np.uint8)

        yield samples


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True, help="Path to reference samples.")
    parser.add_argument("--ckpt", type=int, required=None, help="Checkpoint to use, defaults to the last. Do not include leading zeroes or .pt extension.")
    parser.add_argument("--disable-adapter", action="store_false", dest="use_adapter", help="Disable adapter and use CFG instead.")
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    args.ckpt = get_checkpoint(args.ckpt, args.result_dir, args.use_adapter)
    main(args)
