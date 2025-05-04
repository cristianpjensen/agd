from typing import Iterable, List

import numpy as np
import torch
import pandas as pd
import tensorflow.compat.v1 as tf
import argparse
import yaml
import os
from tqdm import tqdm
from diffusers import DDPMScheduler

from .sample_trajectories import encode_prompt
from .adapter import get_trained_adapter_pipeline
from agd.utils import default_disable_tqdm, get_checkpoint
from agd.eval.evaluator import Evaluator


@default_disable_tqdm
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # Construct pipeline
    pipe = get_trained_adapter_pipeline(device, args.result_dir, args.ckpt, args.use_adapter)
    
    if args.scheduler == "ddpm":
        scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")
        pipe.scheduler = scheduler

    # Compute metrics
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    ref_acts = evaluator.read_activations(args.ref)
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref, ref_acts)

    captions = list(pd.read_csv(args.captions_file)["caption"])
    samples_iterable = get_samples(pipe, captions, args.batch_size, args.num_inference_steps, args.guidance_scale, args.use_adapter)
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
        "checkpoint": args.ckpt,
        "use_adapter": args.use_adapter,
    }

    # Append to metrics.yaml
    with open(os.path.join(args.result_dir, "metrics.yaml"), "a") as f:
        yaml.safe_dump([{ **relevant_args, **metrics }], f)


@torch.no_grad()
def get_samples(
    pipe,
    captions: List[str],
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    use_adapter: bool,
) -> Iterable[np.ndarray]:
    num_samples = len(captions)

    if hasattr(pipe, "unet"):
        denoiser = pipe.unet
    else:
        denoiser = pipe.transformer
    
    denoiser.eval()

    for i in tqdm(range(0, num_samples, batch_size), desc="sampling", disable=False):
        bs = min(batch_size, num_samples - i)
        caption = captions[i:i+bs]

        if use_adapter:
            g = torch.tensor([guidance_scale] * bs, device=pipe.device)
            encoder_hidden_states = encode_prompt(pipe, caption, pipe.device)
            denoiser.set_adapter_kwargs(guidance_scale=g, encoder_hidden_states=encoder_hidden_states)

        samples = pipe(
            prompt=caption,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0 if use_adapter else guidance_scale,
            output_type="np",
        ).images
        samples = (samples * 255).astype(np.uint8)

        yield samples


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True, help="Path to reference samples.")
    parser.add_argument("--captions-file", type=str, required=True, help="Path to captions of reference samples.")
    parser.add_argument("--ckpt", type=int, required=None, help="Checkpoint to use, defaults to the last. Do not include leading zeroes or .pt extension.")
    parser.add_argument("--disable-adapter", action="store_false", dest="use_adapter", help="Disable adapter and use CFG instead.")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--scheduler", type=str, choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    args.ckpt = get_checkpoint(args.ckpt, args.result_dir, args.use_adapter)
    main(args)
