import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL
import argparse

from agd.dit.DiT.diffusion import create_diffusion

from agd.dit.adapter import get_trained_adapter_dit
from agd.utils.training_loop import get_config
from agd.utils import get_checkpoint


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = len(args.class_labels)

    # Check arguments
    config = get_config(args.dir)
    model = get_trained_adapter_dit(device, args.dir, args.ckpt, args.use_adapter).to(device)
    model.eval()

    diffusion = create_diffusion(str(args.inference_steps))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    latent_size = config["image_size"] // 8

    z = torch.randn(bs, 4, latent_size, latent_size, device=device)
    y = torch.tensor(args.class_labels, device=device)

    if args.use_adapter:
        model_kwargs = dict(y=y)
        sample_fn = model.forward
        model.set_adapter_kwargs(
            class_label=y,
            guidance_scale=torch.tensor([args.guidance_scale] * bs, device=device),
        )
    elif args.guidance_scale > 1:
        z = torch.cat([z, z], dim=0)
        y_null = torch.tensor([1000] * bs, device=device)
        y = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=args.guidance_scale)
        sample_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        sample_fn = model.forward

    latents = diffusion.p_sample_loop(
        sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    latents = latents[:bs]

    samples = vae.decode(latents / 0.18215).sample
    samples = samples.clamp(-1, 1)

    # Save images
    save_image(samples, args.output_file, nrow=args.num_cols, normalize=True, value_range=(-1, 1), padding=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--output-file", type=str, default="sample.png")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--class-labels", type=int, choices=list(range(1000)), nargs="+", default=[105, 88, 250, 29, 970, 1, 555, 888])
    parser.add_argument("--disable-adapter", action="store_false", dest="use_adapter", help="Disable adapter and use CFG instead.")
    parser.add_argument("--num-cols", type=int, default=4)
    parser.add_argument("--inference-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    args.ckpt = get_checkpoint(args.ckpt, args.dir, args.use_adapter)
    main(args)
