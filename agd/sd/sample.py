import torch
from torchvision.utils import save_image
from diffusers import ControlNetModel
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import cv2
import argparse

from .sample_trajectories import encode_prompt
from .adapter import get_trained_adapter_pipeline
from agd.utils import get_checkpoint, CatchCalls
from agd.utils.training_loop import get_config


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config(args.dir)

    if args.controlnet_condition is not None:
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            # torch_dtype=torch.float16,
            # use_safetensors=True,
        ).to(device)
    else:
        controlnet = None

    # Check arguments
    pipe = get_trained_adapter_pipeline(
        device,
        args.dir,
        ckpt=args.ckpt,
        use_adapter=args.use_adapter,
        controlnet=controlnet,
    ).to(device)

    def model_to_cpu(*args, **kwargs):
        pipe.to("cpu")
        pipe.vae.to(device)

    pipe.vae.decode = CatchCalls(pipe.vae.decode, model_to_cpu)

    if args.ip_condition is not None:
        assert config["base_model"] == "stabilityai/stable-diffusion-xl-base-1.0", "IP adapter is only supported for SDXL"

        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        )
        pipe.set_ip_adapter_scale(0.6)
        ip_image = load_image(args.ip_condition)
    else:
        ip_image = None

    if args.controlnet_condition is not None:
        assert config["base_model"] == "stabilityai/stable-diffusion-xl-base-1.0", "ControlNet is only supported for SDXL"
        controlnet_image = load_image(args.controlnet_condition)
        controlnet_image = get_canny_edge(controlnet_image)
        controlnet_image.save("condition.png")
    else:
        controlnet_image = None

    if hasattr(pipe, "unet"):
        denoiser = pipe.unet
    else:
        denoiser = pipe.transformer

    denoiser.eval()

    if args.use_adapter:
        g = torch.tensor([args.guidance_scale] * args.num_images, device=device)
        encoder_hidden_states = encode_prompt(pipe, args.prompt, device, batch_size=args.num_images)
        denoiser.set_adapter_kwargs(guidance_scale=g, encoder_hidden_states=encoder_hidden_states)

    if args.controlnet_condition is None:
        samples = pipe(
            prompt=args.prompt,
            num_inference_steps=args.inference_steps,
            num_images_per_prompt=args.num_images,
            ip_adapter_image=ip_image,
            guidance_scale=1.0 if args.use_adapter else args.guidance_scale,
            output_type="pt",
        )
    else:
        samples = pipe(
            prompt=args.prompt,
            image=controlnet_image,
            controlnet_conditioning_scale=0.5,
            num_inference_steps=args.inference_steps,
            num_images_per_prompt=args.num_images,
            guidance_scale=1.0 if args.use_adapter else args.guidance_scale,
            output_type="pt",
        )

    samples = samples.images.cpu()

    # Save images
    save_image(samples, args.output_file, nrow=args.num_cols, normalize=True, value_range=(0, 1), padding=0)


def get_canny_edge(image: torch.Tensor):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--output-file", type=str, default="sample.png")
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--disable-adapter", action="store_false", dest="use_adapter", help="Disable adapter and use CFG instead.")
    parser.add_argument("--ip-condition", type=str, default=None, help="Path to image for conditioning using IP-adapter.")
    parser.add_argument("--controlnet-condition", type=str, default=None, help="Path to image for conditioning using ControlNet.")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--num-cols", type=int, default=4)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    args.ckpt = get_checkpoint(args.ckpt, args.dir, args.use_adapter)

    with torch.inference_mode():
        main(args)
