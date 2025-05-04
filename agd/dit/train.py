import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from time import time
from glob import glob
import argparse
import os

from agd.models import ADAPTER_MODELS
from agd.dit.adapter import get_untrained_adapter_dit, get_pretrained_dit
from agd.dit.DiT.diffusion import create_diffusion
from agd.utils.training_loop import create_logger, log_memory_usage, save_config, get_opt_scheduler, save_checkpoint


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(os.path.join(args.dir, "checkpoints"), exist_ok=True)
    logger = create_logger(args.dir, verbose=args.verbose)
    save_config(vars(args), args.dir)

    # Set up accelerator
    accelerator = Accelerator(project_dir=args.dir)
    logger.info(f"experiment directory created at {args.dir}")

    # Create model and only make adapters trainable
    student = get_untrained_adapter_dit(
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_timestep=args.use_timestep,
        use_class_label=args.use_class_label,
        zero_init=args.zero_init,
        arch=args.arch,
        cfg_mult=args.cfg_mult,
    )
    student.freeze_base_model()
    student.train()

    if args.loss == "cos":
        teacher = get_pretrained_dit(args.image_size)
        teacher.eval()
        sigmas = None
    elif args.loss == "noise":
        teacher = None
        sigmas = torch.sqrt(1.0 - torch.from_numpy(create_diffusion("").alphas_cumprod))
    else:
        teacher = None
        sigmas = None

    # Optimizer and scheduler
    opt, scheduler = get_opt_scheduler(student.adapter_parameters(), args.lr, args.weight_decay, args.num_steps)

    # Log number of parameters
    logger.info(f"student parameters: {sum(p.numel() for p in student.parameters()):,}")
    logger.info(f"adapter parameters: {sum(p.numel() for p in student.adapter_parameters()):,}")
    logger.info(f"requires_grad parameters: {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")

    # Data
    dataset = TrajectoryDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    logger.info(f"dataset contains {len(dataset):,} points ({args.data_path})")

    # Prepare model, optimizer, and loader
    loader, student, opt, scheduler, teacher = accelerator.prepare(loader, student, opt, scheduler, teacher)
    sigmas = sigmas.to(device=accelerator.device) if sigmas is not None else None

    log_memory_usage(logger)

    # Variables for monitoring/logging purposes
    train_steps = 0
    epochs = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"training for {args.num_steps:,} steps...")

    while train_steps < args.num_steps:
        epochs += 1
        logger.info(f"beginning epoch {epochs}...")

        for traj_input, eps_cfg, timestep, condition in loader:
            if train_steps >= args.num_steps:
                break

            # Give conditioning variables to adapter blocks
            guidance_scale = condition["cfg_scale"]
            class_label = condition.get("class_label", None)
            student.set_adapter_kwargs(timestep=timestep, class_label=class_label, guidance_scale=guidance_scale)

            eps_student = student(traj_input, timestep, class_label)

            # Compute loss
            if args.loss == "l2":
                loss = F.mse_loss(eps_student, eps_cfg)
            elif args.loss == "l1":
                loss = F.l1_loss(eps_student, eps_cfg)
            elif args.loss == "cos":
                with torch.no_grad():
                    eps_cond = teacher(traj_input, timestep, class_label)
                    bs = eps_student.shape[0]
                    weight = 0.5 * torch.abs(1 - F.cosine_similarity(eps_cond.view(bs, -1), eps_cfg.view(bs, -1), dim=-1))

                loss = torch.mean(weight * torch.mean(torch.square(eps_student - eps_cfg), dim=(1, 2, 3)))
            elif args.loss == "noise":
                weight = sigmas[timestep].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                loss = torch.mean(weight * torch.square(eps_student - eps_cfg))
            else:
                raise ValueError(f"loss {args.loss} not supported")

            # Compute gradients
            opt.zero_grad()
            accelerator.backward(loss)

            # Step optimizer and scheduler
            opt.step()
            scheduler.step()

            # Log loss values
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed and reduce loss history
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / (log_steps * accelerator.num_processes)

                # Log metrics
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) train loss: {avg_loss:.4f}, train steps/sec: {steps_per_sec:.2f}")
                    log_memory_usage(logger)

                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if accelerator.is_main_process and train_steps % args.ckpt_every == 0 and train_steps > 0:
                ckpt_path = os.path.join(args.dir, "checkpoints", f"{train_steps:07d}.pt")
                save_checkpoint(student.adapter_state_dicts(), ckpt_path)
                logger.info(f"checkpoint saved at {ckpt_path}")

    logger.info("done!")


class TrajectoryDataset(Dataset):
    def __init__(self, dir: str):
        self.trajectory_dirs = sorted(glob(os.path.join(dir, "*/")))
        assert len(self.trajectory_dirs) > 0, "no trajectories found in directory"

        # Read in information about the dataset
        self.timesteps = torch.load(os.path.join(dir, "timesteps.pt"), weights_only=True)
        self.latent_shape = torch.load(os.path.join(dir, "latent_shape.pt"), weights_only=True)
        self.num_steps = len(self.timesteps)

    def __len__(self):
        return len(self.trajectory_dirs) * self.num_steps

    def __getitem__(self, idx):
        trajectory_idx = idx // self.num_steps
        step = idx % self.num_steps

        # Get model in- and output
        trajectory = np.memmap(
            os.path.join(self.trajectory_dirs[trajectory_idx], "trajectory.npy"),
            dtype=np.float16,
            mode="r",
            shape=(self.num_steps, 3, *self.latent_shape),
        )
        model_input, eps_teacher, var_teacher = torch.from_numpy(trajectory[step].copy())
        model_output = torch.cat([eps_teacher, var_teacher], dim=0)

        # Get conditioning
        conditioning = torch.load(
            os.path.join(self.trajectory_dirs[trajectory_idx], "conditioning.pt"),
            weights_only=True,
        )

        return model_input.float(), model_output.float(), self.timesteps[step], conditioning



if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    # Training loop and model definition
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-steps", type=int, default=5_000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    # Adapter configuration
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-timestep", action="store_true")
    parser.add_argument("--use-class-label", action="store_true")
    parser.add_argument("--zero-init", action="store_true")
    parser.add_argument("--arch", type=str, choices=list(ADAPTER_MODELS.keys()), default="additive")
    parser.add_argument("--loss", type=str, choices=["l2", "l1", "cos", "noise"], default="l2")
    parser.add_argument("--cfg-mult", type=float, default=1000)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    args = parser.parse_args()
    main(args)
