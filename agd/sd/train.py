import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from glob import glob
from time import time
import argparse
import os

from agd.models import ADAPTER_MODELS
from agd.utils.training_loop import create_logger, log_memory_usage, save_config, get_opt_scheduler, save_checkpoint
from .adapter import get_untrained_adapter, SUPPORTED_MODELS


def main(args):
    assert torch.cuda.is_available(), "gpu is necessary"

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(os.path.join(args.dir, "checkpoints"), exist_ok=True)
    logger = create_logger(args.dir, verbose=args.verbose)
    save_config(vars(args), args.dir)

    # Set up
    accelerator = Accelerator(project_dir=args.dir)
    logger.info(f"experiment directory created at {args.dir}")

    # Create model and only make adapters trainable
    model = get_untrained_adapter(
        args.base_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_timestep=args.use_timestep,
        use_prompt=args.use_prompt,
        zero_init=args.zero_init,
        arch=args.arch,
    )
    model.freeze_base_model()
    model.train()

    # Log number of parameters
    logger.info(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"adapter parameters: {sum(p.numel() for p in model.adapter_parameters()):,}")
    logger.info(f"requires_grad parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer and scheduler
    opt, scheduler = get_opt_scheduler(model.adapter_parameters(), args.lr, args.weight_decay, args.num_steps)

    # Data
    dataset = TrajectoryDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    logger.info(f"dataset contains {len(dataset):,} points ({args.data_path})")

    loader, model, opt, scheduler = accelerator.prepare(loader, model, opt, scheduler)

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

        for traj_input, eps_teacher, timestep, condition in loader:
            if train_steps >= args.num_steps:
                break

            # Move everything to GPU
            guidance_scale = condition["cfg_scale"]
            # encoder_hidden_states = condition["last_hidden_state"]
            additional_model_kwargs = condition.get("additional_model_kwargs", {})
            additional_model_kwargs["encoder_hidden_states"] = additional_model_kwargs["encoder_hidden_states"].squeeze(1)
            encoder_hidden_states = additional_model_kwargs["encoder_hidden_states"]

            # Give conditioning variables to adapter blocks
            model.set_adapter_kwargs(
                guidance_scale=guidance_scale,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

            eps_student = model(
                traj_input,
                timestep=timestep,
                **additional_model_kwargs,
            ).sample

            loss = F.mse_loss(eps_student, eps_teacher)

            # Backward pass
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
                avg_loss = running_loss / log_steps

                # Log metrics
                logger.info(f"(step={train_steps:07d}) train loss: {avg_loss:.4f}, train steps/sec: {steps_per_sec:.2f}")
                log_memory_usage(logger)

                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                ckpt_path = os.path.join(args.dir, "checkpoints", f"{train_steps:07d}.pt")
                save_checkpoint(model.adapter_state_dicts(), ckpt_path)
                logger.info(f"checkpoint saved at {ckpt_path}")

    logger.info("done!")


class TrajectoryDataset(Dataset):
    def __init__(self, dir: str):
        self.trajectory_dirs = sorted(glob(os.path.join(dir, "*/")))
        assert len(self.trajectory_dirs) > 0, "no trajectories found in directory"

        # Read in information about the dataset
        self.timesteps = torch.load(os.path.join(dir, "timesteps.pt"), weights_only=True)
        self.latent_shape = torch.load(os.path.join(dir, "latent_shape.pt"), weights_only=True)
        self.num_steps = self.timesteps.shape[0]

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
            shape=(self.num_steps, 2, *self.latent_shape),
        )
        model_input, model_output = torch.from_numpy(trajectory[step].copy())

        # Get conditioning
        conditioning = torch.load(
            os.path.join(self.trajectory_dirs[trajectory_idx], "conditioning.pt"),
            weights_only=True,
        )

        for key, val in conditioning.items():
            if isinstance(val, torch.Tensor) and val.shape[0] == 1:
                conditioning[key] = val.squeeze(0)

        conditioning["additional_model_kwargs"].pop("hidden_states", None)
        conditioning.pop("last_hidden_state", None)

        return model_input.float(), model_output.float(), self.timesteps[step], conditioning


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    # Training loop and model definition
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--num-steps", type=int, default=5_000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    # Adapter configuration
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-timestep", action="store_true")
    parser.add_argument("--use-prompt", action="store_true")
    parser.add_argument("--zero-init", action="store_true")
    parser.add_argument("--arch", type=str, choices=list(ADAPTER_MODELS.keys()), default="attention")

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    args = parser.parse_args()
    main(args)
