from typing import Optional

import torch
import logging
import yaml
import os


def create_logger(logging_dir: Optional[str]=None, verbose=False):
    handlers = [logging.StreamHandler()]
    if logging_dir is not None:
        handlers.append(logging.FileHandler(os.path.join(logging_dir, "log.txt")))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)
    return logger


def log_memory_usage(logger):
    """Log current and maximum memory usage. Useful for debugging memory."""

    logger.debug(f"(memory usage) current: {bytes_to_gb(torch.cuda.memory_allocated()):.2f} GB, max: {bytes_to_gb(torch.cuda.max_memory_allocated()):.2f} GB")


def bytes_to_gb(n_bytes):
    return n_bytes * 1e-9


def append_dims(val, n):
    return val.view(val.shape[0], *([1] * (n - 1)))


def get_x0_from_eps_ddpm(x_noisy, noise, timestep, scheduler):
    device, dtype = x_noisy.device, x_noisy.dtype
    alpha_bar = scheduler.alphas_cumprod[timestep.cpu()].to(device=device, dtype=dtype)
    sigma = torch.sqrt(1 - alpha_bar)
    alpha_bar, sigma = append_dims(alpha_bar, x_noisy.ndim), append_dims(sigma, x_noisy.ndim)
    x0 = (x_noisy - sigma * noise) / torch.sqrt(alpha_bar)
    return x0


def get_config(result_dir: os.PathLike) -> dict:
    with open(os.path.join(result_dir, "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict, result_dir: os.PathLike):
    with open(os.path.join(result_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)


def get_opt_scheduler(params, lr: float, weight_decay: float, num_steps: int):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    lr_warmup_steps = num_steps // 10
    scheduler1 = torch.optim.lr_scheduler.LinearLR(opt, 1e-4, 1, lr_warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_steps - lr_warmup_steps, eta_min=lr * 1e-4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[scheduler1, scheduler2], milestones=[lr_warmup_steps])
    return opt, scheduler


def save_checkpoint(params, path: os.PathLike):
    torch.save({ "model": params }, path)
