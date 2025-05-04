from typing import Optional

import torch
import os
from glob import glob


class CatchCalls:
    def __init__(self, instance, fn: callable):
        self.instance = instance
        self.fn = fn

    def __call__(self, *args, **kwargs):
        self.fn(*args, **kwargs)
        return self.instance(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.instance, name)


def to_cpu(data):
    if hasattr(data, "to"):
        return data.cpu()
    
    if isinstance(data, dict):
        return { k: to_cpu(v) for k, v in data.items() }

    if isinstance(data, list):
        return [to_cpu(v) for v in data]
    
    if isinstance(data, tuple):
        return tuple([to_cpu(v) for v in data])
    
    return data 


def remove_none(data):
    if isinstance(data, dict):
        return { k: remove_none(v) for k, v in data.items() if v is not None }

    if isinstance(data, list):
        return [remove_none(v) for v in data if v is not None]
    
    if isinstance(data, tuple):
        return tuple([remove_none(v) for v in data if v is not None])
    
    return data


def remove_uncond_dim(data):
    if isinstance(data, torch.Tensor):
        return data[1]
    
    if isinstance(data, dict):
        return { k: remove_uncond_dim(v) for k, v in data.items() }

    if isinstance(data, list):
        return [remove_uncond_dim(v) for v in data]
    
    if isinstance(data, tuple):
        return tuple([remove_uncond_dim(v) for v in data])

    raise ValueError(f"unsupported data type: {type(data)}")


def default_disable_tqdm(fn: callable):
    from tqdm import tqdm
    from functools import partialmethod

    def wrapper(*args, **kwargs):
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        fn(*args, **kwargs)
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)

    return wrapper


def get_checkpoint(ckpt: Optional[os.PathLike], result_dir: os.PathLike, use_adapter=True) -> int:
    if not use_adapter:
        return 0

    if ckpt is not None:
        return ckpt

    # Take final checkpoint if none is specified
    checkpoints = glob(os.path.join(result_dir, "checkpoints", "*.pt"))
    checkpoints = [int(os.path.basename(ckpt).split(".")[0]) for ckpt in checkpoints]
    if len(checkpoints) > 0:
        return max(checkpoints)
    
    raise ValueError("no checkpoints found")
