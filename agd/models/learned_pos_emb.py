from typing import Optional

import torch
import torch.nn as nn
import math

from agd.adapters import Adapter


class LearnedPosEmbedCFGAdapter(Adapter):
    def __init__(
        self,
        block: nn.Module,
        input_dim: int,
        hidden_dim: int,
        dropout=0.0,
        use_timestep=False,
        timestep_embedder: Optional[callable]=None,
        timestep_dim: Optional[int]=None,
        use_class_label=False,
        class_embedder: Optional[callable]=None,
        class_dim: Optional[int]=None,
        use_prompt=False,
        prompt_dim: Optional[int]=None,
        mlp_ratio=4.0,
        zero_init=True,
        cfg_mult=1000,
    ):
        super().__init__(block)

        self.cfg_mult = cfg_mult
        self.dropout = dropout
        self.use_timestep = use_timestep
        self.use_class_label = use_class_label
        self.use_prompt = use_prompt

        # Guidance scale embedder
        self.guidance_scale_embedder = ScalarEncoder(hidden_dim, mlp_ratio)

        # Timestep embedder
        if use_timestep:
            if timestep_embedder is not None and timestep_dim is not None:
                self.timestep_proj = nn.Linear(timestep_dim, hidden_dim)
                self.timestep_embedder = lambda x: self.timestep_proj(timestep_embedder(x))
            else:
                self.timestep_embedder = ScalarEncoder(hidden_dim, mlp_ratio)

        # Embedding table for class label
        if use_class_label:
            if class_embedder is not None and class_dim is not None:
                self.class_proj = nn.Linear(class_dim, hidden_dim)
                self.class_embedder = lambda x: self.class_proj(class_embedder(x))
            else:
                self.class_embedder = nn.Embedding(1000, hidden_dim)
                nn.init.xavier_uniform_(self.class_embedder.weight, gain=1.0)

        # Prompt projection to embed in same space as other embeddings
        if use_prompt:
            assert prompt_dim is not None, "prompt_dim must be set if using prompt condition"
            self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)

        self.seq_timestep_embedder = ScalarEncoder(hidden_dim, mlp_ratio)
        self.out_proj = nn.Linear(2 * hidden_dim, input_dim)

        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.zeros_(self.out_proj.bias)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)

    def _adapter_forward(
        self,
        x: torch.Tensor,
        guidance_scale: torch.Tensor,
        timestep: Optional[torch.Tensor]=None,
        class_label: Optional[torch.Tensor]=None,
        pooled_prompt_embed: Optional[torch.Tensor]=None,
    ):
        """
        Args:
            x: (...B, T, hidden_dim)
            guidance_scale: (...B)
            timestep: (...B) or None
            class_label: (...B) or None
            pooled_prompt_embed: (...B, prompt_dim) or None

        Returns: (...B, T, hidden_dim)

        """

        # Map all embeddings to a shared space of `hidden_dim` dimensions
        c = self.guidance_scale_embedder(guidance_scale * self.cfg_mult)  # (...B, hidden_dim)

        if self.use_timestep:
            assert timestep is not None
            if isinstance(timestep, (int, float)) or timestep.ndim == 0:
                timestep = torch.tensor([timestep], device=x.device)

            c = c + self.timestep_embedder(timestep)

        if self.use_class_label:
            assert class_label is not None
            c = c + self.class_embedder(class_label)

        if self.use_prompt:
            assert pooled_prompt_embed is not None
            c = c + self.prompt_proj(pooled_prompt_embed.squeeze(-2))

        seq_timesteps = torch.arange(x.shape[-2], device=x.device).float()
        t = self.seq_timestep_embedder(seq_timesteps)  # (T, hidden_dim)
        t = t.unsqueeze(0).expand(*x.shape[:-1], -1)   # (...B, T, hidden_dim)
        c = c.unsqueeze(-2).expand(*x.shape[:-1], -1)  # (...B, T, hidden_dim)

        return self.out_proj(torch.cat([c, t], dim=-1))

    def forward(self, *args, **kwargs):
        """
        Args:
            x: (...B, T, hidden_dim)

        Returns: (...B, T, hidden_dim)
        """

        assert self.kwargs is not None, f"kwargs must be set in {self.__class__.__name__}"
        assert "guidance_scale" in self.kwargs, f"`guidance_scale` must be set in kwargs in {self.__class__.__name__}"

        x = args[0]
        guidance_scale = self.kwargs["guidance_scale"]
        timestep = self.kwargs.get("timestep", None)
        class_label = self.kwargs.get("class_label", None)
        prompt = self.kwargs.get("prompt", None)

        return self.block(*args, **kwargs) + self._adapter_forward(x, guidance_scale, timestep, class_label, prompt)


class ScalarEncoder(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio=4.0):
        super().__init__()

        self.net = nn.Sequential(
            FourierEncoding(hidden_dim),
            MLP(hidden_dim, hidden_dim, mlp_ratio),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, cfg_scale: torch.Tensor) -> torch.Tensor:
        return self.net(cfg_scale)

    
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, mlp_ratio=4.0):
        super().__init__()

        hidden_dim = int(input_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        nn.init.xavier_uniform_(self.net[0].weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.net[2].weight, gain=math.sqrt(2))
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierEncoding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.register_buffer("frequencies", torch.randn(hidden_dim))
        self.register_buffer("phases", torch.rand(hidden_dim))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: (...B)
        
        Returns: (...B, hidden_dim)
        """

        pos = pos.float().unsqueeze(-1)
        return torch.cos(2 * torch.pi * (self.frequencies * pos + self.phases))
