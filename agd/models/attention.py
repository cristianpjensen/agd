
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agd.adapters import Adapter


class CrossAttentionCFGAdapter(Adapter):
    """Adapter that conditions on the CFG scale."""

    def __init__(
        self,
        block: nn.Module,
        input_dim: int,
        hidden_dim: int,
        num_heads=1,
        dropout=0.0,
        qkv_bias=False,
        use_timestep=False,
        timestep_embedder: Optional[callable]=None,
        timestep_dim: Optional[int]=None,
        use_class_label=False,
        class_embedder: Optional[callable]=None,
        class_dim: Optional[int]=None,
        use_prompt=False,
        prompt_dim: Optional[int]=None,
        mlp_ratio=4.0,
        zero_init=False,
    ):
        super().__init__(block)

        self.scale = 1.0 / math.sqrt(hidden_dim)

        self.num_heads = num_heads
        self.dropout = dropout
        self.use_timestep = use_timestep
        self.use_class_label = use_class_label
        self.use_prompt = use_prompt

        # Cross-attention
        self.q_proj = nn.Linear(input_dim, hidden_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize weights
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0)
        if qkv_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)

        nn.init.zeros_(self.out_proj.bias)
        if zero_init:
            nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        else:
            nn.init.zeros_(self.out_proj.weight)

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

        self.pos_enc = PositionalEncoding(hidden_dim)

    def _adapter_forward(
        self,
        x: torch.Tensor,
        guidance_scale: torch.Tensor,
        timestep: Optional[torch.Tensor]=None,
        class_label: Optional[torch.Tensor]=None,
        prompt: Optional[torch.Tensor]=None,
    ):
        """
        Args:
            x: (...B, T, hidden_dim)
            guidance_scale: (...B)
            timestep: (...B)
            class_label: (...B) or None
            prompt: (...B, T_p, hidden_dim) or None

        Returns: (...B, T, hidden_dim)

        """

        # Map all embeddings to a shared space of `hidden_dim` dimensions
        embeddings = self.guidance_scale_embedder(guidance_scale).unsqueeze(-2)  # (...B, S, hidden_dim)

        if self.use_timestep:
            assert timestep is not None
            t_emb = self.timestep_embedder(timestep).unsqueeze(-2)
            embeddings = torch.cat([embeddings, t_emb], dim=-2)

        if self.use_class_label:
            assert class_label is not None
            c_emb = self.class_embedder(class_label).unsqueeze(-2)
            embeddings = torch.cat([embeddings, c_emb], dim=-2)

        if self.use_prompt:
            assert prompt is not None
            prompt_emb = self.prompt_proj(prompt)
            embeddings = torch.cat([embeddings, prompt_emb], dim=-2)

        q = self.q_proj(x)
        embeddings = self.pos_enc(embeddings)
        k = self.k_proj(embeddings)
        v = self.v_proj(embeddings)
        att = multihead_attention(q, k, v, num_heads=self.num_heads, dropout_p=self.dropout if self.training else 0.0, scale=self.scale)
        return self.out_proj(att)

    def forward(self, *args, **kwargs):
        """
        Args:
            x: (...B, T, hidden_dim)

        Returns: (...B, T, hidden_dim)
        """

        assert self.kwargs is not None, f"kwargs must be set in {self.__class__.__name__}"
        assert "guidance_scale" in self.kwargs, f"`guidance_scale` must be set in kwargs in {self.__class__.__name__}"
    
        if len(args) > 0:
            x = args[0]
        elif "hidden_states" in kwargs:
            x = kwargs["hidden_states"]
        else:
            raise ValueError("hidden_states not found in kwargs")

        guidance_scale = self.kwargs["guidance_scale"]
        timestep = self.kwargs.get("timestep", None)
        class_label = self.kwargs.get("class_label", None)
        prompt = self.kwargs.get("encoder_hidden_states", None)

        block_out = self.block(*args, **kwargs)
        adapter_out = self._adapter_forward(x, guidance_scale, timestep, class_label, prompt)

        if isinstance(block_out, tuple):
            block_out = (block_out[0] + adapter_out,) + block_out[1:]
        else:
            block_out += adapter_out

        return block_out

def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads=1,
    dropout_p=0.0,
    scale: Optional[float]=None
) -> torch.Tensor:
    """
    Args:
        q: (...B, T, hidden_dim)
        k: (...B, L, hidden_dim)
        v: (...B, L, hidden_dim)
        num_heads: int
        scale: float or None (defaults to 1.0 / sqrt(hidden_dim))

    Returns: (...B, T, hidden_dim)
    """

    q_heads = q.view(*q.shape[:-1], num_heads, -1).transpose(-3, -2)                                   # (...B, num_heads, T, head_dim) where head_dim = hidden_dim / num_heads
    k_heads = k.view(*k.shape[:-1], num_heads, -1).transpose(-3, -2)                                   # (...B, num_heads, L, head_dim)
    v_heads = v.view(*v.shape[:-1], num_heads, -1).transpose(-3, -2)                                   # (...B, num_heads, L, head_dim)
    out = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=dropout_p, scale=scale)  # (...B, num_heads, T, head_dim)
    out = out.transpose(-3, -2)                                                                        # (...B, T, num_heads, head_dim)
    return out.reshape(*q.shape)                                                                       # (...B, T, hidden_dim)


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


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len=250, max_period=10000.0):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be an even number"

        position = torch.arange(max_len).unsqueeze(1)                                                                 # (max_len, 1)
        div_term = torch.exp(-math.log(max_period) * torch.arange(0, hidden_dim, 2, dtype=torch.float) / hidden_dim)  # (hidden_dim / 2)
        pos_div = position * div_term                                                                                 # (max_len, hidden_dim / 2)
        self.register_buffer("pe", torch.cat([torch.sin(pos_div), torch.cos(pos_div)], dim=-1))                       # (max_len, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (...B, T, hidden_dim)

        Returns: (...B, T, hidden_dim)
        """

        return x + self.pe[:x.shape[-2]]
