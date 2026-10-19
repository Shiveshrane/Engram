"""Causal multi-head self-attention with RoPE, via PyTorch scaled_dot_product_attention.

T4 (sm_75) has no FlashAttention-2. SDPA picks the memory-efficient or math kernel there.
`window` > 0 restricts every query to the last `window` keys (sliding-window attention, arm A5).

Shape legend: B batch, T time, D d_model, H heads, Dh head_dim.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cache(T: int, head_dim: int, base: float, device, dtype=torch.float32):
    """cos, sin of shape [T, Dh/2] for rotary position embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                     # [T, Dh/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, Dh]. Rotates pairs (x1, x2) = (first half, second half) of the head dim.
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, rope_base: float = 10000.0, window: int = 0):
        super().__init__()
        assert head_dim % 2 == 0
        self.n_heads, self.head_dim, self.rope_base, self.window = n_heads, head_dim, rope_base, window
        inner = n_heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.o_proj = nn.Linear(inner, d_model, bias=False)

    def _mask(self, T: int, device) -> Optional[torch.Tensor]:
        if self.window <= 0:
            return None
        i = torch.arange(T, device=device)
        # allowed[i, j] = (j <= i) and (i - j < window)
        return (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, T, D]
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # [B, H, T, Dh]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope_cache(T, self.head_dim, self.rope_base, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask = self._mask(T, x.device)
        if mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))
