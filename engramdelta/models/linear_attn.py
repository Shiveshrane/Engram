"""Vanilla causal linear attention (arm LA, Phase 0 only: the known-result check of test 7).

phi(x) = elu(x) + 1 feature map, normalized output:
    S_t = S_{t-1} + phi(k_t) v_t^T,   z_t = z_{t-1} + phi(k_t)
    o_t = S_t^T phi(q_t) / (z_t^T phi(q_t))
No decay, no delta rule, no short conv: the additive-memory baseline that GDN should beat on MQAR.
Computed chunk-wise in fp32 (quadratic inside a chunk, a running state across chunks).

Shape legend: B batch, T time, D d_model, H heads, K head_dim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_linear_attention(q, k, v, chunk_size: int = 64, eps: float = 1e-6):
    """q, k: [B, H, T, K] (feature-mapped, non-negative); v: [B, H, T, V]. Returns [B, H, T, V], fp32."""
    B, H, T, K = q.shape
    V = v.shape[-1]
    ones = v.new_ones(B, H, T, 1)
    v1 = torch.cat([v, ones], dim=-1)                  # the last column accumulates the normalizer z
    S = q.new_zeros(B, H, K, V + 1)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril()
    outs = []
    for s in range(0, T, chunk_size):
        q_c, k_c, v_c = q[:, :, s:s + chunk_size], k[:, :, s:s + chunk_size], v1[:, :, s:s + chunk_size]
        c = q_c.shape[2]
        scores = (q_c @ k_c.transpose(-1, -2)).masked_fill(~causal[:c, :c], 0)
        o = q_c @ S + scores @ v_c                     # [B, H, c, V + 1]
        outs.append(o[..., :V] / (o[..., V:] + eps))
        S = S + k_c.transpose(-1, -2) @ v_c
    return torch.cat(outs, dim=2)


class LinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        inner = n_heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.o_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        shape = (B, T, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            o = causal_linear_attention(F.elu(q.float()) + 1, F.elu(k.float()) + 1, v.float())
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1).to(x.dtype))
