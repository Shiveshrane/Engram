"""Gated DeltaNet (GDN) sequence mixer.

Projections, gates, short convolutions and the gated output norm follow FLA's
fla/layers/gated_deltanet.py (class GatedDeltaNet, flash-linear-attention, MIT). They are rewritten
here in plain PyTorch, so the layer runs without Triton (MacBook / MPS / CPU) and can expose its
recurrent state for eval/state_metrics.py.

The core recurrence has two backends:
  "fla"   FLA's Triton kernel fla.ops.gated_delta_rule.chunk_gated_delta_rule. Fast, if it works on T4 (checked in NB00).
  "torch" chunk_gated_delta_rule_torch below. It is FLA's naive_chunk_gated_delta_rule
          (vendored in third_party/fla_naive.py) with one change: the per-chunk triangular inverse is
          computed with torch.linalg.solve_triangular instead of a Python loop over rows.
          That loop keeps one autograd copy per row, which is too slow and memory-hungry for training.
Both are checked against FLA's naive_recurrent_gated_delta_rule in tests/test_kernels.py.

Recurrence per head (state S in R^{K x V}; q and k are L2-normalized; beta in (0, 1); g <= 0 is a log-decay):
    S_t = exp(g_t) * S_{t-1}
    S_t = S_t + k_t (beta_t * (v_t - S_t^T k_t))^T        # delta rule: overwrite the value stored under k_t
    o_t = S_t^T q_t / sqrt(K)

Shape legend: B batch, T time, D d_model, H heads, K head_dim (key), V head_v_dim (value).
"""
import inspect
import math
import os
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# Core recurrence: pure PyTorch
# ----------------------------------------------------------------------------------------------

def chunk_gated_delta_rule_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    chunk_size: int = 64, scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None, output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked gated delta rule, computed in fp32.

    q, k: [B, T, H, K]   v: [B, T, H, V]   g, beta: [B, T, H]
    Returns o [B, T, H, V] (in v's dtype), and the final state [B, H, K, V] or None.
    Variable names follow FLA's naive_chunk_gated_delta_rule.
    """
    out_dtype = v.dtype
    BT = chunk_size
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q, k, v, beta, g = (x.transpose(1, 2).float() for x in (q, k, v, beta, g))   # [B, H, T, ...]
    T = q.shape[2]
    pad = (BT - T % BT) % BT
    if pad:
        # Padded steps have k = 0, beta = 0 and g = 0, so they leave the state unchanged.
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    B, H, L, K = q.shape
    V = v.shape[-1]
    n = L // BT
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = (x.reshape(B, H, n, BT, x.shape[-1]) for x in (q, k, v, k_beta))   # [B, H, n, BT, *]
    decay = g.reshape(B, H, n, BT).cumsum(-1)                                            # [B, H, n, BT]
    # L_mask[i, j] = exp(decay_i - decay_j) for j <= i, else 0. tril() before exp() avoids overflow.
    L_mask = (decay[..., :, None] - decay[..., None, :]).tril().exp().tril()            # [B, H, n, BT, BT]
    upper_incl_diag = torch.ones(BT, BT, dtype=torch.bool, device=q.device).triu(0)
    A = ((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(upper_incl_diag, 0)       # strictly lower
    # FLA naive: attn = -A, then a row loop builds (I - attn)^{-1} = (I + A)^{-1}. Same thing as a unit-triangular solve:
    eye = torch.eye(BT, dtype=q.dtype, device=q.device).expand(A.shape)
    Tinv = torch.linalg.solve_triangular(eye + A, eye, upper=False, unitriangular=True)   # [B, H, n, BT, BT]
    u = Tinv @ v                                            # naive: k_cumsum
    w = Tinv @ (k_beta * decay.exp()[..., None])            # naive: k_cumdecay
    S = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.float()
    strict_upper = torch.ones(BT, BT, dtype=torch.bool, device=q.device).triu(1)
    outs = []
    for i in range(n):
        q_i, k_i, u_i, decay_i = q[:, :, i], k[:, :, i], u[:, :, i], decay[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill(strict_upper, 0)
        v_new = u_i - w[:, :, i] @ S                                     # [B, H, BT, V]
        o_inter = (q_i * decay_i[..., None].exp()) @ S                   # contribution of earlier chunks
        outs.append(o_inter + attn @ v_new)
        S = S * decay_i[..., -1, None, None].exp() + \
            (k_i * (decay_i[..., -1, None] - decay_i).exp()[..., None]).transpose(-1, -2) @ v_new
    o = torch.stack(outs, dim=2).reshape(B, H, L, V)[:, :, :T].transpose(1, 2)    # [B, T, H, V]
    return o.to(out_dtype), (S if output_final_state else None)


def recurrent_gated_delta_rule_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    scale: Optional[float] = None, capture_positions: Sequence[int] = (),
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Step-by-step recurrence (same math as FLA's naive_recurrent_gated_delta_rule), in fp32.

    Slow, so it is used only for analysis. Returns o [B, T, H, V] and {t: S_t [B, H, K, V]}, where S_t is the
    state after token t has been written, for every t in capture_positions.
    """
    out_dtype = v.dtype
    q, k, v, beta, g = (x.transpose(1, 2).float() for x in (q, k, v, beta, g))   # [B, H, T, ...]
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q = q * scale
    S = q.new_zeros(B, H, K, V)
    want = set(int(p) for p in capture_positions)
    states: Dict[int, torch.Tensor] = {}
    o = q.new_zeros(B, H, T, V)
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        v_t = beta[:, :, t, None] * (v[:, :, t] - (S * k[:, :, t, :, None]).sum(-2))   # [B, H, V]
        S = S + k[:, :, t, :, None] * v_t[:, :, None, :]
        o[:, :, t] = torch.einsum("bhk,bhkv->bhv", q[:, :, t], S)
        if t in want:
            states[t] = S.clone()
    return o.transpose(1, 2).to(out_dtype), states


# ----------------------------------------------------------------------------------------------
# Core recurrence: FLA Triton kernel
# ----------------------------------------------------------------------------------------------

_FLA_FN = None


def fla_available() -> bool:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        return True
    except Exception:
        return False


def chunk_gated_delta_rule_fla(q, k, v, g, beta):
    """Calls FLA's chunk kernel with pre-normalized q/k, post-sigmoid beta and log-space g (its basic API)."""
    global _FLA_FN
    if _FLA_FN is None:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        kwargs = {}
        if "head_first" in inspect.signature(chunk_gated_delta_rule).parameters:
            kwargs["head_first"] = False     # old FLA releases defaulted to [B, H, T, K]
        _FLA_FN = (chunk_gated_delta_rule, kwargs)
    fn, kwargs = _FLA_FN
    o, _ = fn(q, k, v, g, beta, **kwargs)
    return o


def resolve_backend(name: str) -> str:
    """The environment variable ENGRAMDELTA_GDN_BACKEND overrides the config. The notebooks set it from NB00's result."""
    name = os.environ.get("ENGRAMDELTA_GDN_BACKEND", name)
    if name not in ("torch", "fla"):
        raise ValueError(f"unknown GDN backend {name!r}")
    if name == "fla" and not fla_available():
        raise RuntimeError("GDN backend 'fla' requested but fla.ops.gated_delta_rule cannot be imported")
    return name


# ----------------------------------------------------------------------------------------------
# Layer
# ----------------------------------------------------------------------------------------------

def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x / sqrt(sum(x^2) + eps) over the last dim, in fp32 (FLA's l2norm uses eps = 1e-6)."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


class CausalConv1d(nn.Module):
    """Depthwise causal conv + SiLU over [B, T, C]. Equivalent to FLA's ShortConvolution(activation='silu')."""

    def __init__(self, channels: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=bias, padding=kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))


class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, head_dim: int, expand_v: float = 1.0,
        use_short_conv: bool = True, conv_size: int = 4, use_gate: bool = True,
        norm_eps: float = 1e-5, backend: str = "torch", chunk_size: int = 64,
    ):
        super().__init__()
        self.n_heads, self.head_k_dim = n_heads, head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = n_heads * self.head_k_dim
        self.value_dim = n_heads * self.head_v_dim
        self.use_short_conv, self.use_gate = use_short_conv, use_gate
        self.backend = resolve_backend(backend)
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)     # decay input
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)     # beta (write strength) input
        # Decay parameters, initialized as in FLA (Mamba-2 style).
        A = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.dt_bias._no_weight_decay = True
        if use_short_conv:      # GDN's built-in short conv; switched off only in the H4 ablation
            self.q_conv = CausalConv1d(self.key_dim, conv_size)
            self.k_conv = CausalConv1d(self.key_dim, conv_size)
            self.v_conv = CausalConv1d(self.value_dim, conv_size)
        if use_gate:
            self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

        # Analysis hooks (eval/state_metrics.py). When capture_positions is set, forward uses the recurrent
        # reference and stores {t: S_t}. When record_beta is set, forward stores beta [B, T, H].
        self.capture_positions: Optional[Sequence[int]] = None
        self.captured: Optional[Dict[int, torch.Tensor]] = None
        self.record_beta = False
        self.last_beta: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] (already normalized by the block) -> [B, T, D]
        B, T, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.use_short_conv:
            q, k, v = self.q_conv(q), self.k_conv(k), self.v_conv(v)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
        q = l2norm(q.view(B, T, self.n_heads, self.head_k_dim))           # [B, T, H, K] fp32
        k = l2norm(k.view(B, T, self.n_heads, self.head_k_dim))           # [B, T, H, K] fp32
        v = v.view(B, T, self.n_heads, self.head_v_dim)                   # [B, T, H, V]
        beta = self.b_proj(x).float().sigmoid()                           # [B, T, H]
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias.float())   # [B, T, H] <= 0
        if self.record_beta:
            self.last_beta = beta.detach()

        if self.capture_positions is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                o, self.captured = recurrent_gated_delta_rule_torch(
                    q, k, v.float(), g, beta, capture_positions=self.capture_positions)
        elif self.backend == "fla":
            dt = v.dtype
            o = chunk_gated_delta_rule_fla(q.to(dt), k.to(dt), v, g, beta.to(dt))
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                o, _ = chunk_gated_delta_rule_torch(q, k, v.float(), g, beta, chunk_size=self.chunk_size)

        o = self.o_norm(o.float())                                        # [B, T, H, V]
        if self.use_gate:   # FLA FusedRMSNormGated: RMSNorm(o) * swish(g_proj(x))
            o = o * F.silu(self.g_proj(x).view(B, T, self.n_heads, self.head_v_dim).float())
        return self.o_proj(o.reshape(B, T, self.value_dim).to(x.dtype))
