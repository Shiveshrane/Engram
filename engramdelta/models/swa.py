"""Sliding-window attention: the local helper of arm A5.

A5 = GDN backbone + one SWA branch, at the same block(s) as Engram in A2, added to the residual
stream as its own pre-norm residual branch. The window is kept small (default 16): design §10 warns
that large local helpers can make the recurrent layers lazy.
"""
from .transformer_block import CausalSelfAttention


class SlidingWindowAttention(CausalSelfAttention):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, window: int = 16, rope_base: float = 10000.0):
        assert window > 0
        super().__init__(d_model, n_heads, head_dim, rope_base=rope_base, window=window)
