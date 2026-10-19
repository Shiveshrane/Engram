"""Engram: hashed n-gram memory with context-aware gating.

Port of DeepSeek's standalone reference `engram_demo_v1.py`
(https://github.com/deepseek-ai/Engram, Apache-2.0). The reference is vendored unchanged in
third_party/engram_demo_v1.py, and tests/test_engram_parity.py checks that this file reproduces it.

Differences from the reference (also listed in DEVIATIONS.md):
  1. One residual stream (hc_mult = 1): one key projection and one (key, query) norm pair
     instead of one per hyper-connection branch.
  2. Hashing runs in torch on the model's device, not numpy on the CPU.
  3. `orders` is an explicit list of n-gram orders. The reference always uses 2..max_ngram_size.
     Order 1 (unigram) is used only by arm A4.
  4. The conv dilation is a parameter. The reference uses dilation = max_ngram_size;
     the default 3 keeps that value for orders {2, 3}.

Shape legend:
  B  batch            T  sequence length      D  d_model
  Hn hash heads per n-gram order              Ho total hash heads = len(orders) * Hn
  De embedding dim per head = embed_per_order // Hn
  E  retrieved memory dim = len(orders) * embed_per_order = Ho * De
"""
import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import isprime


# ----------------------------------------------------------------------------------------------
# Hash parameters (deterministic, no learned weights)
# ----------------------------------------------------------------------------------------------

def find_next_prime(start: int, seen_primes: set) -> int:
    """Smallest prime > start that is not already in seen_primes. Same as the reference."""
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


def table_sizes_for_layers(
    layer_ids: Sequence[int],
    n_orders: int,
    rows_per_head: Union[int, Sequence[int]],
    n_heads_per_order: int,
) -> Dict[int, List[List[int]]]:
    """Prime table size of every (layer, order, head).

    Same search as the reference's NgramHashMapping.calculate_vocab_size_across_layers:
    primes are unique across ALL Engram layers and heads, and the search for each head starts at
    the previous head's prime. So the sizes of one layer depend on the full `layer_ids` list,
    and every Engram in a model must be built from one call to this function.

    rows_per_head: one int for all orders, or one entry per order
                   (the reference's `engram_vocab_size` list).
    Returns {layer_id: [[prime for each head] for each order]}.
    """
    if isinstance(rows_per_head, int):
        rows = [rows_per_head] * n_orders
    else:
        rows = list(rows_per_head)
        assert len(rows) == n_orders, "need one rows_per_head entry per order"
    seen: set = set()
    sizes: Dict[int, List[List[int]]] = {}
    for layer_id in layer_ids:
        per_order = []
        for oi in range(n_orders):
            search_start = rows[oi] - 1
            heads = []
            for _ in range(n_heads_per_order):
                prime = find_next_prime(search_start, seen)
                seen.add(prime)
                heads.append(prime)
                search_start = prime
            per_order.append(heads)
        sizes[layer_id] = per_order
    return sizes


def layer_multipliers(layer_id: int, seed: int, max_order: int, hash_vocab_size: int) -> np.ndarray:
    """Odd random multipliers for the multiplicative-XOR hash of one layer (same as the reference).

    The upper bound keeps token_id * multiplier < 2**63 for every token_id < hash_vocab_size, so the
    int64 products never overflow.
    """
    max_long = np.iinfo(np.int64).max
    m_max = int(max_long // hash_vocab_size)
    half_bound = max(1, m_max // 2)
    rng = np.random.default_rng(int(seed + 10007 * int(layer_id)))
    r = rng.integers(low=0, high=half_bound, size=(max_order,), dtype=np.int64)
    return r * 2 + 1


def build_compressed_lookup(tokenizer) -> "tuple[np.ndarray, int]":
    """Tokenizer compression (paper: vocabulary projection P: V -> V').

    Port of the reference's CompressedTokenizer._build_lookup_table, for a `tokenizers.Tokenizer`
    (the reference takes a transformers AutoTokenizer). Token ids whose decoded text is equal after
    NFKC / accent stripping / lowercasing / whitespace collapsing share one canonical id.
    Returns (lookup [vocab_size] int64, number of canonical ids). Used only in Phase 2. Synthetic
    tasks use the identity map.
    """
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    key2new: Dict[str, int] = {}
    lookup = np.empty(vocab_size, dtype=np.int64)
    for tid in range(vocab_size):
        text = tokenizer.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tokenizer.id_to_token(tid)
        else:
            norm = normalizer.normalize_str(text)
            key = norm if norm else text
        if key not in key2new:
            key2new[key] = len(key2new)
        lookup[tid] = key2new[key]
    return lookup, len(key2new)


# ----------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------

class NgramHasher(nn.Module):
    """Token ids -> one table index per (order, head). Causal: index at t uses only ids[<= t]."""

    def __init__(
        self,
        orders: Sequence[int],
        primes: List[List[int]],          # [order][head] table sizes
        multipliers: np.ndarray,          # [max(orders)] odd int64
        pad_id: int,
        compress_lookup: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.orders = list(orders)
        self.max_order = max(self.orders)
        self.n_heads_per_order = len(primes[0])
        assert len(multipliers) >= self.max_order
        if compress_lookup is not None:
            # The reference compresses pad_id too.
            pad_id = int(compress_lookup[pad_id])
            self.register_buffer("compress_lookup", torch.as_tensor(compress_lookup, dtype=torch.long),
                                 persistent=False)
        else:
            self.compress_lookup = None
        self.pad_id = int(pad_id)
        self.register_buffer("multipliers", torch.as_tensor(np.asarray(multipliers), dtype=torch.long),
                             persistent=False)
        self.register_buffer("primes", torch.tensor([p for heads in primes for p in heads], dtype=torch.long),
                             persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, T] int64  ->  [B, T, Ho] int64
        if self.compress_lookup is not None:
            ids = self.compress_lookup[ids]
        B, T = ids.shape
        # shifts[k][:, t] = ids[:, t - k], left-padded with pad_id where t - k < 0 (reference: np.pad + [:, :T]).
        shifts = [ids]
        for k in range(1, self.max_order):
            pad = ids.new_full((B, k), self.pad_id)
            shifts.append(torch.cat([pad, ids], dim=1)[:, :T])
        hashes = []
        h = self.n_heads_per_order
        for oi, n in enumerate(self.orders):
            # The n-gram ending at t is (ids[t-n+1], ..., ids[t]). Mix it with a multiplicative XOR.
            mix = shifts[0] * self.multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, shifts[k] * self.multipliers[k])
            for j in range(h):
                hashes.append(torch.remainder(mix, self.primes[oi * h + j]))
        return torch.stack(hashes, dim=-1)


class MultiHeadEmbedding(nn.Module):
    """All hash-head tables stored in one nn.Embedding, with per-head row offsets (as in the reference)."""

    def __init__(self, list_of_n: List[int], dim: int):
        super().__init__()
        offsets = [0]
        for n in list_of_n[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.embedding = nn.Embedding(num_embeddings=sum(list_of_n), embedding_dim=dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T, Ho] -> [B, T, Ho, De]
        return self.embedding(idx + self.offsets)


class ShortConv(nn.Module):
    """RMSNorm -> causal depthwise Conv1d (dilated) -> SiLU. The reference's ShortConv with hc_mult = 1."""

    def __init__(self, dim: int, kernel_size: int = 4, dilation: int = 1, norm_eps: float = 1e-5,
                 activation: bool = True):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, bias=False,
                              padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, T, D]. Left padding (k-1)*dilation, then keep the first T outputs, so the conv is causal.
        T = x.shape[1]
        y = self.conv(self.norm(x).transpose(1, 2))[..., :T]
        if self.activation:
            y = F.silu(y)
        return y.transpose(1, 2)


class Engram(nn.Module):
    """Retrieve hashed n-gram embeddings, gate them with the hidden state, add a short conv.

    The block computes h <- h + Engram(h, ids) BEFORE its mixer (reference TransformerBlock.forward).
    """

    def __init__(
        self,
        d_model: int,
        orders: Sequence[int],
        primes: List[List[int]],
        multipliers: np.ndarray,
        embed_per_order: int,
        n_heads_per_order: int,
        pad_id: int,
        kernel_size: int = 4,
        conv_dilation: int = 3,
        use_conv: bool = True,
        frozen_table: bool = False,
        compress_lookup: Optional[np.ndarray] = None,
    ):
        super().__init__()
        assert embed_per_order % n_heads_per_order == 0
        assert len(primes) == len(orders) and all(len(p) == n_heads_per_order for p in primes)
        self.d_model = d_model
        self.hasher = NgramHasher(orders, primes, multipliers, pad_id, compress_lookup)
        self.table = MultiHeadEmbedding([p for heads in primes for p in heads], embed_per_order // n_heads_per_order)
        if frozen_table:
            # Arm A3: table values stay at their random init; everything else trains.
            self.table.embedding.weight.requires_grad_(False)
        e_dim = len(orders) * embed_per_order
        self.value_proj = nn.Linear(e_dim, d_model)
        self.key_proj = nn.Linear(e_dim, d_model)
        self.norm_key = nn.RMSNorm(d_model)      # reference: norm1
        self.norm_query = nn.RMSNorm(d_model)    # reference: norm2
        self.short_conv = ShortConv(d_model, kernel_size=kernel_size, dilation=conv_dilation) if use_conv else None
        # Analysis hook: when record_gate is True, the last gate values alpha_t [B, T] are kept here.
        self.record_gate = False
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, D] residual stream (not normalized); ids: [B, T] int64.  Returns [B, T, D].
        idx = self.hasher(ids)                                   # [B, T, Ho]
        e = self.table(idx).flatten(start_dim=-2)               # [B, T, E]
        key = self.norm_key(self.key_proj(e))                   # [B, T, D]
        query = self.norm_query(hidden)                         # [B, T, D]
        gate = (key * query).sum(dim=-1) / math.sqrt(self.d_model)   # [B, T]
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()       # signed square root (reference)
        gate = gate.sigmoid().unsqueeze(-1)                          # alpha_t in (0, 1), [B, T, 1]
        if self.record_gate:
            self.last_gate = gate.detach().squeeze(-1)
        value = gate * self.value_proj(e)                       # [B, T, D]
        if self.short_conv is None:                             # H4 ablation: Engram conv off
            return value
        return value + self.short_conv(value)
