"""MQAR (multi-query associative recall), with optional "local clutter".

Follows the MQAR task of Zoology (Arora et al., arXiv 2312.04927): N key-value pairs, then every key is
queried and the model must output its value. The clutter variant is our addition (design §7.2).

Vocabulary (all ranges disjoint):
    0 PAD, 1 BLANK, 2-3 reserved
    keys    [key_offset,    key_offset + n_keys)
    values  [value_offset,  value_offset + n_values)
    filler  [filler_offset, filler_offset + n_filler)

Sequence layout, length T = 4N + F:
    placement "a"  [k1 v1 k2 v2 ... kN vN] [F clutter tokens] [q1 BLANK q2 BLANK ... qN BLANK]
    placement "b"  [k1 v1 <gap> k2 v2 <gap> ... kN vN]         [q1 BLANK ... qN BLANK]
                   The F clutter tokens are split as evenly as possible into the N-1 gaps between pairs.
  * Keys are distinct within a sequence. Values are drawn with replacement. Both are RESAMPLED FOR EVERY
    SEQUENCE, so no lookup table can memorize key -> value across sequences (design §6 test 6).
  * Queries: every key exactly once, in a random order. The answer (its value) is the target AT the query
    key position. The next input token is BLANK, so the answer never appears in the input.
  * Clutter: a stream of phrases drawn uniformly at random from a pool of P fixed phrases of
    `phrase_len` filler tokens. The pool depends only on (pool_seed, P, phrase_len, n_filler). It is
    shared by train / val / eval and by every arm. Each gap in placement "b" starts a fresh phrase stream.

Loss targets (`kinds`):
    1  answer: at each query key position, target = its value
    2  filler: at every position t whose NEXT token is a filler token, target = that token
       (next-token prediction of the clutter; weighted by filler_weight = lambda in the loss)
    0  no loss
With lambda = 0 the model is never asked to model the clutter. With lambda = 1 it must process the local
patterns, and those can then compete with the stored pairs (see the plan, design issue 1).

Randomness: every batch comes from its own torch.Generator, seeded from (namespace, seed, rank, step,
micro) for training and from (namespace, eval_seed, cell) for the fixed val / eval sets. So the data is
identical across arms, deterministic, resumable from a step number, and the training ranks never share examples.
"""
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Tuple

import torch

PAD, BLANK = 0, 1
N_SPECIAL = 4

# Seed namespaces, so train / val / eval streams can never collide.
_NS_TRAIN, _NS_EVAL, _NS_VAL, _NS_POOL = 1, 2, 3, 4


def stream_seed(*parts: int) -> int:
    """Deterministic 63-bit seed from a tuple of ints (Python's hash() is salted per process, so it is not used)."""
    h = 0
    for p in parts:
        h = (h * 1_000_003 + int(p) + 1) % (2 ** 63 - 1)
    return h


@dataclass
class MQARConfig:
    N: int = 64                 # number of key-value pairs (= number of queries)
    F: int = 0                  # clutter length in tokens
    P: int = 64                 # phrase pool size
    placement: str = "a"        # "a": clutter between pairs and queries; "b": clutter interleaved between pairs
    phrase_len: int = 3
    n_keys: int = 2048
    n_values: int = 2048
    n_filler: int = 512
    filler_weight: float = 1.0  # lambda
    pool_seed: int = 1234
    eval_seed: int = 999_999
    eval_size: int = 2000

    @classmethod
    def from_dict(cls, d: dict) -> "MQARConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def key_offset(self) -> int:
        return N_SPECIAL

    @property
    def value_offset(self) -> int:
        return N_SPECIAL + self.n_keys

    @property
    def filler_offset(self) -> int:
        return N_SPECIAL + self.n_keys + self.n_values

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + self.n_keys + self.n_values + self.n_filler

    @property
    def seq_len(self) -> int:
        return 4 * self.N + self.F


def layout(cfg: MQARConfig) -> Dict:
    """Token positions of every segment. The same for every sequence of a cell.

    Returns dict with
      key_pos, val_pos     [N] positions of the i-th pair's key / value
      clutter              list of (start, length) clutter segments
      query_pos            [N] positions of the query keys (each followed by BLANK)
      end_of_writing       position of the last value token
      end_of_clutter       last position before the queries (= end_of_writing when there is no clutter after the pairs)
      T                    sequence length
    """
    N, F = cfg.N, cfg.F
    assert N >= 1 and F >= 0 and N <= cfg.n_keys
    assert cfg.placement in ("a", "b")
    key_pos: List[int] = []
    clutter: List[Tuple[int, int]] = []
    if cfg.placement == "a" or N == 1:
        key_pos = [2 * i for i in range(N)]
        pairs_end = 2 * N
        if F > 0:
            clutter.append((pairs_end, F))
        query_start = pairs_end + F
    else:
        gaps = [F // (N - 1) + (1 if i < F % (N - 1) else 0) for i in range(N - 1)]
        p = 0
        for i in range(N):
            key_pos.append(p)
            p += 2
            if i < N - 1:
                if gaps[i] > 0:
                    clutter.append((p, gaps[i]))
                p += gaps[i]
        query_start = p
    val_pos = [k + 1 for k in key_pos]
    query_pos = [query_start + 2 * i for i in range(N)]
    T = query_start + 2 * N
    assert T == cfg.seq_len
    return {"key_pos": key_pos, "val_pos": val_pos, "clutter": clutter, "query_pos": query_pos,
            "end_of_writing": val_pos[-1], "end_of_clutter": query_start - 1, "T": T}


def phrase_pool(cfg: MQARConfig) -> torch.Tensor:
    """[P, phrase_len] filler-token phrases. Fixed by (pool_seed, P, phrase_len, n_filler)."""
    g = torch.Generator().manual_seed(stream_seed(_NS_POOL, cfg.pool_seed, cfg.P, cfg.phrase_len, cfg.n_filler))
    return cfg.filler_offset + torch.randint(cfg.n_filler, (cfg.P, cfg.phrase_len), generator=g)


def _clutter_tokens(pool: torch.Tensor, B: int, length: int, g: torch.Generator) -> torch.Tensor:
    """[B, length]: phrases drawn uniformly from the pool, concatenated, truncated to `length`."""
    P, L = pool.shape
    n_phrases = -(-length // L)
    idx = torch.randint(P, (B, n_phrases), generator=g)
    return pool[idx].reshape(B, n_phrases * L)[:, :length]


def generate(cfg: MQARConfig, B: int, g: torch.Generator) -> Dict[str, torch.Tensor]:
    """B sequences. Returns ids, targets, kinds: [B, T] int64 each (on the CPU)."""
    lay = layout(cfg)
    T, N = lay["T"], cfg.N
    key_pos = torch.tensor(lay["key_pos"])
    val_pos = torch.tensor(lay["val_pos"])
    query_pos = torch.tensor(lay["query_pos"])

    ids = torch.full((B, T), PAD, dtype=torch.long)
    keys = torch.rand(B, cfg.n_keys, generator=g).argsort(dim=1)[:, :N] + cfg.key_offset     # distinct per row
    vals = torch.randint(cfg.n_values, (B, N), generator=g) + cfg.value_offset
    ids[:, key_pos] = keys
    ids[:, val_pos] = vals
    if lay["clutter"]:
        pool = phrase_pool(cfg)
        for start, length in lay["clutter"]:
            ids[:, start:start + length] = _clutter_tokens(pool, B, length, g)
    order = torch.rand(B, N, generator=g).argsort(dim=1)          # query order (a permutation per row)
    ids[:, query_pos] = keys.gather(1, order)
    ids[:, query_pos + 1] = BLANK

    targets = torch.zeros((B, T), dtype=torch.long)
    kinds = torch.zeros((B, T), dtype=torch.long)
    targets[:, query_pos] = vals.gather(1, order)
    kinds[:, query_pos] = 1
    next_is_filler = ids[:, 1:] >= cfg.filler_offset               # [B, T-1]
    targets[:, :-1] = torch.where(next_is_filler, ids[:, 1:], targets[:, :-1])
    kinds[:, :-1] = torch.where(next_is_filler, torch.full_like(kinds[:, :-1], 2), kinds[:, :-1])
    return {"ids": ids, "targets": targets, "kinds": kinds}


def train_batch(cfg: MQARConfig, B: int, seed: int, rank: int, step: int, micro: int = 0) -> Dict[str, torch.Tensor]:
    """Training batch for one (rank, step, micro-step). Fresh sequences every call."""
    g = torch.Generator().manual_seed(stream_seed(_NS_TRAIN, seed, rank, step, micro))
    return generate(cfg, B, g)


_FIXED_CACHE: Dict[tuple, Dict[str, torch.Tensor]] = {}


def fixed_set(cfg: MQARConfig, split: str = "eval", size: int = None) -> Dict[str, torch.Tensor]:
    """The fixed held-out set of a cell (identical for every arm and seed).

    split="eval": final metric. split="val": used only to choose hyperparameters (LR pilot E1.0a).
    """
    assert split in ("eval", "val")
    size = cfg.eval_size if size is None else size
    ns = _NS_EVAL if split == "eval" else _NS_VAL
    key = (split, size, cfg.N, cfg.F, cfg.P, cfg.placement, cfg.phrase_len, cfg.n_keys, cfg.n_values,
           cfg.n_filler, cfg.pool_seed, cfg.eval_seed)
    if key not in _FIXED_CACHE:
        g = torch.Generator().manual_seed(stream_seed(ns, cfg.eval_seed, cfg.N, cfg.F, cfg.P,
                                                      ord(cfg.placement), cfg.phrase_len))
        _FIXED_CACHE[key] = generate(cfg, size, g)
    return _FIXED_CACHE[key]
