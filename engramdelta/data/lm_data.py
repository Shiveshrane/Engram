"""Language-model data (Phase 2): fixed-length windows from pre-tokenized uint16 memmaps.

Files (written by scripts/prepare_lm_data.py, published as a Kaggle Dataset):
    train.bin, val.bin        uint16 token ids. Documents are separated by eos_id. val docs never appear in train.
    meta.json                 vocab_size, eos_id, token counts, source info
    tokenizer.json            the 16k BPE tokenizer
    engram_compress.npy       Engram tokenizer-compression lookup [vocab_size] int64

A window is seq_len + 1 consecutive tokens (inputs = w[:-1], targets = w[1:]). Windows do not overlap.
Training order: one deterministic permutation of all windows per epoch, seeded by (seed, epoch). Global
sample g goes to epoch g // n_windows. In each (step, micro-step), rank r takes the r-th block of
per-rank-batch samples, so DDP ranks never share samples and resuming from a step number is exact.
"""
import json
import os
from typing import Dict, Iterator

import numpy as np
import torch

from .mqar import stream_seed

_NS_LM = 11


class LMData:
    def __init__(self, path: str, seq_len: int, split: str):
        self.path, self.seq_len, self.split = path, seq_len, split
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.tokens = np.memmap(os.path.join(path, f"{split}.bin"), dtype=np.uint16, mode="r")
        self.n_windows = (len(self.tokens) - 1) // seq_len
        assert self.n_windows > 0, f"{split}.bin is shorter than one window"
        self._perm_epoch, self._perm = None, None

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def window(self, i: int) -> torch.Tensor:
        s = i * self.seq_len
        return torch.from_numpy(self.tokens[s:s + self.seq_len + 1].astype(np.int64))

    def _batch_from_windows(self, idx) -> Dict[str, torch.Tensor]:
        w = torch.stack([self.window(int(i)) for i in idx])          # [b, L+1]
        ids, targets = w[:, :-1].contiguous(), w[:, 1:].contiguous()
        return {"ids": ids, "targets": targets, "kinds": torch.ones_like(ids)}

    def _perm_for(self, epoch: int, seed: int) -> torch.Tensor:
        if self._perm_epoch != (epoch, seed):
            g = torch.Generator().manual_seed(stream_seed(_NS_LM, seed, epoch))
            self._perm, self._perm_epoch = torch.randperm(self.n_windows, generator=g), (epoch, seed)
        return self._perm

    def train_batch(self, B: int, seed: int, rank: int, world: int, step: int, micro: int,
                    grad_accum: int) -> Dict[str, torch.Tensor]:
        first = ((step * grad_accum + micro) * world + rank) * B
        idx = []
        for g_idx in range(first, first + B):
            epoch, j = divmod(g_idx, self.n_windows)
            idx.append(self._perm_for(epoch, seed)[j])
        return self._batch_from_windows(idx)

    def eval_batches(self, B: int, rank: int, world: int, max_windows: int = None) -> Iterator[Dict[str, torch.Tensor]]:
        """Sequential windows 0..n-1 (n = min(n_windows, max_windows)), sharded as rank::world."""
        n = self.n_windows if max_windows is None else min(self.n_windows, max_windows)
        mine = list(range(rank, n, world))
        for s in range(0, len(mine), B):
            yield {**self._batch_from_windows(mine[s:s + B]), "window_idx": torch.tensor(mine[s:s + B])}
