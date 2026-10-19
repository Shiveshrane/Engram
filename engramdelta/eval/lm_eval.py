"""Phase 2 language-model evaluation (design §8). Any change to this file must be noted in CHANGELOG.md.

For each held-out window (inputs x[0..T-1], targets x[1..T]) the per-token cross-entropy is computed at
every input position p (predicting x[p+1]) and aggregated as:
  * overall          mean over all positions -> loss and perplexity exp(loss)
  * position buckets p in [0, 64), [64, 256), [256, 1024)
  * in-context copy  positions p whose current bigram (x[p-1], x[p]) already occurred earlier in the same
                     window, ending at p' with p - p' > 64. This is an induction-style proxy for long-range
                     recall: the earlier occurrence tells the model what followed that bigram. Computed by
                     copy_mask() below. The mask depends only on the tokens, so it is the same for every arm.
Sums and counts are all-reduced over DDP ranks.
"""
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import dist

BUCKETS = ((0, 64), (64, 256), (256, 1024))
COPY_MIN_DISTANCE = 64


def copy_mask(x: np.ndarray, min_distance: int = COPY_MIN_DISTANCE) -> np.ndarray:
    """x: one window's INPUT tokens [T]. Returns a bool mask [T].

    mask[p] is True iff p >= 1 and the bigram (x[p-1], x[p]) also ends at some p' < p with p - p' > min_distance.
    Such a p' exists iff the FIRST occurrence of that bigram ends before p - min_distance.
    """
    x = np.asarray(x, dtype=np.int64)
    T = len(x)
    mask = np.zeros(T, dtype=bool)
    if T < 2:
        return mask
    vocab = int(x.max()) + 1
    bigram = x[:-1] * vocab + x[1:]                      # bigram ending at p = 1..T-1, stored at index p-1
    _, first_idx, inverse = np.unique(bigram, return_index=True, return_inverse=True)
    first_end = first_idx[inverse] + 1                   # position where the first occurrence ends
    p = np.arange(1, T)
    mask[1:] = first_end < p - min_distance
    return mask


@torch.no_grad()
def evaluate_lm(model, data, device, batch_size: int = 4, max_windows: int = None, amp_dtype=None) -> Dict:
    was_training = model.training
    model.eval()
    rank, world = dist.rank(), dist.world_size()
    names = ["all"] + [f"pos_{a}_{b}" for a, b in BUCKETS] + ["copy"]
    sums = torch.zeros(2 * len(names), dtype=torch.float64, device=device)   # [loss_sum, count] per name
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
    for batch in data.eval_batches(batch_size, rank, world, max_windows):
        ids, tgt = batch["ids"].to(device), batch["targets"].to(device)
        with ctx:
            logits = model(ids)                                              # [b, T, vocab]
        ce = F.cross_entropy(logits.float().flatten(0, 1), tgt.flatten(), reduction="none").view(tgt.shape)
        T = ids.shape[1]
        pos = torch.arange(T, device=device).expand_as(ce)
        cmask = torch.from_numpy(np.stack([copy_mask(r) for r in batch["ids"].numpy()])).to(device)
        masks = [torch.ones_like(cmask)] + [(pos >= a) & (pos < b) for a, b in BUCKETS] + [cmask]
        for i, m in enumerate(masks):
            sums[2 * i] += ce[m].double().sum()
            sums[2 * i + 1] += m.sum()
    dist.all_reduce_sum(sums)
    model.train(was_training)
    out = {}
    for i, n in enumerate(names):
        loss_sum, count = sums[2 * i].item(), sums[2 * i + 1].item()
        out[f"loss_{n}"] = loss_sum / count if count else None
        out[f"tokens_{n}"] = int(count)
    out["ppl"] = float(np.exp(out["loss_all"]))
    return out
