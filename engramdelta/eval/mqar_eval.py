"""MQAR evaluation: recall accuracy on a cell's fixed held-out set, sharded over DDP ranks.

recall = (# query positions whose argmax prediction equals the value) / (# queries), with argmax over the
full vocabulary. The counts are summed over ranks with all_reduce, so the number is exact and independent of
world size. For the "facts" variant (data/mqar.py) it also reports recall_ctx (in-context queries) and recall_fact
(fixed fact queries), from a second forward pass over the same sequences. Any change to this file must be noted in CHANGELOG.md (design rule 3).
"""
from contextlib import nullcontext
from typing import Dict

import torch

from ..data.mqar import MQARConfig, fixed_set
from ..utils import dist


@torch.no_grad()
def evaluate_mqar(model, cfg: MQARConfig, device, split: str = "eval", size: int = None,
                  batch_size: int = 100, amp_dtype=None) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    data = fixed_set(cfg, split, size)
    rank, world = dist.rank(), dist.world_size()
    mine = torch.arange(rank, data["ids"].shape[0], world)
    by_kind = "qtype" in data
    # correct, n_answer, answer_loss_sum, n_filler, filler_loss_sum, correct_ctx, n_ctx, correct_fact, n_fact
    sums = torch.zeros(9, dtype=torch.float64, device=device)
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
    for s in range(0, len(mine), batch_size):
        idx = mine[s:s + batch_size]
        ids, tgt, kinds = (data[k][idx].to(device) for k in ("ids", "targets", "kinds"))
        with ctx:
            out = model(ids, tgt, kinds, filler_weight=1.0)
        n_a, n_f = out["n_main"].double(), out["n_filler"].double()
        sums[:5] += torch.stack([out["correct_main"].double(), n_a, out["loss_main"].double() * n_a,
                                 n_f, out["loss_filler"].double() * n_f])
        if by_kind:
            qt = data["qtype"][idx].to(device)
            sel = qt > 0
            with ctx:
                pred = model.lm_head(model.hidden(ids)[sel]).argmax(-1)
            hit, q = (pred == tgt[sel]).double(), qt[sel]
            sums[5:] += torch.stack([hit[q == 1].sum(), (q == 1).sum().double(), hit[q == 2].sum(), (q == 2).sum().double()])
    dist.all_reduce_sum(sums)
    model.train(was_training)
    correct, n_a, la, n_f, lf, c_ctx, n_ctx, c_fact, n_fact = sums.tolist()
    res = {"recall": correct / max(n_a, 1), "answer_loss": la / max(n_a, 1),
           "filler_loss": lf / max(n_f, 1) if n_f > 0 else None, "n_queries": int(n_a)}
    if by_kind:
        res.update({"recall_ctx": c_ctx / max(n_ctx, 1), "recall_fact": c_fact / max(n_fact, 1)})
    return res
