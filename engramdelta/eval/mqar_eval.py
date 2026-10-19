"""MQAR evaluation: recall accuracy on a cell's fixed held-out set, sharded over DDP ranks.

recall = (# query positions whose argmax prediction equals the value) / (# queries), with argmax over the
full vocabulary. The counts are summed over ranks with all_reduce, so the number is exact and independent of
world size. Any change to this file must be noted in CHANGELOG.md (design rule 3).
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
    sums = torch.zeros(5, dtype=torch.float64, device=device)    # correct, n_answer, answer_loss_sum, n_filler, filler_loss_sum
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
    for s in range(0, len(mine), batch_size):
        idx = mine[s:s + batch_size]
        ids, tgt, kinds = (data[k][idx].to(device) for k in ("ids", "targets", "kinds"))
        with ctx:
            out = model(ids, tgt, kinds, filler_weight=1.0)
        n_a, n_f = out["n_main"].double(), out["n_filler"].double()
        sums += torch.stack([out["correct_main"].double(), n_a, out["loss_main"].double() * n_a,
                             n_f, out["loss_filler"].double() * n_f])
    dist.all_reduce_sum(sums)
    model.train(was_training)
    correct, n_a, la, n_f, lf = sums.tolist()
    return {"recall": correct / max(n_a, 1), "answer_loss": la / max(n_a, 1),
            "filler_loss": lf / max(n_f, 1) if n_f > 0 else None, "n_queries": int(n_a)}
