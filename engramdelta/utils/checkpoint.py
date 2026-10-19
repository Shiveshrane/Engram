"""Resumable checkpoints (free-tier sessions get killed).

Rank 0 writes model + optimizer + GradScaler + LR scheduler + step + the RNG states of EVERY rank
(gathered), atomically (tmp file, then os.replace). Every rank loads the same file and restores its own RNG
state. The data streams are seeded by the step number, so they need no saved state.
"""
import os
from typing import Optional

import torch

from . import dist


def _rng_state():
    st = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def save_checkpoint(path: str, model, optimizer, scaler, scheduler, step: int, extra: Optional[dict] = None):
    rng_all = dist.gather_objects(_rng_state())
    if dist.is_main():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
                    "step": step, "rng": rng_all, "extra": extra or {}}, tmp)
        os.replace(tmp, path)
    dist.barrier()


def load_checkpoint(path: str, model, optimizer, scaler, scheduler, device) -> dict:
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    scheduler.load_state_dict(ck["scheduler"])
    rng = ck["rng"][dist.rank()] if dist.rank() < len(ck["rng"]) else ck["rng"][0]
    torch.set_rng_state(rng["torch"].cpu())
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["cuda"].cpu())
    return {"step": ck["step"], "extra": ck["extra"]}


def save_weights(path: str, model, meta: Optional[dict] = None):
    """Final weights only (small), for later analysis such as state metrics."""
    if dist.is_main():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save({"model": model.state_dict(), "meta": meta or {}}, tmp)
        os.replace(tmp, path)
    dist.barrier()
