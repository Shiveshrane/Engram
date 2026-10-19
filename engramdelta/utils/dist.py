"""Synchronous data-parallel helpers (torch.distributed). They also work in a plain single process.

Launch with torchrun (one process per GPU). torchrun sets RANK / WORLD_SIZE / LOCAL_RANK. Backend: NCCL on
CUDA, gloo on CPU (so the DDP logic can be tested on a laptop). With WORLD_SIZE unset, everything here is a no-op.
"""
import os
from dataclasses import dataclass

import torch
import torch.distributed as tdist


@dataclass
class DistInfo:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    backend: str


def _pick_device(pref: str, local_rank: int) -> torch.device:
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if pref == "cuda":
        raise RuntimeError("train.device=cuda but CUDA is not available")
    if pref == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def setup(device_pref: str = "auto") -> DistInfo:
    """Idempotent: safe to call more than once in the same process."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _pick_device(device_pref, local_rank)
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not tdist.is_initialized():
        try:
            tdist.init_process_group(backend=backend, device_id=device if device.type == "cuda" else None)
        except TypeError:   # torch < 2.3 has no device_id argument
            tdist.init_process_group(backend=backend)
    return DistInfo(rank, world, local_rank, device, backend if world > 1 else "none")


def is_dist() -> bool:
    return tdist.is_available() and tdist.is_initialized()


def rank() -> int:
    return tdist.get_rank() if is_dist() else 0


def world_size() -> int:
    return tdist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def barrier():
    if is_dist():
        tdist.barrier()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_dist():
        tdist.all_reduce(t, op=tdist.ReduceOp.SUM)
    return t


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if is_dist():
        tdist.all_reduce(t, op=tdist.ReduceOp.SUM)
        t /= world_size()
    return t


def any_rank(flag: bool, device) -> bool:
    """True on every rank if flag is True on any rank. Keeps decisions like 'stop now' in sync."""
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    if is_dist():
        tdist.all_reduce(t, op=tdist.ReduceOp.MAX)
    return bool(t.item() > 0)


@torch.no_grad()
def params_in_sync(model: torch.nn.Module, device) -> bool:
    """Sync guard: every rank must hold bit-identical parameters. Compares per-parameter sums and abs-sums."""
    if not is_dist():
        return True
    sig = torch.stack([torch.stack([p.double().sum(), p.double().abs().sum()]) for p in model.parameters()]).flatten()
    sig = sig.to(device)
    gathered = [torch.empty_like(sig) for _ in range(world_size())]
    tdist.all_gather(gathered, sig)
    return all(torch.equal(gathered[0], g) for g in gathered[1:])


def gather_objects(obj):
    """List of obj from every rank (on every rank)."""
    if not is_dist():
        return [obj]
    out = [None] * world_size()
    tdist.all_gather_object(out, obj)
    return out


def cleanup():
    if is_dist():
        tdist.destroy_process_group()
