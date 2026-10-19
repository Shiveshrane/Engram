"""NCCL smoke test for Kaggle's 2 x T4 (NB00).

    torchrun --standalone --nproc_per_node=2 -m engramdelta.scripts.nccl_smoke

Checks that all_reduce returns the right sum on both GPUs, then measures all_reduce bandwidth for a 64 MB
tensor (the Phase 2 Engram tables make the gradient all-reduce about this large).
If this hangs, retry with NCCL_P2P_DISABLE=1 (and NCCL_IB_DISABLE=1), and set NCCL_DEBUG=INFO to see the logs.
"""
import sys
import time

import torch
import torch.distributed as tdist

from ..utils import dist


def main():
    info = dist.setup("auto")
    assert info.world >= 2, "run under torchrun --nproc_per_node=2"
    x = torch.full((1024,), float(info.rank + 1), device=info.device)
    tdist.all_reduce(x)
    expected = info.world * (info.world + 1) / 2
    ok = bool(torch.allclose(x, torch.full_like(x, expected)))
    big = torch.ones(16 * 1024 * 1024, device=info.device)        # 64 MB fp32
    for _ in range(3):
        tdist.all_reduce(big)
    if info.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n = 10
    for _ in range(n):
        tdist.all_reduce(big)
    if info.device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    if info.rank == 0:
        print(f"[nccl_smoke] world={info.world} backend={info.backend} device={info.device} "
              f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
        print(f"[nccl_smoke] all_reduce sum correct: {ok}")
        print(f"[nccl_smoke] 64 MB all_reduce: {dt * 1000:.1f} ms  (algorithm bandwidth {64 / 1024 / dt:.2f} GB/s)")
    dist.cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
