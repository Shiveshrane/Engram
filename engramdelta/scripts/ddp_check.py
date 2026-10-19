"""DDP correctness checks. Run under torchrun with 2 processes (CPU/gloo on a laptop, CUDA/NCCL on Kaggle):

    torchrun --standalone --nproc_per_node=2 -m engramdelta.scripts.ddp_check --workdir /tmp/ddp_check

  1. gradient equivalence  2-rank DDP with B/2 per rank gives the same gradients as one process with batch B
                           (arms A1, A2, A3; A3 has frozen parameters)
  2. parameter sync        after 30 optimizer steps on different data per rank, all ranks hold identical parameters
  3. exact resume          stop at step 10, checkpoint, resume to step 20 == an uninterrupted 20-step run
Prints PASS/FAIL per check; exit code 1 if any check fails.
"""
import argparse
import copy
import os
import shutil
import sys

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import mqar
from ..models.model import build_model, merge
from ..utils import dist
from .train import train

TINY_DATA = {"N": 8, "F": 24, "P": 8, "placement": "a", "n_keys": 64, "n_values": 64, "n_filler": 32,
             "filler_weight": 1.0, "eval_size": 64}
TINY_MODEL = {"d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "mlp_hidden": 64,
              "engram": {"layers": [1], "n_heads_per_order": 2, "embed_per_order": 32, "rows_per_head": 101}}
ARMS = {"A1": {}, "A2": {"engram": {"enabled": True}}, "A3": {"engram": {"enabled": True, "frozen_table": True}}}


def tiny_cfg(arm="A2", **train_over):
    return {"task": "mqar", "arm": arm, "phase": "scratch", "group": "ddp_check", "seed": 0, "data": dict(TINY_DATA),
            "model": merge(TINY_MODEL, ARMS[arm]),
            "train": {"steps": 20, "global_batch": 8, "lr": 3e-3, "warmup": 2, "log_every": 2, "ckpt_every": 0,
                      "sync_check_every": 5, "precision": "fp32", "eval_batch": 32, **train_over}}


def grad_equivalence(device, tol=1e-5) -> bool:
    ok = True
    mq = mqar.MQARConfig.from_dict(TINY_DATA)
    world, rank = dist.world_size(), dist.rank()
    batch = mqar.train_batch(mq, 4 * world, seed=0, rank=0, step=0)        # identical full batch on every rank
    batch = {k: v.to(device) for k, v in batch.items()}
    for arm, over in ARMS.items():
        torch.manual_seed(0)
        model = build_model(merge(TINY_MODEL, {**over, "vocab_size": mq.vocab_size}), device=device)
        ref = copy.deepcopy(model)
        ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False)
        b = 4
        sl = slice(rank * b, (rank + 1) * b)
        ddp(batch["ids"][sl], batch["targets"][sl], batch["kinds"][sl])["loss"].backward()
        ref(batch["ids"], batch["targets"], batch["kinds"])["loss"].backward()
        worst = 0.0
        for (name, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
            if p.grad is None and q.grad is None:
                continue
            diff = (p.grad - q.grad).abs().max().item() / (q.grad.abs().max().item() + 1e-12)
            worst = max(worst, diff)
        passed = worst < tol
        ok &= passed
        if dist.is_main():
            print(f"[ddp_check] grad equivalence {arm}: max relative diff {worst:.2e} -> {'PASS' if passed else 'FAIL'}")
    return ok


def param_sync(workdir) -> bool:
    cfg = tiny_cfg("A2", steps=30, sync_check_every=10)
    train(cfg, "sync", os.path.join(workdir, "results"), os.path.join(workdir, "ckpt"), write_result_file=False, verbose=False)
    return True   # train() raises if the sync guard finds diverged ranks


def exact_resume(workdir, device) -> bool:
    ck = os.path.join(workdir, "ckpt")
    full = train(tiny_cfg("A2"), "full", os.path.join(workdir, "r"), ck, write_result_file=False, verbose=False)
    train(tiny_cfg("A2"), "resumed", os.path.join(workdir, "r"), ck, stop_at_step=10, write_result_file=False, verbose=False)
    res = train(tiny_cfg("A2"), "resumed", os.path.join(workdir, "r"), ck, write_result_file=False, verbose=False)
    a = torch.tensor([c[1:3] for c in full["curve"]])
    b = torch.tensor([c[1:3] for c in res["curve"]])
    tol = 0.0 if device.type == "cpu" else 1e-4
    diff = (a - b).abs().max().item() if a.shape == b.shape else float("inf")
    ok = diff <= tol and full["metrics"]["recall"] == res["metrics"]["recall"]
    if dist.is_main():
        print(f"[ddp_check] exact resume: max loss-curve diff {diff:.2e} (tol {tol}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/engramdelta_ddp_check")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    info = dist.setup(args.device)
    if info.world < 2:
        print("[ddp_check] needs WORLD_SIZE >= 2 (run under torchrun --nproc_per_node=2)")
        return 2
    if info.rank == 0:
        shutil.rmtree(args.workdir, ignore_errors=True)
        os.makedirs(args.workdir)
    dist.barrier()
    results = {}
    results["grad_equivalence"] = grad_equivalence(info.device)
    try:
        results["param_sync"] = param_sync(args.workdir)
    except RuntimeError as e:
        results["param_sync"] = False
        print(f"[ddp_check] rank {info.rank}: {e}")
    if info.rank == 0:
        print(f"[ddp_check] parameter sync after 30 steps -> {'PASS' if results['param_sync'] else 'FAIL'}")
    for name in ("full", "resumed"):
        if info.rank == 0:
            shutil.rmtree(os.path.join(args.workdir, "ckpt", name), ignore_errors=True)
    dist.barrier()
    results["exact_resume"] = exact_resume(args.workdir, info.device)
    all_ok = all(results.values())
    if info.rank == 0:
        print(f"[ddp_check] {'ALL PASS' if all_ok else 'SOME CHECKS FAILED'}: {results}")
    dist.cleanup()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
