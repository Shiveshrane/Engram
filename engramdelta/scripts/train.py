"""Train one run (MQAR or LM) with synchronous data-parallel training.

    # 2 GPUs (Kaggle T4 x2):
    torchrun --standalone --nproc_per_node=2 -m engramdelta.scripts.train --config configs/phase1/A2.yaml \
        --set data.N=64 --set data.F=512 --set seed=1 --run-id my_run
    # single process (CPU / one GPU):
    python -m engramdelta.scripts.train --config ... --set train.device=cpu

Synchronized DDP, so every rank applies the identical update on every optimizer step:
  * Identical init: same seed on every rank, and DDP also broadcasts rank 0's weights when it wraps the model.
  * Gradients are all-reduced (averaged) on every backward pass. Non-final micro-steps of gradient accumulation
    run under no_sync().
  * fp16: GradScaler's inf check runs on the already all-reduced grads, so every rank skips the same steps.
    Clipping runs on identical grads, so it gives identical results.
  * Stop decisions (time limit) are all-reduced, and a parameter checksum guard runs every
    train.sync_check_every steps.
  * The global batch = per-rank batch x world x grad_accum, fixed in the config independent of the world size.

Exit codes: 0 = finished (or the result already exists), 3 = stopped at the time limit after writing a checkpoint
(rerun the same command to resume), anything else = error.
"""
import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import mqar as mqar_data
from ..data.lm_data import LMData
from ..eval.lm_eval import evaluate_lm
from ..eval.mqar_eval import evaluate_mqar
from ..models.gdn_block import resolve_backend
from ..models.model import (MODEL_DEFAULTS, build_model, count_params, count_params_for_config,
                            match_mlp_hidden, merge)
from ..utils import dist, hub_sync
from ..utils.checkpoint import load_checkpoint, save_checkpoint, save_weights
from ..utils.config import load_config, sibling_arm_config
from ..utils.run_log import env_info, git_commit, now, result_path, write_result

TRAIN_DEFAULTS = {
    "steps": 2000, "global_batch": 64, "grad_accum": 1,
    "lr": 1e-3, "min_lr_ratio": 0.1, "warmup": 100, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8,
    "clip": 1.0, "precision": "fp16", "device": "auto",
    "log_every": 50, "eval_every": 0, "eval_size_during": 500, "eval_batch": 100,
    "ckpt_every": 500, "save_final": False, "hub_ckpt": False, "sync_check_every": 500,
    "final_val_windows": 200,
}


def resolve(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["train"] = merge(TRAIN_DEFAULTS, cfg.get("train"))
    cfg["model"] = merge(MODEL_DEFAULTS, cfg.get("model"))
    cfg.setdefault("seed", 0)
    cfg.setdefault("phase", "scratch")
    cfg.setdefault("group", "misc")
    cfg.setdefault("arm", "unknown")
    return cfg


def default_run_id(cfg: dict) -> str:
    t, d = cfg["train"], cfg["data"]
    if cfg["task"] == "mqar":
        cell = f"N{d['N']}_F{d.get('F', 0)}_P{d.get('P', 64)}_{d.get('placement', 'a')}_lam{d.get('filler_weight', 1.0)}"
    else:
        cell = f"steps{t['steps']}"
    return f"{cfg['arm']}__{cell}_lr{t['lr']}_s{cfg['seed']}"


def param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = name == "embed.weight" or name.endswith("table.embedding.weight")
        if p.ndim < 2 or is_embedding or getattr(p, "_no_weight_decay", False):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda_factory(steps: int, warmup: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


def setup_task(cfg: dict):
    """Returns (vocab_size, compress_lookup, seq_len, batch_fn(step, micro, B), eval_fn(model, final), data objects)."""
    d = cfg["data"]
    if cfg["task"] == "mqar":
        mq = mqar_data.MQARConfig.from_dict(d)
        return {"vocab": mq.vocab_size, "compress": None, "seq_len": mq.seq_len, "mqar": mq,
                "pad_id": mqar_data.PAD}
    if cfg["task"] == "lm":
        train = LMData(d["path"], d["seq_len"], "train")
        val = LMData(d["path"], d["seq_len"], "val")
        compress = None
        if cfg["model"]["engram"]["enabled"] and cfg["model"]["engram"]["compress"]:
            compress = np.load(os.path.join(d["path"], "engram_compress.npy"))
        return {"vocab": train.vocab_size, "compress": compress, "seq_len": d["seq_len"], "train": train,
                "val": val, "pad_id": int(train.meta["eos_id"])}
    raise ValueError(f"unknown task {cfg['task']!r}")


def resolve_model_config(cfg: dict, task: dict) -> dict:
    """Fill vocab / pad id, then apply parameter matching (design §5.3): if model.match_nonembed_of names
    an arm, mlp_hidden is set so this arm's non-embedding parameter count matches that arm's."""
    mc = merge(cfg["model"], {"vocab_size": task["vocab"], "engram": {"pad_id": task["pad_id"]}})
    ref_arm = mc.get("match_nonembed_of")
    if ref_arm:
        ref = resolve(sibling_arm_config(cfg, ref_arm))
        ref_mc = merge(ref["model"], {"vocab_size": task["vocab"], "engram": {"pad_id": task["pad_id"]}})
        target = count_params_for_config(ref_mc, task["compress"])["non_embedding"]
        mc["mlp_hidden"] = match_mlp_hidden(mc, target, compress_lookup=task["compress"])
        mc["matched_target_non_embedding"] = target
    return mc


def train(cfg: dict, run_id: Optional[str] = None, results_dir: str = "results", ckpt_dir: str = "ckpt",
          max_minutes: Optional[float] = None, stop_at_step: Optional[int] = None,
          write_result_file: bool = True, verbose: bool = True) -> dict:
    cfg = resolve(cfg)
    t_cfg = cfg["train"]
    info = dist.setup(t_cfg["device"])
    device, rank, world = info.device, info.rank, info.world
    log = print if (verbose and rank == 0) else (lambda *a, **k: None)
    run_id = run_id or default_run_id(cfg)
    wall0 = time.time()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    task = setup_task(cfg)
    mc = resolve_model_config(cfg, task)
    torch.manual_seed(cfg["seed"])                       # same on every rank -> identical init
    model = build_model(mc, task["compress"], device=device)
    params = count_params(model)
    use_amp = t_cfg["precision"] == "fp16" and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else None
    gdn_backend = resolve_backend(mc["gdn"]["backend"]) if mc["mixer"] == "gdn" else None

    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    broadcast_buffers=False) if world > 1 else model
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(param_groups(model, t_cfg["weight_decay"]), lr=t_cfg["lr"],
                            betas=tuple(t_cfg["betas"]), eps=t_cfg["eps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(t_cfg["steps"], t_cfg["warmup"], t_cfg["min_lr_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    accum = t_cfg["grad_accum"]
    assert t_cfg["global_batch"] % (world * accum) == 0, "global_batch must be divisible by world_size * grad_accum"
    B = t_cfg["global_batch"] // (world * accum)
    if cfg["task"] == "mqar":
        mq = task["mqar"]
        def get_batch(step, micro):
            return mqar_data.train_batch(mq, B, cfg["seed"], rank, step, micro)
    else:
        def get_batch(step, micro):
            return task["train"].train_batch(B, cfg["seed"], rank, world, step, micro, accum)
    filler_weight = float(cfg["data"].get("filler_weight", 1.0)) if cfg["task"] == "mqar" else 1.0

    def run_eval(final: bool) -> dict:
        if cfg["task"] == "mqar":
            if final:
                ev = evaluate_mqar(model, mq, device, "eval", batch_size=t_cfg["eval_batch"], amp_dtype=amp_dtype)
                va = evaluate_mqar(model, mq, device, "val", batch_size=t_cfg["eval_batch"], amp_dtype=amp_dtype)
                return {**ev, "val_recall": va["recall"]}
            return evaluate_mqar(model, mq, device, "val", size=t_cfg["eval_size_during"],
                                 batch_size=t_cfg["eval_batch"], amp_dtype=amp_dtype)
        n = t_cfg["final_val_windows"] if final else max(1, t_cfg["final_val_windows"] // 4)
        return evaluate_lm(model, task["val"], device, max_windows=n, amp_dtype=amp_dtype)

    # ---- resume ----
    ck_path = os.path.join(ckpt_dir, run_id, "last.pt")
    if not os.path.exists(ck_path) and t_cfg["hub_ckpt"]:
        hub_sync.download_file(f"ckpt/{run_id}/last.pt", ck_path)
    start_step, extra = 0, {}
    if os.path.exists(ck_path):
        st = load_checkpoint(ck_path, model, opt, scaler, sched, device)
        start_step, extra = st["step"], st["extra"]
        log(f"[{run_id}] resumed from step {start_step}")
    curve, eval_curve = extra.get("curve", []), extra.get("eval_curve", [])
    skipped, nonfinite = extra.get("skipped_steps", 0), extra.get("nonfinite_losses", 0)
    train_seconds = extra.get("train_seconds", 0.0)
    tokens_per_step = t_cfg["global_batch"] * task["seq_len"]

    def checkpoint(step):
        save_checkpoint(ck_path, model, opt, scaler, sched, step, {
            "curve": curve, "eval_curve": eval_curve, "skipped_steps": skipped, "nonfinite_losses": nonfinite,
            "train_seconds": train_seconds})
        if t_cfg["hub_ckpt"] and dist.is_main():
            hub_sync.push_file(ck_path, f"ckpt/{run_id}/last.pt")

    log(f"[{run_id}] world={world} device={device} B/rank={B} accum={accum} T={task['seq_len']} "
        f"amp={use_amp} gdn_backend={gdn_backend} params={params}")
    model.train()
    log_sums = torch.zeros(3, device=device)          # loss_main, loss_filler, n
    seg_t0 = time.time()
    status = "done"
    step = start_step
    for step in range(start_step, t_cfg["steps"]):
        if stop_at_step is not None and step >= stop_at_step:
            status = "stopped"
            break
        opt.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = {k: v.to(device, non_blocking=True) for k, v in get_batch(step, micro).items()}
            sync_ctx = ddp_model.no_sync() if (world > 1 and micro < accum - 1) else nullcontext()
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with sync_ctx:
                with amp_ctx:
                    out = ddp_model(batch["ids"], batch["targets"], batch["kinds"], filler_weight=filler_weight)
                scaler.scale(out["loss"] / accum).backward()
            log_sums += torch.stack([out["loss_main"].float(), out["loss_filler"].float(), torch.ones((), device=device)])
            if not torch.isfinite(out["loss"].detach()):
                nonfinite += 1
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, t_cfg["clip"])
        scale_before = scaler.get_scale() if use_amp else 1.0
        scaler.step(opt)
        scaler.update()
        if use_amp and scaler.get_scale() < scale_before:
            skipped += 1
        sched.step()
        done = step + 1

        if done % t_cfg["log_every"] == 0 or done == t_cfg["steps"]:
            dist.all_reduce_sum(log_sums)
            lm_, lf_, n_ = log_sums.tolist()
            curve.append([done, lm_ / n_, lf_ / n_, float(gnorm), sched.get_last_lr()[0]])
            log(f"[{run_id}] step {done}/{t_cfg['steps']} loss_main {lm_ / n_:.4f} loss_filler {lf_ / n_:.4f} "
                f"gnorm {float(gnorm):.3f} lr {sched.get_last_lr()[0]:.2e}")
            log_sums.zero_()
        if t_cfg["eval_every"] and done % t_cfg["eval_every"] == 0 and done < t_cfg["steps"]:
            train_seconds += time.time() - seg_t0
            ev = run_eval(final=False)
            eval_curve.append([done, ev])
            log(f"[{run_id}] eval@{done} {ev}")
            seg_t0 = time.time()
        if world > 1 and t_cfg["sync_check_every"] and done % t_cfg["sync_check_every"] == 0:
            if not dist.params_in_sync(model, device):
                raise RuntimeError(f"DDP ranks diverged at step {done} (parameter checksums differ)")
        if t_cfg["ckpt_every"] and done % t_cfg["ckpt_every"] == 0 and done < t_cfg["steps"]:
            train_seconds += time.time() - seg_t0
            checkpoint(done)
            seg_t0 = time.time()
        if max_minutes is not None and done % 10 == 0:
            over = (time.time() - wall0) / 60.0 > max_minutes
            if dist.any_rank(over, device) and done < t_cfg["steps"]:
                status = "preempted"
                break
    train_seconds += time.time() - seg_t0

    if status != "done":
        stop_step = step if status == "stopped" else step + 1
        checkpoint(stop_step)
        log(f"[{run_id}] {status} at step {stop_step}; checkpoint saved to {ck_path}")
        return {"status": status, "step": stop_step, "run_id": run_id, "curve": curve}

    metrics = run_eval(final=True)
    steps_this_session = t_cfg["steps"] - start_step
    record = {
        "run_id": run_id, "phase": cfg["phase"], "group": cfg["group"], "arm": cfg["arm"], "seed": cfg["seed"],
        "task": cfg["task"], "status": "done", "metrics": metrics, "params": params,
        "data": cfg["data"], "model": mc, "train": t_cfg,
        "curve_columns": ["step", "loss_main", "loss_filler", "grad_norm", "lr"], "curve": curve,
        "eval_curve": eval_curve, "skipped_steps": skipped, "nonfinite_losses": nonfinite,
        "train_seconds": train_seconds, "wall_seconds_last_session": time.time() - wall0,
        "throughput_tokens_per_s": tokens_per_step * steps_this_session / max(train_seconds, 1e-9)
        if start_step == 0 else None,
        "seq_len": task["seq_len"], "tokens_per_step": tokens_per_step, "per_rank_batch": B,
        "world_size": world, "dist_backend": info.backend, "amp": use_amp, "gdn_backend": gdn_backend,
        "git_commit": git_commit(), "env": env_info(), "config_source": cfg.get("_source"),
        "overrides": cfg.get("_overrides"), "finished_at": now(),
    }
    if t_cfg["save_final"]:
        save_weights(os.path.join(ckpt_dir, run_id, "final.pt"), model, {"model_config": mc, "data": cfg["data"]})
        if dist.is_main() and hub_sync.enabled():
            hub_sync.push_file(os.path.join(ckpt_dir, run_id, "final.pt"), f"ckpt/{run_id}/final.pt")
    if write_result_file and dist.is_main():
        path = write_result(result_path(results_dir, cfg["phase"], cfg["group"], run_id), record)
        hub_sync.push_file(path, os.path.relpath(path, os.path.dirname(os.path.abspath(results_dir))))
        log(f"[{run_id}] wrote {path}")
    dist.barrier()
    log(f"[{run_id}] final metrics {metrics}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. data.N=64 (repeatable)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default=os.environ.get("ENGRAMDELTA_CKPT_DIR", "ckpt"))
    ap.add_argument("--max-minutes", type=float, default=None, help="checkpoint and exit(3) after this wall time")
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config, args.set))
    run_id = args.run_id or default_run_id(cfg)
    if os.path.exists(result_path(args.results_dir, cfg["phase"], cfg["group"], run_id)):
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[{run_id}] result exists, skipping")
        return 0
    try:
        out = train(cfg, run_id, args.results_dir, args.ckpt_dir, args.max_minutes)
    finally:
        dist.cleanup()
    return 3 if out.get("status") == "preempted" else 0


if __name__ == "__main__":
    sys.exit(main())
