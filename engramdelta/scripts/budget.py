"""GPU-hour estimate for a set of manifests, from MEASURED throughput (plan §6.5).

    python -m engramdelta.scripts.budget manifests/E1.*.yaml --results-dir results --quota-hours 30

Per run: seconds = overhead + steps * global_batch * seq_len / throughput(arm).
Throughput (tokens/s) is taken from finished results with the same arm family and the closest sequence length
(E0.4 and E0.6 give A1/A2/T1/LA measurements). Arms without a measurement fall back to the mean over all arms,
and are flagged with '*'. Already-finished runs count as 0.
"""
import argparse
import glob
import os
from collections import defaultdict

import yaml

from ..data.mqar import MQARConfig
from ..utils.config import load_config
from ..utils.run_log import read_results, result_path
from .sweep import phase_of
from .train import resolve

FAMILY = {"A3": "A2", "A4": "A2", "A6": "A2", "A2_noEconv": "A2", "A2_noGconv": "A2", "A2_noboth": "A2",
          "A5": "A1", "A1_noGconv": "A1", "T2": "T1"}


def measured(results_dir: str):
    by = defaultdict(list)   # arm family -> [(seq_len, tok/s)]
    for r in read_results(results_dir):
        if r.get("status") == "done" and r.get("throughput_tokens_per_s"):
            by[FAMILY.get(r["arm"], r["arm"])].append((r["seq_len"], r["throughput_tokens_per_s"], r["world_size"]))
    return by


def estimate(manifests, results_dir: str, overhead_s: float, nproc: int):
    meas = measured(results_dir)
    all_tp = [tp for v in meas.values() for (_, tp, w) in v if w == nproc]
    fallback = sum(all_tp) / len(all_tp) if all_tp else None
    rows = []
    for m in manifests:
        with open(m) as f:
            runs = yaml.safe_load(f) or []
        for run in runs:
            cfg = resolve(load_config(run["config"], run["set"]))
            if cfg["task"] == "mqar":
                T = MQARConfig.from_dict(cfg["data"]).seq_len
            else:
                T = cfg["data"]["seq_len"]
            tokens = cfg["train"]["steps"] * cfg["train"]["global_batch"] * T
            fam = FAMILY.get(cfg["arm"], cfg["arm"])
            cands = [(abs(t - T), tp) for (t, tp, w) in meas.get(fam, []) if w == nproc]
            flag = ""
            if cands:
                tp = sorted(cands)[0][1]
            elif fallback:
                tp, flag = fallback, "*"
            else:
                tp, flag = None, "?"
            done = os.path.exists(result_path(results_dir, phase_of(run), run["group"], run["run_id"]))
            secs = 0.0 if done else (overhead_s + tokens / tp if tp else float("nan"))
            rows.append({"group": run["group"], "run_id": run["run_id"], "T": T, "tokens": tokens,
                         "tok_per_s": tp, "flag": flag, "done": done, "seconds": secs})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--overhead-s", type=float, default=30.0, help="torchrun start + NCCL init + final eval per run")
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--quota-hours", type=float, default=None)
    args = ap.parse_args(argv)
    files = [f for m in args.manifests for f in sorted(glob.glob(m))]
    rows = estimate(files, args.results_dir, args.overhead_s, args.nproc)
    by = defaultdict(lambda: [0, 0, 0.0, set()])
    for r in rows:
        g = by[r["group"]]
        g[0] += 1
        g[1] += r["done"]
        g[2] += r["seconds"]
        if r["flag"]:
            g[3].add(r["flag"])
    print(f"{'group':8s} {'runs':>5s} {'done':>5s} {'GPU-h left':>11s}  notes")
    total = 0.0
    for g, (n, d, s, flags) in by.items():
        total += s
        note = " ".join(sorted(flags))
        print(f"{g:8s} {n:5d} {d:5d} {s / 3600:11.2f}  {note}")
    print(f"{'TOTAL':8s} {len(rows):5d} {sum(r['done'] for r in rows):5d} {total / 3600:11.2f}")
    print("  * = no throughput measured for this arm family (mean of all arms used); ? = nothing measured yet")
    if args.quota_hours:
        print(f"  quota {args.quota_hours:.1f} h -> {'FITS' if total / 3600 <= args.quota_hours else 'OVER BUDGET'}")


if __name__ == "__main__":
    main()
