"""Build run manifests: the exact list of runs of each experiment group (plan §6).

    python -m engramdelta.scripts.make_manifest --group E1.1 --out manifests/
    python -m engramdelta.scripts.make_manifest --group all-phase1 --prereg configs/prereg.yaml --out manifests/

A manifest is a YAML list of {run_id, group, config, set: [dotted overrides]}. Values that are decided after
Phase 0 (N_star, step budget, LRs, Phase 2 token budget) come from configs/prereg.yaml. "auto" entries are
resolved from finished results by the pre-registered rules below, and the tool refuses to build a group whose
inputs are missing.

Pre-registered resolution rules (also in PREREG.md):
  LR "auto"       arm's LR = the candidate with the highest val_recall in E1.0a (ties -> the smaller LR).
                  A3, A4 use A2's LR; A5, A6, A1_noGconv use A1's; A2_* use A2's.
  Phase 2 tokens  tokens = min(phase2_max_tokens, 0.9 * min(throughput of A1, A2 in E2.0) * hours_per_run * 3600),
                  rounded down to a whole number of 65,536-token steps.
  E0.6 / E0.7 N   N = N_star from prereg.yaml if set, otherwise N_star computed from the finished E0.4 results
                  (largest N with A1 mean recall >= 0.9). E0.6 / E0.7 therefore run after E0.4.
  Primary F       primary_cell.F comes from the E0.7 clutter pilot (largest F in {256, 128} where A1 reaches 0.5 val
                  recall). Every Phase 1 cell that compares arms "under clutter" uses it; E1.1 keeps F in {128, 256, 512}.
"""
import argparse
import os
import sys
from typing import Dict, List

import yaml

from ..utils.run_log import read_results

PHASE0_N = [16, 32, 64, 128, 256]
PHASE0_STEPS = 4000      # E0.4 / E0.5 training steps (pre-registered; the value of the first NB02 run)
DESIGN_F = (128, 256, 512)   # the design's clutter levels (design §7.2), besides F = 0
E07_CELLS = [(64, 1.0), (128, 1.0), (256, 1.0), (512, 0.0)]   # clutter pilot: (F, lambda)
LR_INHERIT = {"A3": "A2", "A4": "A2", "A5": "A1", "A6": "A1", "A1_noGconv": "A1",
              "A2_noEconv": "A2", "A2_noGconv": "A2", "A2_noboth": "A2"}


def fmt(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def mqar_run(group, phase_dir, arm, seed, N, F=0, P=64, placement="a", lam=1.0, lr=None, steps=None,
             extra=(), tag="") -> dict:
    sets = [f"group={group}", f"seed={seed}", f"data.N={N}", f"data.F={F}", f"data.P={P}",
            f"data.placement={placement}", f"data.filler_weight={lam}"]
    if lr is not None:
        sets.append(f"train.lr={lr}")
    if steps is not None:
        sets.append(f"train.steps={steps}")
    sets += list(extra)
    rid = f"{group}__{arm}__N{N}_F{F}_P{P}_{placement}_lam{fmt(lam)}" + (f"_lr{fmt(lr)}" if lr is not None else "") \
          + f"_s{seed}{tag}"
    return {"run_id": rid, "group": group, "config": f"configs/{phase_dir}/{arm}.yaml", "set": sets}


def load_prereg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def need(pr: dict, key: str):
    if pr.get(key) in (None, "", "null"):
        sys.exit(f"configs/prereg.yaml: '{key}' is not set yet (fill it after Phase 0, commit, then rerun)")
    return pr[key]


def resolve_lrs(pr: dict, results_dir: str) -> Dict[str, float]:
    lrs = {}
    pilot = [r for r in read_results(os.path.join(results_dir, "phase1", "E1.0a")) if r.get("status") == "done"]
    for arm, v in (pr.get("lr") or {}).items():
        if v != "auto":
            lrs[arm] = float(v)
            continue
        cands = [r for r in pilot if r["arm"] == arm]
        if len(cands) < len(pr["lr_candidates"]):
            sys.exit(f"LR for {arm} is 'auto' but E1.0a has only {len(cands)}/{len(pr['lr_candidates'])} finished runs")
        best = sorted(cands, key=lambda r: (-r["metrics"]["val_recall"], r["train"]["lr"]))[0]
        lrs[arm] = float(best["train"]["lr"])
    for arm, src in LR_INHERIT.items():
        lrs.setdefault(arm, lrs.get(src))
    return lrs


def phase2_steps(pr: dict, results_dir: str, tokens_per_step: int = 64 * 1024) -> int:
    tok = pr.get("phase2_train_tokens")
    if tok in (None, "auto"):
        pilot = [r for r in read_results(os.path.join(results_dir, "phase2", "E2.0"))
                 if r.get("status") == "done" and r.get("throughput_tokens_per_s")]
        tputs = [r["throughput_tokens_per_s"] for r in pilot if r["arm"] in ("A1", "A2")]
        if len(tputs) < 2:
            sys.exit("phase2_train_tokens is 'auto' but the E2.0 throughput pilot (A1, A2) has not finished")
        tok = min(pr["phase2_max_tokens"], 0.9 * min(tputs) * pr["phase2_hours_per_run"] * 3600)
    return max(1, int(tok) // tokens_per_step)


def e06_n(pr: dict, results_dir: str) -> int:
    """N for E0.6: prereg N_star if set, else N* from the finished E0.4 results."""
    if pr.get("N_star") not in (None, "", "null"):
        return int(pr["N_star"])
    from ..analysis import load, n_star
    df = load(results_dir)
    n = n_star(df) if not df.empty and "E0.4" in set(df["group"]) else None
    if n is None:
        sys.exit("E0.6 needs N*: run E0.4 first (no A1 cell reached 0.9 mean recall, or no E0.4 results found)")
    return int(n)


def build(group: str, pr: dict, results_dir: str) -> List[dict]:
    runs: List[dict] = []
    seeds = pr.get("seeds", [0, 1, 2])
    if group == "E0.3":        # determinism: same run twice (2-GPU DDP)
        for rep in (0, 1):
            r = mqar_run(group, "phase1", "A2", 0, 16, extra=["phase=phase0", "train.steps=300"], tag=f"_rep{rep}")
            runs.append(r)
    elif group == "E0.4":      # known result: GDN vs vanilla linear attention, plain MQAR
        for N in PHASE0_N:
            for arm in ("LA", "A1"):
                runs += [mqar_run(group, "phase0", arm, s, N, steps=PHASE0_STEPS) for s in seeds]
            runs.append(mqar_run(group, "phase0", "T1", 0, N, steps=PHASE0_STEPS))
    elif group == "E0.5":      # leak test: Engram-only probe must be at chance
        runs += [mqar_run(group, "phase0", "A6", 0, N, steps=PHASE0_STEPS) for N in PHASE0_N]
    elif group == "E0.6":      # step-budget curve and real per-run time, at the Phase 1 primary cell (N = N*)
        N = e06_n(pr, results_dir)
        for arm in ("A1", "A2"):
            runs.append(mqar_run(group, "phase1", arm, 0, N, 512, 64, "a",
                                 extra=["phase=phase0", "train.steps=6000", "train.eval_every=250"]))
    elif group == "E0.7":      # clutter pilot: which clutter level still lets A1 learn (sets primary F + step budget)
        N = e06_n(pr, results_dir)
        for F, lam in E07_CELLS:
            runs.append(mqar_run(group, "phase1", "A1", 0, N, F, 64, "a", lam,
                                 extra=["phase=phase0", "train.steps=6000", "train.eval_every=250"]))
    elif group.startswith("E1"):
        Ns = need(pr, "N_star")
        steps = need(pr, "phase1_steps")
        pc = pr["primary_cell"]
        if pc.get("F") in (None, "", "null"):
            sys.exit("configs/prereg.yaml: 'primary_cell.F' is not set yet (from the E0.7 clutter pilot, NB04)")
        Fp = int(pc["F"])
        cells_6 = [(N, 0, "a") for N in (Ns, 2 * Ns)] + [(N, Fp, pl) for N in (Ns, 2 * Ns) for pl in ("a", "b")]
        if group == "E1.0a":
            for arm in ("A1", "A2", "T1"):
                for lr in pr["lr_candidates"]:
                    runs.append(mqar_run(group, "phase1", arm, 0, Ns, Fp, 64, "a", 1.0, lr, steps))
            return runs
        lrs = resolve_lrs(pr, results_dir)
        if group == "E1.0c":
            for arm in ("A1", "A2"):
                runs += [mqar_run(group, "phase1", arm, s, Ns, pc["F"], pc["P"], pc["placement"], 0.0, lrs[arm], steps)
                         for s in seeds]
        elif group == "E1.1":
            cells = []
            for N in (Ns, 2 * Ns):
                cells.append((N, 0, 64, "a"))
                cells += [(N, F, 64, pl) for F in DESIGN_F for pl in ("a", "b")]
            cells += [(Ns, Fp, P, pl) for P in (8, 512) for pl in ("a", "b")]
            for (N, F, P, pl) in cells:
                runs += [mqar_run(group, "phase1", "A1", s, N, F, P, pl, 1.0, lrs["A1"], steps,
                                  extra=["train.save_final=true"]) for s in seeds]
        elif group == "E1.2":
            for arm in ("A2", "A3", "A4", "A5"):
                for (N, F, pl) in cells_6:
                    runs += [mqar_run(group, "phase1", arm, s, N, F, 64, pl, 1.0, lrs[arm], steps,
                                      extra=["train.save_final=true"]) for s in seeds]
            for F in [f for f in DESIGN_F if f != Fp]:      # A2 dose-response at the other design levels
                runs += [mqar_run(group, "phase1", "A2", s, Ns, F, 64, "a", 1.0, lrs["A2"], steps,
                                  extra=["train.save_final=true"]) for s in seeds]
        elif group == "E1.2r":
            runs += [mqar_run(group, "phase1", "T1", 0, N, F, 64, pl, 1.0, lrs["T1"], steps) for (N, F, pl) in cells_6]
        elif group == "E1.2l":
            runs += [mqar_run(group, "phase1", "A6", 0, 2 * Ns, Fp, 64, pl, 1.0, lrs["A6"], steps) for pl in ("a", "b")]
        elif group == "E1.3":
            for arm in ("A2_noEconv", "A2_noGconv", "A2_noboth", "A1_noGconv"):
                runs += [mqar_run(group, "phase1", arm, s, Ns, pc["F"], pc["P"], pc["placement"], 1.0, lrs[arm], steps)
                         for s in seeds]
        else:
            sys.exit(f"unknown group {group}")
    elif group == "E2.0":      # throughput pilot
        for arm in ("A1", "A2"):
            runs.append({"run_id": f"E2.0__{arm}__pilot", "group": group, "config": f"configs/phase2/{arm}.yaml",
                         "set": ["group=E2.0", "seed=0", "train.steps=100", "train.warmup=10",
                                 "train.hub_ckpt=false", "train.save_final=false", "train.final_val_windows=16"]})
    elif group in ("E2.1", "E2.2"):
        steps = phase2_steps(pr, results_dir)
        arms = ("A1", "A2", "A3") if group == "E2.1" else ("T1", "T2")
        for arm in arms:
            runs.append({"run_id": f"{group}__{arm}__steps{steps}_s0", "group": group,
                         "config": f"configs/phase2/{arm}.yaml",
                         "set": [f"group={group}", "seed=0", f"train.steps={steps}"]})
    else:
        sys.exit(f"unknown group {group}")
    return runs


GROUP_SETS = {
    "all-phase0": ["E0.3", "E0.4", "E0.5", "E0.6", "E0.7"],
    "all-phase1": ["E1.0a", "E1.0c", "E1.1", "E1.2", "E1.2r", "E1.2l", "E1.3"],
    "all-phase2": ["E2.0", "E2.1", "E2.2"],
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, help="E0.4, E1.1, ... or all-phase0 / all-phase1 / all-phase2")
    ap.add_argument("--prereg", default="configs/prereg.yaml")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--extra-set", action="append", default=[],
                    help="override appended to every run, e.g. data.path=/kaggle/input/x (repeatable)")
    args = ap.parse_args(argv)
    pr = load_prereg(args.prereg)
    groups = GROUP_SETS.get(args.group, [args.group])
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for g in groups:
        runs = build(g, pr, args.results_dir)
        for r in runs:
            r["set"] += list(args.extra_set)
        path = os.path.join(args.out, f"{g}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(runs, f, sort_keys=False)
        print(f"{g}: {len(runs)} runs -> {path}")
        total += len(runs)
    print(f"total: {total} runs")


if __name__ == "__main__":
    main()
