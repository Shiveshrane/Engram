"""Result tables and the pre-registered rule checks (NB02, NB14, NB22).

Rule 6 of the design: this module prints numbers and applies the PRE-REGISTERED rules mechanically
(PREREG.md). It does not interpret them. "exceeds_2sigma" is the rule
|mean_A - mean_B| > 2 * pooled seed std (n >= 3 per arm). The human decides what the results mean.
"""
import math
import os
from typing import Dict, List, Optional

import pandas as pd

from .utils.run_log import read_results

CELL = ["N", "F", "P", "placement", "lam"]
PHASE1_MAIN_GROUPS = {"E1.1", "E1.2", "E1.2r", "E1.2l", "E1.3"}


def load(results_dir: str = "results", phase: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for r in read_results(os.path.join(results_dir, phase) if phase else results_dir):
        if r.get("status") != "done" or "metrics" not in r:
            continue
        d, m, p, t = r.get("data") or {}, r["metrics"], r.get("params") or {}, r.get("train") or {}
        rows.append({
            "run_id": r["run_id"], "phase": r.get("phase"), "group": r.get("group"), "arm": r.get("arm"),
            "seed": r.get("seed"), "N": d.get("N"), "F": d.get("F"), "P": d.get("P"),
            "placement": d.get("placement"), "lam": d.get("filler_weight"), "lr": t.get("lr"), "steps": t.get("steps"),
            "recall": m.get("recall"), "val_recall": m.get("val_recall"), "answer_loss": m.get("answer_loss"),
            "filler_loss": m.get("filler_loss"), "ppl": m.get("ppl"), "loss_all": m.get("loss_all"),
            "loss_copy": m.get("loss_copy"), "non_embedding": p.get("non_embedding"), "total_params": p.get("total"),
            "engram_table": p.get("engram_table"), "world_size": r.get("world_size"),
            "tok_per_s": r.get("throughput_tokens_per_s"), "train_seconds": r.get("train_seconds"),
            "seq_len": r.get("seq_len"), "skipped_steps": r.get("skipped_steps"),
            "nonfinite_losses": r.get("nonfinite_losses"), "gdn_backend": r.get("gdn_backend"),
            "git_commit": r.get("git_commit"),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, keys: List[str], metric: str = "recall") -> pd.DataFrame:
    """mean, sample std (ddof=1) and n of `metric` per group of `keys`."""
    g = df.groupby(keys, dropna=False)[metric]
    return g.agg(mean="mean", std="std", n="count").reset_index()


def pooled_std(sa, na, sb, nb) -> float:
    if na + nb <= 2:
        return float("nan")
    sa = 0.0 if pd.isna(sa) else sa
    sb = 0.0 if pd.isna(sb) else sb
    return math.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))


def compare(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> Dict:
    """Pre-registered significance rule between two sets of per-seed values."""
    na, nb = int(a.count()), int(b.count())
    ma, mb = a.mean(), b.mean()
    sp = pooled_std(a.std(), na, b.std(), nb)
    diff = ma - mb
    return {"A": label_a, "B": label_b, "mean_A": ma, "mean_B": mb, "n_A": na, "n_B": nb,
            "diff_A_minus_B": diff, "pooled_std": sp, "threshold_2sigma": 2 * sp,
            "exceeds_2sigma": bool(na >= 3 and nb >= 3 and abs(diff) > 2 * sp),
            "rule_applicable": bool(na >= 3 and nb >= 3)}


def select(df: pd.DataFrame, arm: str, groups=None, **cell) -> pd.Series:
    m = df["arm"] == arm
    if groups is not None:
        m &= df["group"].isin(groups)
    for k, v in cell.items():
        m &= df[k] == v
    return df.loc[m, "recall"]


# ---------------- Phase 0 ----------------

def known_result_check(df: pd.DataFrame, knee_threshold: float = 0.9) -> Dict:
    """Test 7 (pre-registered): (1) A1 > LA + 2 sigma at >= 2 values of N; (2) A1 mean recall is non-increasing in
    N at and beyond the knee (an increase smaller than 2 x the A1 seed std counts as noise, not an increase)."""
    e = df[df["group"] == "E0.4"]
    if e.empty:
        return {"available": False}
    rows, wins = [], 0
    for N in sorted(e["N"].unique()):
        c = compare(select(e, "A1", N=N), select(e, "LA", N=N), "A1", "LA")
        t1 = select(e, "T1", N=N)
        c.update({"N": int(N), "T1_recall": t1.mean() if len(t1) else None, "A1_std": select(e, "A1", N=N).std()})
        wins += int(c["exceeds_2sigma"] and c["diff_A_minus_B"] > 0)
        rows.append(c)
    table = pd.DataFrame(rows)
    knee = n_star(df, knee_threshold)
    mono = True
    beyond = table[table["N"] >= (knee or 0)].sort_values("N")
    means, stds = beyond["mean_A"].tolist(), beyond["A1_std"].fillna(0).tolist()
    for i in range(1, len(means)):
        if means[i] - means[i - 1] > 2 * max(stds[i], stds[i - 1]):
            mono = False
    return {"available": True, "table": table, "n_wins": wins, "criterion_1_pass": wins >= 2,
            "criterion_2_pass": mono, "pass": wins >= 2 and mono, "N_star": knee}


def n_star(df: pd.DataFrame, threshold: float = 0.9) -> Optional[int]:
    """Largest N with A1 mean recall >= threshold on plain MQAR (E0.4)."""
    s = summarize(df[(df["group"] == "E0.4") & (df["arm"] == "A1")], ["N"])
    ok = s[s["mean"] >= threshold]
    return int(ok["N"].max()) if len(ok) else None


def curve_stats(r: dict, frac: float = 0.98, round_to: int = 500, min_learned: float = 0.5) -> Dict:
    """Val-recall curve of one run with eval_every > 0, and the pre-registered step rule applied to it."""
    curve = [(s, ev["recall"]) for s, ev in r.get("eval_curve", [])] + [(r["train"]["steps"], r["metrics"]["val_recall"])]
    best = max(v for _, v in curve)
    first = min(s for s, v in curve if v >= frac * best)
    learned = best >= min_learned
    return {"N": r["data"]["N"], "F": r["data"]["F"], "lam": r["data"]["filler_weight"], "max_val_recall": best,
            "final_val_recall": r["metrics"]["val_recall"], "first_step_at_98pct": first, "learned": learned,
            "proposal": int(math.ceil(first / round_to) * round_to) if learned else None, "curve": curve,
            "train_seconds": r["train_seconds"], "tok_per_s": r["throughput_tokens_per_s"]}


def clutter_pilot(results_dir: str = "results", N: Optional[int] = None) -> List[Dict]:
    """E0.7 runs (A1 at N*, one clutter level each), sorted by (lambda, F)."""
    rows = [dict(curve_stats(r), run_id=r["run_id"]) for r in read_results(os.path.join(results_dir, "phase0", "E0.7"))
            if N is None or r["data"]["N"] == N]
    return sorted(rows, key=lambda x: (-x["lam"], x["F"]))


def primary_f_rule(rows: List[Dict], candidates=(256, 128)) -> Optional[Dict]:
    """Pre-registered: primary F = the largest candidate F (lambda = 1) where A1's max val recall >= 0.5.
    Returns that run's stats (its "proposal" is the Phase 1 step budget), or None (stop: the human decides)."""
    by_f = {x["F"]: x for x in rows if x["lam"] == 1.0}
    for F in candidates:
        if F in by_f and by_f[F]["learned"]:
            return by_f[F]
    return None


def step_budget_proposal(results_dir: str = "results", N: Optional[int] = None, frac: float = 0.98,
                         round_to: int = 500, min_learned: float = 0.5) -> Dict:
    """E0.6 rule: first eval step where the arm's val recall >= frac * its maximum, rounded up to a multiple of round_to.

    Only E0.6 runs with data.N == N are used (all runs if N is None). The rule is meaningless on a curve that never
    leaves chance, so "learned" is False (and "proposal" None) when the maximum val recall is below min_learned.
    """
    return {r["arm"]: curve_stats(r, frac, round_to, min_learned)
            for r in read_results(os.path.join(results_dir, "phase0", "E0.6"))
            if N is None or r["data"]["N"] == N}


def leak_table(df: pd.DataFrame, n_values: int = 2048) -> pd.DataFrame:
    e = df[df["arm"] == "A6"][["group", "N", "F", "placement", "recall"]].copy()
    e["chance"] = 1.0 / n_values
    e["pass_lt_1pct"] = e["recall"] < 0.01
    return e


# ---------------- Phase 1 ----------------

def h1_grid(df: pd.DataFrame) -> pd.DataFrame:
    e = df[(df["arm"] == "A1") & (df["group"] == "E1.1")]
    return summarize(e, ["N", "placement", "P", "F"]).sort_values(["N", "placement", "P", "F"])


def arms_table(df: pd.DataFrame) -> pd.DataFrame:
    e = df[df["group"].isin(PHASE1_MAIN_GROUPS) & (df["lam"] == 1.0)]
    return summarize(e, ["N", "F", "P", "placement", "arm"]).sort_values(["N", "F", "placement", "arm"])


def gate1_table(df: pd.DataFrame, prereg: dict) -> pd.DataFrame:
    """The pre-registered primary-cell comparisons for Gate 1 (design §7.4). Numbers and rule checks only."""
    Ns = prereg["N_star"]
    pc = prereg["primary_cell"]
    cell = dict(N=Ns, P=pc["P"], placement=pc["placement"], lam=pc["filler_weight"])
    g = PHASE1_MAIN_GROUPS
    rows = [{"hypothesis": "H1: A1(F=0) vs A1(F=512)",
             **compare(select(df, "A1", g, F=0, N=Ns, P=pc["P"], lam=pc["filler_weight"]),
                       select(df, "A1", g, F=pc["F"], **cell), "A1 F=0", f"A1 F={pc['F']}")}]
    for other in ("A1", "A3", "A4"):
        rows.append({"hypothesis": f"H2: A2 vs {other} at the primary cell",
                     **compare(select(df, "A2", g, F=pc["F"], **cell), select(df, other, g, F=pc["F"], **cell),
                               "A2", other)})
    rows.append({"hypothesis": "context: A5 (SWA) vs A1 at the primary cell",
                 **compare(select(df, "A5", g, F=pc["F"], **cell), select(df, "A1", g, F=pc["F"], **cell), "A5", "A1")})
    return pd.DataFrame(rows)


def lambda_check(df: pd.DataFrame, prereg: dict) -> pd.DataFrame:
    pc = prereg["primary_cell"]
    e = df[(df["N"] == prereg["N_star"]) & (df["F"] == pc["F"]) & (df["P"] == pc["P"]) &
           (df["placement"] == pc["placement"]) & df["arm"].isin(["A1", "A2"]) &
           df["group"].isin(PHASE1_MAIN_GROUPS | {"E1.0c"})]
    return summarize(e, ["arm", "lam"])


def h4_table(df: pd.DataFrame, prereg: dict) -> pd.DataFrame:
    pc = prereg["primary_cell"]
    cell = dict(N=prereg["N_star"], F=pc["F"], P=pc["P"], placement=pc["placement"], lam=pc["filler_weight"])
    full = select(df, "A2", PHASE1_MAIN_GROUPS, **cell)
    rows = []
    for arm, econv, gconv in [("A2", "on", "on"), ("A2_noEconv", "off", "on"), ("A2_noGconv", "on", "off"),
                              ("A2_noboth", "off", "off"), ("A1", "-", "on"), ("A1_noGconv", "-", "off")]:
        v = select(df, arm, PHASE1_MAIN_GROUPS, **cell)
        c = compare(v, full, arm, "A2") if arm != "A2" else {}
        rows.append({"arm": arm, "engram_conv": econv, "gdn_short_conv": gconv, "mean": v.mean(), "std": v.std(),
                     "n": int(v.count()), "diff_vs_A2": c.get("diff_A_minus_B"),
                     "within_2sigma_of_A2": (not c.get("exceeds_2sigma")) if c.get("rule_applicable") else None})
    return pd.DataFrame(rows)


def lr_pilot_table(df: pd.DataFrame) -> pd.DataFrame:
    e = df[df["group"] == "E1.0a"]
    return e[["arm", "lr", "val_recall", "recall"]].sort_values(["arm", "lr"])


# ---------------- Phase 2 ----------------

def phase2_table(results_dir: str = "results") -> pd.DataFrame:
    rows = []
    for r in read_results(os.path.join(results_dir, "phase2", "E2.eval")):
        m = r["metrics"]
        rows.append({"run_id": r["run_id"], "arm": r["arm"], "seed": r["seed"], "ppl": m["ppl"],
                     "loss_all": m["loss_all"], "loss_pos_0_64": m["loss_pos_0_64"],
                     "loss_pos_64_256": m["loss_pos_64_256"], "loss_pos_256_1024": m["loss_pos_256_1024"],
                     "loss_copy": m["loss_copy"], "tokens_copy": m["tokens_copy"],
                     "non_embedding": r["params"]["non_embedding"], "total": r["params"]["total"]})
    return pd.DataFrame(rows)


def phase2_relative(t: pd.DataFrame) -> pd.DataFrame:
    """Relative change of each loss vs the arm's own baseline (A2 vs A1, A3 vs A1, T2 vs T1): the H3 numbers."""
    rows = []
    base_of = {"A2": "A1", "A3": "A1", "T2": "T1"}
    cols = ["loss_all", "loss_pos_0_64", "loss_pos_64_256", "loss_pos_256_1024", "loss_copy"]
    for arm, base in base_of.items():
        a, b = t[t["arm"] == arm], t[t["arm"] == base]
        if a.empty or b.empty:
            continue
        rows.append({"arm": arm, "baseline": base, "n_seeds": len(a),
                     **{f"rel_{c}": (a[c].mean() - b[c].mean()) / b[c].mean() for c in cols}})
    return pd.DataFrame(rows)
