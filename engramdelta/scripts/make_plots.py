"""Figures from the result JSONs (static PNGs for the notebooks; the same numbers are printed as tables).

    python -m engramdelta.scripts.make_plots --results-dir results --out figures

Each arm has a fixed color for its identity (never its rank). Error bars are the seed std. No interpretation.
"""
import argparse
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import analysis  # noqa: E402

# Validated categorical palette (dataviz reference instance, light mode), fixed order per arm.
ARM_COLOR = {"A1": "#2a78d6", "A2": "#eb6834", "A3": "#1baf7a", "A4": "#eda100", "A5": "#e87ba4",
             "A6": "#008300", "T1": "#6250d6", "T2": "#e34948", "LA": "#52514e"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED)


def line(ax, x, y, err, arm, label=None):
    c = ARM_COLOR.get(arm, MUTED)
    ax.errorbar(x, y, yerr=err, color=c, linewidth=2, marker="o", markersize=7, capsize=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label or arm)
    if len(x):
        ax.annotate(label or arm, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")


def known_result(df, out):
    e = df[df["group"] == "E0.4"]
    if e.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=SURFACE)
    for arm in ("A1", "LA", "T1"):
        s = analysis.summarize(e[e["arm"] == arm], ["N"]).sort_values("N")
        if len(s):
            line(ax, s["N"].tolist(), s["mean"].tolist(), s["std"].fillna(0).tolist(), arm)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(e["N"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, "E0.4 plain MQAR: recall vs number of pairs", "N (pairs)", "query recall")
    ax.legend(frameon=False)
    p = os.path.join(out, "E0.4_known_result.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def h1(df, out):
    g = analysis.h1_grid(df)
    g = g[g["P"] == 64]
    if g.empty:
        return None
    combos = sorted(set(zip(g["N"], g["placement"])))
    fig, axes = plt.subplots(1, len(combos), figsize=(4 * len(combos), 3.6), facecolor=SURFACE, sharey=True)
    axes = [axes] if len(combos) == 1 else axes
    for ax, (N, pl) in zip(axes, combos):
        s = g[(g["N"] == N) & ((g["placement"] == pl) | (g["F"] == 0))].sort_values("F")
        line(ax, s["F"].tolist(), s["mean"].tolist(), s["std"].fillna(0).tolist(), "A1")
        style(ax, f"A1, N={N}, placement {pl}", "clutter length F", "query recall")
    p = os.path.join(out, "E1.1_H1_recall_vs_clutter.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def arms(df, out):
    t = analysis.arms_table(df)
    t = t[t["P"] == 64]
    if t.empty:
        return None
    n_arms = t.groupby(["N", "F", "placement"])["arm"].nunique()
    cells = sorted(c for c, n in n_arms.items() if n >= 2)      # only cells where arms are compared
    order = [a for a in ("A1", "A2", "A3", "A4", "A5", "T1") if a in set(t["arm"])]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(cells)), 4), facecolor=SURFACE)
    w = 0.8 / max(1, len(order))
    for i, arm in enumerate(order):
        xs, ys, es = [], [], []
        for j, c in enumerate(cells):
            r = t[(t["N"] == c[0]) & (t["F"] == c[1]) & (t["placement"] == c[2]) & (t["arm"] == arm)]
            if len(r):
                xs.append(j - 0.4 + w * (i + 0.5))
                ys.append(r["mean"].iloc[0])
                es.append(0 if r["std"].isna().iloc[0] else r["std"].iloc[0])
        ax.bar(xs, ys, width=w * 0.9, yerr=es, color=ARM_COLOR[arm], label=arm, capsize=2,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"N={n}\nF={f} {pl}" for n, f, pl in cells], fontsize=8)
    style(ax, "Phase 1 arms: recall per cell (P=64, lambda=1)", "", "query recall")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    p = os.path.join(out, "E1.2_arms.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def step_curve(results_dir, out):
    sb = analysis.step_budget_proposal(results_dir)
    if not sb:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=SURFACE)
    for arm, v in sb.items():
        xs, ys = zip(*v["curve"])
        line(ax, list(xs), list(ys), [0] * len(xs), arm)
        ax.axvline(v["proposal"], color=ARM_COLOR.get(arm, MUTED), linestyle=":", linewidth=1)
    style(ax, "E0.6 val recall during training (N=64, F=512, P=64, a)", "step", "val recall")
    p = os.path.join(out, "E0.6_step_curve.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    df = analysis.load(args.results_dir)
    made = [known_result(df, args.out), step_curve(args.results_dir, args.out)]
    if not df.empty and "E1.1" in set(df["group"]):
        made.append(h1(df, args.out))
    if not df.empty and df["group"].isin(analysis.PHASE1_MAIN_GROUPS).any():
        made.append(arms(df, args.out))
    for p in made:
        if p:
            print("wrote", p)


if __name__ == "__main__":
    main()
