"""Parameter table for all arms of a phase (design §6 test 3, §5.3 parameter matching).

    python -m engramdelta.scripts.param_table --phase phase1
    python -m engramdelta.scripts.param_table --phase phase2 --vocab 16384     # before the LM data exists

Embedding = token embedding (+ untied head) + Engram tables. Non-embedding = everything else (Engram
projections, norms and conv included). The matched arms' MLP width comes from model.match_nonembed_of.
"""
import argparse
import glob
import os
from typing import List

import numpy as np

from ..models.model import count_params_for_config
from ..utils.config import load_config
from .train import resolve, resolve_model_config, setup_task

ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "T1", "T2", "LA", "A2_noEconv", "A2_noGconv", "A2_noboth", "A1_noGconv"]


def param_rows(phase: str, config_dir: str = "configs", vocab: int = None) -> List[dict]:
    files = [f for f in glob.glob(os.path.join(config_dir, phase, "*.yaml")) if not f.endswith("base.yaml")]
    arms = sorted((os.path.basename(f)[:-5] for f in files), key=lambda a: (ORDER.index(a) if a in ORDER else 99, a))
    rows = []
    for arm in arms:
        cfg = resolve(load_config(os.path.join(config_dir, phase, f"{arm}.yaml")))
        if cfg["task"] == "lm" and (vocab is not None or not os.path.exists(os.path.join(cfg["data"]["path"], "meta.json"))):
            v = vocab or 16384
            task = {"vocab": v, "compress": np.arange(v), "pad_id": 0}
        else:
            task = setup_task(cfg)
        mc = resolve_model_config(cfg, task)
        c = count_params_for_config(mc, task["compress"])
        rows.append({"arm": arm, "mixer": mc["mixer"], "mlp_hidden": mc["mlp_hidden"],
                     "matched_to": mc.get("match_nonembed_of") or "", **c})
    return rows


def format_table(rows: List[dict]) -> str:
    cols = ["arm", "mixer", "mlp_hidden", "total", "non_embedding", "embedding", "token_embedding",
            "engram_table", "engram_non_table", "trainable", "matched_to"]
    ref = {r["arm"]: r for r in rows}
    lines = ["| " + " | ".join(cols + ["non_emb vs A2"]) + " |", "|" + "---|" * (len(cols) + 1)]
    for r in rows:
        rel = ""
        if "A2" in ref:
            rel = f"{100 * (r['non_embedding'] / ref['A2']['non_embedding'] - 1):+.2f}%"
        vals = [f"{r[c]:,}" if isinstance(r[c], int) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(vals + [rel]) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--vocab", type=int, default=None)
    args = ap.parse_args(argv)
    print(format_table(param_rows(args.phase, vocab=args.vocab)))


if __name__ == "__main__":
    main()
