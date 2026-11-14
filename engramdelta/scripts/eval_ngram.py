"""Study E2.ngram (PREREG.md): val loss of finished LM runs split by the training-set count of the bigram /
trigram ending at each input position (the n-grams Engram looks up), eval/lm_eval.py evaluate_lm_ngram.

    python -m engramdelta.scripts.eval_ngram --group E2.1 --data-path results/lmdata

Needs <data>/val_ngram_counts.npz (scripts/ngram_counts.py). Output: results/phase2/E2.ngram/<run_id>.json
(immutable; runs that already have one are skipped).
"""
import argparse
import os
import sys

import numpy as np
import torch

from ..data.lm_data import LMData
from ..eval.lm_eval import NGRAM_EDGES, NGRAM_MIN_POS, evaluate_lm_ngram
from ..scripts.eval_lm import load_run_model
from ..scripts.ngram_counts import OUT_NAME
from ..utils import dist
from ..utils.run_log import env_info, git_commit, now, read_results, result_path, write_result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default=os.environ.get("ENGRAMDELTA_CKPT_DIR", "ckpt"))
    ap.add_argument("--data-path", default=None, help="override the data path stored in the run config")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--run-id", action="append", default=None, help="evaluate only these runs (repeatable)")
    args = ap.parse_args(argv)
    info = dist.setup("auto")
    runs = [r for g in args.group for r in read_results(os.path.join(args.results_dir, "phase2", g))
            if r.get("status") == "done" and (args.run_id is None or r["run_id"] in args.run_id)]
    for r in runs:
        out = result_path(args.results_dir, "phase2", "E2.ngram", r["run_id"])
        if os.path.exists(out):
            continue
        model, path = load_run_model(r, args.ckpt_dir, args.data_path, info.device)
        if model is None:
            continue
        with np.load(os.path.join(path, OUT_NAME)) as z:
            counts = {k: z[k] for k in z.files}
        val = LMData(path, r["data"]["seq_len"], "val")
        amp = torch.float16 if info.device.type == "cuda" else None
        metrics = evaluate_lm_ngram(model, val, info.device, counts, batch_size=args.batch,
                                    max_windows=args.max_windows, amp_dtype=amp)
        if dist.is_main():
            rec = {"run_id": r["run_id"], "phase": "phase2", "group": "E2.ngram", "arm": r["arm"], "seed": r["seed"],
                   "status": "done", "source_group": r["group"], "metrics": metrics, "edges": list(NGRAM_EDGES),
                   "min_pos": NGRAM_MIN_POS, "val_windows": val.n_windows if args.max_windows is None else
                   min(val.n_windows, args.max_windows), "git_commit": git_commit(), "env": env_info(),
                   "finished_at": now()}
            write_result(out, rec)
            print(f"[eval_ngram] {r['run_id']}: loss_all_p2 {metrics['loss_all_p2']:.4f} | trigram buckets "
                  + " ".join(f"{metrics[k]:.3f}" for k in sorted(metrics) if k.startswith("loss_trigram_") and metrics[k]))
        dist.barrier()
    dist.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
