"""E1.4: state occupancy and beta / alpha statistics for finished Phase 1 runs (NB14). Runs fine on CPU.

    python -m engramdelta.scripts.run_state_metrics --groups E1.1 E1.2 --arms A1 A2 A3 A4

For each finished run with a final.pt, it loads the trained weights and computes, on the first 256 sequences of
the run's own eval cell:
  * state occupancy (eval/state_metrics.py) at the end of writing and the end of clutter, per GDN layer
  * mean beta (GDN write strength) and mean Engram gate alpha on pair / filler / query tokens
Output: results/phase1/E1.4/<run_id>.json (immutable; existing ones are skipped).
"""
import argparse
import os

import torch

from ..data.mqar import MQARConfig
from ..eval.state_metrics import mqar_state_occupancy, token_kind_stats
from ..models.model import build_model
from ..utils import hub_sync
from ..utils.run_log import git_commit, now, read_results, result_path, write_result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["E1.1", "E1.2"])
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "A3", "A4"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default=os.environ.get("ENGRAMDELTA_CKPT_DIR", "ckpt"))
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    n_done = 0
    for g in args.groups:
        for r in read_results(os.path.join(args.results_dir, "phase1", g)):
            if r.get("status") != "done" or r["arm"] not in args.arms:
                continue
            out = result_path(args.results_dir, "phase1", "E1.4", r["run_id"])
            if os.path.exists(out):
                continue
            ck = os.path.join(args.ckpt_dir, r["run_id"], "final.pt")
            if not os.path.exists(ck):
                hub_sync.download_file(f"ckpt/{r['run_id']}/final.pt", ck)
            if not os.path.exists(ck):
                print(f"[E1.4] {r['run_id']}: no final.pt, skipped")
                continue
            state = torch.load(ck, map_location="cpu", weights_only=False)
            model = build_model(state["meta"]["model_config"], device=device)
            model.load_state_dict(state["model"])
            model.eval()
            mq = MQARConfig.from_dict(r["data"])
            rec = {"run_id": r["run_id"], "phase": "phase1", "group": "E1.4", "arm": r["arm"], "seed": r["seed"],
                   "status": "done", "source_group": g, "data": r["data"],
                   "state_occupancy": mqar_state_occupancy(model, mq, device, args.n_seq),
                   "token_kind_stats": token_kind_stats(model, mq, device, args.n_seq),
                   "recall": r["metrics"]["recall"], "git_commit": git_commit(), "finished_at": now()}
            p = write_result(out, rec)
            hub_sync.push_file(p, os.path.relpath(p, os.path.dirname(os.path.abspath(args.results_dir))))
            n_done += 1
            print(f"[E1.4] {r['run_id']} done")
    print(f"[E1.4] {n_done} new records")


if __name__ == "__main__":
    main()
