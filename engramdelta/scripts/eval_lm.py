"""Phase 2 full evaluation of finished LM runs (NB21).

    torchrun --standalone --nproc_per_node=2 -m engramdelta.scripts.eval_lm --group E2.1 --group E2.2

For every finished run of the groups with a final.pt checkpoint, it computes the val loss / perplexity over ALL
val windows, the position-bucket losses, the in-context-copy loss (eval/lm_eval.py) and, for GDN arms, state
occupancy on 64 val windows at positions {256, 512, 1023}. Output: results/phase2/E2.eval/<run_id>.json
(immutable; runs that already have one are skipped).
"""
import argparse
import os
import sys

import numpy as np
import torch

from ..data.lm_data import LMData
from ..eval.lm_eval import evaluate_lm
from ..eval.state_metrics import state_occupancy
from ..models.model import build_model, gdn_modules
from ..utils import dist, hub_sync
from ..utils.run_log import env_info, git_commit, now, read_results, result_path, write_result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default=os.environ.get("ENGRAMDELTA_CKPT_DIR", "ckpt"))
    ap.add_argument("--data-path", default=None, help="override the data path stored in the run config")
    ap.add_argument("--state-windows", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args(argv)
    info = dist.setup("auto")
    runs = [r for g in args.group for r in read_results(os.path.join(args.results_dir, "phase2", g))
            if r.get("status") == "done"]
    for r in runs:
        out = result_path(args.results_dir, "phase2", "E2.eval", r["run_id"])
        if os.path.exists(out):
            continue
        ck = os.path.join(args.ckpt_dir, r["run_id"], "final.pt")
        if not os.path.exists(ck) and dist.is_main():
            hub_sync.download_file(f"ckpt/{r['run_id']}/final.pt", ck)
        dist.barrier()
        if not os.path.exists(ck):
            if dist.is_main():
                print(f"[eval_lm] {r['run_id']}: no final.pt, skipped")
            continue
        state = torch.load(ck, map_location="cpu", weights_only=False)
        mc = state["meta"]["model_config"]
        path = args.data_path or r["data"]["path"]
        compress = np.load(os.path.join(path, "engram_compress.npy")) if mc["engram"]["enabled"] and mc["engram"]["compress"] else None
        model = build_model(mc, compress, device=info.device)
        model.load_state_dict(state["model"])
        val = LMData(path, r["data"]["seq_len"], "val")
        amp = torch.float16 if info.device.type == "cuda" else None
        metrics = evaluate_lm(model, val, info.device, batch_size=args.batch, amp_dtype=amp)
        occupancy = None
        if dist.is_main() and gdn_modules(model):
            ids = torch.stack([val.window(i)[:-1] for i in range(min(args.state_windows, val.n_windows))]).to(info.device)
            occupancy = state_occupancy(model, ids, {"pos_256": 255, "pos_512": 511, "pos_1023": 1023}, batch_size=8)
        if dist.is_main():
            rec = {"run_id": r["run_id"], "phase": "phase2", "group": "E2.eval", "arm": r["arm"], "seed": r["seed"],
                   "status": "done", "source_group": r["group"], "metrics": metrics, "state_occupancy": occupancy,
                   "params": r["params"], "val_windows": val.n_windows, "git_commit": git_commit(),
                   "env": env_info(), "finished_at": now()}
            p = write_result(out, rec)
            hub_sync.push_file(p, os.path.relpath(p, os.path.dirname(os.path.abspath(args.results_dir))))
            print(f"[eval_lm] {r['run_id']}: ppl {metrics['ppl']:.3f} copy-loss {metrics['loss_copy']}")
        dist.barrier()
    dist.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
