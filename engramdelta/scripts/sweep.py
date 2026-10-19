"""Run every pending run of one or more manifests, one after another, each as its own torchrun job.

    python -m engramdelta.scripts.sweep manifests/E1.1.yaml manifests/E1.2.yaml --nproc 2 --session-hours 11

  * Skips runs whose result JSON already exists, so re-running after a killed session resumes the sweep, and an
    unfinished run resumes from its checkpoint.
  * Session budget: a run is not started with less than --min-start-minutes left. Every run gets
    --max-minutes = the remaining budget, so it checkpoints and exits with code 3 before the session dies. The sweep
    then stops; rerun it in the next session.
  * A failed run (exit code other than 0 / 3) is logged to <results>/_failures/<run_id>.log and the sweep continues.
    Failures are listed at the end and must be looked at before the results are used (design rule 4).
"""
import argparse
import os
import subprocess
import sys
import time

import yaml

from ..utils.config import load_config
from ..utils.run_log import result_path


def phase_of(run: dict) -> str:
    """The run's phase (results/<phase>/...), resolved through config inheritance and overrides."""
    return load_config(run["config"], run["set"]).get("phase", "scratch")


def command(run: dict, nproc: int, results_dir: str, ckpt_dir: str, max_minutes: float):
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}"] \
        if nproc > 1 else [sys.executable]
    cmd = launcher + ["-m", "engramdelta.scripts.train", "--config", run["config"], "--run-id", run["run_id"],
                      "--results-dir", results_dir, "--ckpt-dir", ckpt_dir, "--max-minutes", f"{max_minutes:.1f}"]
    for s in run["set"]:
        cmd += ["--set", s]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ckpt-dir", default=os.environ.get("ENGRAMDELTA_CKPT_DIR", "ckpt"))
    ap.add_argument("--session-hours", type=float, default=11.0)
    ap.add_argument("--min-start-minutes", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    runs = []
    for m in args.manifests:
        with open(m) as f:
            runs += yaml.safe_load(f) or []
    t0 = time.time()
    done, skipped, failed, pending = 0, 0, [], []
    for run in runs:
        rp = result_path(args.results_dir, phase_of(run), run["group"], run["run_id"])
        if os.path.exists(rp):
            skipped += 1
            continue
        pending.append(run)
    print(f"[sweep] {len(runs)} runs in manifests, {skipped} already done, {len(pending)} pending")
    if args.dry_run:
        for r in pending:
            print("  ", r["run_id"])
        return 0
    for i, run in enumerate(pending):
        left = args.session_hours * 60 - (time.time() - t0) / 60
        if left < args.min_start_minutes:
            print(f"[sweep] session budget exhausted ({left:.1f} min left); {len(pending) - i} runs remain. Rerun to continue.")
            break
        print(f"\n[sweep] ({i + 1}/{len(pending)}) {run['run_id']}  [{left:.0f} min left]", flush=True)
        rc = subprocess.run(command(run, args.nproc, args.results_dir, args.ckpt_dir, left - 2)).returncode
        if rc == 0:
            done += 1
        elif rc == 3:
            print("[sweep] run stopped at the time limit (checkpoint saved). Stopping the sweep; rerun to resume.")
            break
        else:
            failed.append(run["run_id"])
            fdir = os.path.join(args.results_dir, "_failures")
            os.makedirs(fdir, exist_ok=True)
            with open(os.path.join(fdir, f"{run['run_id']}.log"), "a") as f:
                f.write(f"{time.ctime()} exit code {rc}\n{' '.join(command(run, args.nproc, args.results_dir, args.ckpt_dir, 0))}\n")
    print(f"\n[sweep] finished {done}, failed {len(failed)}, elapsed {(time.time() - t0) / 3600:.2f} h")
    for f in failed:
        print("  FAILED:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
