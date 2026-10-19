"""Push / pull results and checkpoints to the optional HF Hub dataset repo (needs HF_REPO and HF_TOKEN).

    python -m engramdelta.scripts.sync pull results            # at the start of a Kaggle session
    python -m engramdelta.scripts.sync pull ckpt               # e.g. final.pt files for NB14 / NB21
    python -m engramdelta.scripts.sync push results            # at the end of a session (runs also push as they finish)
"""
import argparse
import os
import sys

from ..utils import hub_sync


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("prefix", help="results or ckpt (a repo-relative directory)")
    ap.add_argument("--local", default=None, help="local directory (default: same name as prefix)")
    args = ap.parse_args(argv)
    if not hub_sync.enabled():
        print("[sync] HF_REPO / HF_TOKEN not set: nothing to do (results persist via Kaggle outputs only)")
        return 0
    hub_sync.ensure_repo()
    local = args.local or args.prefix
    if args.action == "push":
        ok = hub_sync.push_folder(local, args.prefix)
    else:
        ok = hub_sync.pull_prefix(args.prefix, os.path.dirname(os.path.abspath(local)) if args.local else ".") is not None
    print(f"[sync] {args.action} {args.prefix}: {'ok' if ok else 'failed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
