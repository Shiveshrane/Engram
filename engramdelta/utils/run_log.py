"""Immutable per-run result files (design rule 5).

Each finished run writes exactly one JSON file: results/<phase>/<group>/<run_id>.json, with the full resolved
config, git commit, seed, parameter counts, every metric, the loss curve, world size, GPU and library versions.
write_result refuses to overwrite and marks the file read-only.
"""
import datetime
import json
import os
import platform
import subprocess
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_commit() -> str:
    """Commit hash of the code being run: $GIT_COMMIT, else a GIT_COMMIT file at the repo root, else `git rev-parse`.
    A '+dirty' suffix means there were uncommitted changes."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    f = os.path.join(REPO_ROOT, "GIT_COMMIT")
    if os.path.exists(f):
        return open(f).read().strip()
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
            "cuda": torch.version.cuda, "gpu": None, "gpu_count": 0, "triton": None, "fla": None}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    for mod, key in (("triton", "triton"), ("fla", "fla")):
        try:
            info[key] = __import__(mod).__version__
        except Exception:
            pass
    return info


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def result_path(results_dir: str, phase: str, group: str, run_id: str) -> str:
    return os.path.join(results_dir, phase, group, f"{run_id}.json")


def write_result(path: str, record: dict) -> str:
    if os.path.exists(path):
        raise FileExistsError(f"result already exists, refusing to overwrite: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1, default=str)
    os.replace(tmp, path)
    os.chmod(path, 0o444)
    return path


def read_results(results_dir: str):
    """All result records under results_dir (recursively)."""
    out = []
    for root, _, files in os.walk(results_dir):
        for fn in sorted(files):
            if fn.endswith(".json"):
                with open(os.path.join(root, fn)) as f:
                    try:
                        rec = json.load(f)
                    except json.JSONDecodeError:
                        continue
                if isinstance(rec, dict) and "run_id" in rec:
                    rec["_path"] = os.path.join(root, fn)
                    out.append(rec)
    return out
