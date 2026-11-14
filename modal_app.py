"""Phase 2 (exploratory, PREREG.md) on Modal: every run in its own A100 container, all running at the same time.

One-time setup:
    pip install modal && modal token new
    modal run modal_app.py::upload_data          # copies results/lmdata (NB03 output, 292 MB) to the Modal volume

Then, in order:
    modal run modal_app.py::bench                # ~$0.5: GDN kernel test + E2.0 throughput pilot (A1, A2) on A100;
                                                 # prints the measured minutes and cost per E2.1 run
    modal run modal_app.py::train                # E2.1: A1/A2/A3 x phase2_seeds, all in parallel. Refuses to start if
                                                 # the projected cost is above --max-dollars (default 20). --dry-run plans only.
    modal run modal_app.py::evaluate             # E2.eval for every finished run, in parallel; prints the Phase 2 tables
    modal run modal_app.py::download             # results + logs -> results/Modal-phase2/ (--ckpt also pulls final.pt)

Every command first syncs the volume's results into results/Modal-phase2/results/ and finishes by syncing again.
Logs: each run's full training / eval output is in results/_logs/<group>/<run_id>.log (written live on the volume,
part of the downloaded results). Each command also appends what it prints to results/Modal-phase2/run.log.

Cost control: the GPU is billed per second while a container runs. Every run gets --max-minutes (it checkpoints
and exits 3 before that, and `train` resumes it next time) and a hard container timeout. `train` prints the
projected and the measured cost. Nothing stays running after a command ends.
"""
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parent
LOCAL_OUT = REPO / "results" / "Modal-phase2"
REMOTE_REPO = "/root/engram"
VOL_MOUNT = "/vol"
RES = f"{VOL_MOUNT}/runs/results"            # results/<phase>/<group>/<run_id>.json and results/_logs/...
CKPT = f"{VOL_MOUNT}/runs/ckpt"
LMDATA = f"{VOL_MOUNT}/lmdata"

GPU = "A100-40GB"
CPU_CORES, MEMORY_MB = 4.0, 16384
# Modal list prices (modal.com/pricing, 2026-11-14): A100 40 GB $0.000583/s, CPU $0.0000131/core/s, RAM $0.00000222/GiB/s.
DOLLARS_PER_HOUR = 3600 * (0.000583 + CPU_CORES * 0.0000131 + MEMORY_MB / 1024 * 0.00000222)    # ~= $2.42
STARTUP_MIN, FINAL_EVAL_MIN = 2.0, 2.0       # per run, on top of the training time (conservative)
MAX_RUN_MINUTES = 110                        # hard cap per run, whatever the estimate says
PHASE2_SETS = [f"data.path={LMDATA}", "train.precision=bf16", "train.tf32=true", "train.hub_ckpt=false"]

vol = modal.Volume.from_name("engramdelta", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "numpy", "sympy", "pyyaml", "pandas", "tokenizers", "einops")
    .run_commands("pip install --no-deps fla-core==0.5.2")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO, REMOTE_REPO, ignore=[
        ".git", ".git/**", "results", "results/**", "notebooks", "notebooks/**", ".changelog", ".changelog/**",
        "figures", "figures/**", "ckpt", "ckpt/**", "manifests", "manifests/**", ".pytest_cache", ".pytest_cache/**",
        "**/__pycache__", "**/__pycache__/**", "ENGRAMDELTA_DESIGN.md"])
)
app = modal.App("engramdelta-phase2", image=image)
FN = dict(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MB, volumes={VOL_MOUNT: vol})


# ------------------------------------------------------------------------------------------------
# Remote side (runs in the container)
# ------------------------------------------------------------------------------------------------

def _enter(commit: str, backend: str = "torch"):
    os.chdir(REMOTE_REPO)
    sys.path.insert(0, REMOTE_REPO)
    os.environ.update({"GIT_COMMIT": commit, "ENGRAMDELTA_GDN_BACKEND": backend, "ENGRAMDELTA_CKPT_DIR": CKPT})


@app.function(timeout=20 * 60, **FN)
def kernels(commit: str) -> dict:
    """NB00's test 4 on this GPU: FLA vs our torch kernel against the reference; picks the GDN backend."""
    _enter(commit)
    from engramdelta.scripts.sweep import run_logged
    out = f"{RES}/phase0/E0.kernels-modal"
    rc = run_logged([sys.executable, "-m", "engramdelta.scripts.bench_kernels", "--out", out],
                    f"{RES}/_logs/E0.kernels-modal/kernels_{int(time.time())}.log")
    files = sorted(pathlib.Path(out).glob("*.json"))
    rec = json.loads(files[-1].read_text()) if files else {}
    vol.commit()
    return {"rc": rc, "backend": rec.get("recommended_backend", "torch"), "verdict": rec.get("verdict"),
            "fla_import_error": rec.get("fla_import_error"), "gpu": (rec.get("env") or {}).get("gpu")}


@app.function(timeout=(MAX_RUN_MINUTES + 15) * 60, **FN)
def run_one(run: dict, commit: str, backend: str, max_minutes: float) -> dict:
    """One training run (single GPU, no DDP), logged to results/_logs/<group>/<run_id>.log on the volume."""
    _enter(commit, backend)
    from engramdelta.scripts.sweep import command, phase_of, run_logged
    from engramdelta.utils.run_log import result_path
    rp = result_path(RES, phase_of(run), run["group"], run["run_id"])
    t0 = time.time()
    rc = 0
    if not os.path.exists(rp):
        rc = run_logged(command(run, 1, RES, CKPT, max_minutes), f"{RES}/_logs/{run['group']}/{run['run_id']}.log")
    vol.commit()
    out = {"run_id": run["run_id"], "rc": rc, "wall_minutes": (time.time() - t0) / 60}
    if os.path.exists(rp):
        r = json.loads(pathlib.Path(rp).read_text())
        out.update({k: r.get(k) for k in ("throughput_tokens_per_s", "train_seconds", "peak_gpu_mem_gb", "gdn_backend")})
        out["metrics"] = r.get("metrics")
    return out


@app.function(timeout=60 * 60, **FN)
def eval_one(run_id: str, group: str, commit: str, backend: str) -> dict:
    _enter(commit, backend)
    from engramdelta.scripts.sweep import run_logged
    t0 = time.time()
    rc = run_logged([sys.executable, "-m", "engramdelta.scripts.eval_lm", "--group", group, "--run-id", run_id,
                     "--results-dir", RES, "--ckpt-dir", CKPT, "--data-path", LMDATA],
                    f"{RES}/_logs/E2.eval/{run_id}.log")
    vol.commit()
    return {"run_id": run_id, "rc": rc, "wall_minutes": (time.time() - t0) / 60}


# ------------------------------------------------------------------------------------------------
# Local side
# ------------------------------------------------------------------------------------------------

def log(*parts):
    """Print and append to results/Modal-phase2/run.log with a UTC timestamp."""
    msg = " ".join(str(p) for p in parts)
    print(msg, flush=True)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(LOCAL_OUT / "run.log", "a") as f:
        for line in msg.splitlines() or [""]:
            f.write(f"[{stamp}] {line}\n")


def git_commit() -> str:
    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    return (h or "unknown") + ("+dirty" if dirty else "")


def pull(include_ckpt: bool = False) -> pathlib.Path:
    """Copy new files of the volume's results (and optionally checkpoints) into results/Modal-phase2/."""
    n = 0
    for sub in ["runs/results"] + (["runs/ckpt"] if include_ckpt else []):
        try:
            entries = vol.listdir(sub, recursive=True)
        except Exception:       # nothing there yet
            continue
        for e in entries:
            if e.type != modal.volume.FileEntryType.FILE:
                continue
            rel = pathlib.PurePosixPath(e.path).relative_to("runs")
            dst = LOCAL_OUT / rel
            if dst.exists() and dst.stat().st_size == e.size and not str(rel).endswith(".log"):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"".join(vol.read_file(e.path)))
            n += 1
    log(f"[pull] {n} files updated in {LOCAL_OUT}")
    return LOCAL_OUT / "results"


def settings_path() -> pathlib.Path:
    return LOCAL_OUT / "modal_settings.json"


def load_settings() -> dict:
    p = settings_path()
    return json.loads(p.read_text()) if p.exists() else {}


@app.local_entrypoint()
def upload_data(path: str = "results/lmdata"):
    src = (REPO / path).resolve()
    need = ["meta.json", "train.bin", "val.bin", "engram_compress.npy"]
    missing = [f for f in need if not (src / f).exists()]
    if missing:
        raise SystemExit(f"{src} is missing {missing}: it must be the output folder of notebook 03")
    log(f"[upload] {src} -> volume engramdelta:/lmdata")
    with vol.batch_upload(force=True) as b:
        b.put_directory(str(src), "/lmdata")
    log("[upload] done")


def _plan_e21(results_local: pathlib.Path, s: dict):
    """E2.1 runs (make_manifest rule) and their projected minutes from the E2.0 throughput on this GPU."""
    sys.path.insert(0, str(REPO))
    from engramdelta.scripts.make_manifest import build, load_prereg
    from engramdelta.utils.run_log import read_results, result_path
    pr = load_prereg(str(REPO / "configs" / "prereg.yaml"))
    tput = {r["arm"]: r["throughput_tokens_per_s"] for r in read_results(str(results_local / "phase2" / "E2.0"))
            if r.get("status") == "done"}
    if not {"A1", "A2"} <= set(tput):
        raise SystemExit("E2.0 throughput pilot not found: run `modal run modal_app.py::bench` first")
    runs = build("E2.1", pr, str(results_local))
    plan = []
    for r in runs:
        r["set"] += PHASE2_SETS
        steps = int(next(x for x in r["set"] if x.startswith("train.steps=")).split("=")[1])
        arm = r["config"].split("/")[-1][:-5]
        tps = tput.get(arm, tput["A2"])                     # A3 has A2's architecture
        minutes = steps * 64 * 1024 / tps / 60 + STARTUP_MIN + FINAL_EVAL_MIN
        done = os.path.exists(result_path(str(results_local), "phase2", "E2.1", r["run_id"]))
        plan.append({"run": r, "arm": arm, "steps": steps, "minutes": minutes, "done": done})
    return plan, tput


@app.local_entrypoint()
def bench():
    commit = git_commit()
    log(f"===== bench | code {commit} | {GPU} at ~${DOLLARS_PER_HOUR:.2f}/h")
    k = kernels.remote(commit)
    log(f"[kernels] gpu {k['gpu']} | recommended GDN backend: {k['backend']} | verdict {k['verdict']}"
        + (f" | FLA import error: {k['fla_import_error']}" if k.get("fla_import_error") else ""))
    sys.path.insert(0, str(REPO))
    from engramdelta.scripts.make_manifest import build, load_prereg
    runs = build("E2.0", load_prereg(str(REPO / "configs" / "prereg.yaml")), "results")
    for r in runs:
        r["set"] += PHASE2_SETS
    log(f"[E2.0] {len(runs)} pilot runs in parallel (100 steps each)")
    outs = list(run_one.map(runs, kwargs={"commit": commit, "backend": k["backend"], "max_minutes": 30},
                            order_outputs=False, return_exceptions=True))
    for o in outs:
        log(f"[E2.0] {o}")
    settings = {"backend": k["backend"], "gpu": k["gpu"], "commit": commit, "kernel_verdict": k["verdict"]}
    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text(json.dumps(settings, indent=1))
    res = pull()
    plan, tput = _plan_e21(res, settings)
    log(f"[E2.0] throughput tokens/s: { {a: round(t) for a, t in tput.items()} } | peak GPU memory (GB): "
        f"{ {o.get('run_id', '?').split('__')[1]: round(o.get('peak_gpu_mem_gb') or 0, 1) for o in outs if isinstance(o, dict)} }")
    _print_plan(plan)


def _print_plan(plan):
    todo = [p for p in plan if not p["done"]]
    cost = sum(p["minutes"] for p in todo) / 60 * DOLLARS_PER_HOUR
    log(f"[plan] E2.1: {len(plan)} runs ({len(plan) - len(todo)} done), {plan[0]['steps']} steps = "
        f"{plan[0]['steps'] * 65536 / 1e6:.0f}M tokens each")
    for p in plan:
        log(f"   {p['run']['run_id']:40s} ~{p['minutes']:5.1f} min  ~${p['minutes'] / 60 * DOLLARS_PER_HOUR:.2f}"
            + ("  (done)" if p["done"] else ""))
    log(f"[plan] all pending runs start at the same time: wall clock ~{max([p['minutes'] for p in todo] or [0]):.0f} min, "
        f"projected cost ~${cost:.2f}")
    return cost


@app.local_entrypoint()
def train(max_dollars: float = 20.0, dry_run: bool = False):
    commit = git_commit()
    s = load_settings()
    if not s:
        raise SystemExit("no modal_settings.json: run `modal run modal_app.py::bench` first")
    log(f"===== train E2.1 | code {commit} | backend {s['backend']} | cap ${max_dollars:.2f}")
    if commit.endswith("+dirty"):
        log("!! uncommitted changes: the result files will record a '+dirty' commit")
    plan, _ = _plan_e21(pull(), s)
    cost = _print_plan(plan)
    todo = [p for p in plan if not p["done"]]
    if not todo:
        log("[train] nothing to do")
        return
    if cost > max_dollars:
        raise SystemExit(f"projected ${cost:.2f} > cap ${max_dollars:.2f}: not starting. Lower phase2_seeds or "
                         f"phase2_max_tokens in configs/prereg.yaml, or raise --max-dollars.")
    if dry_run:
        log("[train] dry run: nothing started")
        return
    t0 = time.time()
    calls = [(p["run"], commit, s["backend"], min(MAX_RUN_MINUTES, 1.5 * p["minutes"] + 10)) for p in todo]
    spent = 0.0
    for o in run_one.starmap(calls, order_outputs=False, return_exceptions=True):
        if not isinstance(o, dict):
            log(f"[train] FAILED (exception): {o}")
            continue
        spent += (o["wall_minutes"] + 1) / 60 * DOLLARS_PER_HOUR
        m = o.get("metrics") or {}
        status = {0: "done", 3: "stopped at its time limit (rerun train to resume)"}.get(o["rc"], f"FAILED rc={o['rc']}")
        log(f"[train] {o['run_id']}: {status} | {o['wall_minutes']:.1f} min | val ppl {m.get('ppl')} | "
            f"peak mem {o.get('peak_gpu_mem_gb')} GB | ~${spent:.2f} so far")
    log(f"[train] finished in {(time.time() - t0) / 60:.1f} min wall clock, ~${spent:.2f} GPU cost")
    pull()


@app.local_entrypoint()
def evaluate(groups: str = "E2.1"):
    commit = git_commit()
    s = load_settings()
    res = pull()
    sys.path.insert(0, str(REPO))
    from engramdelta.utils.run_log import read_results, result_path
    todo = [(r["run_id"], r["group"]) for g in groups.split(",") for r in read_results(str(res / "phase2" / g))
            if r.get("status") == "done" and not os.path.exists(result_path(str(res), "phase2", "E2.eval", r["run_id"]))]
    log(f"===== evaluate {groups} | code {commit} | {len(todo)} runs in parallel")
    for o in eval_one.starmap([(rid, g, commit, s.get("backend", "torch")) for rid, g in todo],
                              order_outputs=False, return_exceptions=True):
        log(f"[eval] {o}")
    res = pull()
    import pandas as pd
    from engramdelta import analysis
    pd.set_option("display.width", 200)
    t = analysis.phase2_table(str(res))
    if not t.empty:
        log(t.to_string(index=False))
        log(analysis.phase2_relative(t).to_string(index=False))
        log(analysis.phase2_tests(t)[["metric", "A", "B", "mean_A", "mean_B", "diff_A_minus_B",
                                      "threshold_2sigma", "exceeds_2sigma"]].to_string(index=False))


@app.local_entrypoint()
def download(ckpt: bool = False):
    pull(include_ckpt=ckpt)
