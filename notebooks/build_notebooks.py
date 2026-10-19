"""Generates the Kaggle notebooks (one per experiment step) from the cell sources below.

    python notebooks/build_notebooks.py        # rewrites notebooks/*.ipynb

Edit this file, not the .ipynb files, so all notebooks keep the same setup / persistence logic.
"""
import json
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(s).strip("\n")}


def notebook(cells, gpu=True):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isInternetEnabled": True,
                       "isGpuEnabled": gpu}}
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


# ------------------------------------------------------------------------------------------------
# Shared cells
# ------------------------------------------------------------------------------------------------

SETTINGS = '''
# ===================== SETTINGS: edit this cell, then "Run All" (or "Save Version -> Save & Run All") =====================
REPO_URL     = "https://github.com/Shiveshrane/Engram.git"   # None -> copy the code from CODE_DATASET instead
REPO_REF     = "main"           # pin the PREREG commit hash for Phase 1 / 2 runs
CODE_DATASET = "/kaggle/input/engramdelta-src"   # Kaggle Dataset with the repo (used when REPO_URL is None)
HF_REPO      = None             # e.g. "your-hf-user/engramdelta-runs" (private dataset repo; needs Kaggle Secret HF_TOKEN)
GDN_BACKEND  = "torch"          # "torch" (pure PyTorch, always works) or "fla" (only if NB00 says PASS)
FLA_SPEC     = "fla-core==0.5.2"                 # FLA's kernel package (fla.ops); installed with --no-deps
PREV_OUTPUTS = []               # outputs of earlier sessions attached as inputs; globs work, e.g. ["/kaggle/input/*"]
NCCL_ENV     = {}               # set to {"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"} if NB00's first NCCL test hung
'''

SETUP = '''
# ===================== Session setup (no edits needed) =====================
import os, sys, subprocess, shutil, pathlib, json, glob, time, datetime, traceback, zipfile
NEEDS_GPU, INSTALL_FLA = __NEEDS_GPU__, __INSTALL_FLA__
NB_NAME = "__NB_NAME__"
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = pathlib.Path("/kaggle/working") if ON_KAGGLE else pathlib.Path.cwd()
# On Kaggle the code is cloned to /tmp, which is NOT saved, so the notebook output holds only what this session
# creates: results/, figures/, logs/, ckpt/ (unfinished runs only) and the results zip. results/ and figures/ are
# symlinked from the repo into /kaggle/working. Locally (notebook opened from the repo's notebooks/ folder) the
# repo is the parent directory.
REPO = pathlib.Path("/tmp/engramdelta") if ON_KAGGLE else (WORK.parent if WORK.name == "notebooks" else WORK)

# 0. run.log: everything this notebook prints (every command, every training step, every error traceback) is also
#    written, live, to run.log at the top of the notebook output, with elapsed seconds per line. Each training run
#    also gets its own log, results/_logs/<group>/<run_id>.log. After every cell, results/ (result JSONs + all logs,
#    no checkpoints) is zipped into <notebook>_results.zip. All three are in the Output tab, ready to download.
class _Tee:
    _engram_tee = True
    def __init__(self, stream, f, t0):
        self._s, self._f, self._t0, self._bol = stream, f, t0, True
    def write(self, s):
        out = []
        for part in s.splitlines(keepends=True):
            if self._bol:
                out.append(f"[{time.time() - self._t0:9.1f}s] ")
            out.append(part)
            self._bol = part.endswith("\\n")
        self._f.write("".join(out)); self._f.flush()
        return self._s.write(s)
    def flush(self):
        self._s.flush(); self._f.flush()
    def __getattr__(self, k):
        return getattr(self._s, k)

if not getattr(sys.stdout, "_engram_tee", False):      # setup cell re-run: keep the existing log
    _STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    LOG_PATH = WORK / "run.log"
    _LOG = open(LOG_PATH, "a", buffering=1)
    _LOG.write(f"===== {NB_NAME} | started {_STAMP} (UTC) =====\\n")
    _T0 = time.time()
    sys.stdout, sys.stderr = _Tee(sys.stdout, _LOG, _T0), _Tee(sys.stderr, _LOG, _T0)
ZIP_PATH = WORK / f"{NB_NAME}_results.zip"

def bundle():
    """Copy the session log into results/_logs/notebooks/ and zip results/ into ZIP_PATH (no checkpoints)."""
    try:
        res = REPO / "results"
        if not res.is_dir():
            return
        dst = res / "_logs" / "notebooks"; dst.mkdir(parents=True, exist_ok=True)
        _LOG.flush(); shutil.copy2(LOG_PATH, dst / f"{NB_NAME}__{_STAMP}.log")
        tmp = ZIP_PATH.with_name(ZIP_PATH.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(res):
                for fn in files:
                    fp = os.path.join(root, fn); z.write(fp, os.path.relpath(fp, REPO))
        os.replace(tmp, ZIP_PATH)
    except Exception as e:      # never let logging break the experiment
        sys.__stdout__.write(f"!! results zip failed: {e}\\n")

_ip = get_ipython()
for _ev in ("pre_run_cell", "post_run_cell"):
    for _cb in list(_ip.events.callbacks[_ev]):
        if getattr(_cb, "__name__", "").startswith("_engram_"):
            _ip.events.unregister(_ev, _cb)
def _engram_pre(info=None):
    src = (getattr(info, "raw_cell", "") or "").strip().splitlines()
    _LOG.write(f"\\n===== cell start {time.strftime('%H:%M:%S')} | {src[0][:100] if src else ''}\\n")
def _engram_post(result=None):
    err = getattr(result, "error_in_exec", None) or getattr(result, "error_before_exec", None)
    if err is not None:
        _LOG.write("!! CELL ERROR\\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__)))
    bundle()
_ip.events.register("pre_run_cell", _engram_pre)
_ip.events.register("post_run_cell", _engram_post)
print("=" * 100)
print(f"LOGS  run.log (this whole notebook, live)  -> {LOG_PATH}")
print(f"      one log per training run             -> {REPO / 'results' / '_logs'}/<group>/<run_id>.log")
print(f"      everything zipped after every cell   -> {ZIP_PATH}")
print("      Download from the notebook's Output tab (run.log and the zip are at the top level).")
print("=" * 100)

def run(cmd, check=True, env=None, timeout=None, show=None):
    """Run a shell command in the repo and stream its output into this notebook."""
    print("$", show or cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})}, cwd=str(REPO) if REPO.exists() else str(WORK))
    t0 = time.time()
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            if timeout and time.time() - t0 > timeout:
                raise subprocess.TimeoutExpired(cmd, timeout)
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill(); print(f"!! timed out after {timeout}s"); rc = -9
    if check and rc != 0:
        raise RuntimeError(f"exit code {rc}: {show or cmd}")
    return rc

def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name)

# 1. code
if not REPO.exists():
    if REPO_URL:
        url, tok = REPO_URL, secret("GITHUB_TOKEN")
        if tok and url.startswith("https://"):
            url = url.replace("https://", f"https://{tok}@")
        run(f"git clone -q {url} {REPO}", show=f"git clone {REPO_URL}")
        run(f"git -C {REPO} checkout -q {REPO_REF}")
    else:
        shutil.copytree(CODE_DATASET, REPO)
if ON_KAGGLE:
    for d in ("results", "figures"):
        out, link = WORK / d, REPO / d
        out.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            if link.exists():
                shutil.copytree(link, out, dirs_exist_ok=True); shutil.rmtree(link)
            link.symlink_to(out, target_is_directory=True)
os.chdir(REPO); sys.path.insert(0, str(REPO))
try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    commit = open("GIT_COMMIT").read().strip() if os.path.exists("GIT_COMMIT") else "unknown"
os.environ["GIT_COMMIT"] = commit
print("code commit:", commit)

# 2. packages (never let pip replace Kaggle's torch)
run("pip install -q --no-deps einops sympy")
if INSTALL_FLA or GDN_BACKEND == "fla":
    run(f"pip install -q --no-deps {FLA_SPEC}", check=False)

# 3. environment for every launched run
os.environ["ENGRAMDELTA_GDN_BACKEND"] = GDN_BACKEND
os.environ["ENGRAMDELTA_CKPT_DIR"] = str(WORK / "ckpt")
os.environ.update(NCCL_ENV)
if HF_REPO and secret("HF_TOKEN"):
    os.environ["HF_REPO"], os.environ["HF_TOKEN"] = HF_REPO, secret("HF_TOKEN")
    print("HF Hub sync: ON ->", HF_REPO)
else:
    print("HF Hub sync: OFF (results persist via this notebook's output only)")

# 4. restore results / checkpoints from earlier sessions (never overwrites existing files)
def restore(src, dst):
    if not os.path.isdir(src):
        return 0
    n = 0
    for root, _, files in os.walk(src):
        for f in files:
            s = os.path.join(root, f); d = os.path.join(dst, os.path.relpath(s, src))
            if not os.path.exists(d):
                os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d); n += 1
    return n
if ON_KAGGLE and os.path.isdir("/kaggle/input"):
    print("attached inputs:", sorted(glob.glob("/kaggle/input/*")) or "none")
def find_dirs(prev, name):
    """Folders called `name` up to 4 levels inside an attached output (e.g. <prev>/results, <prev>/engramdelta/results)."""
    c = [p for k in range(5) for p in glob.glob(os.path.join(prev, *(["*"] * k), name))]
    return [d for d in dict.fromkeys(c) if os.path.isdir(d) and "/_logs/" not in d]
_prevs = [p for e in PREV_OUTPUTS for p in (sorted(glob.glob(e)) or [e])]     # entries may be globs, e.g. "/kaggle/input/*"
for prev in _prevs:
    if not os.path.isdir(prev):
        print(f"!! PREV_OUTPUTS entry not found: {prev}  (check the 'attached inputs' list above)"); continue
    dirs = find_dirs(prev, "results")
    n1 = sum(restore(d, str(REPO / "results")) for d in dirs)
    # checkpoints: only for runs that have no result yet (a finished run never resumes, so its checkpoint is skipped)
    done_ids = {os.path.splitext(f)[0] for _, _, fs in os.walk(REPO / "results") for f in fs if f.endswith(".json")}
    n2 = 0
    for ck in [c for d in find_dirs(prev, "ckpt") for c in glob.glob(os.path.join(d, "*"))]:
        if os.path.basename(ck) not in done_ids:
            n2 += restore(ck, str(WORK / "ckpt" / os.path.basename(ck)))
    print(f"restored from {prev}: {n1} result files (from {dirs}), {n2} checkpoint files of unfinished runs")
if os.environ.get("HF_REPO"):
    run("python -m engramdelta.scripts.sync pull results", check=False)

# 5. hardware
if NEEDS_GPU:
    run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", check=False)
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPUs:", torch.cuda.device_count())
NPROC = max(1, torch.cuda.device_count())
if NEEDS_GPU and NPROC < 2:
    print("!! WARNING: fewer than 2 GPUs. Set Accelerator = 'GPU T4 x2' in the notebook settings.")
'''

FINISH = '''
# ===================== End of session: persist =====================
if os.environ.get("HF_REPO"):
    run("python -m engramdelta.scripts.sync push results", check=False)
n = sum(len(f) for _, _, f in os.walk(REPO / "results"))
print(f"{n} files under results/. They are part of this notebook's output: Save Version keeps them.")
print("Next session: attach this notebook's output as an input and add its path to PREV_OUTPUTS (or use HF_REPO).")
bundle()
_mb = ZIP_PATH.stat().st_size / 1e6 if ZIP_PATH.exists() else 0.0
print(f"\\nTo send the results back: Output tab -> download {ZIP_PATH.name} ({_mb:.1f} MB) "
      "and unzip it into the repo folder:")
print(f"    unzip -n {ZIP_PATH.name} -d <your Engram repo>     # -n: never overwrite existing files")
print("It holds results/ (result JSONs), results/_logs/<group>/<run_id>.log (each run's training log)")
print(f"and results/_logs/notebooks/{NB_NAME}__{_STAMP}.log (a copy of run.log).")
print(f"run.log alone (this notebook's full printed output) is also in the Output tab: {LOG_PATH}")
'''


LOGS_MD_MARK = "### Logs and downloads"
LOGS_MD = '''
### Logs and downloads (created automatically while this notebook runs)

Everything appears in this notebook's **Output** tab (`/kaggle/working`) and can be downloaded directly:

| File | What it contains |
|---|---|
| `run.log` | **The whole run**: everything this notebook prints, live, with elapsed seconds per line: every command, every training step, every warning and the full traceback of any failed cell |
| `results/_logs/<group>/<run_id>.log` | One log per training run (the full output of that run, both GPUs) |
| `__NB_NAME___results.zip` | Rebuilt after every cell: `results/` (result JSONs + all the logs above, no checkpoints). **Download this one** |

To hand the results back: download `__NB_NAME___results.zip` and run
`unzip -n __NB_NAME___results.zip -d <your Engram repo>`. If the session dies, `run.log` and the per-run logs
still hold everything up to that moment. The code itself is cloned to `/tmp` and is **not** part of the output.
'''


def setup_cells(gpu=True, fla=False):
    return [code(SETTINGS), code(SETUP.replace("__NEEDS_GPU__", str(gpu)).replace("__INSTALL_FLA__", str(fla)))]


def sweep_cells(groups, hours, extra_expr='""'):
    return [code(f'''
GROUPS = {groups!r}     # run in this order (later groups may need earlier results, e.g. LRs from E1.0a)
SESSION_HOURS = {hours}   # leave margin below Kaggle's 12 h session limit
EXTRA = {extra_expr}
t_start = time.time()
for g in GROUPS:
    left = SESSION_HOURS - (time.time() - t_start) / 3600
    if left < 0.15:
        print("session budget used up; rerun this notebook to continue"); break
    run(f"python -m engramdelta.scripts.make_manifest --group {{g}} --out manifests {{EXTRA}}")
    run(f"python -m engramdelta.scripts.budget manifests/{{g}}.yaml --nproc {{NPROC}}", check=False)
    rc = run(f"python -m engramdelta.scripts.sweep manifests/{{g}}.yaml --nproc {{NPROC}} --session-hours {{left:.2f}}", check=False)
    if rc != 0:
        print(f"!! sweep for {{g}} reported failures or stopped early (exit {{rc}}). Check results/_failures before continuing.")
''')]


# ------------------------------------------------------------------------------------------------
# Notebooks
# ------------------------------------------------------------------------------------------------

NB = {}

NB["00_env_and_kernels"] = notebook([
    md('''
    # 00 · Environment, NCCL, GDN kernels, DDP speedup  (Phase 0, ~0.75 GPU-h)

    **Accelerator:** GPU T4 ×2 · **Internet:** on

    **What it does**
    1. Prints the hardware and software report.
    2. Runs an NCCL all-reduce smoke test across the two T4s, with a fallback to `NCCL_P2P_DISABLE=1`.
    3. Installs FLA's kernel package `fla-core` (`--no-deps`, so Kaggle's torch is never replaced) and runs **test 4**: FLA's Triton GDN kernel and our pure-PyTorch
       kernel against FLA's naive reference (outputs + gradients, fp32 and fp16), plus fwd+bwd speed at T ∈ {512, 1024, 2048}.
    4. Measures 1-GPU vs 2-GPU (DDP) training throughput for A1/A2 at a Phase 1 cell, for each usable backend.

    **Outputs:** `results/phase0/E0.kernels/*.json`, `results/phase0/E0.thr/*.json`, and the recommended `GDN_BACKEND`.

    **Stop and ask if:** NCCL fails with and without the fallback; or the *torch* kernel FAILs correctness (that is a bug).
    An FLA failure is fine: use `GDN_BACKEND = "torch"`.
    '''),
    *setup_cells(gpu=True, fla=True),
    md("## 1 · NCCL smoke test (2 processes, one per T4)"),
    code('''
    rc = run("torchrun --standalone --nproc_per_node=2 -m engramdelta.scripts.nccl_smoke", check=False, timeout=300)
    if rc != 0:
        print("\\n!! NCCL test failed or hung. Retrying with NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 ...")
        fb = {"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1", "NCCL_DEBUG": "INFO"}
        rc = run("torchrun --standalone --nproc_per_node=2 -m engramdelta.scripts.nccl_smoke", check=False, env=fb, timeout=300)
        if rc == 0:
            os.environ.update({"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"})
            print('>> Fallback works. In EVERY later notebook set NCCL_ENV = {"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"}')
        else:
            raise RuntimeError("NCCL does not work on this session: stop and investigate (NCCL_DEBUG=INFO output above).")
    '''),
    md("## 2 · FLA install check"),
    code('''
    run("python -c \\"import importlib.metadata as m, triton; print('fla-core', m.version('fla-core'), '| triton', triton.__version__); "
        "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla.ops import OK')\\"", check=False)
    run("python -c \\"import torch; print('compute capability', torch.cuda.get_device_capability(0))\\"", check=False)
    '''),
    md("## 3 · Test 4: kernel correctness (vs FLA naive reference) and speed"),
    code('''
    run("python -m engramdelta.scripts.bench_kernels --out results/phase0/E0.kernels")
    latest = sorted(glob.glob("results/phase0/E0.kernels/*.json"))[-1]
    kern = json.load(open(latest))
    print(json.dumps(kern["verdict"], indent=1))
    RECOMMENDED = kern["recommended_backend"]
    print("\\n>> recommended GDN_BACKEND =", RECOMMENDED)
    assert all(v == "PASS" for k, v in kern["verdict"].items() if k.startswith("torch_chunk")), "torch kernel FAILED: stop and ask"
    '''),
    code('''
    # The same comparison as a pytest (includes the CUDA-only FLA test)
    run("python -m pytest tests/test_kernels.py -q -rs", check=False)
    '''),
    md('''
    ## 4 · DDP throughput: 1 GPU vs 2 GPUs (synchronized DDP)
    150 steps of A1 and A2 at (N=64, F=512, P=64, a), for each usable backend. The first steps include Triton
    compilation, so FLA numbers are slightly pessimistic. Speedup = tokens/s at 2 GPUs ÷ at 1 GPU.
    The plan's decision point: if the speedup for these tiny models is below 1.3×, report it before Phase 1.
    '''),
    code('''
    backends = ["torch"] + (["fla"] if RECOMMENDED == "fla" else [])
    for be in backends:
        for arm in ("A1", "A2"):
            for nproc in (1, 2):
                rid = f"E0.thr__{arm}__{be}__np{nproc}"
                launcher = f"torchrun --standalone --nproc_per_node={nproc}" if nproc > 1 else "python"
                run(f"{launcher} -m engramdelta.scripts.train --config configs/phase1/{arm}.yaml --run-id {rid} "
                    f"--set phase=phase0 --set group=E0.thr --set data.N=64 --set data.F=512 "
                    f"--set train.steps=150 --set train.log_every=50 --set data.eval_size=400",
                    env={"ENGRAMDELTA_GDN_BACKEND": be}, check=False)
    import pandas as pd
    rows = []
    for f in sorted(glob.glob("results/phase0/E0.thr/*.json")):
        r = json.load(open(f))
        rows.append({"run": r["run_id"], "arm": r["arm"], "backend": r["gdn_backend"], "world": r["world_size"],
                     "tok/s": round(r["throughput_tokens_per_s"]), "sec_per_step": r["train_seconds"] / r["train"]["steps"]})
    t = pd.DataFrame(rows); print(t.to_string(index=False))
    for (arm, be), g in t.groupby(["arm", "backend"]):
        if set(g["world"]) == {1, 2}:
            sp = g[g.world == 2]["tok/s"].iloc[0] / g[g.world == 1]["tok/s"].iloc[0]
            print(f"DDP speedup {arm} [{be}]: {sp:.2f}x")
    '''),
    md('''
    ## Decision (write it into PREREG.md → "Phase 0 outcomes")
    * `GDN_BACKEND` for all later notebooks = the recommended value above (keep `FLA_SPEC` as is).
    * If NCCL needed the fallback, set `NCCL_ENV` in all later notebooks.
    * Tokens/s at 2 GPUs (torch / fla) feed the Phase 1 budget in NB02.
    '''),
    code(FINISH),
])

NB["01_phase0_tests"] = notebook([
    md('''
    # 01 · Phase 0 test suite on 2×T4  (~0.5 GPU-h)

    **What it does**
    * The full pytest suite on GPU:
      * test 1 causality
      * test 2 overfit-one-batch (E0.1)
      * test 3 parameter tables
      * test 4 kernels
      * test 5 Engram parity
      * test 6 MQAR format and leak
      * test 8 fp16 stability vs fp32 for A1/A2/T1 at the real Phase 1 size (E0.2)
      * test 9 determinism
      * the gloo-DDP test
    * Synchronized-DDP checks with **NCCL** on the two T4s: gradient equivalence vs single-process, parameter sync, exact resume.
    * Parameter tables for all arms of all phases.
    * E0.3: determinism of a real 2-GPU DDP run (A2, the same seed twice).

    **Stop and ask if:** any test fails (do not patch a test to make it pass: design rule 4).
    '''),
    *setup_cells(gpu=True),
    md("## 1 · Full test suite (GPU)"),
    code('''run("python -m pytest tests -q -rs -s --durations=15", check=False)'''),
    md("## 2 · Synchronized DDP checks with NCCL (2 × T4)"),
    code('''run(f"torchrun --standalone --nproc_per_node={NPROC} -m engramdelta.scripts.ddp_check --workdir {WORK}/ddp_check")'''),
    md("## 3 · Parameter tables (test 3)"),
    code('''
    for ph in ("phase0", "phase1"):
        run(f"python -m engramdelta.scripts.param_table --phase {ph}")
    run("python -m engramdelta.scripts.param_table --phase phase2 --vocab 16384")
    '''),
    md("## 4 · E0.3 determinism: A2 twice with the same seed, 2-GPU DDP, 300 steps"),
    *sweep_cells(["E0.3"], 0.5),
    code('''
    a, b = [json.load(open(f)) for f in sorted(glob.glob("results/phase0/E0.3/*.json"))[:2]]
    import numpy as np
    ca, cb = np.array([c[1:3] for c in a["curve"]]), np.array([c[1:3] for c in b["curve"]])
    print("max |loss difference| over the curve:", float(np.abs(ca - cb).max()))
    print("recall:", a["metrics"]["recall"], b["metrics"]["recall"])
    '''),
    code(FINISH),
])

NB["02_phase0_known_result"] = notebook([
    md("""
    # 02 · Phase 0 known result, leak test, step budget  (~3 GPU-h fresh; ~0.5 GPU-h when resuming)

    **Stage 1 · E0.4 (test 7) and E0.5 (leak test)**, 4,000 steps each (pre-registered):
    * **E0.4:** vanilla linear attention (LA) vs GDN (A1) on plain MQAR, N ∈ {16, 32, 64, 128, 256}, 3 seeds; T1 at 1 seed.
      Gives the test-7 verdict and **N\\*** = the largest N with A1 mean recall ≥ 0.9.
    * **E0.5:** the Engram-only probe A6 at every N; it must be at chance (recall < 1%).

    **Stage 2 · E0.6 at N = N\\*** (the Phase 1 primary cell: N\\*, F=512, P=64, a): A1 and A2, 6,000 steps, val recall
    every 250 steps (~25 min). MQAR has a long plateau (recall stuck at 1/N) before the key→value binding is learned;
    this curve gives the Phase 1 step budget and the real per-run time. **If A1 never gets above 50% val recall, the
    notebook stops**: the step rule would be meaningless (you decide what to change).

    **Resuming / re-running:** attach the earlier output of this notebook (Add Input → Your Work) and put its path in
    `PREV_OUTPUTS`. Finished runs are skipped, so only missing runs train.

    **Outputs:** test-7 table and verdict, N\\*, leak table, step-budget proposal, `configs/prereg.proposed.yaml`, and the
    Phase 1 GPU-hour estimate from measured throughput.

    **Stop if test 7 fails** (design §6: the pipeline is wrong). **Then you:** fill `configs/prereg.yaml` (N_star,
    phase1_steps) and the PREREG.md "Phase 0 outcomes" table, commit, push, and pin that commit as `REPO_REF` in
    every Phase 1 notebook.
    """),
    *setup_cells(gpu=True),
    md("## Stage 1 · E0.4 (test 7) and E0.5 (leak test)"),
    *sweep_cells(["E0.4", "E0.5"], 2.5),
    md("## Test 7: known result (GDN vs vanilla linear attention) and N*"),
    code("""
    from engramdelta import analysis
    df = analysis.load("results")
    chk = analysis.known_result_check(df)
    print(chk["table"][["N", "mean_A", "mean_B", "pooled_std", "diff_A_minus_B", "exceeds_2sigma", "T1_recall"]]
          .rename(columns={"mean_A": "A1", "mean_B": "LA"}).to_string(index=False))
    print("\\ncriterion 1 (A1 > LA + 2 sigma at >= 2 N):", chk["criterion_1_pass"], f"({chk['n_wins']} N)")
    print("criterion 2 (A1 non-increasing beyond the knee):", chk["criterion_2_pass"])
    print("TEST 7:", "PASS" if chk["pass"] else "FAIL -> STOP, the pipeline is wrong (design §6)")
    N_STAR = chk["N_star"]
    print("\\nN* (largest N with A1 mean recall >= 0.9):", N_STAR)
    if not chk["pass"] or N_STAR is None:
        raise RuntimeError("STOP: test 7 failed or no N reached 0.9. Do not run E0.6 or Phase 1; report this table.")
    """),
    md("## Test 6 at scale: A6 leak check"),
    code("""
    print(analysis.leak_table(df).sort_values("N").to_string(index=False))
    missing = sorted(set([16, 32, 64, 128, 256]) - set(df[(df["group"] == "E0.5")]["N"]))
    if missing:
        print("!! E0.5 runs missing for N =", missing, "(rerun this notebook with PREV_OUTPUTS to finish them)")
    run("python -m pytest tests/test_repro_known.py -q -s -rs", check=False)
    """),
    md("## Stage 2 · E0.6 at N = N*: training curve and real per-run time"),
    *sweep_cells(["E0.6"], 1.0),
    md("## Step budget (E0.6) and measured run time"),
    code("""
    sb = analysis.step_budget_proposal("results", N=N_STAR)
    for arm, v in sb.items():
        print(f"{arm} (N={v['N']}): max val recall {v['max_val_recall']:.3f}, first step >= 98% of max: "
              f"{v['first_step_at_98pct']}, proposal {v['proposal']} | {v['tok_per_s']:.0f} tok/s, "
              f"{v['train_seconds'] / 60:.1f} min for 6000 steps")
        print("   curve (step, val recall):", [(s, round(r, 3)) for s, r in v["curve"]])
    run("python -m engramdelta.scripts.make_plots --results-dir results --out figures", check=False)
    from IPython.display import Image, display
    for f in sorted(glob.glob("figures/E0.*.png")):
        display(Image(f))
    if "A1" not in sb:
        raise RuntimeError(f"STOP: no E0.6 A1 result at N={N_STAR} (did the E0.6 sweep run out of time? rerun to resume)")
    if not sb["A1"]["learned"]:
        raise RuntimeError(f"STOP: A1 never exceeded 50% val recall at the primary cell (N={N_STAR}, F=512) in 6000 steps. "
                         "The step-budget rule is meaningless here; bring this curve back for a decision.")
    """),
    md("## Proposed pre-registered values and the Phase 1 budget (from measured throughput)"),
    code("""
    import yaml
    pr = yaml.safe_load(open("configs/prereg.yaml"))
    prop = dict(pr, N_star=N_STAR, phase1_steps=sb["A1"]["proposal"])     # rule: A1's E0.6 proposal (PREREG.md)
    yaml.safe_dump(prop, open("configs/prereg.proposed.yaml", "w"), sort_keys=False)
    shutil.copy2("configs/prereg.proposed.yaml", "results/prereg.proposed.yaml")   # also in the output / zip
    print(open("configs/prereg.proposed.yaml").read())
    # Budget: LRs are not known yet, so use placeholder LRs (they do not change the run time)
    tmp = dict(prop, lr={"A1": 1e-3, "A2": 1e-3, "T1": 1e-3})
    yaml.safe_dump(tmp, open("/tmp/prereg_budget.yaml", "w"))
    run("python -m engramdelta.scripts.make_manifest --group all-phase1 --prereg /tmp/prereg_budget.yaml --out /tmp/manifests_p1")
    run(f"python -m engramdelta.scripts.budget '/tmp/manifests_p1/*.yaml' --nproc {NPROC} --quota-hours 13", check=False)
    """),
    md("""
    ## Your turn (no GPU needed)
    1. Check the test-7 verdict. If it FAILs, stop here.
    2. Copy `N_star` and `phase1_steps` from `configs/prereg.proposed.yaml` (also `results/prereg.proposed.yaml` in the
       zip) into `configs/prereg.yaml`. If the budget above says OVER BUDGET, decide on cuts (plan §6.5) **before** committing.
       (If this notebook stopped because A1 did not learn in E0.6, `phase1_steps` and the primary clutter level come from
       NB04, the clutter pilot.)
    3. Fill the "Phase 0 outcomes" table in PREREG.md, commit, push, and put the commit hash in PREREG.md and as `REPO_REF`
       in notebooks 10–14.
    """),
    code(FINISH),
])

NB["04_phase0_clutter_pilot"] = notebook([
    md("""
    # 04 · Phase 0 clutter pilot (E0.7): primary clutter level and Phase 1 step budget  (~0.55 GPU-h)

    **Accelerator:** GPU T4 ×2 · **Internet:** on · **Inputs:** none (N\\* = 32 is read from `configs/prereg.yaml`)

    **Why:** NB02 showed that A1 (GDN) solves N = N\\* = 32 pairs at 99% *without* clutter (it breaks the plateau at
    ~1,900 steps), but with F = 512 clutter tokens it never leaves chance in 6,000 steps (A1 and A2 at 0.04%). With
    every arm at zero, Phase 1 could not show whether Engram helps. This pilot finds the clutter level where A1 still
    learns, and gives the Phase 1 step budget from that run's curve.

    **Runs (A1 only, N = N\\*, P = 64, placement a, 6,000 steps, val recall every 250 steps):**

    | F (clutter tokens) | filler loss λ | purpose |
    |---|---|---|
    | 64 | 1 | backup level, in case 128 already fails |
    | 128 | 1 | design level |
    | 256 | 1 | design level |
    | 512 | 0 | is the filler loss what blocks learning at 512? |

    **Pre-registered rule (PREREG.md):** primary F = the largest of {256, 128} where A1's max val recall ≥ 0.5;
    `phase1_steps` = the 98% rule on that run's curve. If neither learns, the notebook **stops** and you decide
    (the F = 64 and λ = 0 runs inform that decision).

    **Outputs:** the pilot table and curves, `results/prereg.proposed.yaml` (N_star, phase1_steps, primary F) and the
    Phase 1 GPU-hour estimate.
    """),
    *setup_cells(gpu=True),
    md("## E0.7 · clutter pilot runs"),
    *sweep_cells(["E0.7"], 1.0),
    md("## Result: which clutter level does A1 still learn?"),
    code("""
    import yaml
    from engramdelta import analysis
    pr = yaml.safe_load(open("configs/prereg.yaml"))
    N_STAR = int(pr["N_star"])
    rows = analysis.clutter_pilot("results", N=N_STAR)
    print(f"A1 at N={N_STAR}, P=64, placement a, 6000 steps   (chance = 1/2048 = 0.0005)")
    print(f"{'F':>4} {'lambda':>6} {'max val recall':>15} {'final':>7} {'learned':>8} {'first step >= 98% of max':>25} {'min':>5}")
    for x in rows:
        print(f"{x['F']:>4} {x['lam']:>6g} {x['max_val_recall']:>15.3f} {x['final_val_recall']:>7.3f} {str(x['learned']):>8} "
              f"{x['first_step_at_98pct'] if x['learned'] else '-':>25} {x['train_seconds'] / 60:>5.1f}")
    for x in rows:
        print(f"  F={x['F']} lambda={x['lam']:g} curve:", [(st, round(v, 3)) for st, v in x["curve"]])

    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for x in rows:
        st, v = zip(*x["curve"])
        ax.plot(st, v, linewidth=2, label=f"F={x['F']}, lambda={x['lam']:g}")
    ax.set_xlabel("step"); ax.set_ylabel("val recall"); ax.set_ylim(0, 1.02)
    ax.set_title(f"E0.7 clutter pilot: A1 at N={N_STAR}"); ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig("figures/E0.7_clutter_pilot.png", dpi=150, bbox_inches="tight"); plt.show()

    if len(rows) < 4:
        print("!! only", len(rows), "of 4 pilot runs finished (session time?). Rerun with this notebook's output in PREV_OUTPUTS.")
    choice = analysis.primary_f_rule(rows)
    if choice is None:
        raise RuntimeError("STOP: A1 did not reach 0.5 val recall at F=256 or F=128. Bring this table back for a decision "
                           "(see the F=64 and lambda=0 rows).")
    PRIMARY_F, PHASE1_STEPS = choice["F"], choice["proposal"]
    print(f"\\n>> primary F = {PRIMARY_F}   (largest of 256, 128 where A1 learns)")
    print(f">> phase1_steps = {PHASE1_STEPS}   (first step >= 98% of max at F={PRIMARY_F}: {choice['first_step_at_98pct']}, rounded up to 500)")
    """),
    md("## Proposed pre-registered values and the Phase 1 budget (from measured throughput)"),
    code("""
    prop = dict(pr, phase1_steps=PHASE1_STEPS, primary_cell=dict(pr["primary_cell"], F=PRIMARY_F))
    yaml.safe_dump(prop, open("configs/prereg.proposed.yaml", "w"), sort_keys=False)
    shutil.copy2("configs/prereg.proposed.yaml", "results/prereg.proposed.yaml")   # also in the output / zip
    print(open("configs/prereg.proposed.yaml").read())
    tmp = dict(prop, lr={"A1": 1e-3, "A2": 1e-3, "T1": 1e-3})     # LRs are not known yet; they do not change run time
    yaml.safe_dump(tmp, open("/tmp/prereg_budget.yaml", "w"))
    run("python -m engramdelta.scripts.make_manifest --group all-phase1 --prereg /tmp/prereg_budget.yaml --out /tmp/manifests_p1")
    run(f"python -m engramdelta.scripts.budget '/tmp/manifests_p1/*.yaml' --nproc {NPROC} --quota-hours 13", check=False)
    """),
    md("""
    ## Your turn (no GPU needed)
    1. Download `04_phase0_clutter_pilot_results.zip` and give it to Claude, or read the two `>>` lines above.
    2. If the budget says OVER BUDGET, decide on cuts **before** committing.
    3. Copy `phase1_steps` and `primary_cell.F` from `results/prereg.proposed.yaml` into `configs/prereg.yaml`, fill the
       PREREG.md "Phase 0 outcomes" table, commit, push, and put the commit hash in PREREG.md and as `REPO_REF` in
       notebooks 10–14. Then run NB10.
    """),
    code(FINISH),
])

NB["03_lm_data_prep"] = notebook([
    md('''
    # 03 · Phase 2 data prep  (CPU session: uses no GPU quota)

    **Accelerator:** None · **Internet:** on

    It streams FineWeb-Edu `sample-10BT`. The first 2,000 documents become the val set (exact-duplicate texts are later
    dropped from train). It trains a 16k byte-level BPE on train documents only, writes ~150M train tokens and the val
    tokens as uint16 memmaps, and builds the Engram tokenizer-compression lookup. Takes about 20–40 min on a Kaggle CPU.

    **After it finishes:** Save Version. Then in NB20/NB21 use "Add Input → Notebook Output → this notebook".
    The data shows up at `/kaggle/input/<this-notebook-slug>/lmdata` (NB20 finds it automatically). You can also turn
    the output into a Dataset named `engramdelta-lmdata`.
    '''),
    *setup_cells(gpu=False),
    code('''
    TRAIN_TOKENS = 150_000_000    # NB20's token budget is <= this (plan: ~120M per run)
    run(f"python -m engramdelta.scripts.prepare_lm_data --out {WORK}/lmdata --train-tokens {TRAIN_TOKENS} "
        f"--val-docs 2000 --tok-docs 50000 --vocab 16384")
    print(open(f"{WORK}/lmdata/meta.json").read())
    run(f"ls -la {WORK}/lmdata")
    '''),
], gpu=False)

NB["10_phase1_pilot"] = notebook([
    md('''
    # 10 · Phase 1 pilot: LR (E1.0a) and filler-loss check (E1.0c)  (~1.1 GPU-h)

    **Needs:** `configs/prereg.yaml` with N_star and phase1_steps filled **and committed** (pin `REPO_REF` to that commit).

    * **E1.0a:** A1, A2, T1 × LR {3e-4, 1e-3, 3e-3} at (N\\*, F=256, P=64, a), 1 seed (9 runs). Pre-registered rule: best
      **val** recall wins, ties go to the smaller LR. A3/A4 inherit A2's LR; A5/A6 inherit A1's.
    * **E1.0c:** A1, A2 at the primary cell with λ = 0 (no filler loss), 3 seeds (6 runs). The λ = 1 counterparts come from
      E1.1/E1.2.

    To save session overhead you can append later groups to `GROUPS` (e.g. `"E1.1"`): finished runs are always skipped.
    '''),
    *setup_cells(gpu=True),
    code('''print(open("configs/prereg.yaml").read())'''),
    *sweep_cells(["E1.0a", "E1.0c"], 1.5),
    code('''
    from engramdelta import analysis
    df = analysis.load("results")
    print(analysis.lr_pilot_table(df).to_string(index=False))
    from engramdelta.scripts.make_manifest import resolve_lrs, load_prereg
    print("\\nresolved LRs (pre-registered rule):", resolve_lrs(load_prereg("configs/prereg.yaml"), "results"))
    '''),
    code(FINISH),
])

NB["11_phase1_H1_grid"] = notebook([
    md('''
    # 11 · Phase 1 · E1.1: H1 grid on the GDN baseline A1  (~4.1 GPU-h, 54 runs)

    A1 at N ∈ {N\\*, 2N\\*} × (F = 0, or F ∈ {128, 256, 512} × placement {a, b}) at P = 64, plus P ∈ {8, 512} at F = 512, N\\*,
    {a, b}. 3 seeds, λ = 1. Final weights are saved for the state metrics (E1.4).

    Resumable: rerun (with PREV_OUTPUTS / HF_REPO) until the sweep reports 0 pending.
    '''),
    *setup_cells(gpu=True),
    *sweep_cells(["E1.1"], 11.0),
    code('''
    from engramdelta import analysis
    print(analysis.h1_grid(analysis.load("results")).to_string(index=False))
    '''),
    code(FINISH),
])

NB["12_phase1_arms"] = notebook([
    md('''
    # 12 · Phase 1 · E1.2 (H2 + controls), E1.2r (T1 reference), E1.2l (leak under clutter)  (~6.5 GPU-h, 86 runs)

    * **E1.2:** A2, A3 (frozen table), A4 (unigram), A5 (SWA-16) × N ∈ {N\\*, 2N\\*} × (F = 0, or F = 512 × {a, b}), 3 seeds;
      plus A2 at F ∈ {128, 256} (N\\*, a).
    * **E1.2r:** T1 on the same 6 cells, 1 seed (reference only; expected near ceiling).
    * **E1.2l:** A6 at (2N\\*, F = 512, a/b): must stay at chance.
    '''),
    *setup_cells(gpu=True),
    *sweep_cells(["E1.2", "E1.2r", "E1.2l"], 11.0),
    code('''
    from engramdelta import analysis
    df = analysis.load("results")
    print(analysis.arms_table(df).to_string(index=False))
    print(analysis.leak_table(df).to_string(index=False))
    '''),
    code(FINISH),
])

NB["13_phase1_H4"] = notebook([
    md('''
    # 13 · Phase 1 · E1.3: H4 conv ablation  (~0.9 GPU-h, 12 runs)

    A2 with {Engram conv off}, {GDN short conv off} and {both off}, plus A1 with the GDN short conv off, at the primary cell
    (N\\*, F=512, P=64, a, λ=1), 3 seeds. Full A2 and A1 come from E1.2 / E1.1.
    '''),
    *setup_cells(gpu=True),
    *sweep_cells(["E1.3"], 2.0),
    code('''
    import yaml
    from engramdelta import analysis
    pr = yaml.safe_load(open("configs/prereg.yaml"))
    print(analysis.h4_table(analysis.load("results"), pr).to_string(index=False))
    '''),
    code(FINISH),
])

NB["14_phase1_analysis"] = notebook([
    md('''
    # 14 · Phase 1 analysis and Gate 1  (CPU session)

    Tables and figures from the immutable result files, the pre-registered rule checks, and the state metrics (E1.4:
    effective rank / Frobenius norm of the GDN state, plus β and Engram-gate α on pair vs filler tokens).
    **This notebook does not interpret anything. The Gate 1 decision is yours** (design §7.4).

    **Inputs:** every Phase 1 result and final.pt. Attach the outputs of NB10–13 via `PREV_OUTPUTS`, or use `HF_REPO`.
    '''),
    *setup_cells(gpu=False),
    code('''
    if os.environ.get("HF_REPO"):
        run("python -m engramdelta.scripts.sync pull ckpt", check=False)
    import yaml, pandas as pd
    from engramdelta import analysis
    pd.set_option("display.width", 200)
    pr = yaml.safe_load(open("configs/prereg.yaml"))
    df = analysis.load("results")
    p1 = df[df["phase"] == "phase1"]
    print(f"{len(p1)} finished Phase 1 runs; failures logged:", os.listdir("results/_failures") if os.path.isdir("results/_failures") else "none")
    print("fp16 health: runs with non-finite losses:", int((p1["nonfinite_losses"] > 0).sum()),
          "| max GradScaler skipped steps:", p1["skipped_steps"].max())
    print("code commits used:", p1["git_commit"].value_counts().to_dict())
    print("non-embedding params per arm:", p1.groupby("arm")["non_embedding"].first().to_dict())
    '''),
    md("## LR pilot (E1.0a) and the λ check (E1.0c vs λ = 1)"),
    code('''
    print(analysis.lr_pilot_table(df).to_string(index=False))
    print(analysis.lambda_check(df, pr).to_string(index=False))
    '''),
    md("## H1 grid (A1)"),
    code('''print(analysis.h1_grid(df).to_string(index=False))'''),
    md("## All arms per cell (λ = 1)"),
    code('''print(analysis.arms_table(df).to_string(index=False))'''),
    md("## Gate 1 table: pre-registered primary-cell comparisons (rule: |diff| > 2 × pooled std, n ≥ 3)"),
    code('''print(analysis.gate1_table(df, pr).to_string(index=False))'''),
    md("## H4: 2×2 conv ablation"),
    code('''print(analysis.h4_table(df, pr).to_string(index=False))'''),
    md("## Leak checks (A6 must be < 1%)"),
    code('''print(analysis.leak_table(df).to_string(index=False))'''),
    md("## E1.4 state metrics (exploratory)"),
    code('''
    run("python -m engramdelta.scripts.run_state_metrics --groups E1.1 E1.2 --arms A1 A2 A3 A4", check=False)
    from engramdelta.utils.run_log import read_results
    rows = []
    for r in read_results("results/phase1/E1.4"):
        d = r["data"]
        for layer, segs in r["state_occupancy"].items():
            for seg, m in segs.items():
                rows.append({"arm": r["arm"], "N": d["N"], "F": d["F"], "P": d["P"], "placement": d["placement"],
                             "layer": layer, "segment": seg, "eff_rank": m["effective_rank"], "fro": m["fro"]})
    if rows:
        st = pd.DataFrame(rows)
        print(st.groupby(["N", "F", "placement", "layer", "segment", "arm"])[["eff_rank", "fro"]].agg(["mean", "std"]).to_string())
        kinds = []
        for r in read_results("results/phase1/E1.4"):
            for name, cats in r["token_kind_stats"].items():
                kinds.append({"arm": r["arm"], "F": r["data"]["F"], "placement": r["data"]["placement"], "stat": name, **cats})
        print(pd.DataFrame(kinds).groupby(["stat", "arm", "F", "placement"]).mean(numeric_only=True).to_string())
    '''),
    md("## Figures"),
    code('''
    run("python -m engramdelta.scripts.make_plots --results-dir results --out figures", check=False)
    from IPython.display import Image, display
    for f in sorted(glob.glob("figures/*.png")):
        display(Image(f))
    '''),
    md('''
    ## Gate 1 (your decision, design §7.4)
    Continue to Phase 2 only if H1 holds AND A2 > A1 at the primary cell AND the A3/A4 controls do not explain the gain.
    If H1 fails, stop and write up the negative result (or reframe). If A3/A4 explain the gain, reframe as "gated side-pathway".
    '''),
    code(FINISH),
], gpu=False)

LMDATA_CELL = '''
# Locate the Phase 2 data (output of notebook 03, attached as an input)
cands = [p for p in ["/kaggle/input/engramdelta-lmdata"] + glob.glob("/kaggle/input/*/lmdata") + glob.glob("/kaggle/input/*")
         if os.path.exists(os.path.join(p, "meta.json"))]
assert cands, "Phase 2 data not found: attach the output of notebook 03 (Add Input -> Notebook Output)"
LMDATA = cands[0]
print("LM data:", LMDATA); print(open(os.path.join(LMDATA, "meta.json")).read())
'''

NB["20_phase2_train"] = notebook([
    md('''
    # 20 · Phase 2: small LM runs  (~9.5 GPU-h; only if Gate 1 passed)

    * **E2.0:** throughput pilot (A1, A2, 100 steps). The pre-registered rule then sets the token budget:
      min(150M, 0.9 × slower tok/s × 1.7 h), the same for every arm.
    * **E2.1:** A1, A2, A3 (seed 0). **E2.2:** T1, T2 (seed 0), needed for H3.
    * Model: d=384, 8 blocks, GDN 6×64 heads (or attention), ≈16.2M non-embedding params for every arm, Engram at blocks
      {1, 3} with ≈15M table params, 16k vocab, context 1024, AdamW + cosine, fp16 + GradScaler, 2-GPU synchronized DDP.
    * Checkpoints every 200 steps (about 10 min), pushed to HF if `HF_REPO` is set. A killed session resumes where it stopped.

    **Inputs:** the output of notebook 03 (LM data).
    '''),
    *setup_cells(gpu=True),
    code(LMDATA_CELL),
    *sweep_cells(["E2.0", "E2.1", "E2.2"], 11.3, extra_expr='f"--extra-set data.path={LMDATA}"'),
    code('''
    from engramdelta import analysis
    d = analysis.load("results")
    print(d[d["phase"] == "phase2"][["run_id", "arm", "steps", "loss_all", "ppl", "tok_per_s", "train_seconds", "non_embedding"]].to_string(index=False))
    '''),
    code(FINISH),
])

NB["21_phase2_eval"] = notebook([
    md('''
    # 21 · Phase 2 evaluation  (~0.75 GPU-h)

    For every finished Phase 2 run: val loss / perplexity over **all** val windows, loss by position bucket
    ([0,64), [64,256), [256,1024)), the **in-context-copy loss** (tokens whose current bigram already occurred > 64 positions
    earlier in the window), and GDN state occupancy at positions 256 / 512 / 1023. Sharded over both GPUs.
    '''),
    *setup_cells(gpu=True),
    code(LMDATA_CELL),
    code('''
    if os.environ.get("HF_REPO"):
        run("python -m engramdelta.scripts.sync pull ckpt", check=False)
    run(f"torchrun --standalone --nproc_per_node={NPROC} -m engramdelta.scripts.eval_lm --group E2.1 --group E2.2 --data-path {LMDATA}")
    from engramdelta import analysis
    print(analysis.phase2_table("results").to_string(index=False))
    '''),
    code(FINISH),
])

NB["22_phase2_analysis"] = notebook([
    md('''
    # 22 · Phase 2 analysis  (CPU session)

    Tables only. H3 compares the relative change of the in-context-copy loss and the long-position bucket for A2 vs A1 with
    T2 vs T1. **One seed per arm: any write-up must say so** (design §8). No interpretation here.
    '''),
    *setup_cells(gpu=False),
    code('''
    import pandas as pd
    from engramdelta import analysis
    pd.set_option("display.width", 200)
    t = analysis.phase2_table("results")
    print(t.to_string(index=False))
    print("\\nRelative change vs the arm's baseline (negative = lower loss):")
    print(analysis.phase2_relative(t).to_string(index=False))
    from engramdelta.utils.run_log import read_results
    for r in read_results("results/phase2/E2.eval"):
        if r.get("state_occupancy"):
            print(r["arm"], {l: {s: round(m["effective_rank"], 2) for s, m in segs.items()} for l, segs in r["state_occupancy"].items()})
    '''),
    code(FINISH),
], gpu=False)


def main():
    for name, nb in NB.items():
        for c in nb["cells"]:
            c["source"] = c["source"].replace("__NB_NAME__", name)
        if not any(LOGS_MD_MARK in c["source"] for c in nb["cells"]):
            nb["cells"].insert(1, md(LOGS_MD.replace("__NB_NAME__", name)))
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
