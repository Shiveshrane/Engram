# Local change log

Maintained by the change-log skill. Newest entries at the bottom.
Line numbers refer to the file *after* the change unless marked as removed.

---

## [0001] Add model code: Engram port, GDN mixer, attention/SWA/linear-attention mixers, full model, vendored references
*2026-10-19 00:23:10*

- **File:** `engramdelta/__init__.py` (created, +1/-0)
  - **Lines:** 1 added
- **File:** `engramdelta/models/engram.py` (created, +281/-0)
  - **Lines:** 1-281 added
- **File:** `engramdelta/models/gdn_block.py` (created, +257/-0)
  - **Lines:** 1-257 added
- **File:** `engramdelta/models/transformer_block.py` (created, +61/-0)
  - **Lines:** 1-61 added
- **File:** `engramdelta/models/swa.py` (created, +13/-0)
  - **Lines:** 1-13 added
- **File:** `engramdelta/models/linear_attn.py` (created, +53/-0)
  - **Lines:** 1-53 added
- **File:** `engramdelta/models/model.py` (created, +251/-0)
  - **Lines:** 1-251 added
- **File:** `third_party/README.md` (created, +16/-0)
  - **Lines:** 1-16 added
- **File:** `third_party/engram_demo_v1.py` (created, +423/-0)
  - **Lines:** 1-423 added
- **File:** `third_party/fla_naive.py` (created, +161/-0)
  - **Lines:** 1-161 added
- **File:** `third_party/LICENSE-ENGRAM` (created, +202/-0)
  - **Lines:** 1-202 added
- **File:** `third_party/LICENSE-FLA` (created, +21/-0)
  - **Lines:** 1-21 added

**Summary:** Add model code: Engram port, GDN mixer, attention/SWA/linear-attention mixers, full model, vendored references

**What the change does:** New package engramdelta.models. engram.py ports DeepSeek's engram_demo_v1.py (hash, prime tables, multi-head embedding, gating, dilated short conv) to hc_mult=1 with torch-side hashing. gdn_block.py reimplements FLA's GatedDeltaNet layer with a switchable core (FLA Triton kernel or a pure-PyTorch chunked delta rule using solve_triangular) plus a recurrent reference that captures states for analysis. model.py assembles arms A1-A6, T1, T2, LA from one config, computes loss only at loss positions (answers + lambda-weighted filler), counts parameters (embedding vs non-embedding) and matches MLP width for parameter matching. third_party/ holds the unchanged reference files with licenses and provenance.

Diff: `.changelog/entries/0001/diff.patch`

---

## [0002] Add MQAR/LM data, evaluation (recall, state occupancy, LM buckets/copy loss) and DDP/checkpoint/logging/config utilities
*2026-10-19 00:27:12*

- **File:** `engramdelta/data/mqar.py` (created, +209/-0)
  - **Lines:** 1-209 added
- **File:** `engramdelta/data/lm_data.py` (created, +69/-0)
  - **Lines:** 1-69 added
- **File:** `engramdelta/eval/mqar_eval.py` (created, +38/-0)
  - **Lines:** 1-38 added
- **File:** `engramdelta/eval/state_metrics.py` (created, +120/-0)
  - **Lines:** 1-120 added
- **File:** `engramdelta/eval/lm_eval.py` (created, +74/-0)
  - **Lines:** 1-74 added
- **File:** `engramdelta/utils/dist.py` (created, +114/-0)
  - **Lines:** 1-114 added
- **File:** `engramdelta/utils/checkpoint.py` (created, +54/-0)
  - **Lines:** 1-54 added
- **File:** `engramdelta/utils/run_log.py` (created, +85/-0)
  - **Lines:** 1-85 added
- **File:** `engramdelta/utils/hub_sync.py` (created, +78/-0)
  - **Lines:** 1-78 added
- **File:** `engramdelta/utils/config.py` (created, +55/-0)
  - **Lines:** 1-55 added

**Summary:** Add MQAR/LM data, evaluation (recall, state occupancy, LM buckets/copy loss) and DDP/checkpoint/logging/config utilities

**What the change does:** data/mqar.py generates MQAR with clutter placements a/b from a fixed phrase pool, per-sequence key-value resampling, answer + filler-next-token targets, and step-seeded deterministic streams; data/lm_data.py serves rank-strided deterministic windows from uint16 memmaps. eval/ adds DDP-sharded MQAR recall, GDN state effective-rank/Frobenius/spectrum via the recurrent reference, beta/alpha stats per token category, and LM eval with position buckets and the in-context-copy mask. utils/ adds idempotent torchrun/DDP setup with a parameter-sync guard, atomic checkpoints with per-rank RNG, immutable read-only result JSON, optional HF Hub sync, and YAML configs with inheritance and dotted overrides.

Diff: `.changelog/entries/0002/diff.patch`

---

## [0003] Add training (synchronized DDP), sweep/manifest/budget tooling, checks, data prep, evals and analysis
*2026-10-19 11:44:57*

- **File:** `engramdelta/scripts/train.py` (created, +322/-0)
  - **Lines:** 1-322 added
- **File:** `engramdelta/scripts/sweep.py` (created, +93/-0)
  - **Lines:** 1-93 added
- **File:** `engramdelta/scripts/make_manifest.py` (created, +195/-0)
  - **Lines:** 1-195 added
- **File:** `engramdelta/scripts/budget.py` (created, +97/-0)
  - **Lines:** 1-97 added
- **File:** `engramdelta/scripts/param_table.py` (created, +64/-0)
  - **Lines:** 1-64 added
- **File:** `engramdelta/scripts/ddp_check.py` (created, +123/-0)
  - **Lines:** 1-123 added
- **File:** `engramdelta/scripts/nccl_smoke.py` (created, +47/-0)
  - **Lines:** 1-47 added
- **File:** `engramdelta/scripts/bench_kernels.py` (created, +147/-0)
  - **Lines:** 1-147 added
- **File:** `engramdelta/scripts/prepare_lm_data.py` (created, +134/-0)
  - **Lines:** 1-134 added
- **File:** `engramdelta/scripts/eval_lm.py` (created, +76/-0)
  - **Lines:** 1-76 added
- **File:** `engramdelta/scripts/run_state_metrics.py` (created, +65/-0)
  - **Lines:** 1-65 added
- **File:** `engramdelta/scripts/sync.py` (created, +34/-0)
  - **Lines:** 1-34 added
- **File:** `engramdelta/scripts/make_plots.py` (created, +145/-0)
  - **Lines:** 1-145 added
- **File:** `engramdelta/analysis.py` (created, +213/-0)
  - **Lines:** 1-213 added

**Summary:** Add training (synchronized DDP), sweep/manifest/budget tooling, checks, data prep, evals and analysis

**What the change does:** train.py runs one MQAR or LM run under torchrun with DDP gradient all-reduce, no_sync accumulation, fp16 GradScaler, cosine LR, parameter-sync guard, atomic resumable checkpoints, time-limit exit code 3 and immutable result JSON. make_manifest.py encodes the exact experiment groups E0.3-E2.2 and resolves prereg 'auto' values by pre-registered rules; sweep.py runs manifests with skip-if-done and a session budget; budget.py estimates GPU-hours from measured throughput. ddp_check/nccl_smoke/bench_kernels implement the Phase 0 DDP, NCCL and kernel tests; prepare_lm_data/eval_lm/run_state_metrics/sync cover Phase 2 data, LM eval, state metrics and HF Hub sync; analysis.py and make_plots.py produce tables, rule checks and figures without interpretation.

Diff: `.changelog/entries/0003/diff.patch`

---

## [0004] Add Phase 0 test suite (tests 1-9, DDP, LM eval, manifests)
*2026-10-19 12:00:39*

- **File:** `tests/conftest.py` (created, +7/-0)
  - **Lines:** 1-7 added
- **File:** `tests/helpers.py` (created, +41/-0)
  - **Lines:** 1-41 added
- **File:** `tests/test_causality.py` (created, +88/-0)
  - **Lines:** 1-88 added
- **File:** `tests/test_overfit.py` (created, +22/-0)
  - **Lines:** 1-22 added
- **File:** `tests/test_params.py` (created, +27/-0)
  - **Lines:** 1-27 added
- **File:** `tests/test_kernels.py` (created, +114/-0)
  - **Lines:** 1-114 added
- **File:** `tests/test_engram_parity.py` (created, +105/-0)
  - **Lines:** 1-105 added
- **File:** `tests/test_mqar_leak.py` (created, +100/-0)
  - **Lines:** 1-100 added
- **File:** `tests/test_repro_known.py` (created, +39/-0)
  - **Lines:** 1-39 added
- **File:** `tests/test_fp16.py` (created, +31/-0)
  - **Lines:** 1-31 added
- **File:** `tests/test_determinism.py` (created, +42/-0)
  - **Lines:** 1-42 added
- **File:** `tests/test_ddp.py` (created, +32/-0)
  - **Lines:** 1-32 added
- **File:** `tests/test_lm_eval.py` (created, +63/-0)
  - **Lines:** 1-63 added
- **File:** `tests/test_manifest.py` (created, +29/-0)
  - **Lines:** 1-29 added

**Summary:** Add Phase 0 test suite (tests 1-9, DDP, LM eval, manifests)

**What the change does:** Causality (Engram module, hash padding, every arm), overfit-one-batch for every arm, parameter table with matching invariants, kernel correctness vs FLA naive reference, Engram parity vs the vendored DeepSeek demo (stubbed tokenizer), MQAR format/leak checks with an A6 chance probe, fp16 vs fp32 (CUDA only), determinism/immutable results/single-process resume, 2-process gloo DDP checks via ddp_check, LM copy-mask/sampler/eval checks, and manifest run counts (Phase 1 = 167). test_repro_known reads Kaggle E0.4/E0.5 results. 67 pass locally; CUDA/FLA/Kaggle-result tests skip.

Diff: `.changelog/entries/0004/diff.patch`

---

## [0005] Add --extra-set option to make_manifest
*2026-10-19 12:07:16*

- **File:** `engramdelta/scripts/make_manifest.py` (modified, +4/-0)
  - **Lines:** 179-180 added; 188-189 added

**Summary:** Add --extra-set option to make_manifest

**What the change does:** make_manifest previously wrote runs with only the group's own overrides. It now accepts repeatable --extra-set key=value overrides appended to every run, used by NB20 to point Phase 2 runs at the attached LM data path.

Diff: `.changelog/entries/0005/diff.patch`

---

## [0006] Fix plot layout and make the DDP test use loopback rendezvous
*2026-10-19 12:07:16*

- **File:** `engramdelta/scripts/make_plots.py` (modified, +6/-2)
  - **Lines:** 11 added; 57-58 added; 90-91 modified (were 87 before); 108 modified (were 104 before)

**Summary:** Fix plot layout and make the DDP test use loopback rendezvous

**What the change does:** make_plots: the arms chart now shows only cells where at least two arms were run, the legend sits outside the plot, and the E0.4 N axis shows plain numbers. test_ddp: torchrun --standalone timed out resolving the macOS .local hostname; the test now uses --master_addr=127.0.0.1 with a free port and GLOO_SOCKET_IFNAME=lo0 on macOS.

Diff: `.changelog/entries/0006/diff.patch`

---

## [0007] Add per-arm YAML configs for phases 0-2 and the machine-readable pre-registration
*2026-10-19 12:07:16*

- **File:** `configs/phase0/A1.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase0/A2.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase0/A3.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase0/A4.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase0/A5.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase0/A6.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase0/base.yaml` (created, +6/-0)
  - **Lines:** 1-6 added
- **File:** `configs/phase0/LA.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase0/T1.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase1/A1_noGconv.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase1/A1.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase1/A2_noboth.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase1/A2_noEconv.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase1/A2_noGconv.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase1/A2.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase1/A3.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase1/A4.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase1/A5.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase1/A6.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase1/base.yaml` (created, +59/-0)
  - **Lines:** 1-59 added
- **File:** `configs/phase1/T1.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase2/A1.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase2/A2.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase2/A3.yaml` (created, +4/-0)
  - **Lines:** 1-4 added
- **File:** `configs/phase2/A5.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase2/base.yaml` (created, +49/-0)
  - **Lines:** 1-49 added
- **File:** `configs/phase2/T1.yaml` (created, +5/-0)
  - **Lines:** 1-5 added
- **File:** `configs/phase2/T2.yaml` (created, +6/-0)
  - **Lines:** 1-6 added

**Summary:** Add per-arm YAML configs for phases 0-2 and the machine-readable pre-registration

**What the change does:** phase1/base.yaml fixes the MQAR task and the d=128, 2-block, 4x32-head GDN model with Engram at block 1 (2.4k rows/head); each arm file (A1-A6, T1, H4 variants) is a small override, with match_nonembed_of: A2 for arms without Engram. phase0 inherits phase1 with F=0 and adds LA. phase2 defines the d=384, 8-block LM with Engram at blocks 1 and 3 (14.7k rows/head), A1/A2/A3/A5/T1/T2. prereg.yaml holds N_star, phase1_steps, LRs and Phase 2 tokens, to be filled after Phase 0.

Diff: `.changelog/entries/0007/diff.patch`

---

## [0008] Add project docs, pre-registration, deviations log and the 12 Kaggle notebooks
*2026-10-19 12:07:16*

- **File:** `README.md` (created, +73/-0)
  - **Lines:** 1-73 added
- **File:** `DEVIATIONS.md` (created, +40/-0)
  - **Lines:** 1-40 added
- **File:** `CHANGELOG.md` (created, +9/-0)
  - **Lines:** 1-9 added
- **File:** `PREREG.md` (created, +52/-0)
  - **Lines:** 1-52 added
- **File:** `requirements.txt` (created, +14/-0)
  - **Lines:** 1-14 added
- **File:** `.gitignore` (created, +10/-0)
  - **Lines:** 1-10 added
- **File:** `notebooks/build_notebooks.py` (created, +653/-0)
  - **Lines:** 1-653 added
- **File:** `notebooks/00_env_and_kernels.ipynb` (created, +107/-0)
  - **Lines:** 1-107 added
- **File:** `notebooks/01_phase0_tests.ipynb` (created, +102/-0)
  - **Lines:** 1-102 added
- **File:** `notebooks/02_phase0_known_result.ipynb` (created, +107/-0)
  - **Lines:** 1-107 added
- **File:** `notebooks/03_lm_data_prep.ipynb` (created, +47/-0)
  - **Lines:** 1-47 added
- **File:** `notebooks/10_phase1_pilot.ipynb` (created, +68/-0)
  - **Lines:** 1-68 added
- **File:** `notebooks/11_phase1_H1_grid.ipynb` (created, +61/-0)
  - **Lines:** 1-61 added
- **File:** `notebooks/12_phase1_arms.ipynb` (created, +61/-0)
  - **Lines:** 1-61 added
- **File:** `notebooks/13_phase1_H4.ipynb` (created, +61/-0)
  - **Lines:** 1-61 added
- **File:** `notebooks/14_phase1_analysis.ipynb` (created, +155/-0)
  - **Lines:** 1-155 added
- **File:** `notebooks/20_phase2_train.ipynb` (created, +68/-0)
  - **Lines:** 1-68 added
- **File:** `notebooks/21_phase2_eval.ipynb` (created, +61/-0)
  - **Lines:** 1-61 added
- **File:** `notebooks/22_phase2_analysis.ipynb` (created, +54/-0)
  - **Lines:** 1-54 added

**Summary:** Add project docs, pre-registration, deviations log and the 12 Kaggle notebooks

**What the change does:** README explains layout, local use, the Kaggle notebook order and synchronized DDP. DEVIATIONS lists every departure from the DeepSeek/FLA references and the design doc; PREREG fixes the decision rules and resolution rules before results exist; CHANGELOG tracks eval/data changes (design rule 3). notebooks/build_notebooks.py generates notebooks 00-22 with a shared setup cell (clone/copy code, --no-deps installs, secrets, HF or output-chaining restore) and manifest-driven resumable sweeps.

Diff: `.changelog/entries/0008/diff.patch`

---

## [0009] Point notebooks at the GitHub repo Shiveshrane/Engram
*2026-10-19 16:18:01*

- **File:** `notebooks/build_notebooks.py` (modified, +1/-1)
  - **Lines:** 37 modified (were 37 before)
- **File:** `notebooks/00_env_and_kernels.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/01_phase0_tests.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/02_phase0_known_result.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/03_lm_data_prep.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/10_phase1_pilot.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/11_phase1_H1_grid.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/12_phase1_arms.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/13_phase1_H4.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/14_phase1_analysis.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/20_phase2_train.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/21_phase2_eval.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/22_phase2_analysis.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)

**Summary:** Point notebooks at the GitHub repo Shiveshrane/Engram

**What the change does:** The settings cell's REPO_URL default was a placeholder (YOUR_GITHUB_USER/engramdelta.git). It now defaults to the public repo https://github.com/Shiveshrane/Engram.git, so every notebook clones the code on Kaggle without edits or a GitHub token.

Diff: `.changelog/entries/0009/diff.patch`

---

## [0010] Install fla-core (FLA's kernels) instead of the wrapper and surface FLA import errors
*2026-10-19 17:06:08*

- **File:** `engramdelta/models/gdn_block.py` (modified, +6/-1)
  - **Lines:** 126 added; 130-131 added; 134 added; 136-137 modified (were 132 before)
- **File:** `engramdelta/scripts/bench_kernels.py` (modified, +5/-1)
  - **Lines:** 20 added; 123-126 modified (were 122 before)
- **File:** `notebooks/build_notebooks.py` (modified, +5/-4)
  - **Lines:** 42 modified (were 42 before); 189 modified (were 189 before); 214-215 modified (were 214 before); 263 modified (were 262 before)
- **File:** `notebooks/00_env_and_kernels.ipynb` (modified, +4/-4)
  - **Lines:** 6 modified (were 6 before); 13 modified (were 13 before); 44 modified (were 44 before); 80 modified (were 80 before)
- **File:** `notebooks/01_phase0_tests.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/02_phase0_known_result.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/03_lm_data_prep.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/10_phase1_pilot.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/11_phase1_H1_grid.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/12_phase1_arms.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/13_phase1_H4.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/14_phase1_analysis.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/20_phase2_train.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/21_phase2_eval.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `notebooks/22_phase2_analysis.ipynb` (modified, +1/-1)
  - **Lines:** 13 modified (were 13 before)
- **File:** `requirements.txt` (modified, +1/-1)
  - **Lines:** 14 modified (were 14 before)

**Summary:** Install fla-core (FLA's kernels) instead of the wrapper and surface FLA import errors

**What the change does:** NB00 on Kaggle showed fla.ops was not importable: flash-linear-attention 0.5.x is only a wrapper and the Triton kernels live in fla-core, which --no-deps skipped, so the FLA kernel was never tested. FLA_SPEC now defaults to fla-core==0.5.2; fla_available() records the import error in FLA_IMPORT_ERROR and bench_kernels prints and stores it; the NB00 version cell uses importlib.metadata instead of the missing fla.__version__.

Diff: `.changelog/entries/0010/diff.patch`

---

## [0011] NB02: run E0.6 first and train E0.4/E0.5 for the measured step count
*2026-10-19 21:04:35*

- **File:** `notebooks/build_notebooks.py` (modified, +28/-5)
  - **Lines:** 316 modified (were 316 before); 318-323 modified (were 318-320 before); 326-327 added; 337-355 modified (were 332 before)
- **File:** `notebooks/02_phase0_known_result.ipynb` (modified, +26/-2)
  - **Lines:** 6 modified (were 6 before); 23-27 added; 32-51 modified (were 27 before)
- **File:** `PREREG.md` (modified, +1/-0)
  - **Lines:** 29 added

**Summary:** NB02: run E0.6 first and train E0.4/E0.5 for the measured step count

**What the change does:** NB02 previously ran E0.6, E0.4 and E0.5 in one sweep with E0.4 fixed at 2000 steps. A local check showed MQAR's plateau (recall stuck at 1/N) can last ~1000 steps even at N=8, so 2000 steps may be too short at large N and waste ~1.8 GPU-h. NB02 now runs E0.6 alone, prints the val-recall curve, sets E04_STEPS by a pre-registered rule (A1's 98%-of-max step clamped to [2000, 4000], overridable), and passes it to the E0.4/E0.5 sweep via --extra-set; PREREG.md records the rule.

Diff: `.changelog/entries/0011/diff.patch`

---

## [0012] Write every sweep run's full output to results/_logs/<group>/<run_id>.log
*2026-10-20 01:10:14*

- **File:** `engramdelta/scripts/sweep.py` (modified, +23/-2)
  - **Lines:** 12-13 added; 40-56 added; 93-94 modified (were 74 before); 105-106 modified (were 85 before)

**Summary:** Write every sweep run's full output to results/_logs/<group>/<run_id>.log

**What the change does:** sweep.main() used to launch each run with subprocess.run, so the training output only reached the notebook stream. The new run_logged() starts the run with Popen (PYTHONUNBUFFERED=1), prints every line live as before, and also appends it to results/_logs/<group>/<run_id>.log with start/exit-code markers. The _failures entry now points to that log. Because the logs live under results/, they travel with HF sync, PREV_OUTPUTS restore and the notebook results zip.

Diff: `.changelog/entries/0012/diff.patch`

---

## [0013] Notebooks: session log file, cell-error tracebacks, and a downloadable results zip
*2026-10-20 01:10:14*

- **File:** `notebooks/build_notebooks.py` (modified, +75/-1)
  - **Lines:** 49 modified (were 49 before); 51 added; 56-119 added; 214-220 added; 742-743 added
- **File:** `.gitignore` (modified, +3/-0)
  - **Lines:** 11-13 added
- **File:** `README.md` (modified, +6/-0)
  - **Lines:** 64-69 added

**Summary:** Notebooks: session log file, cell-error tracebacks, and a downloadable results zip

**What the change does:** The shared SETUP cell now tees sys.stdout/stderr into logs/<notebook>__<UTC start>.log with elapsed seconds on every line, and registers IPython pre/post_run_cell hooks that mark each cell's start and write the full traceback of a failing cell. After every cell, bundle() copies the session log into results/_logs/notebooks/ and zips results/ (result JSONs + per-run logs, no checkpoints) into <notebook>_results.zip in the output, so a partial zip exists even if the session dies. FINISH prints how to download and unzip it. main() substitutes the notebook name for __NB_NAME__. .gitignore ignores local logs/ and *_results.zip; README documents the download step. All 12 .ipynb files were regenerated from build_notebooks.py.

Diff: `.changelog/entries/0013/diff.patch`

---

## [0014] Delete a finished run's resume checkpoint after its result file is written
*2026-10-20 01:37:42*

- **File:** `engramdelta/scripts/train.py` (modified, +4/-0)
  - **Lines:** 295-298 added

**Summary:** Delete a finished run's resume checkpoint after its result file is written

**What the change does:** train() kept ckpt/<run_id>/last.pt after a run finished, although sweep skips any run whose result JSON exists, so the file is never used again. It cost about 15 MB per Phase 1 run (most of NB02's 633 MB output). After the result is written (rank 0, write_result_file=True) last.pt and its then-empty folder are removed. Preempted runs keep their checkpoint; final.pt (save_final) is untouched; a copy already pushed to the HF Hub is not deleted.

Diff: `.changelog/entries/0014/diff.patch`

---

## [0015] Kaggle: clone the code to /tmp so the notebook output holds only generated files
*2026-10-20 01:37:42*

- **File:** `notebooks/build_notebooks.py` (modified, +15/-3)
  - **Lines:** 54-58 modified (were 54-55 before); 159-166 added; 202-203 modified (were 191 before)
- **File:** `README.md` (modified, +2/-0)
  - **Lines:** 64-65 added

**Summary:** Kaggle: clone the code to /tmp so the notebook output holds only generated files

**What the change does:** SETUP used to clone the repo into /kaggle/working/engramdelta, so every notebook output contained the whole code tree. REPO is now /tmp/engramdelta on Kaggle (not saved), and results/ and figures/ are symlinked from the repo to /kaggle/working/results and /kaggle/working/figures, so all existing relative paths still work. The output now contains only results/, figures/, logs/, ckpt/ (unfinished runs) and the results zip. restore() reads PREV_OUTPUTS from <prev>/results and, for outputs made before this change, <prev>/engramdelta/results. All .ipynb files were regenerated.

Diff: `.changelog/entries/0015/diff.patch`

---

## [0016] Notebook log is now run.log at the top of the output; every notebook documents its logs
*2026-10-20 01:40:57*

- **File:** `notebooks/build_notebooks.py` (modified, +36/-9)
  - **Lines:** 60-63 modified (were 60-62 before); 83-84 modified (were 82-83 before); 86 added; 98 modified (were 96 before); 123-128 modified (were 121-122 before); 238-257 modified (were 232 before); 781-782 added
- **File:** `.gitignore` (modified, +1/-0)
  - **Lines:** 14 added
- **File:** `README.md` (modified, +2/-2)
  - **Lines:** 64 modified (were 64 before); 67 modified (were 67 before)

**Summary:** Notebook log is now run.log at the top of the output; every notebook documents its logs

**What the change does:** The session log used to be logs/<notebook>__<UTC>.log, which was easy to miss. SETUP now writes it live to /kaggle/working/run.log (fixed name, top level of the Output tab) with a header line naming the notebook and start time, and prints a banner with the paths of run.log, the per-run logs (results/_logs/<group>/<run_id>.log) and the results zip. bundle() copies run.log into results/_logs/notebooks/<notebook>__<UTC>.log so the zip carries it. main() inserts a 'Logs and downloads' markdown cell as the second cell of every notebook explaining the three files and how to download/unzip them. run.log added to .gitignore.

Diff: `.changelog/entries/0016/diff.patch`

---

## [0017] Keep ENGRAMDELTA_DESIGN.md out of the repo
*2026-10-20 01:42:38*

- **File:** `.gitignore` (modified, +3/-0)
  - **Lines:** 15-17 added
- **File:** `README.md` (modified, +1/-1)
  - **Lines:** 5 modified (were 5 before)

**Summary:** Keep ENGRAMDELTA_DESIGN.md out of the repo

**What the change does:** Added ENGRAMDELTA_DESIGN.md to .gitignore and removed it from the git index with git rm --cached (the local file is untouched). It had been committed in 61388d4, so the ignore rule alone would not have stopped tracking it. The README link to it became plain text, since the file is no longer on GitHub. The file still exists in the history of earlier commits.

Diff: `.changelog/entries/0017/diff.patch`

---

## [0018] E0.6 runs at N = N*; E0.4/E0.5 fixed at 4000 steps; step rule requires A1 to learn
*2026-10-20 01:53:38*

- **File:** `engramdelta/scripts/make_manifest.py` (modified, +21/-5)
  - **Lines:** 16-17 added; 29 added; 92-103 added; 114-115 modified (were 99-100 before); 117-119 modified (were 102-103 before); 121 modified (were 105 before)
- **File:** `engramdelta/analysis.py` (modified, +12/-4)
  - **Lines:** 110-116 modified (were 110-111 before); 119-120 added; 124-126 modified (were 117-118 before)
- **File:** `engramdelta/scripts/make_plots.py` (modified, +7/-3)
  - **Lines:** 116-118 modified (were 116 before); 121 added; 126-128 modified (were 123-124 before)
- **File:** `tests/test_manifest.py` (modified, +13/-0)
  - **Lines:** 30-42 added

**Summary:** E0.6 runs at N = N*; E0.4/E0.5 fixed at 4000 steps; step rule requires A1 to learn

**What the change does:** E0.6 was hard-coded at N=64, which E0.4 showed is 2N* (A1 2% recall even without clutter), so its step-budget proposal was noise. make_manifest now builds E0.6 at N = prereg N_star, or N* computed from finished E0.4 results via e06_n() (exits if neither exists). E0.4/E0.5 runs carry train.steps=4000 (PHASE0_STEPS) themselves; run IDs are unchanged, so the 40 finished runs are still recognised. analysis.step_budget_proposal() takes N to select E0.6 runs and returns learned=False / proposal=None when max val recall < 0.5. make_plots' E0.6 plot uses N* and its title shows N. New tests check the 4000 steps, the E0.6 cell and the missing-N* exit.

Diff: `.changelog/entries/0018/diff.patch`

---

## [0019] NB02: E0.4/E0.5 first, then E0.6 at N*, with hard stops; resume restore skips finished checkpoints
*2026-10-20 01:53:38*

- **File:** `notebooks/build_notebooks.py` (modified, +73/-58)
  - **Lines:** 43 modified (were 43 before); 207-225 modified (were 207-211 before); 438-439 modified (were 424-425 before); 441-444 modified (were 427-429 before); 446-449 modified (were 431-434 before); 451-452 modified (were 436 before); 454-455 modified (were 438-439 before); 460 modified (were 444 before); 462-465 modified (were 446-466 before); 474-478 modified (were 475-476 before); 480-484 modified (were 478-479 before); 486-488 modified (were 481 before); 490-491 modified (were 483-484 before); 493-496 modified (were 486-487 before); 501-506 modified (were 492 before); 508 modified (were 494 before); 511 modified (were 497 before); 513 added; 520-521 modified (were 505-506 before); 524-525 modified (were 509-510 before); 528 modified (were 513 before)
- **File:** `PREREG.md` (modified, +15/-2)
  - **Lines:** 29-30 modified (were 29 before); 32-33 modified (were 31 before); 39-49 added

**Summary:** NB02: E0.4/E0.5 first, then E0.6 at N*, with hard stops; resume restore skips finished checkpoints

**What the change does:** NB02 now runs E0.4+E0.5 (4000 steps), computes test 7 and N*, stops with an error if test 7 fails, reports missing E0.5 runs, runs E0.6 at N*, and stops if A1 never exceeds 50% val recall instead of proposing a step budget. phase1_steps is A1's proposal (the written rule; it previously took the max over A1 and A2), and prereg.proposed.yaml is also copied into results/. SETUP prints the attached inputs, accepts globs in PREV_OUTPUTS, finds results/ and ckpt/ folders up to 4 levels deep (old outputs had engramdelta/results), and restores checkpoints only for runs without a result, so ~600 MB of finished-run checkpoints are not copied into the new output. PREREG.md: the rule text is updated and a 'Changes to these rules before the pre-registration commit' section records why and when.

Diff: `.changelog/entries/0019/diff.patch`

---

## [0020] E0.7 clutter pilot group; Phase 1 cells follow the pilot's primary F
*2026-10-20 02:44:39*

- **File:** `engramdelta/scripts/make_manifest.py` (modified, +21/-9)
  - **Lines:** 16-19 modified (were 16-17 before); 32-33 added; 127-131 added; 136-139 modified (were 127 before); 143 modified (were 131 before); 154-155 modified (were 142-143 before); 164 modified (were 152 before); 170 modified (were 158 before); 195 modified (were 183 before)
- **File:** `engramdelta/analysis.py` (modified, +32/-12)
  - **Lines:** 110-138 added; 146-148 modified (were 117-128 before)
- **File:** `configs/prereg.yaml` (modified, +3/-3)
  - **Lines:** 4-5 modified (were 4-5 before); 12 modified (were 12 before)
- **File:** `tests/test_manifest.py` (modified, +22/-0)
  - **Lines:** 43-64 added

**Summary:** E0.7 clutter pilot group; Phase 1 cells follow the pilot's primary F

**What the change does:** New manifest group E0.7: A1 at N=N*, P=64, a, 6000 steps with eval every 250, at F=64/128/256 (lambda=1) and F=512 (lambda=0). Phase 1 groups used a hard-coded F=512 as the clutter condition (and F=256 for the LR pilot); they now use primary_cell.F everywhere arms are compared under clutter (E1.0a, E1.0c, E1.2/E1.2r/E1.2l, the E1.1 P sweep, E1.3), refuse to build while it is null, and the A2 dose-response runs at the design levels other than the primary F. E1.1 keeps F in {128,256,512}; run counts stay 167. analysis gains curve_stats() (shared with step_budget_proposal), clutter_pilot() and primary_f_rule() (largest of 256,128 with max val recall >= 0.5). prereg.yaml: N_star=32 (the E0.4 rule result), primary_cell.F=null until the pilot. Tests cover the E0.7 cells and the primary-F wiring.

Diff: `.changelog/entries/0020/diff.patch`

---

## [0021] Add notebook 04: E0.7 clutter pilot that proposes primary F and phase1_steps
*2026-10-20 02:44:39*

- **File:** `notebooks/build_notebooks.py` (modified, +91/-0)
  - **Lines:** 526-527 added; 530-618 added
- **File:** `PREREG.md` (modified, +16/-4)
  - **Lines:** 12 modified (were 12 before); 31-39 modified (were 31-33 before); 57-62 added
- **File:** `README.md` (modified, +1/-0)
  - **Lines:** 52 added

**Summary:** Add notebook 04: E0.7 clutter pilot that proposes primary F and phase1_steps

**What the change does:** New 04_phase0_clutter_pilot notebook (no inputs; N* from prereg.yaml): runs the 4 E0.7 runs (~0.55 GPU-h), prints the pilot table and curves, saves figures/E0.7_clutter_pilot.png, applies the pre-registered primary-F rule and stops with an error if neither F=256 nor F=128 learns, then writes results/prereg.proposed.yaml (phase1_steps, primary_cell.F) and prints the Phase 1 budget from measured throughput. NB02's instructions point to NB04 when E0.6 did not learn. PREREG.md: new primary-F and step-budget rules, primary cell F = primary F, and a dated entry explaining the change. README lists NB04.

Diff: `.changelog/entries/0021/diff.patch`

---

## [0022] Option A Phase 1 grid: 67 runs at N*=32, F=512, lambda=0, 6000 steps
*2026-10-20 13:13:15*

- **File:** `engramdelta/scripts/make_manifest.py` (modified, +24/-32)
  - **Lines:** 18-19 modified (were 18-19 before); 133-134 added; 139-142 modified (were 137-141 before); 144 modified (were 143 before); 147 modified (were 146 before); 149-152 modified (were 148-157 before); 154 modified (were 159 before); 156-157 modified (were 161-162 before); 159-160 modified (were 164-165 before); 162-164 modified (were 167-171 before); 166 modified (were 173-174 before); 188 modified (were 196 before)
- **File:** `engramdelta/analysis.py` (modified, +5/-4)
  - **Lines:** 16 modified (were 16 before); 165-167 modified (were 165-166 before); 177 modified (were 176 before)
- **File:** `engramdelta/scripts/make_plots.py` (modified, +1/-1)
  - **Lines:** 107 modified (were 107 before)
- **File:** `configs/prereg.yaml` (modified, +2/-3)
  - **Lines:** 5 modified (were 5 before); 11 modified (were 11-12 before)
- **File:** `tests/test_manifest.py` (modified, +32/-29)
  - **Lines:** 1-4 modified (were 1 before); 7-9 modified (were 4-6 before); 12-17 modified (were 9-10 before); 23 modified (were 16 before); 30-43 modified (were 23-29 before); 49 modified (were 35-36 before); old 40 removed (after line 52); 58 modified (were 46 before); 65 modified (were 53-62 before); 67 modified (were 64 before)

**Summary:** Option A Phase 1 grid: 67 runs at N*=32, F=512, lambda=0, 6000 steps

**What the change does:** After the E0.7 pilot (lambda=1 never learns; lambda=0 learns at F=512) the human chose option A. make_manifest's Phase 1 groups now read F, P, placement and lambda from prereg primary_cell and use N* only, placement a: E1.0a = A1, A2 x 3 LRs (T1 dropped), E1.0c = A1, A2 at the other lambda (1), E1.1 = A1 x F in {0,128,256,512}, E1.2 = A2-A5 x F in {0,512} + A2 dose at 128/256, E1.2l = one A6 run, E1.3 unchanged in arms; E1.2r (T1) removed. prereg.yaml: phase1_steps 6000, primary_cell F 512 / filler_weight 0.0, lr only A1/A2. analysis: arms_table takes an optional lam filter, E1.2r dropped from the main groups, H1 label uses the primary F. Tests rewritten for the 67-run grid.

Diff: `.changelog/entries/0022/diff.patch`

---

## [0023] Notebooks 10-14 for the option A grid; PREREG/DEVIATIONS record the decision
*2026-10-20 13:13:15*

- **File:** `notebooks/build_notebooks.py` (modified, +76/-44)
  - **Lines:** 648-649 modified (were 648-649 before); 651-652 modified (were 651 before); 654-656 modified (were 653-656 before); 658-660 modified (were 658-659 before); 662-665 modified (were 661-663 before); 670-675 modified (were 668-669 before); 680-681 modified (were 674-675 before); 683-684 modified (were 677-678 before); 686-693 modified (were 680-681 before); 695-697 modified (were 683-684 before); 699-703 modified (were 686-687 before); 708-709 modified (were 692-693 before); 711-722 modified (were 695-699 before); 724-725 modified (were 701-702 before); 730 modified (were 707 before); 735-736 modified (were 712-713 before); 738-748 modified (were 715-717 before); 750-751 modified (were 719-720 before); 756 modified (were 725 before); 768-769 modified (were 737 before); 794 modified (were 762 before)
- **File:** `PREREG.md` (modified, +41/-14)
  - **Lines:** 12-14 modified (were 12-13 before); 25-27 modified (were 24 before); 66-74 modified (were 63 before); 76 modified (were 65-67 before); 78-92 modified (were 69 before); 96-105 modified (were 73-78 before)
- **File:** `DEVIATIONS.md` (modified, +3/-1)
  - **Lines:** 32 modified (were 32 before); 41-42 added
- **File:** `README.md` (modified, +4/-4)
  - **Lines:** 54-57 modified (were 54-57 before)

**Summary:** Notebooks 10-14 for the option A grid; PREREG/DEVIATIONS record the decision

**What the change does:** NB10 = E1.0a only (A1, A2 LR pilot, ~1.4 GPU-h) and writes results/lrs.resolved.yaml; NB11 = E1.1 + E1.0c (~2.8 h), NB12 = E1.2 + E1.2l (~3.7 h), NB13 = E1.3 (~2.7 h); 11-13 are independent once the LRs are committed, so they can run in parallel. NB14 headings and inputs updated. PREREG.md: primary cell lambda=0, the 67-run grid table, filled Phase 0 outcomes, and a dated entry for the human's option-A decision with its known limitation. DEVIATIONS.md: D1 updated (primary lambda 0), new D10 (grid cuts) and D11 (primary cell). README table updated.

Diff: `.changelog/entries/0023/diff.patch`

---

## [0024] Record the pre-registration commit hash
*2026-10-20 13:13:17*

- **File:** `PREREG.md` (modified, +1/-1)
  - **Lines:** 6 modified (were 6 before)

**Summary:** Record the pre-registration commit hash

**What the change does:** PREREG.md now names commit 6ebba48 (2026-10-20) as the pre-registration commit: it fixes N_star=32, phase1_steps=6000 and the primary cell before any Phase 1 run. The LRs are added later by the pre-registered E1.0a rule.

Diff: `.changelog/entries/0024/diff.patch`
