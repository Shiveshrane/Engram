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
