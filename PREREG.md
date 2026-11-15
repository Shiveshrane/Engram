# Pre-registration

Fill in the **Phase 0 outcomes** section and `configs/prereg.yaml` after NB02, **commit both before any Phase 1
run**, and write the commit hash below. Everything else on this page is fixed now, before any results exist.

- Pre-registration commit: `6ebba48` (date: 2026-10-20). LRs from NB10 (E1.0a) are added by rule in a later commit.

## Decision rules (fixed)

- **Significance.** |mean_A − mean_B| > 2 × pooled seed std, with n ≥ 3 seeds per arm. Pooled std =
  sqrt(((n_A−1)s_A² + (n_B−1)s_B²) / (n_A+n_B−2)). Metric: query recall on the cell's fixed 2,000-sequence eval set.
- **Primary cell:** N = N\* = 32, F = 512, P = 64, placement a, **lambda = 0** (answer-only loss; option A, see the
  2026-10-20 entries below). Every other cell is secondary / descriptive.
- **H1:** A1(F=0) > A1(F=512) significantly (N = N\*, P = 64, a, lambda = 0).
- **H2:** A2 > A1 significantly at the primary cell, **and** A2 > A3 and A2 > A4 significantly. If A2 > A1 but A3 or A4 is
  not significantly below A2, the claim is reframed as "gated side-pathway", not "n-gram offload" (design §7.4).
- **H4:** report the 2×2 table (Engram conv × GDN short conv) on A2 at the primary cell. "Changes little" = within 2σ of
  full A2.
- **H3 (Phase 2):** relative improvement of the in-context-copy loss and the [256, 1024) bucket loss, A2 vs A1 compared with
  T2 vs T1. One seed: descriptive only, stated as such.
- **Gate 1:** continue to Phase 2 only if H1 holds AND A2 > A1 at the primary cell AND the A3/A4 controls do not explain
  the gain. The human decides, using the NB14 table.
- **Test 7 (pipeline check):** A1 > LA + 2σ at ≥ 2 values of N ∈ {16, 32, 64, 128, 256}, and A1 mean recall is
  non-increasing in N beyond the knee (increases smaller than 2 × the A1 seed std count as noise). If this fails: stop.
- **Leak check:** A6 recall < 1% (chance 1/2048) at every N (E0.5) and at the primary cell (E1.2l).
- **Filler-loss check (E1.0c, descriptive):** A1 and A2 at the primary cell with lambda = 1, 3 seeds, reported next to
  the lambda = 0 results.

## Rules that resolve values from Phase 0 / pilots (fixed)

- **N\*** = the largest N in E0.4 with A1 mean recall ≥ 0.9. Phase 1 uses N ∈ {N\*, 2N\*}.
- **Phase 0 (E0.4 / E0.5) step count** = 4,000 steps (fixed).
- **E0.6** runs at the Phase 1 primary cell: N = N\* (from E0.4), F = 512, P = 64, placement a, 6,000 steps, A1 and A2.
- **Primary clutter level F (E0.7 clutter pilot, NB04)**: A1 at N = N\*, P = 64, placement a, 6,000 steps, at
  F ∈ {64, 128, 256} with λ = 1 and at F = 512 with λ = 0. Primary F = the largest of {256, 128} where A1's max val
  recall is ≥ 0.5. If neither, stop: the human decides (the F = 64 and λ = 0 runs inform that decision).
- **Phase 1 step budget** = the first eval step at which A1's val recall reaches ≥ 98% of its maximum, rounded up to a
  multiple of 500, on the E0.7 run at the primary F (E0.6 if its A1 run learned). The rule applies only if A1's maximum
  val recall is ≥ 0.5; otherwise there is no step budget and the human decides what to change before Phase 1.
- **Phase 1 cells use the primary F** wherever arms are compared under clutter: E1.0a, E1.0c, the E1.2 / E1.2r / E1.2l
  cells, the E1.1 P sweep and E1.3. E1.1 keeps the design's levels F ∈ {0, 128, 256, 512}; the A2 dose-response
  (E1.2) runs at the design levels other than the primary F.
- **Learning rate** per arm = the candidate in {3e-4, 1e-3, 3e-3} with the highest **val** recall in E1.0a (a separate
  val set, not the eval set). Ties go to the smaller LR. A3 and A4 use A2's LR; A5, A6 and A1_noGconv use A1's;
  A2_* use A2's.
- **Phase 2 tokens** = min(150M, 0.9 × min(A1, A2 throughput in E2.0) × 1.7 h), rounded down to whole 65,536-token steps.
  The same token count is used for every arm.

## Changes to these rules before the pre-registration commit

- 2026-10-20, after the first NB02 run (code commit 5ceb0ab): E0.6 had been fixed at N = 64 and run *before* E0.4, and its
  step count also set the E0.4 / E0.5 steps. E0.4 then gave N\* = 32, so N = 64 is 2N\*, where A1 stays near 1/N even
  without clutter (2.0% recall). E0.6 at N = 64 never left chance (max val recall 0.001), and the step rule produced
  noise (5,500), which E0.4 / E0.5 then used, clamped to 4,000. Changes: E0.4 / E0.5 are fixed at the 4,000 steps they
  already used (no E0.4 / E0.5 result changes); E0.6 moves after E0.4 and runs at N = N\*, the cell the step budget is
  for; the step rule now requires A1 to reach 0.5 val recall. The N = 64 E0.6 runs are kept as results and reported, but
  not used for the step budget. The Phase 1 step budget now also uses A1's proposal only, as written above (NB02 had
  used the maximum over A1 and A2).

- 2026-10-20, after the NB02 rerun (code commit 43cd9f7): E0.6 at N = N\* = 32, F = 512 also never left chance (A1 and
  A2 max val recall 0.001 in 6,000 steps), while A1 solves N = 32 without clutter (99%, plateau broken at ~1,900
  steps). With every arm at zero at F = 512 the arm comparisons (H2, H4) would be untestable. Change: the primary
  clutter level is no longer fixed at 512 but chosen by the E0.7 rule above, before any Phase 1 run; the design's
  clutter levels {0, 128, 256, 512} are unchanged and F = 512 stays in E1.1.

- 2026-10-20, after the E0.7 clutter pilot (NB04, code commit d7e02ff), **human decision (option A)**: with lambda = 1,
  A1 does not learn within 6,000 steps at any clutter level tried (F = 64: 1.7%, 128: 0.4%, 256: 0.0% val recall), so
  the primary-F rule above found no level. With lambda = 0 (answer-only loss, the standard MQAR loss), A1 learns at
  F = 512 (66% val recall, breaking the plateau at ~3,000 steps vs ~1,900 without clutter), so clutter hurts without
  removing all signal. Changes: primary cell = (N\*, F = 512, P = 64, a, lambda = 0); phase1_steps = 6,000 from that
  run; E1.0c now checks lambda = 1; the grid is cut to 67 runs (table below) to fit the budget at 6,000 steps. Known
  limitation, to be stated in any write-up: with lambda = 0 the model is never asked to predict the filler, so it may
  learn to ignore it rather than offload it to Engram; E1.0c and the E1.4 beta / alpha statistics on filler tokens
  are the evidence on this.

## Grid (one-week version, option A: 67 Phase 1 runs, 6,000 steps each; see `engramdelta/scripts/make_manifest.py`)

| Group | Runs | Cells |
|---|---|---|
| E1.0a LR pilot | 6 | A1, A2 × LR {3e-4, 1e-3, 3e-3} at the primary cell, 1 seed |
| E1.0c filler-loss check | 6 | A1, A2 at the primary cell with lambda = 1, 3 seeds |
| E1.1 H1 | 12 | A1 × F ∈ {0, 128, 256, 512}, 3 seeds |
| E1.2 H2 + controls | 30 | A2, A3, A4, A5 × F ∈ {0, 512}, 3 seeds; A2 × F ∈ {128, 256}, 3 seeds |
| E1.2l leak under clutter | 1 | A6 at the primary cell |
| E1.3 H4 | 12 | A2_noEconv, A2_noGconv, A2_noboth, A1_noGconv at the primary cell, 3 seeds |

All at N = N\* = 32, P = 64, placement a, lambda = 0 unless stated. Estimated 10.5 GPU-h from measured throughput (NB02,
NB04). Cut from the full design for budget: 5 → 3 seeds; the N\*/2 and 2N\* cells; placement b; the P sweep; the
Engram-placement pilot; the A3 dose-response; T1 in Phase 1; Phase 2 at 1 seed. If time is left over, the first thing
added back is 2 extra seeds of A1–A4 at the primary cell.

## Phase 0 outcomes

| Item | Value |
|------|-------|
| GDN backend (NB00) | torch (FLA kernels not used) |
| Test 7 pass? (criterion 1 / 2) | PASS / PASS. A1 99.8%, 98.9%, 2.0% at N = 16, 32, 64; LA 5.9%, 2.7%, 1.2% |
| Leak check (E0.5) | PASS at N = 16 … 256 (A6 0.05%, chance 0.049%) |
| N\* | 32 |
| E0.6 at N\*, F = 512, lambda = 1 | A1 and A2 at chance in 6,000 steps |
| E0.7 clutter pilot (A1, N = 32) | lambda = 1: F = 64 → 1.7%, 128 → 0.4%, 256 → 0.0%; lambda = 0, F = 512 → 66% |
| Primary cell | N = 32, F = 512, P = 64, a, lambda = 0 (option A, human decision) |
| Phase 1 step budget | 6,000 (E0.7, A1 at F = 512, lambda = 0: 98% of max at 5,750) |
| Measured Phase 1 run time (min) | 12.1 train min for A1 at F = 512 (T = 640), 6,000 steps |
| Phase 1 budget estimate (GPU-h) | 10.5 (67 runs) |
| LRs (E1.0a, NB10, code 1997b4d) | A1 1e-3 (val recall 0.659), A2 3e-3 (0.068; best of the three is the largest candidate). A3/A4/A2_* use A2's, A5/A6/A1_noGconv use A1's |

## Phase 1 outcomes and Gate 1 (all 67 runs; NB10–NB13, code 6748062; analysis run locally = NB14)

| Item | Value |
|------|-------|
| H1: A1 F = 0 vs F = 512 | 0.992 vs 0.754, diff 0.238 > 2σ 0.186: **holds** |
| H2: A2 vs A1 / A3 / A4 (primary cell) | A2 0.618 vs 0.754 / 0.900 / 0.693; no comparison exceeds 2σ: **not supported** |
| H4 (2×2 conv ablation) | all within 2σ of full A2 (A2 std 0.47); descriptively GDN short conv off → A1 0.03, A2 0.16; Engram conv off → 0.73 |
| Leak (E1.2l) | A6 0.04% (< 1%): pass |
| E1.0c (λ = 1, descriptive) | A1 3/3 seeds at chance; A2 1/3 seeds learned (0.91) |
| fp16 health | 0 non-finite losses; max 6 GradScaler-skipped steps per run |
| **Gate 1** | **fails** under the rule (H2 not supported) |

## After Gate 1: exploratory studies (human decision 2026-11-14, rules fixed before any of these runs)

Gate 1 failed, so nothing below is confirmatory: it is reported as exploratory, follow-up work, with its own rules
fixed here before any run.

### Phase 2 (LM), exploratory
- Runs on **Modal, 1× A100 40 GB per run, bf16**, every run in its own container at the same time (`modal_app.py`).
  Arms A1, A2, A3 × seeds {0, 1, 2} (`phase2_seeds`). T1 / T2 (H3) only if budget is left.
- Token budget: the existing rule (E2.0 throughput on the same hardware, cap 150M tokens). The GDN kernel
  (FLA or torch) is chosen by the kernel test on the same GPU before E2.0, and is the same for every arm.
- Reported: `phase2_table`, `phase2_relative` and `phase2_tests` (A2 vs A1 and A3 vs A1 per loss with the 2σ rule,
  n = 3), plus state occupancy. With n = 3 these are still descriptive.
- Not comparable bit-for-bit with Phase 0/1 (different GPU, single process, bf16); Phase 2 comparisons are only
  made within Phase 2.

### E3: facts + in-context recall (Kaggle 2×T4, NB30 + NB31)
Question: when the model must also *memorize* fixed facts, does in-context recall suffer (as gated recurrent models
learn the memorization solution first, Tong et al. 2026, arXiv 2609.16540), and does an n-gram memory (Engram)
protect in-context recall by taking over the memorization?
- Task (`data/mqar.py`, facts variant): N = N\* = 32 in-context pairs, F = 0, plus M = 32 fact queries per sequence
  from a fixed table of K = 2,048 two-token facts (a, b) → value (fact tokens: 64, own vocabulary range). Facts never
  appear in the context. Facts **off** = plain MQAR at the same N.
- Arms A1, A2 (E3.1), A3, A4 (E3.2), each with facts off and on, seeds {0..4} (`phase3_seeds`), 6,000 steps,
  Phase 1 LRs, λ = 0, val eval every 250 steps. A6 (leak probe), facts on, 1 seed. 41 runs.
- Metric: **in-context recall** (`recall_ctx`, eval set) at facts on. Rule: 2σ pooled, n = 5.
- **H5a (precondition):** A1 facts off > A1 facts on.
- **H5 (main):** A2 > A1 at facts on, **and** A2 > A3 and A2 > A4 at facts on (A4 = unigram table cannot store
  two-token facts; A3 = frozen random table). If A2 > A1 but not > A3 / A4: "extra lookup capacity", not
  "n-gram offload".
- Context check: A2 vs A1 at facts off. A gain there too means the effect is not about offloading memorization.
- Secondary, descriptive: fact recall; escape step (first eval step with in-context val recall ≥ 0.5) and the number
  of seeds that escape within 6,000 steps.
- Leak: A6 in-context recall < 1% at facts on (its fact recall is expected to be high, and is reported).

### Phase 2 outcome (exploratory; Modal A100, code 7c5a332, 150M tokens per run, full val set, 3 seeds)

| Arm | val loss | ppl | [0, 64) | [64, 256) | [256, 1024) | in-context copy |
|---|---|---|---|---|---|---|
| A1 | 3.812 | 45.2 | 4.156 | 3.840 | 3.776 | 3.635 |
| A2 | 3.847 | 46.8 | 4.185 | 3.875 | 3.811 | 3.674 |
| A3 | 3.845 | 46.8 | 4.184 | 3.874 | 3.810 | 3.674 |

A2 − A1 exceeds 2σ on every loss, in the direction of A2 being **worse** (+0.035 nats overall, +0.039 on the
in-context-copy loss). A3 ≈ A2. Descriptively, the GDN states of A2 / A3 have higher effective rank and Frobenius
norm than A1's at positions 256 / 512 / 1023 (e.g. eff. rank 18.1 vs 15.7 at 1023). Cost: ~$12.5 of Modal credit.

### E3 outcome (exploratory; Modal 1× L4 per run, code d3af24f, 41 runs, DEVIATIONS.md D14)

In-context recall (eval set, mean ± std over 5 seeds); fact recall; median escape step (first eval step with
in-context val recall ≥ 0.5) and the number of seeds that escape within 6,000 steps.

| Arm | ctx, facts off | ctx, facts on | fact recall | escape step off / on | escaped off / on |
|---|---|---|---|---|---|
| A1 | 0.983 ± 0.007 | 0.984 ± 0.012 | 1.000 | 1000 / 1250 | 5 / 5 |
| A2 | 0.955 ± 0.028 | 0.919 ± 0.114 | 1.000 | 2000 / 2500 | 5 / 5 |
| A3 | 0.980 ± 0.021 | 0.947 ± 0.032 | 1.000 | 1250 / 1750 | 5 / 5 |
| A4 | 0.980 ± 0.008 | 0.713 ± 0.433 | 1.000 | 1500 / 2750 | 5 / 4 |
| A6 | — | 0.0006 | 1.000 | — | 0 |

- **H5a not supported:** A1 facts off − on = −0.0002 (2σ threshold 0.020). The memorization load does not reduce
  A1's in-context recall, so the precondition of H5 fails.
- **H5 not supported:** A2 − A1 = −0.065 at facts on (threshold 0.162); A2 − A3 = −0.029 (0.167); A2 − A4 = +0.205
  (0.633). Context check, facts off: A2 − A1 = −0.029 (threshold 0.040).
- Leak: A6 in-context recall 0.06% < 1% (pass); A6 fact recall 100%, as expected.
- Descriptive: every arm stores all 2,048 facts. Every table arm (A2, A3, A4) escapes the in-context recall plateau
  later than A1, more so with facts on. Cost ~$3.75.

## Follow-up studies after E3 (human decision 2026-11-15, rules fixed before any of these runs)

Exploratory and descriptive. They answer two questions about the Phase 2 result (A2 worse than A1 by > 2σ).

### E2.ngram: does Engram help where its lookup is well trained?
- For each val input position p ≥ 2 of a window, the key is the training-set count of the trigram (and, secondary,
  the bigram) of canonical ids ending at p: the n-grams Engram's order-3 / order-2 heads hash
  (`scripts/ngram_counts.py`, counted over all 150M training tokens).
- Buckets of that count: 0, 1–9, 10–99, 100–999, ≥ 1,000 (val token shares for trigrams: 26 / 25 / 26 / 17 / 7%).
- Metric: mean cross-entropy per bucket (`eval/lm_eval.py` `evaluate_lm_ngram`), for the 9 E2.1 runs (and the
  E2.long runs). Comparisons: A2 − A1 and A3 − A1 per bucket, 2σ pooled over 3 seeds.
- Reading fixed in advance: if Engram's table helps with frequent n-grams, A2 − A1 decreases as the count grows
  and A3 − A1 does not (A3's table is frozen and random). A2 < A1 by > 2σ in the ≥ 1,000 bucket would be a positive
  result for Engram on frequent n-grams. A2 − A1 flat or growing with the count means no such effect.

### E2.long: does the A2 − A1 gap close with more training?
- A1 and A2, seed 0, 4,576 steps = 2× E2.1 (300M tokens seen = 2 epochs of the 150M training tokens; no more
  unique data was prepared). All other settings as E2.1 (Modal A100, bf16 + TF32). Cost cap $7.
- Metrics: the E2.eval losses (full val set) and E2.ngram buckets.
- Reading fixed in advance (one seed, descriptive): the gap "closes" if A2 − A1 at 2× steps is less than half of
  E2.1's 3-seed mean gap (+0.035 nats overall); it "persists" if it is at least E2.1's seed-0 gap minus 2× E2.1's
  pooled seed std; otherwise "shrinks".

### E2.ngram outcome (Modal L4, code 0d9b8b4, 9 E2.1 runs, full val set, positions p ≥ 2)

Mean loss by the training count of the trigram ending at p (3 seeds); A2 − A1 and A3 − A1 exceed 2σ in every bucket.

| Trigram count | Val share | A1 | A2 | A3 | A2 − A1 | A3 − A1 |
|---|---|---|---|---|---|---|
| 0 | 26% | 3.675 | 3.719 | 3.717 | +0.043 | +0.041 |
| 1–9 | 25% | 3.756 | 3.795 | 3.793 | +0.038 | +0.037 |
| 10–99 | 26% | 3.829 | 3.860 | 3.859 | +0.031 | +0.030 |
| 100–999 | 17% | 3.858 | 3.885 | 3.884 | +0.027 | +0.026 |
| ≥ 1,000 | 7% | 4.316 | 4.344 | 4.340 | +0.029 | +0.025 |

Reading (fixed in advance): A2 − A1 decreases with the count, **but so does A3 − A1** (frozen random table), so there is
no effect specific to a learned n-gram table. A2 is worse than A1 in the ≥ 1,000 bucket (no positive result).
Bigrams give the same picture (A2 − A1 +0.045 → +0.032).

### E2.long outcome (Modal A100, code 0d9b8b4, seed 0, 4,576 steps = 300M tokens seen, 2 epochs)

| Runs | A1 loss | A2 − A1 all | [0, 64) | [64, 256) | [256, 1024) | in-context copy |
|---|---|---|---|---|---|---|
| E2.1 seed 0 (1×) | 3.818 | +0.039 | +0.021 | +0.037 | +0.041 | +0.053 |
| E2.1 mean of 3 seeds (1×) | 3.812 | +0.035 | +0.029 | +0.034 | +0.036 | +0.039 |
| E2.long seed 0 (2×) | 3.600 | +0.013 | +0.008 | +0.012 | +0.013 | +0.011 |

- By the pre-registered reading the gap **closes**: +0.0128 < half of E2.1's mean gap (0.0175). One seed;
  A2 is still worse than A1.
- E2.ngram at 2× (descriptive): A2 − A1 by trigram count 0 / 1–9 / 10–99 / 100–999 / ≥ 1,000 = +0.019 / +0.016 /
  +0.010 / +0.006 / +0.007, a steeper decrease than at 1×. A3 was not run at 2×, so this cannot be attributed
  to the learned table.
- Cost: ~$5.7 (incl. ~$0.6 lost when the local Modal client disconnected; the runs resumed from step 200).
  Modal total ~$22 of $30. No further runs (human decision 2026-11-15).
