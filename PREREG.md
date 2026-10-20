# Pre-registration

Fill in the **Phase 0 outcomes** section and `configs/prereg.yaml` after NB02, **commit both before any Phase 1
run**, and write the commit hash below. Everything else on this page is fixed now, before any results exist.

- Pre-registration commit: `__________` (date: ______)

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
