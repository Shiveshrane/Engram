# Pre-registration

Fill in the **Phase 0 outcomes** section and `configs/prereg.yaml` after NB02, **commit both before any Phase 1
run**, and write the commit hash below. Everything else on this page is fixed now, before any results exist.

- Pre-registration commit: `__________` (date: ______)

## Decision rules (fixed)

- **Significance.** |mean_A − mean_B| > 2 × pooled seed std, with n ≥ 3 seeds per arm. Pooled std =
  sqrt(((n_A−1)s_A² + (n_B−1)s_B²) / (n_A+n_B−2)). Metric: query recall on the cell's fixed 2,000-sequence eval set.
- **Primary cell:** N = N\*, F = 512, P = 64, placement a, lambda = 1. Every other cell is secondary / descriptive.
- **H1:** A1(F=0) > A1(F=512) significantly at the primary cell (same N, P).
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
- **Leak check:** A6 recall < 1% (chance 1/2048) at every N (E0.5) and under clutter (E1.2l).

## Rules that resolve values from Phase 0 / pilots (fixed)

- **N\*** = the largest N in E0.4 with A1 mean recall ≥ 0.9. Phase 1 uses N ∈ {N\*, 2N\*}.
- **Phase 0 (E0.4 / E0.5) step count** = A1's E0.6 proposal (same 98% rule as below), clamped to [2000, 4000].
- **Phase 1 step budget** = the first E0.6 eval step at which A1's val recall reaches ≥ 98% of its maximum, rounded up to a
  multiple of 500.
- **Learning rate** per arm = the candidate in {3e-4, 1e-3, 3e-3} with the highest **val** recall in E1.0a (a separate
  val set, not the eval set). Ties go to the smaller LR. A3 and A4 use A2's LR; A5, A6 and A1_noGconv use A1's;
  A2_* use A2's.
- **Phase 2 tokens** = min(150M, 0.9 × min(A1, A2 throughput in E2.0) × 1.7 h), rounded down to whole 65,536-token steps.
  The same token count is used for every arm.

## Grid (one-week version, 167 Phase 1 runs; see `engramdelta/scripts/make_manifest.py`)

Cut from the full design for budget: 5 → 3 seeds; the N\*/2 cells; the P sweep at 2N\*; the Engram-placement pilot; the A3
dose-response; T1 at 1 seed; Phase 2 at 1 seed. If time is left over, the first thing added back is 2 extra seeds of
A1–A4 at the primary cell.

## Phase 0 outcomes (fill after NB02)

| Item | Value |
|------|-------|
| GDN backend (NB00) | |
| Test 7 pass? (criterion 1 / 2) | |
| N\* | |
| Phase 1 step budget | |
| Measured Phase 1 run time (min) | |
| Phase 1 budget estimate (GPU-h) | |
