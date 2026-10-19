# EngramDelta: Design Doc (v0.1)

Hand this file to Claude Code as the project spec. Start with Phase 0 only. Do not start Phase 1 or 2 until every Phase 0 test passes and the human has reviewed the core files (see "Rules for the coding agent").

---

## 1. One-paragraph summary

Delta-rule linear-attention models (Gated DeltaNet, Kimi Delta Attention) keep a fixed-size recurrent state matrix per head (d_k x d_v). Engram (DeepSeek, 2026) is a small add-on module that looks up learned vectors from hashed tables keyed by the last 2-3 tokens, gates them with the hidden state, and adds them to the residual stream. Engram has only been published on attention/MoE backbones. **Hypothesis:** in a fixed-size recurrent state, local n-gram patterns compete with long-range associations for capacity, so offloading local patterns to a hashed lookup should protect long-range recall, and the effect should be larger for delta-rule backbones than for Transformers. We test this with cheap synthetic experiments first, then one small language-model check.

This is a mechanism study, not a scaling claim. Compute is one person on free-tier GPUs.

## 2. Hypotheses and falsification criteria

| ID | Claim | Counts as supported | Counts as refuted |
|----|-------|---------------------|-------------------|
| H1 | Local "clutter" degrades long-range recall in a GDN state | Recall of the GDN baseline drops significantly as clutter increases (beyond seed noise) | Recall is flat across clutter levels |
| H2 | An Engram lookup protects recall under clutter in GDN | GDN+Engram recall under clutter is significantly higher than GDN, AND the frozen-random control does not explain the gain | The gain disappears under the controls, or there is no gain |
| H3 | Engram helps GDN more than it helps a Transformer in a real LM | Larger relative improvement on long-range / in-context-copy loss for GDN than for the Transformer (Phase 2 only) | Equal or smaller improvement |
| H4 | Engram's own short conv is partly redundant with GDN's built-in short conv | Removing one of the two changes little | Removing either hurts clearly |

Notes:
- "Significantly" means the mean difference exceeds 2x the seed standard deviation, with at least 3 seeds (5 preferred). Decide thresholds before running, not after.
- H3 cannot be tested on MQAR: a full-attention Transformer is near ceiling there, so any gap would be trivial. H3 belongs to Phase 2.
- A clean negative result is an acceptable outcome. Report it honestly.

## 3. Constraints

- Hardware: free-tier Kaggle T4 (16 GB, session-limited), Modal T4 credits, and a MacBook with an Apple M5 Pro (24 GB unified memory, MPS) for development and tiny runs.
- T4 (sm_75) has no native bf16: train in fp16 with a GradScaler (or fp32 for the tiniest models). FlashAttention-2 is not supported on sm_75: use PyTorch `scaled_dot_product_attention`.
- UNVERIFIED: whether the flash-linear-attention (FLA) Triton kernels run on T4. Phase 0 must check this. Fallback: a chunked or sequential PyTorch reference implementation of the delta rule, fine at <=2K context and <=20M parameters.
- Budget per run: synthetic runs should take minutes (<30 min). Phase 2 runs should take roughly 1-3 hours each. If a run needs much more, shrink the model or the token budget instead of extending it.
- Every run must be resumable from checkpoints (free-tier sessions get killed).

## 3.1 Non-goals

- No billion-token pretraining. No claims about scale.
- No KV-cache compression or quantization work.
- No new framework or library. The deliverable is a small, readable experiment repo.

## 4. Rules for the coding agent (Claude Code)

1. **Follow the reference code, do not reinvent.** Engram: port DeepSeek's standalone reference implementation. GDN: use FLA's implementation, or a PyTorch reference such as the Raschka from-scratch version for the fallback. Flag every deviation in a `DEVIATIONS.md` file.
2. **Tests before models.** Write the Phase 0 tests first. A model file is not "done" until its tests pass.
3. **Never silently change evaluation code or metrics.** Any change to files in `eval/` or `data/` requires a note in `CHANGELOG.md` and human review.
4. **Stop and ask** if a test fails in a way you cannot explain, instead of patching the test to pass.
5. **Log everything:** each run writes an immutable JSON file with config, git commit hash, seed, parameter counts, and all metrics. Never overwrite results.
6. **Do not interpret results** or write claims. Print tables and plots. The human decides what the results mean.
7. **Files the human must read and understand line by line** (the human will have to defend them in interviews): `models/engram.py`, `data/mqar.py`, `eval/state_metrics.py`, and all of `tests/`.
8. Keep code small and boring. Prefer plain PyTorch, explicit shapes in comments, and no unnecessary abstraction.

## 5. Architecture

### 5.1 Backbone
- Small pre-norm residual blocks: mixer (GDN, or attention for the Transformer arm) followed by an MLP.
- GDN mixer: FLA `GatedDeltaNet` if the kernels work on T4, else the PyTorch reference. Keep GDN's built-in short convolution (check the flag, likely `use_short_conv`) ON by default; it is ablated in H4.
- Transformer mixer: causal multi-head attention via PyTorch SDPA, with RoPE.

### 5.2 Engram module (orientation only; the reference code is authoritative)
Per the Engram paper (arXiv 2601.07372) and DeepSeek's reference repo:
1. Compress token ids to canonical ids (tokenizer-level normalization of equivalent tokens). For synthetic tasks, use the identity map.
2. Form suffix n-grams ending at position t for n in {2, 3}. Left-pad so positions with fewer than n tokens still work. **Only tokens at positions <= t may be used.**
3. For each n, use K hash heads (multiplicative-XOR hash) into embedding tables of prime size. Concatenate retrieved rows into a vector e_t.
4. Project e_t to a key and a value; gate the value using the current hidden state as the query.
5. Apply a short depthwise causal convolution, then add the result to the residual stream.
6. Placement: at one or two early layers (configurable). Use identical placement across all arms.

### 5.3 Arms
Synthetic phase (Phase 1), all small GDN models unless noted:

| Arm | Description |
|-----|-------------|
| A1 | GDN baseline |
| A2 | GDN + Engram |
| A3 | GDN + Engram with a frozen random table (table values fixed at init, everything else trainable). Control: does the learned content matter, or only the gated side-pathway? |
| A4 | GDN + Engram with n=1 only (unigram). Control: gated side-pathway with no n-gram information |
| A5 | GDN + sliding-window attention (small window, e.g. 16) as the local helper. Alternative way to offload local work |
| A6 | Engram-only recall probe (embedding + Engram + MLP, no sequence mixer). Leak check: must score near chance on MQAR |
| T1 | Transformer baseline (reference point only; expected near ceiling) |

Language-model phase (Phase 2): A1, A2, A3, plus T1 and T2 (Transformer + Engram) if budget allows. A5 is optional.

H4 ablation: Engram conv ON/OFF crossed with GDN short conv ON/OFF (2x2) on the A2 setup.

Parameter matching: report total and non-embedding parameters for every arm in one table. Primary comparison: match non-embedding parameters as closely as possible, and keep Engram tables within about 1x the backbone's parameters. Record exact numbers.

## 6. Phase 0: setup and sanity tests (must all pass)

Write these in `tests/`. Run them on every model/arm.

1. **Causality test.** For random input x and a random position j: perturb tokens at positions > j, and assert logits at positions <= j are unchanged. Also perturb x_j and assert logits at j DO change (so the test is not vacuous). Run on the Engram module alone and on every full model. This is the most likely place for a silent bug (a leaking n-gram hash would make Engram look amazing for the wrong reason).
2. **Overfit-one-batch.** Each model must reach near-zero loss on a single tiny batch within a few hundred steps.
3. **Parameter table.** Print total / non-embedding / Engram-table parameter counts for all arms side by side.
4. **Kernel correctness.** If FLA kernels are used: outputs and gradients must match the naive PyTorch recurrent reference within tolerance on random inputs (fp32, small shapes). Also confirm they run on a T4 and record speed versus the fallback.
5. **Engram parity.** With identical weights and inputs, `models/engram.py` must reproduce the output of DeepSeek's standalone reference demo.
6. **MQAR leak test.** Arm A6 (no sequence mixer) must score near chance. Key-value assignments must be resampled per sequence from a large vocabulary so that a lookup table cannot memorize answers across sequences.
7. **Known-result reproduction.** On plain MQAR (no clutter): GDN should clearly outperform vanilla linear attention, consistent with the literature, and recall should degrade as the number of pairs approaches state capacity. If this fails, the pipeline is wrong; do not proceed.
8. **fp16 stability.** 500 training steps with no NaNs; compare the loss curve against fp32 for the first 200 steps on a tiny model.
9. **Determinism.** Same seed gives the same result on the same hardware (within numerical tolerance).

## 7. Phase 1: synthetic experiments (cheap, minutes per run)

### 7.1 Task: MQAR (multi-query associative recall)
Follows the task from the Zoology paper (arXiv 2312.04927). A sequence presents N key-value pairs; later, keys are queried and the model must output the matching value.

Model size: 2-4 layers, d_model 128-256, a few million non-embedding parameters. Choose head dimensions so per-head state capacity is a known quantity (e.g., d_k = d_v = 32 gives a 32x32 state per head), then sweep N around that capacity.

### 7.2 The "local clutter" variant (our addition; validate it carefully)
Filler tokens are drawn from a small pool of P fixed multi-token phrases (e.g., 3 tokens each, from a filler vocabulary separate from key/value tokens), emitted at random. These are locally predictable patterns that a hashed n-gram lookup can learn cheaply.

Two placements:
- (a) **Clutter between the pairs and the queries** (tests whether local junk overwrites stored associations).
- (b) **Clutter interleaved between the pairs** (tests competition during writing).

Sweep:
- Number of pairs N: around and above state capacity.
- Filler length F: {0, 128, 256, 512}.
- Phrase pool size P: {8, 64, 512}.
- At least 3 seeds per cell (5 preferred).

Caveat to keep in mind: this synthetic clutter may not resemble real language statistics. Phase 2 exists for that reason. Do not over-claim from Phase 1 alone.

### 7.3 Metrics
- Query recall accuracy (primary).
- **State occupancy** (exploratory), measured on the recurrent state S_t at chosen points (end of the writing segment, end of the clutter segment): effective rank (exp of the entropy of the normalized singular values), the singular-value spectrum, and the Frobenius norm. Average over heads and report per layer.

### 7.4 Gate 1 (decide whether to continue)
Continue to Phase 2 only if H1 holds (clutter hurts GDN) AND A2 beats A1 under clutter AND the A3/A4 controls do not explain the gain. If H1 fails, the premise is wrong: stop and write up the negative result, or reframe. If the gain is explained by the controls, reframe the claim as "gated side-pathway" rather than "n-gram offload".

## 8. Phase 2: small language-model check

- Model: about 10-20M non-embedding parameters (e.g., d_model 384, 8 layers), context 1024, a small tokenizer (16k-32k vocabulary) to keep the embedding table small.
- Data: a FineWeb-Edu subset (Hugging Face), about 200-300M tokens. Hold out a fixed validation set that never overlaps training data.
- Training: AdamW, cosine schedule, fp16 with GradScaler, gradient clipping, checkpoint and resume support.
- Arms: A1, A2, A3 first. Add T1 and T2 only if the first three look interesting and budget allows.
- Seeds: 2 where possible. If only 1 seed fits, say so explicitly in any write-up.
- Metrics:
  - Validation perplexity overall.
  - Loss bucketed by token position ([0,64), [64,256), [256,1024)).
  - **In-context-copy loss:** loss on tokens whose preceding bigram appeared earlier in the window at distance greater than 64 (an induction-style proxy for long-range recall).
  - State occupancy (as in 7.3) on held-out text.
- H4: the 2x2 conv ablation on A2 at the same scale if budget allows; otherwise run it on the synthetic setup.

## 9. Proposed repo layout

```
engramdelta/
  models/      engram.py, gdn_block.py, transformer_block.py, swa.py, model.py
  data/        mqar.py, lm_data.py
  eval/        state_metrics.py, lm_eval.py
  tests/       test_causality.py, test_overfit.py, test_params.py,
               test_kernels.py, test_engram_parity.py, test_mqar_leak.py,
               test_repro_known.py, test_fp16.py
  configs/     one YAML per arm and phase
  scripts/     train.py, sweep_mqar.py, make_plots.py
  results/     immutable JSON per run
  DESIGN.md  DEVIATIONS.md  CHANGELOG.md
```

## 10. Known pitfalls

- **Causality leaks** in the n-gram hash or its left-padding (see test 1).
- **Answer leakage** in MQAR (fixed key-value pairs across sequences let a table memorize them).
- **Unequal parameter counts** across arms.
- **fp16 overflow** in one arm only.
- **Engram gains from the side-pathway, not the lookup content.** Evidence from an image-generation study suggests this can happen (arXiv 2605.13179). That is why controls A3 and A4 exist.
- **Large local helpers can make the recurrent layers lazy.** Evidence from sliding-window work suggests larger windows can hurt long-context behavior (arXiv 2509.24552, 2606.15378). Keep the A5 window small.
- **Seed noise.** Small models vary a lot. Do not draw conclusions from a single seed.

## 11. Prior-art status (as of Oct 2026, low-to-medium confidence)

No paper was found that combines Engram with a delta-rule or other recurrent backbone. Closest neighbours: the Gated DeltaNet-2 hybrid that pairs a recurrent mixer with sliding-window attention for local work (arXiv 2605.22791), and SWAX (arXiv 2509.24552). DeepSeek appears to ship Engram in a V4.1 checkpoint layout (vLLM docs) but on a non-delta-rule backbone.

TODO before writing anything up: read the full text of the Engram follow-ups (FactorEngram, Lngram, MoME and similar) for any non-attention-backbone ablation, and re-run an arXiv keyword search for "Engram" together with "linear attention", "Gated DeltaNet", "Mamba", "RWKV".

## 12. References

Read during the research pass (links as found):
- Engram, "Conditional Memory via Scalable Lookup": https://arxiv.org/abs/2601.07372 ; reference code: https://github.com/deepseek-ai/Engram
- Kimi Linear (Kimi Delta Attention): https://arxiv.org/abs/2510.26692
- Gated DeltaNet-2: https://arxiv.org/abs/2605.22791
- SWAX, "Short window attention enables long-term memorization": https://arxiv.org/abs/2509.24552
- Engram in autoregressive image generation: https://arxiv.org/abs/2605.13179
- Rethinking the role of efficient attention in hybrid architectures: https://arxiv.org/abs/2606.15378
- What attention recalls and recurrence controls: https://arxiv.org/abs/2609.04434
- Balancing memory pathways: https://arxiv.org/abs/2610.06750
- WriteSAE: https://arxiv.org/abs/2605.12770
- Raschka's from-scratch Gated DeltaNet notes: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md

Cited from memory, not opened during the research pass (verify before relying on them):
- Gated Delta Networks (Yang et al.): arXiv 2412.06464
- Zoology / MQAR (Arora et al.): arXiv 2312.04927
- flash-linear-attention library: https://github.com/fla-org/flash-linear-attention
