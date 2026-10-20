# Deviations from the reference code and the design doc

Design rule 1: every deviation from the reference implementations, or from ENGRAMDELTA_DESIGN.md, is listed here.

## Engram (`engramdelta/models/engram.py` vs DeepSeek `engram_demo_v1.py`)

| # | Deviation | Why | Checked by |
|---|-----------|-----|------------|
| E1 | One residual stream (hc_mult = 1): one key projection, one (key, query) RMSNorm pair, and a ShortConv with one norm | Our backbones have no mHC hyper-connections | `test_engram_parity.py` (reference run with hc_mult = 1) |
| E2 | n-gram hashing in torch on the model's device (int64 multiply / XOR / remainder), not numpy on the CPU | Speed, and every DDP rank stays on its GPU | `test_engram_parity.py::test_hash_indices_match` (bit-identical) |
| E3 | `orders` is an explicit list. The reference always uses 2..max_ngram_size. Order 1 is only used by arm A4 | A4 control (unigram side-pathway) | `test_causality.py` (orders (1,), (2,), (3,)) |
| E4 | ShortConv dilation is a parameter (default 3 = the reference's max_ngram_size for orders {2,3}). A4 keeps 3 | So A4 has the same conv as A2 | — |
| E5 | Tiny tables: ~2.4k rows per head (Phase 1), ~14.7k (Phase 2), instead of 646k | Design §5.3: tables about 1x the backbone | `test_params.py` |
| E6 | Tokenizer compression is the identity on MQAR. In Phase 2 it is a port of `CompressedTokenizer` for a `tokenizers.Tokenizer` (the reference takes a transformers AutoTokenizer) | Synthetic vocabulary has no surface variants | — |
| E7 | Engram modules keep PyTorch default init (tables N(0,1), as in the demo). Backbone Linear and embedding weights are N(0, 0.02) | The demo does not specify a model-level init | — |
| E8 | A6 = embedding + Engram + MLP, no mixer. Engram's short conv (receptive field 10 tokens) still mixes locally | As designed. Recall needs reach ≥ 2N tokens | `test_mqar_leak.py`, E0.5 |

## Gated DeltaNet (`engramdelta/models/gdn_block.py` vs FLA `fla/layers/gated_deltanet.py`)

| # | Deviation | Why | Checked by |
|---|-----------|-----|------------|
| G1 | The layer (projections, short convs, decay/beta gates, gated RMSNorm output) is rewritten in plain PyTorch, following FLA's source | It must run without Triton (Mac / fallback) and expose the recurrent state | `test_kernels.py` |
| G2 | Pure-PyTorch chunk kernel = FLA `naive_chunk_gated_delta_rule` with the per-chunk triangular inverse computed by `torch.linalg.solve_triangular` instead of a Python row loop | The row loop keeps one autograd copy per row, which is too slow and too much memory for training | `test_kernels.py` (vs FLA naive recurrent and naive chunk, outputs and grads) |
| G3 | FLA's Triton kernel is called with pre-normalized q/k (eps 1e-6, like FLA's l2norm), post-sigmoid beta and log-space g. The newer fused-gate options are not used | Same API across FLA versions | `bench_kernels.py`, `test_kernels.py` (CUDA) |
| G4 | head_dim 32 with expand_v = 1 (Phase 1) and 64 (Phase 2). FLA defaults to expand_v = 2 and asks for key_dim = 0.75 d_model | Design §7.1 asks for d_k = d_v (a 32 x 32 state) | — |
| G5 | State metrics use the step-by-step recurrence on the trained projections, not the training kernel | FLA's layer does not expose per-position states | `test_kernels.py::test_layer_capture_path_equals_chunk_path` |

## Design-doc changes (plan §1, agreed defaults)

| # | Change | Reason |
|---|--------|--------|
| D1 | MQAR loss = mean CE(answers) + lambda * mean CE(next filler token). Planned: primary lambda = 1, check lambda = 0. **Changed 2026-10-20 (option A): primary lambda = 0 (answer-only, the standard MQAR loss), check lambda = 1 (E1.0c)** | Planned lambda = 1 so that filler must be processed. The E0.7 pilot showed A1 cannot learn at all with lambda = 1 under any clutter level tried, but learns at F = 512 with lambda = 0 (PREREG.md). The two terms are averaged separately |
| D2 | Added arm LA (vanilla causal linear attention, elu+1 feature map, normalized) for test 7 | The design's arm table has no linear-attention arm |
| D3 | Phase 1 model d_model 128, 2 blocks, 4 heads x 32 (the low end of the design's 128-256) | One-week Kaggle budget (plan §6.5) |
| D4 | MQAR vocabulary 2048 keys / 2048 values / 512 filler (chance 1/2048) | Smaller LM head, still far above what a table could memorize |
| D5 | Engram placement fixed at block 1 (Phase 1) and blocks {1, 3} (Phase 2). The placement pilot was cut | Budget. Follows the reference `layer_ids[0] = 1` and the paper's "layer 2 is best" |
| D6 | Phase 1: 3 seeds (the design's minimum, 5 preferred). Cut cells listed in PREREG.md | One-week budget |
| D7 | Phase 2: one seed per arm, with T1 and T2 included (H3 testable at one seed) | One-week budget. Any write-up must say "single seed" |
| D8 | Parameter matching: arms without Engram get their MLP width set so that non-embedding params match A2 within ±2%. Engram tables count as embedding params | Design §5.3 |
| D9 | Logits are computed only at loss positions during training | Speed. The loss is identical |
| D10 | Phase 1 grid (option A): N = N\* only (no 2N\*), placement a only (no interleaved clutter b), no P sweep, no T1 | One-week budget at 6,000 steps per run. A1 fails at 2N\* = 64 even without clutter; T1 fails at N = 32 without clutter (E0.4). Placement b (design §7.2) is untested this week |
| D11 | Primary clutter cell F = 512 with lambda = 0, chosen after the E0.7 pilot | See D1 and PREREG.md |
