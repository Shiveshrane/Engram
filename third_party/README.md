# Vendored reference code (unchanged)

These files are copied byte-for-byte from upstream and are used **only by tests** (as references to check
our ports against). Do not edit them. If upstream changes, re-download and re-run the parity tests.

| File | Upstream | Upstream commit at download (2026-10-19) | License |
|------|----------|------------------------------------------|---------|
| `engram_demo_v1.py` | https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py | `fb7f84a21f91223715394a33a1dc24bbfb7f788e` | Apache-2.0 (`LICENSE-ENGRAM`) |
| `fla_naive.py` | https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py | `9e97506c338769e8031966fb22fe31780addd8fa` | MIT (`LICENSE-FLA`) |

Used by:
- `tests/test_engram_parity.py` imports `engram_demo_v1.py`. Its module-level `engram_cfg` / `backbone_config`
  are replaced with small configs, and `transformers.AutoTokenizer` is replaced with a stub tokenizer,
  so that nothing is downloaded.
- `tests/test_kernels.py` uses `naive_recurrent_gated_delta_rule` / `naive_chunk_gated_delta_rule` as the
  ground truth for `engramdelta/models/gdn_block.py`.
