"""Test 5 (design §6): models/engram.py reproduces DeepSeek's standalone reference demo.

The vendored reference (third_party/engram_demo_v1.py, unchanged) reads module-level configs. We replace them
with small ones (hc_mult = 1 to match our single residual stream), and replace transformers.AutoTokenizer with
a stub whose tokens all normalize to distinct strings, so tokenizer compression is the identity map and nothing
is downloaded. Then we copy the reference's weights into our module and compare:
  table sizes (primes), hash multipliers, hash indices, and the full forward output.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pytest
import torch

from engramdelta.models.engram import Engram, layer_multipliers, table_sizes_for_layers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 50
LAYER_IDS = [1, 3]


class StubTokenizer:
    def __len__(self):
        return VOCAB

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"tok{i}" for i in ids)

    def convert_ids_to_tokens(self, tid):
        return f"tok{tid}"


class StubAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return StubTokenizer()


@pytest.fixture(scope="module")
def ref():
    pytest.importorskip("tokenizers")
    if importlib.util.find_spec("transformers") is None:
        sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=StubAutoTokenizer)
    spec = importlib.util.spec_from_file_location("engram_demo_v1", os.path.join(ROOT, "third_party", "engram_demo_v1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.AutoTokenizer = StubAutoTokenizer
    mod.engram_cfg = mod.EngramConfig(tokenizer_name_or_path="stub", engram_vocab_size=[97, 103], max_ngram_size=3,
                                      n_embed_per_ngram=32, n_head_per_ngram=4, layer_ids=LAYER_IDS, pad_id=2,
                                      seed=0, kernel_size=4)
    mod.backbone_config = mod.BackBoneConfig(hidden_size=16, hc_mult=1, vocab_size=VOCAB, num_layers=4)
    return mod


def build_pair(ref, layer_id):
    torch.manual_seed(0)
    r = ref.Engram(layer_id=layer_id)
    primes = table_sizes_for_layers(LAYER_IDS, 2, [97, 103], 4)
    ours = Engram(d_model=16, orders=[2, 3], primes=primes[layer_id],
                  multipliers=layer_multipliers(layer_id, 0, 3, VOCAB), embed_per_order=32, n_heads_per_order=4,
                  pad_id=2, kernel_size=4, conv_dilation=3)
    with torch.no_grad():
        ours.table.embedding.weight.copy_(r.multi_head_embedding.embedding.weight)
        ours.value_proj.load_state_dict(r.value_proj.state_dict())
        ours.key_proj.load_state_dict(r.key_projs[0].state_dict())
        ours.norm_key.load_state_dict(r.norm1[0].state_dict())
        ours.norm_query.load_state_dict(r.norm2[0].state_dict())
        ours.short_conv.conv.load_state_dict(r.short_conv.conv.state_dict())
        ours.short_conv.norm.load_state_dict(r.short_conv.norms[0].state_dict())
    return r, ours


@pytest.mark.parametrize("layer_id", LAYER_IDS)
def test_tables_and_multipliers_match(ref, layer_id):
    r, ours = build_pair(ref, layer_id)
    ref_primes = r.hash_mapping.vocab_size_across_layers[layer_id]
    assert table_sizes_for_layers(LAYER_IDS, 2, [97, 103], 4)[layer_id] == ref_primes
    assert np.array_equal(layer_multipliers(layer_id, 0, 3, VOCAB), r.hash_mapping.layer_multipliers[layer_id])
    assert ours.table.embedding.weight.shape == r.multi_head_embedding.embedding.weight.shape


@pytest.mark.parametrize("layer_id", LAYER_IDS)
def test_hash_indices_match(ref, layer_id):
    r, ours = build_pair(ref, layer_id)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, VOCAB, size=(3, 29), dtype=np.int64)
    ids[0, :5] = 2                      # pad id inside the sequence too
    ref_idx = r.hash_mapping.hash(ids)[layer_id]                    # [B, T, heads] numpy
    our_idx = ours.hasher(torch.from_numpy(ids)).numpy()
    assert np.array_equal(ref_idx, our_idx)


@pytest.mark.parametrize("layer_id", LAYER_IDS)
def test_forward_matches_reference(ref, layer_id):
    r, ours = build_pair(ref, layer_id)
    rng = np.random.default_rng(1)
    ids = rng.integers(0, VOCAB, size=(2, 23), dtype=np.int64)
    hidden = torch.randn(2, 23, 1, 16, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        ref_out = r(hidden_states=hidden, input_ids=ids)[:, :, 0, :]
        our_out = ours(hidden[:, :, 0, :], torch.from_numpy(ids))
    assert torch.allclose(ref_out, our_out, atol=1e-6), (ref_out - our_out).abs().max()
