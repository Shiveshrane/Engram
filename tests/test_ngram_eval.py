"""Study E2.ngram: train n-gram counts per val token (scripts/ngram_counts.py) and the bucketed eval loss."""
import json
from collections import Counter

import numpy as np
import torch

from engramdelta.data.lm_data import LMData
from engramdelta.eval.lm_eval import NGRAM_LABELS, evaluate_lm_ngram
from engramdelta.scripts.ngram_counts import build, counts_for


def test_counts_match_brute_force():
    rng = np.random.default_rng(0)
    train, val = rng.integers(0, 5, 3000), rng.integers(0, 6, 400)        # val has an id never seen in train
    for n in (2, 3):
        got = counts_for(train, val, n, base=6)
        c = Counter(tuple(train[i:i + n]) for i in range(len(train) - n + 1))
        want = [0] * (n - 1) + [c.get(tuple(val[j - n + 1:j + 1]), 0) for j in range(n - 1, len(val))]
        assert got.tolist() == want


class Bigramish(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.emb = torch.nn.Embedding(v, 16)
        self.out = torch.nn.Linear(16, v)

    def forward(self, ids):
        return self.out(self.emb(ids))


def test_bucketed_loss_is_consistent(tmp_path):
    V, L = 50, 32
    rng = np.random.default_rng(1)
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": V, "eos_id": 0}))
    rng.integers(0, V, 20000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    (rng.integers(0, V, 10 * L + 1) % 7).astype(np.uint16).tofile(tmp_path / "val.bin")   # frequent n-grams too
    np.save(tmp_path / "engram_compress.npy", np.arange(V) // 2)                           # a non-identity map
    build(str(tmp_path))
    counts = dict(np.load(tmp_path / "val_ngram_counts.npz"))
    torch.manual_seed(0)
    model, val = Bigramish(V), LMData(str(tmp_path), L, "val")
    m = evaluate_lm_ngram(model, val, torch.device("cpu"), counts, batch_size=3)
    assert m["tokens_all_p2"] == val.n_windows * (L - 2)
    for name in ("bigram", "trigram"):
        toks = [m[f"tokens_{name}_{b}"] for b in NGRAM_LABELS]
        assert sum(toks) == m["tokens_all_p2"]
        weighted = sum(m[f"loss_{name}_{b}"] * t for b, t in zip(NGRAM_LABELS, toks) if t) / sum(toks)
        assert abs(weighted - m["loss_all_p2"]) < 1e-9
    # direct check of the overall p >= 2 loss
    ids = torch.stack([val.window(i) for i in range(val.n_windows)])
    ce = torch.nn.functional.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten(), reduction="none")
    assert abs(ce.view(val.n_windows, L)[:, 2:].mean().item() - m["loss_all_p2"]) < 1e-5
