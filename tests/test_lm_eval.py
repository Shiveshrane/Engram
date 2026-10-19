"""Phase 2 data and evaluation: the in-context-copy mask, the LM window sampler and the eval aggregation."""
import json

import numpy as np
import pytest
import torch

from engramdelta.data.lm_data import LMData
from engramdelta.eval.lm_eval import copy_mask, evaluate_lm
from helpers import tiny_model


def brute_copy_mask(x, d):
    T = len(x)
    m = np.zeros(T, dtype=bool)
    for p in range(1, T):
        m[p] = any(x[q - 1] == x[p - 1] and x[q] == x[p] and p - q > d for q in range(1, p))
    return m


def test_copy_mask_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 6, size=150)
        assert np.array_equal(copy_mask(x, 10), brute_copy_mask(x, 10))


def test_copy_mask_simple():
    x = np.array([5, 7] + [0] * 70 + [5, 7, 9])
    m = copy_mask(x, 64)
    assert m[73] and m.sum() >= 1          # bigram (5, 7) ends at 1 and again at 73: distance 72 > 64
    assert not copy_mask(np.array([5, 7, 5, 7]), 64).any()


@pytest.fixture
def lm_dir(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 5000), ("val", 1200)):
        rng.integers(0, 50, size=n).astype(np.uint16).tofile(tmp_path / f"{split}.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 50, "eos_id": 0}))
    return str(tmp_path)


def test_lm_sampler_ranks_disjoint_and_resumable(lm_dir):
    d = LMData(lm_dir, 32, "train")
    seen = []
    for step in range(3):
        for rank in range(2):
            b = d.train_batch(4, seed=0, rank=rank, world=2, step=step, micro=0, grad_accum=1)
            assert torch.equal(b["targets"][:, :-1], b["ids"][:, 1:])
            seen += [tuple(r.tolist()) for r in b["ids"]]
    assert len(set(seen)) == len(seen), "ranks / steps must not share windows within an epoch"
    again = d.train_batch(4, seed=0, rank=1, world=2, step=2, micro=0, grad_accum=1)
    assert tuple(again["ids"][0].tolist()) in seen


def test_evaluate_lm_runs(lm_dir):
    d = LMData(lm_dir, 32, "val")
    model = tiny_model("A2", 50)
    out = evaluate_lm(model, d, torch.device("cpu"), batch_size=4)
    assert out["tokens_all"] == d.n_windows * 32
    assert out["tokens_pos_0_64"] == out["tokens_all"]
    assert 2.0 < out["loss_all"] < 6.0
