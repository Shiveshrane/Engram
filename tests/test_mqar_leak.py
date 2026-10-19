"""Test 6 (design §6): MQAR format and answer leakage.

Format: every query's target is the value that followed that key in the pair segment. Answers never appear in
the input after the pair segment. Keys are distinct within a sequence. Clutter is pure filler built from the
phrase pool. Loss kinds have the documented counts.
Leakage: key -> value assignments are resampled per sequence, so the same key maps to the same value across
sequences only at chance rate. The train / val / eval streams differ. An Engram-only model with no sequence
mixer (A6) trained on MQAR stays near chance.
"""
import pytest
import torch

from engramdelta.data import mqar
from engramdelta.eval.mqar_eval import evaluate_mqar
from helpers import tiny_model, tiny_mqar


@pytest.mark.parametrize("placement,F,N", [("a", 0, 6), ("a", 24, 6), ("b", 24, 6), ("b", 7, 5), ("b", 0, 1)])
def test_format(placement, F, N):
    cfg = tiny_mqar(placement=placement, F=F, N=N)
    lay = mqar.layout(cfg)
    b = mqar.train_batch(cfg, 16, seed=0, rank=0, step=0)
    ids, tgt, kinds = b["ids"], b["targets"], b["kinds"]
    assert ids.shape == (16, cfg.seq_len) == (16, 4 * N + F)
    kp, vp, qp = (torch.tensor(lay[k]) for k in ("key_pos", "val_pos", "query_pos"))
    keys, vals = ids[:, kp], ids[:, vp]
    assert ((keys >= cfg.key_offset) & (keys < cfg.value_offset)).all()
    assert ((vals >= cfg.value_offset) & (vals < cfg.filler_offset)).all()
    for r in range(16):
        assert len(set(keys[r].tolist())) == N, "keys must be distinct within a sequence"
        kv = dict(zip(keys[r].tolist(), vals[r].tolist()))
        q = ids[r, qp].tolist()
        assert sorted(q) == sorted(kv), "every key is queried exactly once"
        assert tgt[r, qp].tolist() == [kv[k] for k in q], "target at a query = value of that key"
    assert (ids[:, qp + 1] == mqar.BLANK).all()
    after_pairs = ids[:, lay["end_of_writing"] + 1:]
    assert not ((after_pairs >= cfg.value_offset) & (after_pairs < cfg.filler_offset)).any(), "value token leaked into input"
    clutter_mask = torch.zeros(cfg.seq_len, dtype=torch.bool)
    for s, l in lay["clutter"]:
        clutter_mask[s:s + l] = True
    assert int(clutter_mask.sum()) == F
    assert (ids[:, clutter_mask] >= cfg.filler_offset).all()
    assert (ids[:, ~clutter_mask] < cfg.filler_offset).all()
    assert ((kinds == 1).sum(1) == N).all()
    assert ((kinds == 2).sum(1) == F).all(), "one filler target per clutter token"
    assert (tgt[:, :-1][kinds[:, :-1] == 2] == ids[:, 1:][kinds[:, :-1] == 2]).all()


def test_phrases_come_from_the_pool():
    cfg = tiny_mqar(F=30, P=4, phrase_len=3)
    pool = {tuple(p) for p in mqar.phrase_pool(cfg).tolist()}
    b = mqar.train_batch(cfg, 8, 0, 0, 0)
    s, l = mqar.layout(cfg)["clutter"][0]
    for row in b["ids"][:, s:s + l].tolist():
        for i in range(0, l, 3):
            assert tuple(row[i:i + 3]) in pool
    assert torch.equal(mqar.phrase_pool(cfg), mqar.phrase_pool(tiny_mqar(F=30, P=4)))
    assert not torch.equal(mqar.phrase_pool(cfg), mqar.phrase_pool(tiny_mqar(F=30, P=4, pool_seed=7)))


def test_assignments_resampled_across_sequences():
    cfg = tiny_mqar(N=6, n_keys=64, n_values=64)
    lay = mqar.layout(cfg)
    ids = torch.cat([mqar.train_batch(cfg, 64, 0, 0, s)["ids"] for s in range(4)])
    pairs = {}
    same = total = 0
    for row in ids.tolist():
        for kp, vp in zip(lay["key_pos"], lay["val_pos"]):
            k, v = row[kp], row[vp]
            if k in pairs:
                total += 1
                same += pairs[k] == v
            pairs[k] = v
    assert total > 100 and same / total < 3 / cfg.n_values + 0.02, f"key->value repeats too often: {same}/{total}"


def test_streams_are_distinct_and_deterministic():
    cfg = tiny_mqar()
    a = mqar.train_batch(cfg, 8, seed=0, rank=0, step=5)["ids"]
    assert torch.equal(a, mqar.train_batch(cfg, 8, seed=0, rank=0, step=5)["ids"])
    assert not torch.equal(a, mqar.train_batch(cfg, 8, seed=0, rank=1, step=5)["ids"])
    assert not torch.equal(a, mqar.train_batch(cfg, 8, seed=1, rank=0, step=5)["ids"])
    ev, va = mqar.fixed_set(cfg, "eval")["ids"], mqar.fixed_set(cfg, "val")["ids"]
    assert not torch.equal(ev[:8], va[:8]) and not torch.equal(ev[:8], a)


def test_engram_only_probe_stays_at_chance():
    """A6 (embedding + Engram + MLP, no mixer), trained on fresh MQAR batches, must not beat chance by much."""
    cfg = tiny_mqar(N=6, F=0, n_keys=64, n_values=64, eval_size=256)
    model = tiny_model("A6", cfg.vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(300):
        b = mqar.train_batch(cfg, 32, 0, 0, step)
        out = model(b["ids"], b["targets"], b["kinds"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    ev = evaluate_mqar(model, cfg, torch.device("cpu"))
    chance = 1 / cfg.n_values
    assert ev["recall"] < 3 * chance + 0.01, f"A6 recall {ev['recall']:.3f} vs chance {chance:.3f}: answer leakage?"
