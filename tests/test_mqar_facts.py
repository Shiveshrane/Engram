"""The "facts" MQAR variant (exploratory study E3): format, fixed fact table, and that the plain task is unchanged."""
import hashlib

import pytest
import torch

from engramdelta.data import mqar
from engramdelta.data.mqar import BLANK, MQARConfig, fact_table, fixed_set, generate, layout, train_batch


def digest(d):
    m = hashlib.sha256()
    for k in ("ids", "targets", "kinds"):
        m.update(d[k].numpy().tobytes())
    return m.hexdigest()[:16]


# Fingerprints of the plain task taken before the facts variant was added (code commit 6748062), so every
# Phase 0 / Phase 1 sequence is still bit-identical.
@pytest.mark.parametrize("cfg,train_hash,eval_hash", [
    (dict(N=8, F=6, P=4, placement="a"), "4a8c03e5f60b8fae", "9c799850d694ab30"),
    (dict(N=8, F=7, P=4, placement="b"), "d62e72d30821a5c6", "a0f8c6182ef2a566"),
    (dict(N=32, F=512, filler_weight=0.0), "b8f1746bfc522840", "02dcf7e6a04f8ea0"),
])
def test_plain_task_unchanged(cfg, train_hash, eval_hash):
    c = MQARConfig(**cfg)
    assert c.vocab_size == 4612
    assert digest(train_batch(c, 4, 0, 0, 5, 1)) == train_hash
    assert digest(fixed_set(c, "eval", 50)) == eval_hash
    assert "qtype" not in train_batch(c, 2, 0, 0, 0)


CFG = MQARConfig(N=8, F=6, P=4, n_facts=5, fact_pool=40, fact_vocab=16)


def test_fact_table_fixed_and_distinct():
    a, b, v = fact_table(CFG)
    a2, b2, v2 = fact_table(CFG)
    assert torch.equal(a, a2) and torch.equal(b, b2) and torch.equal(v, v2)
    assert len({(int(x), int(y)) for x, y in zip(a, b)}) == CFG.fact_pool
    assert ((a >= CFG.fact_offset) & (a < CFG.vocab_size)).all() and ((b >= CFG.fact_offset) & (b < CFG.vocab_size)).all()
    assert ((v >= CFG.value_offset) & (v < CFG.filler_offset)).all()


def test_facts_format():
    d = generate(CFG, 64, torch.Generator().manual_seed(0))
    ids, tgt, kinds, qt = d["ids"], d["targets"], d["kinds"], d["qtype"]
    lay = layout(CFG)
    assert ids.shape == (64, CFG.seq_len) == (64, 4 * 8 + 6 + 2 * 5)
    a, b, v = fact_table(CFG)
    facts = {(int(x), int(y)): int(z) for x, y, z in zip(a, b, v)}
    qs = lay["query_pos"][0]
    for r in range(64):
        keys = ids[r, lay["key_pos"]].tolist()
        vals = ids[r, lay["val_pos"]].tolist()
        kv = dict(zip(keys, vals))
        context = ids[r, :qs]
        assert not ((context >= CFG.fact_offset).any()), "facts must never appear in the context"
        n_ctx = n_fact = 0
        for p in lay["query_pos"]:
            x, y = int(ids[r, p]), int(ids[r, p + 1])
            if y == BLANK:          # in-context query: answer at the key position
                n_ctx += 1
                assert qt[r, p] == 1 and kinds[r, p] == 1 and tgt[r, p] == kv[x] and kinds[r, p + 1] == 0
            else:                   # fact query: answer at the second fact token
                n_fact += 1
                assert qt[r, p + 1] == 2 and kinds[r, p + 1] == 1 and tgt[r, p + 1] == facts[(x, y)]
                assert kinds[r, p] == 0
        assert (n_ctx, n_fact) == (CFG.N, CFG.n_facts)
        assert int((kinds[r] == 1).sum()) == CFG.N + CFG.n_facts
    # every answer is absent from the input that precedes it (values only appear in the pairs section)
    assert not (ids[:, qs:] >= CFG.value_offset).logical_and(ids[:, qs:] < CFG.filler_offset).any()


def test_facts_filler_targets_only_on_filler():
    d = generate(CFG, 16, torch.Generator().manual_seed(1))
    k2 = d["kinds"][:, :-1] == 2
    nxt = d["ids"][:, 1:]
    assert torch.equal(k2, (nxt >= CFG.filler_offset) & (nxt < CFG.filler_offset + CFG.n_filler))


def test_facts_eval_set_differs_from_plain_but_is_fixed():
    plain = MQARConfig(N=8, F=6, P=4)
    e1, e2 = fixed_set(CFG, "eval", 20), fixed_set(CFG, "eval", 20)
    assert torch.equal(e1["ids"], e2["ids"])
    assert e1["ids"].shape[1] != fixed_set(plain, "eval", 20)["ids"].shape[1]
    assert not torch.equal(fixed_set(CFG, "val", 20)["ids"], e1["ids"])


def test_facts_split_recall_and_training_smoke(tmp_path):
    """A few CPU steps of A2 with facts: the result has recall_ctx / recall_fact, and recall = their weighted mean."""
    from engramdelta.scripts.train import resolve, train
    from engramdelta.utils.config import load_config
    cfg = resolve(load_config("configs/phase1/A2.yaml", [
        "data.N=4", "data.F=0", "data.n_facts=3", "data.fact_pool=20", "data.fact_vocab=8", "data.eval_size=40",
        "train.steps=6", "train.log_every=3", "train.eval_every=3", "train.eval_size_during=20", "train.device=cpu",
        "train.global_batch=8", "phase=phase3", "group=E3.test"]))
    rec = train(cfg, "facts_smoke", str(tmp_path / "results"), str(tmp_path / "ckpt"), verbose=False)
    m = rec["metrics"]
    assert {"recall_ctx", "recall_fact", "val_recall_ctx"} <= set(m)
    assert abs(m["recall"] - (4 * m["recall_ctx"] + 3 * m["recall_fact"]) / 7) < 1e-9
    assert "recall_ctx" in rec["eval_curve"][0][1]
