"""Every Phase 2 arm resolves its model from an LM data folder like notebook 03's output.

Regression: A1 / A5 / T1 / T2 (no Engram, parameter-matched to A2) crashed on Modal with "engram.compress=True needs
the tokenizer compression lookup", because only Engram arms loaded engram_compress.npy.
"""
import json

import numpy as np
import pytest

from engramdelta.models.model import count_params_for_config
from engramdelta.scripts.train import resolve, resolve_model_config, setup_task
from engramdelta.utils.config import load_config

VOCAB = 16384


@pytest.fixture(scope="module")
def lmdata(tmp_path_factory):
    d = tmp_path_factory.mktemp("lmdata")
    (d / "meta.json").write_text(json.dumps({"vocab_size": VOCAB, "eos_id": 0}))
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, VOCAB, 5000, dtype=np.uint16).tofile(d / f"{split}.bin")
    np.save(d / "engram_compress.npy", (np.arange(VOCAB) % 12578).astype(np.int64))   # 12,578 canonical ids, as in NB03
    return d


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A5", "T1", "T2"])
def test_phase2_arm_resolves_and_is_matched(arm, lmdata):
    cfg = resolve(load_config(f"configs/phase2/{arm}.yaml", [f"data.path={lmdata}", "data.seq_len=64"]))
    task = setup_task(cfg)
    mc = resolve_model_config(cfg, task)
    n = count_params_for_config(mc, task["compress"])["non_embedding"]
    a2 = resolve(load_config("configs/phase2/A2.yaml", [f"data.path={lmdata}", "data.seq_len=64"]))
    t2 = setup_task(a2)
    target = count_params_for_config(resolve_model_config(a2, t2), t2["compress"])["non_embedding"]
    assert abs(n - target) / target < 0.02, (arm, n, target)
