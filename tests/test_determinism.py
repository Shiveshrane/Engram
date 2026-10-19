"""Test 9 (design §6): the same seed gives the same result on the same hardware. Plus the result-file contract
(design rule 5): one immutable JSON per run, never overwritten."""
import json
import os

import pytest

from engramdelta.scripts.train import train
from helpers import tiny_train_cfg


def test_same_seed_same_result(tmp_path):
    a = train(tiny_train_cfg("A2", steps=20, seed=3), "a", str(tmp_path / "r"), str(tmp_path / "c"),
              write_result_file=False, verbose=False)
    b = train(tiny_train_cfg("A2", steps=20, seed=3), "b", str(tmp_path / "r"), str(tmp_path / "c"),
              write_result_file=False, verbose=False)
    c = train(tiny_train_cfg("A2", steps=20, seed=4), "c", str(tmp_path / "r"), str(tmp_path / "c"),
              write_result_file=False, verbose=False)
    assert a["curve"] == b["curve"] and a["metrics"] == b["metrics"]       # bit-identical on CPU
    assert a["curve"] != c["curve"]


def test_result_file_is_immutable(tmp_path):
    rec = train(tiny_train_cfg("A1", steps=4), "imm", str(tmp_path / "r"), str(tmp_path / "c"), verbose=False)
    path = os.path.join(tmp_path, "r", "scratch", "tests", "imm.json")
    with open(path) as f:
        on_disk = json.load(f)
    for key in ("git_commit", "seed", "params", "metrics", "model", "train", "data", "world_size", "env", "curve"):
        assert key in on_disk
    assert on_disk["metrics"]["recall"] == rec["metrics"]["recall"]
    with pytest.raises(FileExistsError):
        train(tiny_train_cfg("A1", steps=4), "imm", str(tmp_path / "r"), str(tmp_path / "c2"), verbose=False)


def test_resume_single_process(tmp_path):
    full = train(tiny_train_cfg("A3", steps=12), "full", str(tmp_path / "r"), str(tmp_path / "c"),
                 write_result_file=False, verbose=False)
    train(tiny_train_cfg("A3", steps=12), "res", str(tmp_path / "r"), str(tmp_path / "c"), stop_at_step=6,
          write_result_file=False, verbose=False)
    res = train(tiny_train_cfg("A3", steps=12), "res", str(tmp_path / "r"), str(tmp_path / "c"),
                write_result_file=False, verbose=False)
    assert full["curve"] == res["curve"] and full["metrics"] == res["metrics"]
