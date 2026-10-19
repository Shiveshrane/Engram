"""Test 8 (design §6): fp16 stability, on CUDA only (T4 has no bf16).

Arms A1, A2, T1 at the real Phase 1 size: 500 fp16 steps with no non-finite loss (GradScaler skips only a few
steps while it calibrates its scale), and the fp16 loss curve over the first 200 steps stays close to an fp32
run with the same seed.
"""
import pytest
import torch

from engramdelta.scripts.train import resolve, train
from engramdelta.utils.config import load_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 test needs CUDA")


@pytest.mark.parametrize("arm", ["A1", "A2", "T1"])
def test_fp16_stable_and_close_to_fp32(arm, tmp_path):
    over = ["data.N=16", "data.F=128", "train.log_every=10", "train.ckpt_every=0", "train.device=cuda"]
    cfg16 = resolve(load_config(f"configs/phase1/{arm}.yaml", over + ["train.steps=500", "train.precision=fp16"]))
    cfg32 = resolve(load_config(f"configs/phase1/{arm}.yaml", over + ["train.steps=500", "train.precision=fp32"]))
    r16 = train(cfg16, f"fp16_{arm}", str(tmp_path / "r"), str(tmp_path / "c"), write_result_file=False, verbose=False)
    r32 = train(cfg32, f"fp32_{arm}", str(tmp_path / "r"), str(tmp_path / "c"), stop_at_step=200,
                write_result_file=False, verbose=False)
    assert r16["nonfinite_losses"] == 0
    assert r16["skipped_steps"] <= 20, f"GradScaler skipped {r16['skipped_steps']} steps"
    c16 = [c[1] + c[2] for c in r16["curve"] if c[0] <= 200]
    c32 = [c[1] + c[2] for c in r32["curve"] if c[0] <= 200]
    n = min(len(c16), len(c32))
    rel = abs(sum(c16[:n]) - sum(c32[:n])) / sum(c32[:n])
    print(f"{arm}: mean |fp16 - fp32| relative over first 200 steps = {rel:.4f}")
    assert rel < 0.05
