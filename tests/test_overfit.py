"""Test 2 (design §6): every model reaches near-zero loss on one tiny fixed batch within a few hundred steps."""
import pytest
import torch

from engramdelta.data import mqar
from helpers import ARMS, tiny_model, tiny_mqar


@pytest.mark.parametrize("arm", list(ARMS))
def test_overfit_one_batch(arm):
    cfg = tiny_mqar(N=4, F=6, P=4, n_keys=32, n_values=32, n_filler=16)
    batch = mqar.train_batch(cfg, 4, seed=0, rank=0, step=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tiny_model(arm, cfg.vocab_size).to(device)
    batch = {k: v.to(device) for k, v in batch.items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
    for step in range(400):
        out = model(batch["ids"], batch["targets"], batch["kinds"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    assert out["loss"].item() < 0.05, f"{arm}: loss {out['loss'].item():.4f} after 400 steps on one batch"
