"""Test 1 (design §6): causality. The most likely place for a silent bug.

For random inputs and a random position j:
  * perturbing tokens at positions > j must leave every output at positions <= j unchanged, and
  * perturbing x_j must change the output at j, so the test is not vacuous.
Run on the Engram module alone (including its hash) and on every full model. For the sequence-mixing arms, we
also check that a token 5 steps back still influences position j, so the mixer is actually connected.
"""
import pytest
import torch

from engramdelta.models.engram import Engram, layer_multipliers, table_sizes_for_layers
from helpers import ARMS, SEQUENCE_MIXING_ARMS, tiny_model

VOCAB = 97
T = 40          # not a multiple of the GDN chunk size (16), so padding is exercised
ATOL = 1e-6


def perturb_after(x: torch.Tensor, j: int, vocab: int, gen) -> torch.Tensor:
    y = x.clone()
    y[:, j + 1:] = torch.randint(0, vocab, y[:, j + 1:].shape, generator=gen)
    return y


def perturb_at(x: torch.Tensor, j: int, vocab: int) -> torch.Tensor:
    y = x.clone()
    y[:, j] = (y[:, j] + 1 + torch.arange(y.shape[0])) % vocab
    return y


def make_engram(orders=(2, 3), heads=2, embed=32):
    primes = table_sizes_for_layers([0], len(orders), 53, heads)[0]
    return Engram(d_model=16, orders=list(orders), primes=primes,
                  multipliers=layer_multipliers(0, 0, max(orders), VOCAB), embed_per_order=embed,
                  n_heads_per_order=heads, pad_id=0)


@pytest.mark.parametrize("orders", [(2, 3), (1,), (2,), (3,)])
def test_engram_module_causal(orders):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    eng = make_engram(orders).eval()
    for trial in range(5):
        x = torch.randint(0, VOCAB, (3, T), generator=gen)
        h = torch.randn(3, T, 16, generator=gen)
        j = int(torch.randint(0, T - 1, (1,), generator=gen))
        out = eng(h, x)
        h2 = h.clone()
        h2[:, j + 1:] = torch.randn(h2[:, j + 1:].shape, generator=gen)
        out2 = eng(h2, perturb_after(x, j, VOCAB, gen))
        assert torch.allclose(out[:, :j + 1], out2[:, :j + 1], atol=ATOL), f"future leaked into position <= {j}"
        out3 = eng(h, perturb_at(x, j, VOCAB))
        assert (out3[:, j] - out[:, j]).abs().max() > 1e-4, "changing x_j did not change the output at j"


def test_engram_hash_is_causal_and_padded():
    eng = make_engram((2, 3))
    x = torch.randint(1, VOCAB, (2, 12))
    idx = eng.hasher(x)
    x2 = x.clone()
    x2[:, 6:] = (x2[:, 6:] + 3) % VOCAB
    assert torch.equal(eng.hasher(x2)[:, :6], idx[:, :6])
    # position 0: the 2-gram is (pad, x0). Changing nothing but x0 changes the index at 0, and never at t >= 3.
    x3 = x.clone()
    x3[:, 0] = (x3[:, 0] + 1) % VOCAB
    i3 = eng.hasher(x3)
    assert not torch.equal(i3[:, 0], idx[:, 0])
    assert torch.equal(i3[:, 3:], idx[:, 3:])


@pytest.mark.parametrize("arm", list(ARMS))
def test_full_model_causal(arm):
    model = tiny_model(arm, VOCAB).eval()
    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for trial in range(3):
            x = torch.randint(0, VOCAB, (2, T), generator=gen)
            j = int(torch.randint(6, T - 1, (1,), generator=gen))
            logits = model(x)
            logits2 = model(perturb_after(x, j, VOCAB, gen))
            assert torch.allclose(logits[:, :j + 1], logits2[:, :j + 1], atol=ATOL), \
                f"{arm}: future tokens changed logits at positions <= {j}"
            logits3 = model(perturb_at(x, j, VOCAB))
            assert (logits3[:, j] - logits[:, j]).abs().max() > 1e-5, f"{arm}: x_j does not affect logits at j"
            if arm in SEQUENCE_MIXING_ARMS:
                logits4 = model(perturb_at(x, j - 5, VOCAB))
                assert (logits4[:, j] - logits[:, j]).abs().max() > 1e-7, f"{arm}: x_(j-5) does not reach position j"
