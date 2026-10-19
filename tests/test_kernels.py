"""Test 4 (design §6): kernel correctness against FLA's naive reference (third_party/fla_naive.py), in fp32.

  * our pure-PyTorch chunk kernel == FLA naive recurrent and FLA naive chunk (outputs and gradients)
  * our recurrent reference (used for state metrics) == FLA naive recurrent, and its captured state at T-1 ==
    the chunk kernel's final state
  * the GatedDeltaNet layer gives the same output through the chunk path and the state-capture path
  * FLA's Triton kernel == the reference (CUDA + fla only; speed and the T4 verdict come from bench_kernels.py)
"""
import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

from engramdelta.models.gdn_block import (GatedDeltaNet, chunk_gated_delta_rule_fla, chunk_gated_delta_rule_torch,
                                          fla_available, recurrent_gated_delta_rule_torch)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def naive():
    pytest.importorskip("einops")
    spec = importlib.util.spec_from_file_location("fla_naive", os.path.join(ROOT, "third_party", "fla_naive.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs(B, T, H, K, V, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    v = torch.randn(B, T, H, V, generator=g)
    gate = F.logsigmoid(torch.randn(B, T, H, generator=g) + 2)
    beta = torch.rand(B, T, H, generator=g)
    return [x.to(device).requires_grad_(True) for x in (q, k, v, gate, beta)]


def run(fn, xs, seed=1):
    xs = [x.detach().clone().requires_grad_(True) for x in xs]
    o = fn(*xs)
    do = torch.randn(o.shape, generator=torch.Generator().manual_seed(seed)).to(o)
    (o * do).sum().backward()
    return o.detach(), [x.grad.detach() for x in xs]


SHAPES = [(2, 1, 2, 8, 8), (2, 17, 2, 16, 8), (1, 64, 3, 16, 16), (2, 150, 2, 32, 32)]


@pytest.mark.parametrize("shape", SHAPES)
def test_torch_chunk_matches_naive_recurrent(naive, shape):
    xs = inputs(*shape)
    ref_o, ref_g = run(lambda q, k, v, g, b: naive.naive_recurrent_gated_delta_rule(q, k, v, b, g)[0], xs)
    for cs in (16, 64):
        o, gr = run(lambda q, k, v, g, b: chunk_gated_delta_rule_torch(q, k, v, g, b, chunk_size=cs)[0], xs)
        assert torch.allclose(o, ref_o, atol=1e-5), (o - ref_o).abs().max()
        for name, a, b in zip("qkvgb", gr, ref_g):
            assert torch.allclose(a, b, atol=1e-4), f"grad {name}: {(a - b).abs().max()}"


@pytest.mark.parametrize("shape", SHAPES[1:])
def test_torch_chunk_matches_naive_chunk(naive, shape):
    q, k, v, g, b = [x.detach() for x in inputs(*shape)]
    ref_o, ref_S = naive.naive_chunk_gated_delta_rule(q, k, v, g, b, chunk_size=16, output_final_state=True)
    o, S = chunk_gated_delta_rule_torch(q, k, v, g, b, chunk_size=16, output_final_state=True)
    assert torch.allclose(o, ref_o, atol=1e-5)
    assert torch.allclose(S, ref_S, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES[1:])
def test_recurrent_reference_and_state_capture(naive, shape):
    q, k, v, g, b = [x.detach() for x in inputs(*shape)]
    T = shape[1]
    ref_o, ref_S = naive.naive_recurrent_gated_delta_rule(q, k, v, b, g, output_final_state=True)
    o, states = recurrent_gated_delta_rule_torch(q, k, v, g, b, capture_positions=[0, T // 2, T - 1])
    assert torch.allclose(o, ref_o, atol=1e-5)
    assert torch.allclose(states[T - 1], ref_S, atol=1e-5)
    _, S_chunk = chunk_gated_delta_rule_torch(q, k, v, g, b, chunk_size=16, output_final_state=True)
    assert torch.allclose(states[T - 1], S_chunk, atol=1e-5)
    # The state after token t only depends on tokens <= t: the prefix run gives the same captured state.
    _, st_prefix = recurrent_gated_delta_rule_torch(q[:, :T // 2 + 1], k[:, :T // 2 + 1], v[:, :T // 2 + 1],
                                                     g[:, :T // 2 + 1], b[:, :T // 2 + 1], capture_positions=[T // 2])
    assert torch.allclose(st_prefix[T // 2], states[T // 2], atol=1e-6)


@pytest.mark.parametrize("short_conv", [True, False])
def test_layer_capture_path_equals_chunk_path(short_conv):
    torch.manual_seed(0)
    layer = GatedDeltaNet(32, 2, 16, use_short_conv=short_conv, chunk_size=16).eval()
    x = torch.randn(2, 37, 32)
    with torch.no_grad():
        y_chunk = layer(x)
        layer.capture_positions = [36]
        y_rec = layer(x)
        layer.capture_positions = None
    assert torch.allclose(y_chunk, y_rec, atol=1e-5)
    assert layer.captured[36].shape == (2, 2, 16, 16)


@pytest.mark.skipif(not (torch.cuda.is_available() and fla_available()), reason="needs CUDA + flash-linear-attention")
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 5e-2), (torch.float32, 1e-3)])
def test_fla_triton_matches_reference(naive, dtype, tol):
    xs = inputs(2, 150, 2, 32, 32, device="cuda")
    ref_o, ref_g = run(lambda q, k, v, g, b: naive.naive_recurrent_gated_delta_rule(q, k, v, b, g)[0], xs)
    xs_dt = [x.detach().to(dtype) if x.dim() == 4 else x.detach() for x in xs]
    try:
        o, gr = run(lambda q, k, v, g, b: chunk_gated_delta_rule_fla(q, k, v, g, b.to(q.dtype)), xs_dt)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"FLA kernel failed on this GPU ({dtype}): {type(e).__name__}: {e}")
    assert (o.float() - ref_o).abs().max() < tol
    for a, b in zip(gr, ref_g):
        assert (a.float() - b).abs().max() < 10 * tol
