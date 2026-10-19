"""Phase 0 test 4: GDN kernel correctness and speed (NB00).

    python -m engramdelta.scripts.bench_kernels --out results/phase0/E0.kernels

Correctness (fp32, small random shapes): our torch chunk kernel and (if installed) FLA's Triton chunk kernel are
compared against FLA's naive recurrent reference (third_party/fla_naive.py), for the outputs and for the gradients
of q, k, v, g, beta. FLA is also run with fp16 inputs (training dtype on T4).
Speed: forward + backward time at the Phase 1 / Phase 2 head shapes for T in {512, 1024, 2048}.
Writes one new JSON file per invocation (timestamped; never overwritten) and prints a recommendation.
"""
import argparse
import importlib.util
import json
import os
import time

import torch
import torch.nn.functional as F

from ..models.gdn_block import chunk_gated_delta_rule_fla, chunk_gated_delta_rule_torch, fla_available
from ..utils.run_log import REPO_ROOT, env_info, now, write_result


def load_fla_naive():
    spec = importlib.util.spec_from_file_location("fla_naive", os.path.join(REPO_ROOT, "third_party", "fla_naive.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def random_inputs(B, T, H, K, V, device, dtype=torch.float32, seed=0):
    g_ = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=g_)
    k = F.normalize(torch.randn(B, T, H, K, generator=g_), p=2, dim=-1)
    q = F.normalize(q, p=2, dim=-1)
    v = torch.randn(B, T, H, V, generator=g_)
    beta = torch.rand(B, T, H, generator=g_).sigmoid()
    g = F.logsigmoid(torch.randn(B, T, H, generator=g_))
    return [x.to(device=device, dtype=dtype if x.dim() == 4 else torch.float32).requires_grad_(True)
            for x in (q, k, v, g, beta)]


def outputs_and_grads(fn, inputs, seed=1):
    inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
    o = fn(*inputs)
    g_ = torch.Generator(device="cpu").manual_seed(seed)
    do = torch.randn(o.shape, generator=g_).to(o.device, o.dtype)
    (o * do).sum().backward()
    return o.detach().float(), [x.grad.detach().float() for x in inputs]


def max_diffs(a, b):
    (oa, ga), (ob, gb) = a, b
    out = {"o": (oa - ob).abs().max().item()}
    for name, x, y in zip(("dq", "dk", "dv", "dg", "dbeta"), ga, gb):
        out[name] = (x - y).abs().max().item()
    return out


def correctness(device):
    naive = load_fla_naive()
    ref_fn = lambda q, k, v, g, b: naive.naive_recurrent_gated_delta_rule(q, k, v, b, g)[0]
    ours_fn = lambda q, k, v, g, b: chunk_gated_delta_rule_torch(q, k, v, g, b)[0]
    res = {}
    for shape in [(2, 200, 4, 32, 32), (2, 129, 2, 64, 64)]:
        inp = random_inputs(*shape, device=device)
        ref = outputs_and_grads(ref_fn, inp)
        key = "x".join(map(str, shape))
        res[f"torch_chunk_fp32_{key}"] = max_diffs(outputs_and_grads(ours_fn, inp), ref)
        if fla_available() and device.type == "cuda":
            for dt in (torch.float32, torch.float16):
                try:
                    inp_dt = [x.detach().to(dt) if x.dim() == 4 else x.detach() for x in inp]
                    fla_fn = lambda q, k, v, g, b: chunk_gated_delta_rule_fla(q, k, v, g, b.to(q.dtype))
                    res[f"fla_chunk_{str(dt)[6:]}_{key}"] = max_diffs(outputs_and_grads(fla_fn, inp_dt), ref)
                except Exception as e:      # noqa: BLE001  (record the failure, e.g. unsupported on sm_75)
                    res[f"fla_chunk_{str(dt)[6:]}_{key}"] = {"error": f"{type(e).__name__}: {e}"[:500]}
    return res


def bench(fn, inputs, iters=10):
    for _ in range(3):
        outputs_and_grads(fn, inputs)
    if inputs[0].device.type == "cuda":
        torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        t0 = time.time()
        outputs_and_grads(fn, inputs)
        if inputs[0].device.type == "cuda":
            torch.cuda.synchronize()
        t.append(time.time() - t0)
    return sorted(t)[len(t) // 2]


def speed(device):
    res = {}
    for B, H, K in [(32, 4, 32), (16, 6, 64)]:          # Phase 1 per-rank shapes / Phase 2 per-rank shapes
        for T in (512, 1024, 2048):
            key = f"B{B}_H{H}_K{K}_T{T}"
            inp = random_inputs(B, T, H, K, K, device=device)
            res[f"torch_{key}"] = bench(lambda q, k, v, g, b: chunk_gated_delta_rule_torch(q, k, v, g, b)[0], inp)
            if fla_available() and device.type == "cuda":
                try:
                    inp16 = [x.detach().half() if x.dim() == 4 else x.detach() for x in inp]
                    res[f"fla_fp16_{key}"] = bench(lambda q, k, v, g, b: chunk_gated_delta_rule_fla(q, k, v, g, b.half()), inp16)
                except Exception as e:      # noqa: BLE001
                    res[f"fla_fp16_{key}"] = f"error: {type(e).__name__}: {e}"[:300]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase0/E0.kernels")
    ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--fwd-tol", type=float, default=1e-4)
    ap.add_argument("--grad-tol", type=float, default=1e-3)
    ap.add_argument("--fp16-tol", type=float, default=5e-2)
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = {"run_id": f"kernels_{now().replace(':', '-')}", "status": "done", "phase": "phase0", "group": "E0.kernels",
           "arm": "-", "env": env_info(), "fla_available": fla_available(), "correctness": correctness(device)}
    if not args.no_speed:
        rec["speed_seconds_fwd_bwd"] = speed(device)
    verdict = {}
    for name, d in rec["correctness"].items():
        if "error" in d:
            verdict[name] = "ERROR"
            continue
        tol_o, tol_g = (args.fp16_tol, args.fp16_tol) if "float16" in name else (args.fwd_tol, args.grad_tol)
        grads_ok = all(d[k] <= tol_g for k in ("dq", "dk", "dv", "dg", "dbeta"))
        verdict[name] = "PASS" if d["o"] <= tol_o and grads_ok else "FAIL"
    rec["verdict"] = verdict
    fla_ok = any(k.startswith("fla_chunk_float16") for k in verdict) and \
        all(v == "PASS" for k, v in verdict.items() if k.startswith("fla_chunk"))
    rec["recommended_backend"] = "fla" if fla_ok else "torch"
    path = write_result(os.path.join(args.out, rec["run_id"] + ".json"), rec)
    print(json.dumps({"correctness": rec["correctness"], "verdict": verdict}, indent=1))
    if "speed_seconds_fwd_bwd" in rec:
        print("speed (median s per fwd+bwd):")
        for k, v in rec["speed_seconds_fwd_bwd"].items():
            print(f"  {k:28s} {v if isinstance(v, str) else f'{v:.4f}'}")
    print(f"recommended GDN backend: {rec['recommended_backend']}   (written to {path})")


if __name__ == "__main__":
    main()
