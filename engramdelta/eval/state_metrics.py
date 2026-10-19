"""State occupancy of the GDN recurrent state S_t (design §7.3, exploratory).

For each GDN layer, the state of each head at chosen positions t is S_t in R^{K x V}, the state AFTER
token t has been written. Per head:
    singular values  sigma_1 >= ... >= sigma_r                  (r = min(K, V))
    effective rank   exp( H(p) ),  p_i = sigma_i / sum_j sigma_j,  H(p) = -sum_i p_i log p_i
                     (1 when one direction holds all the energy; r when all directions are equal)
    Frobenius norm   ||S_t||_F = sqrt(sum_i sigma_i^2)
These are averaged over heads and sequences and reported per layer. The full mean spectrum is reported too.

States come from the step-by-step reference recurrence (GatedDeltaNet.capture_positions), which uses the
model's own trained projections. The model runs in fp32 without autocast.

Also here: mean write strength beta and mean Engram gate alpha per token category (pair / filler / query).
They show directly whether a model learned to stop writing clutter into its state (plan, design issue 2).
"""
from typing import Dict, List, Sequence

import torch

from ..data.mqar import MQARConfig, fixed_set, layout
from ..models.model import engram_modules, gdn_modules


def spectrum_metrics(S: torch.Tensor) -> Dict[str, torch.Tensor]:
    """S: [..., K, V] -> per-matrix effective rank, Frobenius norm, singular values [..., r]."""
    sigma = torch.linalg.svdvals(S.double())                          # [..., r], descending
    p = sigma / sigma.sum(-1, keepdim=True).clamp_min(1e-300)
    entropy = -(p * torch.log(p.clamp_min(1e-300))).sum(-1)           # p = 0 terms contribute 0
    return {"effective_rank": entropy.exp(), "fro": sigma.pow(2).sum(-1).sqrt(), "sigma": sigma}


@torch.no_grad()
def collect_states(model, ids: torch.Tensor, positions: Sequence[int]) -> List[Dict[int, torch.Tensor]]:
    """Run the model once with state capture on. Returns one {t: S_t [B, H, K, V]} dict per GDN layer, in depth order."""
    layers = gdn_modules(model)
    for m in layers:
        m.capture_positions = list(positions)
    try:
        with torch.autocast(device_type=ids.device.type, enabled=False):
            model.hidden(ids)
        return [m.captured for m in layers]
    finally:
        for m in layers:
            m.capture_positions, m.captured = None, None


@torch.no_grad()
def state_occupancy(model, ids: torch.Tensor, named_positions: Dict[str, int], batch_size: int = 32) -> Dict:
    """named_positions: e.g. {"end_of_writing": 127, "end_of_clutter": 639}.

    Returns {"layer_<i>": {name: {"effective_rank", "fro", "spectrum"}}}, means over heads and sequences.
    """
    names = list(named_positions)
    positions = [named_positions[n] for n in names]
    acc: Dict = {}
    n_total = 0
    for s in range(0, ids.shape[0], batch_size):
        per_layer = collect_states(model, ids[s:s + batch_size], positions)
        b = ids[s:s + batch_size].shape[0]
        n_total += b
        for li, states in enumerate(per_layer):
            for name, t in zip(names, positions):
                m = spectrum_metrics(states[t])                         # each [b, H] (sigma: [b, H, r])
                slot = acc.setdefault(f"layer_{li}", {}).setdefault(name, {"effective_rank": 0.0, "fro": 0.0, "spectrum": 0.0})
                slot["effective_rank"] += m["effective_rank"].mean(-1).sum().item()
                slot["fro"] += m["fro"].mean(-1).sum().item()
                slot["spectrum"] = slot["spectrum"] + m["sigma"].mean(1).sum(0)  # sum over sequences of head-mean spectra
    for layer in acc.values():
        for slot in layer.values():
            slot["effective_rank"] /= n_total
            slot["fro"] /= n_total
            slot["spectrum"] = (slot["spectrum"] / n_total).tolist()
    return acc


@torch.no_grad()
def mqar_state_occupancy(model, cfg: MQARConfig, device, n_seq: int = 256) -> Dict:
    """State occupancy on the first n_seq sequences of a cell's eval set, at the end of writing and the end of clutter."""
    lay = layout(cfg)
    ids = fixed_set(cfg, "eval")["ids"][:n_seq].to(device)
    named = {"end_of_writing": lay["end_of_writing"]}
    if lay["end_of_clutter"] != lay["end_of_writing"]:
        named["end_of_clutter"] = lay["end_of_clutter"]
    return state_occupancy(model, ids, named)


@torch.no_grad()
def token_kind_stats(model, cfg: MQARConfig, device, n_seq: int = 256) -> Dict:
    """Mean GDN beta (per layer, mean over heads) and Engram gate alpha (per Engram layer) per token category."""
    lay = layout(cfg)
    ids = fixed_set(cfg, "eval")["ids"][:n_seq].to(device)
    T = ids.shape[1]
    cat = {"pair": torch.zeros(T, dtype=torch.bool), "filler": torch.zeros(T, dtype=torch.bool),
           "query": torch.zeros(T, dtype=torch.bool)}
    cat["pair"][lay["key_pos"] + lay["val_pos"]] = True
    for start, length in lay["clutter"]:
        cat["filler"][start:start + length] = True
    cat["query"][lay["query_pos"]] = True
    gdns, engrams = gdn_modules(model), engram_modules(model)
    for m in gdns:
        m.record_beta = True
    for m in engrams:
        m.record_gate = True
    try:
        with torch.autocast(device_type=device.type, enabled=False):
            model.hidden(ids)
        out: Dict = {}
        for li, m in enumerate(gdns):
            beta = m.last_beta.mean(-1).cpu()                          # [B, T]
            out[f"beta_layer_{li}"] = {k: beta[:, v].mean().item() for k, v in cat.items() if v.any()}
        for li, m in enumerate(engrams):
            gate = m.last_gate.float().cpu()                           # [B, T]
            out[f"alpha_engram_{li}"] = {k: gate[:, v].mean().item() for k, v in cat.items() if v.any()}
        return out
    finally:
        for m in gdns:
            m.record_beta, m.last_beta = False, None
        for m in engrams:
            m.record_gate, m.last_gate = False, None
