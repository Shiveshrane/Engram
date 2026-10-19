"""Full model: token embedding -> pre-norm residual blocks -> RMSNorm -> (tied) LM head.

Every arm is one configuration of this file (see configs/*/A*.yaml):
  A1 GDN                          mixer=gdn
  A2 GDN + Engram                 mixer=gdn, engram.enabled
  A3 GDN + frozen-table Engram    mixer=gdn, engram.enabled, engram.frozen_table
  A4 GDN + unigram Engram         mixer=gdn, engram.enabled, engram.orders=[1] (same table/projection sizes as A2)
  A5 GDN + sliding-window attn    mixer=gdn, swa.enabled
  A6 Engram-only probe            mixer=none, engram.enabled   (no sequence mixer: leak check)
  T1 Transformer                  mixer=attn
  T2 Transformer + Engram         mixer=attn, engram.enabled
  LA vanilla linear attention     mixer=linear_attn            (Phase 0 only)

Block (reference TransformerBlock order):  h = h + Engram(h, ids)      [Engram layers only]
                                           h = h + SWA(norm(h))         [A5 only]
                                           h = h + Mixer(norm(h))       [unless mixer=none]
                                           h = h + MLP(norm(h))
"""
import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engram import Engram, layer_multipliers, table_sizes_for_layers
from .gdn_block import GatedDeltaNet
from .linear_attn import LinearAttention
from .swa import SlidingWindowAttention
from .transformer_block import CausalSelfAttention

MODEL_DEFAULTS = {
    "vocab_size": None,
    "d_model": 128,
    "n_layers": 2,
    "mixer": "gdn",              # gdn | attn | linear_attn | none
    "n_heads": 4,
    "head_dim": 32,
    "mlp_hidden": 512,
    "tie_embeddings": True,
    "norm_eps": 1e-5,
    "rope_base": 10000.0,
    "init_std": 0.02,
    "match_nonembed_of": None,   # arm name; the train script then sets mlp_hidden so non-embedding params match
    "gdn": {"expand_v": 1.0, "use_short_conv": True, "conv_size": 4, "use_gate": True,
            "backend": "torch", "chunk_size": 64},
    "swa": {"enabled": False, "layers": [1], "window": 16},
    "engram": {"enabled": False, "layers": [1], "orders": [2, 3], "n_heads_per_order": 4,
               "embed_per_order": 128, "rows_per_head": 2400, "kernel_size": 4, "conv_dilation": 3,
               "use_conv": True, "frozen_table": False, "seed": 0, "pad_id": 0, "compress": False},
}


def merge(base: dict, over: Optional[dict]) -> dict:
    """Recursive dict merge (over wins). Neither input is modified."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def make_mixer(mc: dict) -> Optional[nn.Module]:
    d, h, hd = mc["d_model"], mc["n_heads"], mc["head_dim"]
    if mc["mixer"] == "gdn":
        return GatedDeltaNet(d, h, hd, norm_eps=mc["norm_eps"], **mc["gdn"])
    if mc["mixer"] == "attn":
        return CausalSelfAttention(d, h, hd, rope_base=mc["rope_base"])
    if mc["mixer"] == "linear_attn":
        return LinearAttention(d, h, hd)
    if mc["mixer"] == "none":
        return None
    raise ValueError(f"unknown mixer {mc['mixer']!r}")


class Block(nn.Module):
    def __init__(self, mc: dict, layer_idx: int, engram: Optional[Engram]):
        super().__init__()
        d = mc["d_model"]
        self.engram = engram
        swa = mc["swa"]
        if swa["enabled"] and layer_idx in swa["layers"]:
            self.swa_norm = nn.RMSNorm(d, eps=mc["norm_eps"])
            self.swa = SlidingWindowAttention(d, mc["n_heads"], mc["head_dim"], window=swa["window"],
                                              rope_base=mc["rope_base"])
        else:
            self.swa = None
        self.mixer = make_mixer(mc)
        if self.mixer is not None:
            self.mix_norm = nn.RMSNorm(d, eps=mc["norm_eps"])
        self.mlp_norm = nn.RMSNorm(d, eps=mc["norm_eps"])
        self.mlp = SwiGLU(d, mc["mlp_hidden"])

    def forward(self, h: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # h: [B, T, D], ids: [B, T]
        if self.engram is not None:
            h = h + self.engram(h, ids)
        if self.swa is not None:
            h = h + self.swa(self.swa_norm(h))
        if self.mixer is not None:
            h = h + self.mixer(self.mix_norm(h))
        return h + self.mlp(self.mlp_norm(h))


class LanguageModel(nn.Module):
    def __init__(self, mc: dict, compress_lookup: Optional[np.ndarray] = None):
        super().__init__()
        mc = merge(MODEL_DEFAULTS, mc)
        assert mc["vocab_size"], "model.vocab_size must be set (the train script sets it from the data)"
        self.cfg = mc
        d, V = mc["d_model"], mc["vocab_size"]
        self.embed = nn.Embedding(V, d)
        engrams = self._build_engrams(mc, compress_lookup)
        self.blocks = nn.ModuleList([Block(mc, i, engrams.get(i)) for i in range(mc["n_layers"])])
        self.norm = nn.RMSNorm(d, eps=mc["norm_eps"])
        self.lm_head = nn.Linear(d, V, bias=False)
        self._init_weights(mc["init_std"])
        if mc["tie_embeddings"]:
            self.lm_head.weight = self.embed.weight

    @staticmethod
    def _build_engrams(mc: dict, compress_lookup) -> Dict[int, Engram]:
        ec = mc["engram"]
        if not ec["enabled"]:
            return {}
        layers = list(ec["layers"])
        assert all(0 <= l < mc["n_layers"] for l in layers)
        if ec["compress"]:
            assert compress_lookup is not None, "engram.compress=True needs the tokenizer compression lookup"
            hash_vocab = int(np.max(compress_lookup)) + 1
        else:
            compress_lookup = None
            hash_vocab = mc["vocab_size"]
        primes = table_sizes_for_layers(layers, len(ec["orders"]), ec["rows_per_head"], ec["n_heads_per_order"])
        out = {}
        for layer_id in layers:
            out[layer_id] = Engram(
                d_model=mc["d_model"], orders=ec["orders"], primes=primes[layer_id],
                multipliers=layer_multipliers(layer_id, ec["seed"], max(ec["orders"]), hash_vocab),
                embed_per_order=ec["embed_per_order"], n_heads_per_order=ec["n_heads_per_order"],
                pad_id=ec["pad_id"], kernel_size=ec["kernel_size"], conv_dilation=ec["conv_dilation"],
                use_conv=ec["use_conv"], frozen_table=ec["frozen_table"], compress_lookup=compress_lookup)
        return out

    def _init_weights(self, std: float):
        # Backbone Linear and token-embedding weights ~ N(0, std). Engram modules keep PyTorch default init,
        # as in the reference demo (nn.Embedding ~ N(0, 1) for the tables). Recorded in DEVIATIONS.md.
        for name, m in self.named_modules():
            if ".engram" in f".{name}":
                continue
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)                       # [B, T, D]
        for blk in self.blocks:
            h = blk(h, ids)
        return self.norm(h)

    def forward(self, ids: torch.Tensor, targets: Optional[torch.Tensor] = None,
                kinds: Optional[torch.Tensor] = None, filler_weight: float = 1.0):
        """Without targets: full logits [B, T, vocab].

        With targets and kinds ([B, T] each), it returns a dict with the training loss, where
        kinds = 1 marks main loss positions (MQAR answers, or every LM token), 2 marks MQAR filler
        next-token positions, and 0 marks no loss.
            loss = mean CE(kind 1) + filler_weight * mean CE(kind 2)
        The two terms are averaged separately, so the strength of the recall signal does not depend on the
        filler length F. Logits are computed only at positions with kind > 0.
        """
        h = self.hidden(ids)
        if targets is None:
            return self.lm_head(h)
        sel = kinds > 0
        logits = self.lm_head(h[sel]).float()                  # [M, vocab]
        tgt, kind = targets[sel], kinds[sel]
        ce = F.cross_entropy(logits, tgt, reduction="none")    # [M]
        main = kind == 1
        filler = kind == 2
        loss_main = ce[main].mean() if main.any() else ce.sum() * 0.0
        loss_filler = ce[filler].mean() if filler.any() else ce.sum() * 0.0
        loss = loss_main + filler_weight * loss_filler
        with torch.no_grad():
            correct = (logits[main].argmax(-1) == tgt[main]).sum()
        return {"loss": loss, "loss_main": loss_main.detach(), "loss_filler": loss_filler.detach(),
                "correct_main": correct, "n_main": main.sum(), "n_filler": filler.sum()}


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

def engram_modules(model: nn.Module) -> List[Engram]:
    return [m for m in model.modules() if isinstance(m, Engram)]


def gdn_modules(model: nn.Module) -> List[GatedDeltaNet]:
    return [m for m in model.modules() if isinstance(m, GatedDeltaNet)]


def count_params(model: nn.Module) -> Dict[str, int]:
    """Embedding parameters = token embedding (+ untied LM head) + Engram tables. Everything else is non-embedding."""
    total = sum(p.numel() for p in model.parameters())          # tied weights are counted once
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    token_embedding = model.embed.weight.numel()
    lm_head = 0 if model.lm_head.weight is model.embed.weight else model.lm_head.weight.numel()
    engram_table = sum(m.table.embedding.weight.numel() for m in engram_modules(model))
    engram_all = sum(p.numel() for m in engram_modules(model) for p in m.parameters())
    embedding = token_embedding + lm_head + engram_table
    return {"total": total, "trainable": trainable, "embedding": embedding, "non_embedding": total - embedding,
            "token_embedding": token_embedding, "engram_table": engram_table,
            "engram_non_table": engram_all - engram_table}


def build_model(mc: dict, compress_lookup=None, device="cpu") -> LanguageModel:
    with torch.device(device):
        return LanguageModel(mc, compress_lookup)


def count_params_for_config(mc: dict, compress_lookup=None) -> Dict[str, int]:
    """Parameter counts without allocating memory (the model is built on the meta device)."""
    return count_params(build_model(mc, compress_lookup, device="meta"))


def match_mlp_hidden(mc: dict, target_non_embedding: int, multiple: int = 8, compress_lookup=None) -> int:
    """MLP hidden width (a multiple of `multiple`) that brings this config's non-embedding count closest to the target.
    Non-embedding params are linear in mlp_hidden (3 * d_model * n_layers per unit), so two probe builds suffice."""
    def nonemb(h):
        return count_params_for_config(merge(mc, {"mlp_hidden": h}), compress_lookup)["non_embedding"]
    h0, h1 = multiple, 2 * multiple
    n0, n1 = nonemb(h0), nonemb(h1)
    per_unit = (n1 - n0) / (h1 - h0)
    h = h0 + (target_non_embedding - n0) / per_unit
    return max(multiple, int(round(h / multiple)) * multiple)
