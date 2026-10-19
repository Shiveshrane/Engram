"""Tiny configs shared by the tests (CPU-sized versions of every arm)."""
import torch

from engramdelta.data import mqar
from engramdelta.models.model import build_model, merge

TINY_DATA = {"N": 6, "F": 24, "P": 8, "placement": "a", "phrase_len": 3, "n_keys": 64, "n_values": 64,
             "n_filler": 32, "filler_weight": 1.0, "eval_size": 64}
TINY_MODEL = {"d_model": 32, "n_layers": 2, "n_heads": 2, "head_dim": 16, "mlp_hidden": 64,
              "gdn": {"chunk_size": 16},
              "engram": {"layers": [1], "n_heads_per_order": 2, "embed_per_order": 32, "rows_per_head": 101}}
ARMS = {
    "A1": {},
    "A2": {"engram": {"enabled": True}},
    "A3": {"engram": {"enabled": True, "frozen_table": True}},
    "A4": {"engram": {"enabled": True, "orders": [1], "n_heads_per_order": 4, "embed_per_order": 64}},
    "A5": {"swa": {"enabled": True, "window": 4}},
    "A6": {"mixer": "none", "engram": {"enabled": True}},
    "T1": {"mixer": "attn"},
    "T2": {"mixer": "attn", "engram": {"enabled": True}},
    "LA": {"mixer": "linear_attn"},
    "A2_noEconv": {"engram": {"enabled": True, "use_conv": False}},
    "A2_noGconv": {"engram": {"enabled": True}, "gdn": {"use_short_conv": False}},
}
SEQUENCE_MIXING_ARMS = [a for a in ARMS if a != "A6"]


def tiny_mqar(**over) -> mqar.MQARConfig:
    return mqar.MQARConfig.from_dict({**TINY_DATA, **over})


def tiny_model(arm: str, vocab: int, seed: int = 0, **over):
    torch.manual_seed(seed)
    return build_model(merge(merge(TINY_MODEL, ARMS[arm]), {"vocab_size": vocab, **over}))


def tiny_train_cfg(arm: str = "A2", steps: int = 20, seed: int = 0, **train_over) -> dict:
    return {"task": "mqar", "arm": arm, "phase": "scratch", "group": "tests", "seed": seed,
            "data": dict(TINY_DATA), "model": merge(TINY_MODEL, ARMS[arm]),
            "train": {"steps": steps, "global_batch": 8, "lr": 3e-3, "warmup": 2, "log_every": 2, "ckpt_every": 0,
                      "precision": "fp32", "device": "cpu", "eval_batch": 32, **train_over}}
