"""YAML configs with single inheritance and command-line overrides.

A config file may contain `inherit: base.yaml` (a path relative to the file). It is merged on top of that base.
Overrides use dotted keys, like ["data.N=64", "seed=1", "model.engram.use_conv=false"], with values parsed by YAML.
The resolved config remembers its source file (`_source`) and overrides (`_overrides`), so the train script can
build the parameter-matching reference arm from a sibling file with the same overrides.
"""
import copy
import os
from typing import Iterable

import yaml

from ..models.model import merge


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "inherit" in cfg:
        base = _load_yaml(os.path.join(os.path.dirname(path), cfg.pop("inherit")))
        cfg = merge(base, cfg)
    return cfg


def set_dotted(cfg: dict, key: str, value):
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or ():
        key, _, raw = ov.partition("=")
        if not _:
            raise ValueError(f"override must be key=value, got {ov!r}")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def load_config(path: str, overrides: Iterable[str] = ()) -> dict:
    overrides = list(overrides or ())
    cfg = apply_overrides(_load_yaml(path), overrides)
    cfg["_source"] = os.path.abspath(path)
    cfg["_overrides"] = overrides
    return cfg


def sibling_arm_config(cfg: dict, arm: str) -> dict:
    """Config of another arm in the same directory, with the same overrides (used for parameter matching)."""
    path = os.path.join(os.path.dirname(cfg["_source"]), f"{arm}.yaml")
    return load_config(path, cfg.get("_overrides", ()))
