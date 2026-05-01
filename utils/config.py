"""Tiny YAML config helper with attribute-style access and dotted-key overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigNode(dict):
    """Dict that also exposes keys as attributes, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(val, dict) and not isinstance(val, ConfigNode):
            val = ConfigNode(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        def _plain(v):
            if isinstance(v, dict):
                return {k: _plain(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [_plain(vv) for vv in v]
            return v
        return _plain(self)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigNode({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path) -> ConfigNode:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return _wrap(raw)


def save_config(cfg: ConfigNode | dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, ConfigNode) else cfg
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _coerce(value: str) -> Any:
    """Best-effort convert CLI string to int/float/bool/None."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def merge_overrides(cfg: ConfigNode, overrides: Iterable[str]) -> ConfigNode:
    """Apply `a.b.c=value` dotted overrides from CLI. Returns a new ConfigNode."""
    merged = _wrap(copy.deepcopy(cfg.to_dict() if isinstance(cfg, ConfigNode) else cfg))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (want key=value): {item!r}")
        key, _, raw_val = item.partition("=")
        parts = key.split(".")
        cursor = merged
        for p in parts[:-1]:
            if p not in cursor or not isinstance(cursor[p], dict):
                cursor[p] = ConfigNode()
            cursor = cursor[p]
        cursor[parts[-1]] = _coerce(raw_val)
    return merged


def resolve_path(path: str | Path, anchor: str | Path | None = None) -> Path:
    """Expand ~ / env vars; resolve relative to anchor if provided."""
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if p.is_absolute():
        return p.resolve()
    if anchor is not None:
        return (Path(anchor) / p).resolve()
    return p.resolve()
