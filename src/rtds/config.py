"""Load config.toml into a read-only nested mapping."""

import tomllib
from pathlib import Path
from types import MappingProxyType

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.toml"


def _freeze(obj):
    # Read-only, so no script can change a setting.
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def load_config(path=DEFAULT_PATH):
    with open(path, "rb") as f:
        return _freeze(tomllib.load(f))
