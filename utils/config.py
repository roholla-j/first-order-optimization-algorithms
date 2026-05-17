import tomllib
from pathlib import Path

_cfg = None


def _load():
    global _cfg
    root = Path(__file__).parent.parent
    with open(root / "params.toml", "rb") as f:
        _cfg = tomllib.load(f)


def get() -> dict:
    if _cfg is None:
        _load()
    return _cfg
