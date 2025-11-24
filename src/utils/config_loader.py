from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML/JSON config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yml", ".yaml"}:
            return yaml.safe_load(f)
        if path.suffix == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported config extension: {path.suffix}")
