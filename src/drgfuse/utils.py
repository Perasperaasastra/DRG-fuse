"""
Small utilities shared across scripts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required to read config files.") from e


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
