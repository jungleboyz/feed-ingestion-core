"""Generic JSON cache helpers for dedup and summary caches."""

import json
import os
from typing import Any, Set


def load_json(path: str, default: Any = None) -> Any:
    """Load a JSON file, returning *default* if it doesn't exist or is corrupt."""
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: str, data: Any) -> None:
    """Write *data* as JSON to *path*, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_set(path: str) -> Set[str]:
    """Load a set stored as ``{"seen": [...]}``."""
    data = load_json(path, {})
    return set(data.get("seen", []))


def save_set(path: str, data: Set[str]) -> None:
    """Save a set as ``{"seen": [...]}``."""
    save_json(path, {"seen": list(data)})
