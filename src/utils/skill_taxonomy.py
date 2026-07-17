from __future__ import annotations
import json
import re
from functools import lru_cache
from pathlib import Path

_TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "skill_taxonomy.json"


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, str]:
    with open(_TAXONOMY_PATH, encoding="utf-8") as f:
        taxonomy: dict[str, list[str]] = json.load(f)
    mapping: dict[str, str] = {}
    for tid, aliases in taxonomy.items():
        mapping[tid.lower()] = tid
        for alias in aliases:
            key = alias.lower().strip()
            if key not in mapping:
                mapping[key] = tid
    return mapping


def normalize(raw: str) -> str:
    """Return taxonomy ID if known, else a slugified fallback."""
    key = raw.lower().strip()
    return _alias_map().get(key, _slugify(raw))


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")
