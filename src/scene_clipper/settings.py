from __future__ import annotations

import json
from pathlib import Path

from .describe import MODEL

DEFAULTS = {"google_api_key": ""}


def settings_path(catalog_root: Path) -> Path:
    return Path(catalog_root) / ".settings.json"


def normalize(values: dict) -> dict:
    settings = {**DEFAULTS, **{key: values[key] for key in DEFAULTS if key in values}}
    settings["google_api_key"] = str(settings["google_api_key"] or "").strip()
    return settings


def load_settings(path: Path) -> dict:
    path = Path(path)
    stored = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        stored = loaded if isinstance(loaded, dict) else {}
    return normalize(stored)


def save_settings(path: Path, updates: dict) -> dict:
    """Merge updates over the stored settings; omitted keys keep their value."""
    path = Path(path)
    settings = normalize({**load_settings(path), **updates})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return settings


def public_settings(settings: dict) -> dict:
    """Settings shaped for the UI — the API key itself never leaves the server."""
    return {
        "has_api_key": bool(settings["google_api_key"]),
        "api_key_hint": _hint(settings["google_api_key"]),
        "model": MODEL,
    }


def _hint(api_key: str) -> str:
    if not api_key:
        return ""
    return f"{api_key[:6]}…{api_key[-4:]}" if len(api_key) > 12 else "stored"
