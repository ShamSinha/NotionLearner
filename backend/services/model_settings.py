from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from services.system_status import get_ollama_stats

STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "model_settings.json"
_lock = threading.Lock()


def _defaults() -> Dict[str, str]:
    return {
        "categorize_model": settings.categorize_model,
        "analyze_model": settings.analyze_model,
        "embedding_model": settings.embedding_model,
    }


def _load() -> Dict[str, str]:
    data = _defaults()
    if STORE_PATH.exists():
        try:
            saved = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for key in data:
                    if saved.get(key):
                        data[key] = str(saved[key])
        except Exception:
            pass
    return data


def _save(data: Dict[str, str]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_model_settings() -> Dict[str, str]:
    with _lock:
        return _load()


def update_model_settings(
    categorize_model: Optional[str] = None,
    analyze_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Dict[str, str]:
    with _lock:
        data = _load()
        if categorize_model:
            data["categorize_model"] = categorize_model
        if analyze_model:
            data["analyze_model"] = analyze_model
        if embedding_model:
            data["embedding_model"] = embedding_model
        _save(data)
        return dict(data)


def get_categorize_model() -> str:
    return get_model_settings()["categorize_model"]


def get_analyze_model() -> str:
    return get_model_settings()["analyze_model"]


def get_embedding_model() -> str:
    return get_model_settings()["embedding_model"]


def list_installed_models() -> Dict[str, Any]:
    stats = get_ollama_stats()
    installed = stats.get("installed") or []
    embed_like = [m for m in installed if "embed" in m.lower()]
    chat_like = [m for m in installed if "embed" not in m.lower()]
    current = get_model_settings()
    return {
        "installed": installed,
        "chat_models": chat_like,
        "embedding_models": embed_like or installed,
        "current": current,
        "ollama_ok": bool(stats.get("ok")),
    }
