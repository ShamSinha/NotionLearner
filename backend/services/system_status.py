from __future__ import annotations

import httpx
import psutil

from config import settings


def get_system_stats() -> dict:
    mem = psutil.virtual_memory()
    return {
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "ram_used_gb": round(mem.used / (1024**3), 1),
        "ram_available_gb": round(mem.available / (1024**3), 1),
        "ram_percent": mem.percent,
        "cpu_percent": psutil.cpu_percent(interval=0.05),
    }


def get_ollama_stats() -> dict:
    base = (settings.openai_base_url or "http://localhost:11434/v1").rstrip("/")
    root = base.removesuffix("/v1") if base.endswith("/v1") else "http://localhost:11434"
    root = root.rstrip("/")
    try:
        with httpx.Client(timeout=2.0) as client:
            tags = client.get(f"{root}/api/tags").json()
            ps = client.get(f"{root}/api/ps").json()
        models = [m.get("name") for m in tags.get("models", [])]
        loaded = []
        for m in ps.get("models", []):
            size = m.get("size") or m.get("size_vram") or 0
            loaded.append(
                {
                    "name": m.get("name") or m.get("model"),
                    "size_gb": round(size / (1024**3), 2),
                    "context_length": m.get("context_length"),
                    "expires_at": m.get("expires_at"),
                }
            )
        return {
            "ok": True,
            "root": root,
            "configured_model": settings.openai_model,
            "installed": models,
            # /api/ps = model kept warm in RAM, NOT necessarily generating
            "running": loaded,
            "loaded": loaded,
            "loaded_in_ram": len(loaded) > 0,
            "busy": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "root": root,
            "configured_model": settings.openai_model,
            "installed": [],
            "running": [],
            "loaded": [],
            "loaded_in_ram": False,
            "busy": False,
            "error": str(exc),
        }
