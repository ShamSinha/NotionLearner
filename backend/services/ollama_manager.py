from __future__ import annotations

import threading
import time
from typing import Optional

import httpx

from config import settings


class OllamaManager:
    """Track last LLM use and unload models after idle to free RAM."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_used = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def touch(self) -> None:
        with self._lock:
            self._last_used = time.time()

    def _root(self) -> str:
        base = (settings.openai_base_url or "http://localhost:11434/v1").rstrip("/")
        root = base.removesuffix("/v1") if base.endswith("/v1") else "http://localhost:11434"
        return root.rstrip("/")

    def unload_all(self) -> None:
        root = self._root()
        try:
            with httpx.Client(timeout=5.0) as client:
                ps = client.get(f"{root}/api/ps").json()
                for m in ps.get("models", []):
                    name = m.get("name") or m.get("model")
                    if not name:
                        continue
                    # keep_alive=0 unloads immediately
                    client.post(
                        f"{root}/api/generate",
                        json={"model": name, "keep_alive": 0, "prompt": ""},
                    )
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.wait(30):
            idle_for = settings.ollama_idle_unload_seconds
            if idle_for <= 0:
                continue
            with self._lock:
                idle = time.time() - self._last_used
            if idle >= idle_for:
                self.unload_all()
                with self._lock:
                    self._last_used = time.time()


ollama_manager = OllamaManager()
