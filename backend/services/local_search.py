from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

import httpx

from config import settings
from services.model_settings import get_embedding_model


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings.db"


class LocalSearch:
    """Lightweight local semantic search over saved learning items via Ollama embeddings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    category TEXT,
                    text TEXT,
                    embedding TEXT,
                    created_at REAL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(DB_PATH))

    def _root(self) -> str:
        base = (settings.openai_base_url or "http://localhost:11434/v1").rstrip("/")
        root = base.removesuffix("/v1") if base.endswith("/v1") else "http://localhost:11434"
        return root.rstrip("/")

    def embed(self, text: str) -> Optional[List[float]]:
        text = (text or "")[:4000]
        if not text.strip():
            return None
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self._root()}/api/embeddings",
                    json={"model": get_embedding_model(), "prompt": text},
                )
                if resp.status_code >= 400:
                    return None
                data = resp.json()
                return data.get("embedding")
        except Exception:
            return None

    def upsert(
        self,
        item_id: str,
        title: str,
        url: str,
        category: str,
        text: str,
    ) -> None:
        emb = self.embed(f"{title}\n{category}\n{text}")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO items (id, title, url, category, text, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, strftime('%s','now'))
                """,
                (
                    item_id,
                    title,
                    url,
                    category,
                    text[:8000],
                    json.dumps(emb) if emb else None,
                ),
            )

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(y * y for y in b)) or 1e-9
        return dot / (na * nb)

    def search(self, query: str, limit: int = 8) -> List[dict]:
        q = self.embed(query)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, url, category, text, embedding FROM items"
            ).fetchall()
        scored = []
        for item_id, title, url, category, text, emb_json in rows:
            score = 0.0
            if q and emb_json:
                try:
                    emb = json.loads(emb_json)
                    score = self._cos(q, emb)
                except Exception:
                    score = 0.0
            elif query.lower() in (title or "").lower() or query.lower() in (text or "").lower():
                score = 0.2
            if score > 0:
                scored.append(
                    {
                        "id": item_id,
                        "title": title,
                        "url": url,
                        "category": category,
                        "score": round(score, 4),
                        "snippet": (text or "")[:240],
                    }
                )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]


local_search = LocalSearch()


def stable_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
