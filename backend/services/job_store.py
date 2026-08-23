from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class Job:
    id: str
    url: str
    title: str
    mode: str  # categorize | summarize | feynman | paper
    status: str = "queued"
    stage: str = "Queued"
    message: str = ""
    category: Optional[str] = None
    notion_url: Optional[str] = None
    notion_page_id: Optional[str] = None
    reused_page: bool = False
    transcript_chars: int = 0
    transcript_preview: str = ""
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    timings: Dict[str, float] = field(default_factory=dict)
    followups: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        elapsed = None
        if self.started_at:
            end = self.finished_at or time.time()
            elapsed = round(end - self.started_at, 1)
        data["elapsed_seconds"] = elapsed
        data["created_at_iso"] = time.strftime("%H:%M:%S", time.localtime(self.created_at))
        return data


class JobStore:
    def __init__(self, maxlen: int = 100) -> None:
        self._lock = threading.Lock()
        self._jobs: Deque[Job] = deque(maxlen=maxlen)
        self._by_id: Dict[str, Job] = {}

    def create(self, url: str, title: str, mode: str) -> Job:
        job = Job(
            id=str(uuid.uuid4())[:8],
            url=url,
            title=title or url,
            mode=mode,
            status="queued",
            stage="Queued",
            started_at=time.time(),
        )
        with self._lock:
            self._jobs.appendleft(job)
            self._by_id[job.id] = job
        return job

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._by_id.get(job_id)
            if not job:
                return
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = time.time()

    def mark_stage(self, job_id: str, status: str, stage: str, message: str = "") -> None:
        self.update(job_id, status=status, stage=stage, message=message)

    def finish_ok(
        self,
        job_id: str,
        category: Optional[str],
        notion_url: Optional[str],
        timings: Dict[str, float],
        followups: Optional[List[str]] = None,
    ) -> None:
        self.update(
            job_id,
            status="done",
            stage="Done",
            message="Saved to Notion",
            category=category,
            notion_url=notion_url,
            timings=timings,
            followups=followups or [],
            finished_at=time.time(),
        )

    def finish_error(self, job_id: str, error: str) -> None:
        self.update(
            job_id,
            status="error",
            stage="Error",
            message=error,
            error=error,
            finished_at=time.time(),
        )

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in self._jobs]

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._by_id.get(job_id)
            return job.to_dict() if job else None

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs if j.status not in ("done", "error"))


jobs = JobStore()
