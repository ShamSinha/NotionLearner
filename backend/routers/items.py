from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, HttpUrl

from config import settings
from services.job_store import jobs
from services.local_search import local_search
from services.pipeline import resolve_mode, run_job

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=1)  # one LLM job at a time on 16GB Mac


def verify_token(authorization: str = Header(...)) -> None:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_secret:
        raise HTTPException(status_code=401, detail="Invalid API token")


class AddItemRequest(BaseModel):
    url: HttpUrl
    title: str = ""
    page_html: str = ""
    selected_text: Optional[str] = None
    summarize: bool = False
    mode: Optional[str] = None  # categorize | summarize | feynman | paper
    async_mode: bool = True


class BatchItem(BaseModel):
    url: HttpUrl
    title: str = ""


class BatchRequest(BaseModel):
    items: List[BatchItem]
    mode: str = "categorize"


class AddItemResponse(BaseModel):
    success: bool
    job_id: str
    accepted: bool = True
    title: str
    mode: str
    message: str = "Job accepted — poll /api/jobs/{id}"


def _submit(job_id: str, url: str, title: str, page_html: str, selected_text: Optional[str], mode: str) -> None:
    executor.submit(run_job, job_id, url, title, page_html, selected_text, mode)


@router.post("/items", response_model=AddItemResponse)
async def add_item(
    body: AddItemRequest,
    _: None = Depends(verify_token),
) -> AddItemResponse:
    url = str(body.url)
    mode = resolve_mode(body.summarize, body.mode)
    job = jobs.create(url=url, title=body.title, mode=mode)
    jobs.mark_stage(job.id, "queued", "Queued", "Waiting for worker…")

    if body.async_mode:
        _submit(job.id, url, body.title, body.page_html, body.selected_text, mode)
        return AddItemResponse(
            success=True,
            job_id=job.id,
            accepted=True,
            title=body.title or url,
            mode=mode,
            message="Accepted. Transcript saves first; AI runs in background. Watch localhost:8000",
        )

    # Sync fallback (legacy)
    run_job(job.id, url, body.title, body.page_html, body.selected_text, mode)
    done = jobs.get(job.id) or {}
    if done.get("status") == "error":
        raise HTTPException(status_code=502, detail=done.get("error") or "Job failed")
    return AddItemResponse(
        success=True,
        job_id=job.id,
        accepted=True,
        title=done.get("title") or body.title,
        mode=mode,
        message="Completed",
    )


@router.post("/items/batch")
async def add_batch(body: BatchRequest, _: None = Depends(verify_token)):
    if not body.items:
        raise HTTPException(status_code=422, detail="No items")
    if len(body.items) > 40:
        raise HTTPException(status_code=422, detail="Max 40 tabs per batch")
    job_ids = []
    for item in body.items:
        mode = body.mode if body.mode in ("categorize", "summarize", "feynman", "paper") else "categorize"
        job = jobs.create(url=str(item.url), title=item.title, mode=mode)
        jobs.mark_stage(job.id, "queued", "Queued", "Batch queue…")
        _submit(job.id, str(item.url), item.title, "", None, mode)
        job_ids.append(job.id)
    return {"success": True, "count": len(job_ids), "job_ids": job_ids}


@router.get("/jobs")
async def list_jobs():
    return {"jobs": jobs.list_jobs(), "active": jobs.active_count()}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/search")
async def search(q: str, limit: int = 8):
    if not q.strip():
        return {"results": []}
    return {"results": local_search.search(q, limit=limit)}
