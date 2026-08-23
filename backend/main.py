from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import settings
from routers import items, models
from services.job_store import jobs
from services.model_settings import get_model_settings
from services.system_status import get_ollama_stats, get_system_stats

# Import to start idle-unload thread
from services import ollama_manager as _ollama_manager  # noqa: F401

app = FastAPI(
    title="NotionLearner API",
    description="Local learning inbox: transcripts, Whisper fallback, chunked LLM, Notion",
    version="0.4.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(items.router, prefix="/api")
app.include_router(models.router, prefix="/api")

DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"


@app.get("/")
async def dashboard():
    return FileResponse(DASHBOARD)


@app.get("/health")
async def health():
    return {"status": "ok", "active_jobs": jobs.active_count()}


@app.get("/api/status")
async def status():
    ollama = get_ollama_stats()
    active = jobs.active_count()
    ollama["busy"] = active > 0
    ollama["generating"] = active > 0
    current = get_model_settings()
    return {
        "status": "ok",
        "model": current["analyze_model"],
        "categorize_model": current["categorize_model"],
        "analyze_model": current["analyze_model"],
        "embedding_model": current["embedding_model"],
        "base_url": settings.openai_base_url,
        "active_jobs": active,
        "categories": settings.category_list,
        "system": get_system_stats(),
        "ollama": ollama,
        "features": {
            "async_jobs": True,
            "whisper_fallback": settings.enable_whisper_fallback,
            "chunked_summarize": True,
            "pdf_extraction": True,
            "feynman_mode": True,
            "local_search": True,
            "model_switcher": True,
            "idle_unload_seconds": settings.ollama_idle_unload_seconds,
        },
    }
