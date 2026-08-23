from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from config import settings
from services.model_settings import get_model_settings, list_installed_models, update_model_settings

router = APIRouter()


def verify_token(authorization: str = Header(None)) -> None:
    # Allow unauthenticated GET for UI convenience; require token for POST switch
    return


def verify_token_write(authorization: str = Header(...)) -> None:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_secret:
        raise HTTPException(status_code=401, detail="Invalid API token")


class ModelUpdateRequest(BaseModel):
    categorize_model: Optional[str] = None
    analyze_model: Optional[str] = None
    embedding_model: Optional[str] = None


@router.get("/models")
async def get_models():
    return list_installed_models()


@router.post("/models")
async def set_models(
    body: ModelUpdateRequest,
    _: None = Depends(verify_token_write),
):
    installed = set(list_installed_models().get("installed") or [])
    for name in (body.categorize_model, body.analyze_model, body.embedding_model):
        if name and installed and name not in installed:
            # Allow setting anyway if ollama list briefly empty, but warn
            if installed:
                raise HTTPException(
                    status_code=422,
                    detail=f"Model '{name}' is not installed in Ollama. Installed: {sorted(installed)}",
                )
    updated = update_model_settings(
        categorize_model=body.categorize_model,
        analyze_model=body.analyze_model,
        embedding_model=body.embedding_model,
    )
    return {"success": True, "current": updated, "installed": sorted(installed)}
