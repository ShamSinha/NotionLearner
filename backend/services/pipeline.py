from __future__ import annotations

import time
from typing import Optional

from services.content_extractor import extract_from_url
from services.job_store import jobs
from services.llm_processor import analyze_content, categorize_content
from services.local_search import local_search, stable_id
from services.notion_client import (
    create_learning_item,
    find_learning_item_by_url,
    get_page_url,
    refresh_learning_item_source,
    update_learning_item,
)


def resolve_mode(summarize: bool, mode: Optional[str], resource_hint: str = "") -> str:
    if mode in ("categorize", "summarize", "feynman", "paper"):
        return mode
    if summarize:
        return "summarize"
    return "categorize"


def run_job(
    job_id: str,
    url: str,
    title: str,
    page_html: str,
    selected_text: Optional[str],
    mode: str,
) -> None:
    timings: dict = {}
    try:
        jobs.mark_stage(job_id, "extracting", "Extracting", "Fetching transcript / PDF / page…")
        t0 = time.time()
        extracted = extract_from_url(
            url=url,
            page_title=title,
            page_html=page_html,
            selected_text=selected_text,
        )
        timings["extract"] = round(time.time() - t0, 1)

        # Auto-upgrade articles that are really papers
        effective_mode = mode
        if mode == "summarize" and extracted.resource_type == "paper":
            effective_mode = "paper"

        jobs.update(
            job_id,
            title=extracted.title,
            transcript_chars=len(extracted.content or ""),
            transcript_preview=(extracted.content or "")[:400],
            mode=effective_mode,
            message=f"Source: {extracted.source}",
        )

        jobs.mark_stage(
            job_id,
            "notion",
            "Preparing Notion",
            "Creating or updating the clean Notion page…",
        )
        t1 = time.time()
        page_id = find_learning_item_by_url(url)
        reused_page = bool(page_id)
        if page_id:
            refresh_learning_item_source(
                page_id=page_id,
                title=extracted.title,
                resource_type=extracted.resource_type,
                transcript=extracted.content or "",
            )
        else:
            page_id = create_learning_item(
                title=extracted.title,
                url=url,
                resource_type=extracted.resource_type,
                transcript=extracted.content or "",
                analysis=None,
            )
        timings["notion_save"] = round(time.time() - t1, 1)
        notion_url = get_page_url(page_id)
        jobs.update(
            job_id,
            notion_url=notion_url,
            notion_page_id=page_id,
            reused_page=reused_page,
            message="Updating existing Notion page" if reused_page else "Created Notion page",
        )

        jobs.mark_stage(job_id, "llm", "Running local LLM", f"Mode={effective_mode}…")
        t2 = time.time()
        if effective_mode == "categorize":
            analysis = categorize_content(
                extracted.title, extracted.content, extracted.resource_type, url
            )
        else:
            analysis = analyze_content(
                extracted.title,
                extracted.content,
                extracted.resource_type,
                url,
                mode=effective_mode,
            )
        timings["llm"] = round(time.time() - t2, 1)
        jobs.update(job_id, category=analysis.category)

        jobs.mark_stage(job_id, "notion", "Updating Notion", "Writing AI fields…")
        t3 = time.time()
        update_learning_item(page_id, analysis)
        timings["notion_update"] = round(time.time() - t3, 1)

        try:
            local_search.upsert(
                item_id=stable_id(url),
                title=extracted.title,
                url=url,
                category=analysis.category,
                text=analysis.summary or extracted.content[:2000],
            )
        except Exception:
            pass

        timings["total"] = round(sum(v for v in timings.values() if isinstance(v, (int, float))), 1)
        jobs.finish_ok(
            job_id,
            analysis.category,
            notion_url,
            timings,
            followups=analysis.followups,
        )
    except Exception as exc:
        jobs.finish_error(job_id, str(exc))
