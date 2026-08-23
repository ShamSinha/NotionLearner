from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openai import OpenAI

from config import settings
from services.content_extractor import extract_youtube_id, parse_timestamp_to_seconds, youtube_watch_url
from services.model_settings import get_analyze_model, get_categorize_model
from services.ollama_manager import ollama_manager


@dataclass
class LearningAnalysis:
    summary: str
    category: str
    domain: str
    subtopic: str
    difficulty: str
    estimated_time_minutes: int
    key_concepts: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    questions_to_explore: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    useful_timestamps: List[str] = field(default_factory=list)
    timestamp_links: List[str] = field(default_factory=list)
    followups: List[str] = field(default_factory=list)
    paper: Dict[str, str] = field(default_factory=dict)
    feynman_notes: str = ""
    mode: str = "summarize"


def _client() -> OpenAI:
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    cleaned = re.sub(
        r"<\|channel\|>thought.*?<channel\|>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        elif start != -1:
            cleaned = cleaned[start:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = cleaned
        if repaired.count('"') % 2 == 1:
            repaired += '"'
        repaired += "]" * max(0, repaired.count("[") - repaired.count("]"))
        repaired += "}" * max(0, repaired.count("{") - repaired.count("}"))
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return {}


def _normalize_category(raw: Optional[str]) -> str:
    categories = settings.category_list
    if not raw:
        return "Other"
    raw_norm = raw.strip().lower()
    for cat in categories:
        if cat.lower() == raw_norm:
            return cat
    for cat in categories:
        if cat.lower() in raw_norm or raw_norm in cat.lower():
            return cat
    return "Other"


def _chunks(text: str, size: int = 4500, overlap: int = 200) -> List[str]:
    text = (text or "").strip()
    if len(text) <= size:
        return [text] if text else []
    lines = text.splitlines()
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for line in lines:
        if buf_len + len(line) + 1 > size and buf:
            chunks.append("\n".join(buf))
            # overlap: keep last few lines
            keep = []
            keep_len = 0
            for prev in reversed(buf):
                if keep_len + len(prev) > overlap:
                    break
                keep.insert(0, prev)
                keep_len += len(prev) + 1
            buf = keep
            buf_len = keep_len
        buf.append(line)
        buf_len += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks[:12]  # hard cap for Mac Mini


def _chat(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> dict:
    ollama_manager.touch()
    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={
            "think": False,
            "options": {
                "num_ctx": 8192,
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        },
    )
    ollama_manager.touch()
    return _parse_json(response.choices[0].message.content or "")


def _chat_text(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1600,
) -> str:
    ollama_manager.touch()
    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={
            "think": False,
            "options": {
                "num_ctx": 8192,
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        },
    )
    ollama_manager.touch()
    text = response.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


CATEGORIZE_SYSTEM = """Classify one learning resource. Reply with ONLY compact JSON:
{{"category":"...","subtopic":"...","domain":"...","followups":["next topic 1","next topic 2"]}}
category MUST be exactly one of:
{categories}
followups = 2-3 suggested next study topics given this material. No markdown."""


CHUNK_SYSTEM = """Extract study notes from this transcript/paper chunk. Reply ONLY JSON:
{{"key_points":["..."],"concepts":["..."],"questions":["..."],"timestamps":["mm:ss — topic"],"claims":["..."]}}
Be concise. Max 5 items per array."""


MERGE_SYSTEM = """Merge chunk notes into one learning analysis. Reply ONLY JSON:
{{
 "summary":"2-4 sentences",
 "category":"one allowed category",
 "subtopic":"short topic",
 "domain":"field",
 "difficulty":"beginner|intermediate|advanced",
 "estimated_time_minutes":60,
 "key_concepts":["..."],
 "prerequisites":["..."],
 "questions_to_explore":["..."],
 "related_topics":["..."],
 "useful_timestamps":["mm:ss — topic"],
 "followups":["what to study next"]
}}
Allowed categories:
{categories}
Prefer the closest category. Arrays max 6 items."""


PAPER_SYSTEM = """You are a research paper analyst. Reply ONLY JSON:
{{
 "summary":"2-4 sentences",
 "category":"one allowed category",
 "subtopic":"...",
 "domain":"...",
 "difficulty":"beginner|intermediate|advanced",
 "estimated_time_minutes":90,
 "key_concepts":["..."],
 "prerequisites":["..."],
 "questions_to_explore":["..."],
 "related_topics":["..."],
 "followups":["..."],
 "paper":{{
   "problem":"...",
   "motivation":"...",
   "method":"...",
   "architecture":"...",
   "loss":"...",
   "dataset":"...",
   "experiments":"...",
   "results":"...",
   "limitations":"...",
   "contributions":"..."
 }}
}}
Allowed categories:
{categories}"""


FEYNMAN_SYSTEM = """You are an expert teacher using the Feynman technique.
Write an in-depth academic-style study note in continuous prose (NO bullet points).
Include direct quotes from the source text in quotation marks where possible.
Explain like teaching a sharp undergrad: simple language, deep ideas.
Cover: core claim, why it matters, key mechanisms, subtle pitfalls, and what to study next.
Write 4-8 short paragraphs."""


def categorize_content(
    title: str,
    content: str,
    resource_type: str,
    url: str,
) -> LearningAnalysis:
    cats = "\n".join(f"- {c}" for c in settings.category_list)
    excerpt = content[:2500]
    raw = _chat(
        CATEGORIZE_SYSTEM.format(categories=cats),
        f"type={resource_type}\ntitle={title}\nurl={url}\n\n{excerpt}",
        model=get_categorize_model(),
        max_tokens=260,
    )
    category = _normalize_category(raw.get("category"))
    return LearningAnalysis(
        summary="",
        category=category,
        domain=raw.get("domain") or category,
        subtopic=raw.get("subtopic") or "",
        difficulty="intermediate",
        estimated_time_minutes=0,
        followups=raw.get("followups") or [],
        mode="categorize",
    )


def _attach_timestamp_links(url: str, timestamps: List[str]) -> List[str]:
    vid = extract_youtube_id(url)
    if not vid:
        return []
    links = []
    for item in timestamps:
        # "12:04 — topic" or "12:04 - topic"
        m = re.match(r"\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[—\-–:]\s*(.+)", item)
        if not m:
            continue
        ts, label = m.group(1), m.group(2).strip()
        sec = parse_timestamp_to_seconds(ts)
        links.append(f"{ts} — {label} → {youtube_watch_url(vid, sec)}")
    return links


def analyze_content(
    title: str,
    content: str,
    resource_type: str,
    url: str,
    mode: str = "summarize",
) -> LearningAnalysis:
    """Hierarchical chunked analysis for long transcripts; paper/feynman modes supported."""
    cats = "\n".join(f"- {c}" for c in settings.category_list)
    model = get_analyze_model()

    if mode == "paper" or resource_type == "paper":
        return _analyze_paper(title, content, url, cats, model)

    parts = _chunks(content, size=4500)
    chunk_notes = []
    for i, part in enumerate(parts, 1):
        note = _chat(
            CHUNK_SYSTEM,
            f"Chunk {i}/{len(parts)} of '{title}' ({resource_type})\n\n{part}",
            model=model,
            max_tokens=500,
        )
        if note:
            chunk_notes.append(note)

    merged_input = json.dumps(chunk_notes, ensure_ascii=False)[:12000]
    raw = _chat(
        MERGE_SYSTEM.format(categories=cats),
        f"title={title}\ntype={resource_type}\nurl={url}\n\nchunk_notes={merged_input}",
        model=model,
        max_tokens=1200,
    )

    try:
        eta = int(raw.get("estimated_time_minutes", 60) or 60)
    except (TypeError, ValueError):
        eta = 60

    timestamps = raw.get("useful_timestamps") or []
    analysis = LearningAnalysis(
        summary=raw.get("summary", "") or "",
        category=_normalize_category(raw.get("category")),
        domain=raw.get("domain") or "",
        subtopic=raw.get("subtopic") or "",
        difficulty=raw.get("difficulty", "intermediate") or "intermediate",
        estimated_time_minutes=eta,
        key_concepts=raw.get("key_concepts") or [],
        prerequisites=raw.get("prerequisites") or [],
        questions_to_explore=raw.get("questions_to_explore") or [],
        related_topics=raw.get("related_topics") or [],
        useful_timestamps=timestamps,
        timestamp_links=_attach_timestamp_links(url, timestamps),
        followups=raw.get("followups") or [],
        mode=mode,
    )

    if mode == "feynman":
        analysis.feynman_notes = _chat_text(
            FEYNMAN_SYSTEM,
            f"Title: {title}\nURL: {url}\n\nSource material:\n{content[:10000]}\n\n"
            f"Structured hints: {json.dumps({'summary': analysis.summary, 'concepts': analysis.key_concepts})}",
            model=model,
            max_tokens=1800,
        )
    return analysis


def _analyze_paper(title: str, content: str, url: str, cats: str, model: str) -> LearningAnalysis:
    # Chunk then ask for paper schema on consolidated text
    parts = _chunks(content, size=5000)
    digests = []
    for i, part in enumerate(parts[:8], 1):
        note = _chat(
            CHUNK_SYSTEM,
            f"Paper chunk {i}/{min(len(parts),8)}: {title}\n\n{part}",
            model=model,
            max_tokens=450,
        )
        digests.append(note)
    raw = _chat(
        PAPER_SYSTEM.format(categories=cats),
        f"title={title}\nurl={url}\n\nchunk_digests={json.dumps(digests)[:11000]}",
        model=model,
        max_tokens=1400,
    )
    try:
        eta = int(raw.get("estimated_time_minutes", 90) or 90)
    except (TypeError, ValueError):
        eta = 90
    paper = raw.get("paper") or {}
    if not isinstance(paper, dict):
        paper = {}
    return LearningAnalysis(
        summary=raw.get("summary", "") or "",
        category=_normalize_category(raw.get("category")),
        domain=raw.get("domain") or "",
        subtopic=raw.get("subtopic") or "",
        difficulty=raw.get("difficulty", "advanced") or "advanced",
        estimated_time_minutes=eta,
        key_concepts=raw.get("key_concepts") or [],
        prerequisites=raw.get("prerequisites") or [],
        questions_to_explore=raw.get("questions_to_explore") or [],
        related_topics=raw.get("related_topics") or [],
        followups=raw.get("followups") or [],
        paper={k: str(v) for k, v in paper.items()},
        mode="paper",
    )
