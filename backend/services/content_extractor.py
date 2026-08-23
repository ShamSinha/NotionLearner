from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote, urlparse

import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from config import settings
from services.pdf_extractor import extract_pdf_text, is_pdf_url
from services.whisper_fallback import transcribe_youtube, whisper_available


@dataclass
class ExtractedContent:
    url: str
    title: str
    content: str
    resource_type: str  # paper | video | article | webpage
    selected_text: Optional[str] = None
    source: str = "page"  # page | captions | whisper | pdf


YOUTUBE_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
]


def detect_resource_type(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "video"
    if "arxiv.org" in host or is_pdf_url(url):
        return "paper"
    if url.lower().endswith(".pdf"):
        return "paper"
    return "article"


def extract_youtube_id(url: str) -> Optional[str]:
    for pattern in YOUTUBE_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def youtube_watch_url(video_id: str, seconds: int = 0) -> str:
    if seconds > 0:
        return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"
    return f"https://www.youtube.com/watch?v={video_id}"


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_timestamp_to_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0


def fetch_youtube_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    try:
        try:
            fetched = api.fetch(video_id, languages=("en", "en-US", "en-GB"))
        except NoTranscriptFound:
            transcript_list = api.list(video_id)
            transcript = next(iter(transcript_list))
            fetched = transcript.fetch()

        lines = []
        for entry in fetched:
            text = (entry.text or "").replace("\n", " ").strip()
            if not text:
                continue
            lines.append(f"[{_format_timestamp(float(entry.start or 0))}] {text}")
        if not lines:
            raise ValueError(f"Empty transcript for video {video_id}")
        return "\n".join(lines)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, StopIteration) as exc:
        raise ValueError(f"No transcript available for video {video_id}") from exc


def is_chatgpt_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "chatgpt.com" in host or "chat.openai.com" in host


def is_reddit_url(url: str) -> bool:
    host = urlparse(url).netloc.lower().split(":", 1)[0]
    return host == "reddit.com" or host.endswith(".reddit.com") or host == "redd.it"


def _clean_reddit_title(value: str) -> str:
    title = html_lib.unescape(value or "").strip()
    title = re.sub(r"\s*[:|–—-]\s*r/[^\s]+\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*[-|–—]\s*Reddit\s*$", "", title, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", title).strip()


def needs_generated_title(value: str, url: str = "") -> bool:
    title = (value or "").strip()
    if not title or title == url:
        return True
    normalized = re.sub(r"\s+", " ", title).strip().lower()
    if normalized in {
        "repost",
        "post",
        "article",
        "document",
        "page",
        "reddit",
        "reddit - dive into anything",
        "reddit – dive into anything",
        "home",
        "just a moment...",
        "sign in",
        "untitled",
    }:
        return True
    if normalized.startswith(("http://", "https://")):
        return True
    # PDF viewers commonly expose only a machine filename instead of the paper title.
    return bool(re.fullmatch(r"[^/]+\.pdf(?:\s*[-|–—].*)?", normalized))


def _useful_title(value: str, url: str = "") -> bool:
    return not needs_generated_title(value, url)


def extract_reddit_title(page_html: str, page_title: str, url: str) -> str:
    """Choose the actual Reddit post title instead of a viewer/UI label."""
    source = page_html or ""
    candidates: list[str] = []

    # New Reddit exposes the original title directly on the post web component.
    post_title = re.search(
        r"<shreddit-post\b[^>]*\bpost-title=(?:\"([^\"]+)\"|'([^']+)')",
        source,
        flags=re.IGNORECASE,
    )
    if post_title:
        candidates.append(post_title.group(1) or post_title.group(2) or "")

    for pattern in (
        r"<meta\b[^>]*(?:property|name)=[\"']og:title[\"'][^>]*content=[\"']([^\"']+)[\"']",
        r"<meta\b[^>]*content=[\"']([^\"']+)[\"'][^>]*(?:property|name)=[\"']og:title[\"']",
        r"<meta\b[^>]*(?:property|name)=[\"']twitter:title[\"'][^>]*content=[\"']([^\"']+)[\"']",
    ):
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            candidates.append(match.group(1))

    headline = re.search(r'"headline"\s*:\s*("(?:\\.|[^"\\])*")', source)
    if headline:
        try:
            candidates.append(json.loads(headline.group(1)))
        except json.JSONDecodeError:
            pass

    candidates.append(page_title)
    slug = re.search(r"/comments/[^/]+/([^/?#]+)/?", url, flags=re.IGNORECASE)
    if slug:
        candidates.append(unquote(slug.group(1)).replace("_", " ").replace("-", " "))

    for candidate in candidates:
        cleaned = _clean_reddit_title(candidate)
        if _useful_title(cleaned, url):
            return cleaned
    return page_title or url


def extract_chatgpt_text(page_html: str, selected_text: Optional[str] = None) -> str:
    """Best-effort extraction from a ChatGPT chat / share page DOM."""
    if selected_text and len(selected_text.strip()) > 80:
        return selected_text.strip()

    html = page_html or ""
    parts: list[str] = []

    # Common ChatGPT message containers
    for pattern in (
        r'data-message-author-role="(user|assistant)"[^>]*>([\s\S]*?)(?:</div>\s*){2,}',
        r'<div[^>]*class="[^"]*markdown[^"]*"[^>]*>([\s\S]*?)</div>',
    ):
        for match in re.finditer(pattern, html, flags=re.IGNORECASE):
            raw = match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(1)
            text = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
            text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+\n", "\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
            if len(text) > 40:
                parts.append(text)

    if parts:
        # de-dupe while preserving order
        seen = set()
        unique = []
        for p in parts:
            key = p[:120]
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return "\n\n---\n\n".join(unique)[:60000]

    # Fallback: strip tags from whole page (noisy but better than nothing)
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:60000]


def extract_from_url(
    url: str,
    page_title: str = "",
    page_html: str = "",
    selected_text: Optional[str] = None,
) -> ExtractedContent:
    resource_type = detect_resource_type(url)

    if resource_type == "video":
        video_id = extract_youtube_id(url)
        if not video_id:
            raise ValueError("Could not parse YouTube video ID")
        title = page_title or f"YouTube Video ({video_id})"
        source = "captions"
        try:
            content = fetch_youtube_transcript(video_id)
        except ValueError:
            if settings.enable_whisper_fallback and whisper_available():
                content = transcribe_youtube(url, settings.whisper_model)
                source = "whisper"
            else:
                content = selected_text or (
                    f"YouTube video without captions.\nURL: {url}\nTitle: {title}\n"
                    "(Install yt-dlp + mlx-whisper/faster-whisper for audio transcription.)"
                )
                source = "page"
        return ExtractedContent(
            url=url,
            title=title,
            content=content,
            resource_type="video",
            selected_text=selected_text,
            source=source,
        )

    if resource_type == "paper" or is_pdf_url(url):
        title = page_title or url
        try:
            content = extract_pdf_text(url=url)
            source = "pdf"
        except Exception as exc:
            content = selected_text or f"PDF extraction failed: {exc}\nURL: {url}\nTitle: {title}"
            source = "page"
            if page_html:
                extracted = trafilatura.extract(page_html, include_comments=False, output_format="txt")
                if extracted:
                    content = extracted
        return ExtractedContent(
            url=url,
            title=title,
            content=content[:80000],
            resource_type="paper",
            selected_text=selected_text,
            source=source,
        )

    content = ""
    title = page_title or url
    source = "page"

    if is_chatgpt_url(url):
        content = extract_chatgpt_text(page_html, selected_text)
        source = "chatgpt"
        if not title or title == url or "ChatGPT" in title:
            title = page_title or "ChatGPT conversation"
        return ExtractedContent(
            url=url,
            title=title,
            content=content or f"ChatGPT link (open to view): {url}",
            resource_type="article",
            selected_text=selected_text,
            source=source,
        )

    if page_html:
        extracted = trafilatura.extract(
            page_html,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )
        if extracted:
            content = extracted
        if is_reddit_url(url):
            title = extract_reddit_title(page_html, page_title, url)
        else:
            meta = trafilatura.extract_metadata(page_html)
            if meta and _useful_title(meta.title, url):
                title = meta.title

    if not content and selected_text:
        content = selected_text

    if not content:
        content = f"URL: {url}\nTitle: {title}"

    return ExtractedContent(
        url=url,
        title=title,
        content=content[:50000],
        resource_type=resource_type,
        selected_text=selected_text,
        source=source,
    )
