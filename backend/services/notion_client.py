from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from notion_client import Client

from config import settings
from services.llm_processor import LearningAnalysis


STATUS_INBOX = "Inbox"
# Notion hard-caps each rich_text content at 2000. Use a safer slice because
# some unicode sequences can trip their length check vs Python len().
NOTION_TEXT_MAX = 1800

TYPE_MAP = {
    "paper": "Paper",
    "video": "Video",
    "article": "Article",
    "webpage": "Article",
}


def _chunk_text(text: str, size: int = NOTION_TEXT_MAX) -> List[str]:
    text = text or ""
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        piece = text[i : i + size]
        # Extra guard if Notion counts differently
        while len(piece) > 2000:
            piece = piece[:-1]
        if piece:
            chunks.append(piece)
        i += len(piece) if piece else size
    return chunks


def _rich_text(text: str) -> list:
    if not text:
        return []
    return [
        {"type": "text", "text": {"content": chunk}}
        for chunk in _chunk_text(str(text))
    ]


def _paragraph_blocks(text: str, limit_chars: int = 40000) -> List[dict]:
    text = (text or "")[:limit_chars]
    blocks = []
    for chunk in _chunk_text(text):
        if not chunk.strip():
            continue
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": chunk[:2000]}}]
                },
            }
        )
    return blocks


def _bullet_list(items: list) -> list:
    blocks = []
    for item in items:
        if not item:
            continue
        # One bullet = one rich_text item max 2000
        content = str(item)[:NOTION_TEXT_MAX]
        blocks.append(
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": content[:2000]}}]
                },
            }
        )
    return blocks


def _numbered_list(items: list) -> list:
    blocks = []
    for item in items:
        if not item:
            continue
        content = str(item)[:NOTION_TEXT_MAX]
        blocks.append(
            {
                "object": "block",
                "type": "numbered_list_item",
                "numbered_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                },
            }
        )
    return blocks


def _heading_3(text: str) -> dict:
    return {
        "object": "block",
        "type": "heading_3",
        "heading_3": {"rich_text": _rich_text(text)},
    }


def _section(blocks: List[dict], title: str, items: list, numbered: bool = False) -> None:
    values = [item for item in items if item]
    if not values:
        return
    blocks.append(_heading_3(title))
    blocks.extend(_numbered_list(values) if numbered else _bullet_list(values))


def get_client() -> Client:
    return Client(auth=settings.notion_api_key)


@lru_cache(maxsize=1)
def _resolve_data_source() -> tuple:
    client = get_client()
    db = client.databases.retrieve(database_id=settings.notion_database_id)
    sources = db.get("data_sources") or []
    if not sources:
        raise RuntimeError("No data sources found on this Notion database.")
    data_source_id = sources[0]["id"]
    ds = client.data_sources.retrieve(data_source_id=data_source_id)
    props = ds.get("properties") or {}
    title_name = "Name"
    for name, meta in props.items():
        if meta.get("type") == "title":
            title_name = name
            break
    return data_source_id, title_name, set(props.keys())


def _append_children(client: Client, page_id: str, children: List[dict]) -> None:
    for i in range(0, len(children), 100):
        batch = children[i : i + 100]
        if batch:
            client.blocks.children.append(block_id=page_id, children=batch)


def create_learning_item(
    title: str,
    url: str,
    resource_type: str,
    transcript: str = "",
    analysis: Optional[LearningAnalysis] = None,
) -> str:
    client = get_client()
    data_source_id, title_prop, prop_names = _resolve_data_source()
    notion_type = TYPE_MAP.get(resource_type, "Article")

    properties = {
        title_prop: {"title": [{"text": {"content": title[:2000]}}]},
        "URL": {"url": url},
        "Type": {"select": {"name": notion_type}},
        "Status": {"select": {"name": STATUS_INBOX}},
    }
    if "Transcript" in prop_names and transcript:
        properties["Transcript"] = {
            "rich_text": [{"type": "text", "text": {"content": transcript[:NOTION_TEXT_MAX]}}]
        }
    if analysis:
        _apply_analysis_properties(properties, analysis, prop_names)

    page = client.pages.create(
        parent={"type": "data_source_id", "data_source_id": data_source_id},
        properties=properties,
    )
    page_id = page["id"]

    children: List[dict] = []
    if transcript:
        children.append(
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": _rich_text("Transcript / Extracted content")},
            }
        )
        children.extend(_paragraph_blocks(transcript))
    if analysis:
        children.extend(_analysis_blocks(analysis))
    if children:
        _append_children(client, page_id, children)
    return page_id


def find_learning_item_by_url(url: str) -> Optional[str]:
    """Return an existing page for the exact source URL, if one exists."""
    client = get_client()
    data_source_id, _, prop_names = _resolve_data_source()
    if "URL" not in prop_names:
        return None
    response = client.data_sources.query(
        data_source_id=data_source_id,
        filter={"property": "URL", "url": {"equals": url}},
        page_size=1,
    )
    results = response.get("results") or []
    return results[0].get("id") if results else None


def refresh_learning_item_source(
    page_id: str,
    title: str,
    resource_type: str,
    transcript: str,
) -> None:
    """Refresh source properties without appending a duplicate transcript body."""
    client = get_client()
    _, title_prop, prop_names = _resolve_data_source()
    properties = {
        title_prop: {"title": [{"text": {"content": title[:2000]}}]},
    }
    if "Type" in prop_names:
        properties["Type"] = {"select": {"name": TYPE_MAP.get(resource_type, "Article")}}
    if "Transcript" in prop_names and transcript:
        properties["Transcript"] = {
            "rich_text": [{"type": "text", "text": {"content": transcript[:NOTION_TEXT_MAX]}}]
        }
    client.pages.update(page_id=page_id, properties=properties)


def update_learning_item(page_id: str, analysis: LearningAnalysis) -> None:
    client = get_client()
    _, _, prop_names = _resolve_data_source()
    properties: dict = {}
    _apply_analysis_properties(properties, analysis, prop_names)
    if properties:
        client.pages.update(page_id=page_id, properties=properties)
    label = analysis.mode.replace("_", " ").title()
    children = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": _rich_text(f"AI Analysis · {label}")},
        }
    ]
    children.extend(_analysis_blocks(analysis))
    _append_children(client, page_id, children)


def _apply_analysis_properties(
    properties: dict, analysis: LearningAnalysis, prop_names: set
) -> None:
    if "Course" in prop_names:
        properties["Course"] = {"select": {"name": analysis.category[:100]}}
    if analysis.domain and "Domain" in prop_names:
        properties["Domain"] = {"select": {"name": analysis.domain[:100]}}
    if analysis.subtopic and "Subtopic" in prop_names:
        properties["Subtopic"] = {
            "rich_text": [{"type": "text", "text": {"content": analysis.subtopic[:NOTION_TEXT_MAX]}}]
        }
    if analysis.estimated_time_minutes and "Estimated Time" in prop_names:
        properties["Estimated Time"] = {"number": analysis.estimated_time_minutes}
    if analysis.estimated_time_minutes and "Priority" in prop_names:
        properties["Priority"] = {
            "select": {
                "name": "High" if analysis.difficulty == "advanced" else "Medium"
            }
        }
    if analysis.summary and "AI Summary" in prop_names:
        properties["AI Summary"] = {
            "rich_text": [{"type": "text", "text": {"content": analysis.summary[:NOTION_TEXT_MAX]}}]
        }
    if analysis.key_concepts and "Key Concepts" in prop_names:
        properties["Key Concepts"] = {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": ", ".join(analysis.key_concepts)[:NOTION_TEXT_MAX]},
                }
            ]
        }
    if analysis.prerequisites and "Prerequisites" in prop_names:
        properties["Prerequisites"] = {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": ", ".join(analysis.prerequisites)[:NOTION_TEXT_MAX]},
                }
            ]
        }
    if analysis.questions_to_explore and "Questions" in prop_names:
        qtext = "\n".join(f"• {q}" for q in analysis.questions_to_explore)[:NOTION_TEXT_MAX]
        properties["Questions"] = {
            "rich_text": [{"type": "text", "text": {"content": qtext}}]
        }


def _analysis_blocks(analysis: LearningAnalysis) -> List[dict]:
    blocks: List[dict] = []
    if analysis.thesis:
        blocks.append(
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": _rich_text(analysis.thesis[:NOTION_TEXT_MAX]),
                    "icon": {"type": "emoji", "emoji": "💡"},
                },
            }
        )
    if analysis.summary:
        blocks.append(_heading_3("Summary"))
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": _rich_text(analysis.summary[:NOTION_TEXT_MAX])},
            }
        )
    if analysis.why_it_matters:
        blocks.append(_heading_3("Why It Matters"))
        blocks.extend(_paragraph_blocks(analysis.why_it_matters, limit_chars=4000))
    if analysis.mental_model:
        blocks.append(_heading_3("Mental Model"))
        blocks.extend(_paragraph_blocks(analysis.mental_model, limit_chars=4000))

    _section(blocks, "Key Takeaways", analysis.key_takeaways)
    _section(blocks, "How It Works", analysis.mechanism_steps, numbered=True)
    _section(blocks, "Examples", analysis.examples)
    _section(blocks, "Misconceptions and Pitfalls", analysis.misconceptions)
    _section(blocks, "Key Concepts", analysis.key_concepts)
    _section(blocks, "Prerequisites", analysis.prerequisites)

    if analysis.feynman_notes:
        blocks.append(_heading_3("Feynman Explanation"))
        blocks.extend(_paragraph_blocks(analysis.feynman_notes, limit_chars=12000))
    if analysis.paper:
        blocks.append(_heading_3("Paper Structure"))
        for key in (
            "problem",
            "motivation",
            "method",
            "architecture",
            "loss",
            "dataset",
            "experiments",
            "results",
            "limitations",
            "contributions",
        ):
            val = analysis.paper.get(key)
            if val:
                blocks.append(
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {
                                        "content": f"{key.title()}: {val}"[:NOTION_TEXT_MAX]
                                    },
                                }
                            ]
                        },
                    }
                )

    recall_items = []
    for item in analysis.recall_questions:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question:
            recall_items.append(f"Q: {question}\nA: {answer or 'Not stated in source'}")
    _section(blocks, "Active Recall", recall_items, numbered=True)

    if analysis.timestamp_links:
        blocks.append(_heading_3("YouTube Timestamp Links"))
        blocks.extend(_bullet_list(analysis.timestamp_links))
    elif analysis.useful_timestamps:
        blocks.append(_heading_3("Useful Timestamps"))
        blocks.extend(_bullet_list(analysis.useful_timestamps))

    _section(blocks, "Evidence From the Source", analysis.source_evidence)
    _section(blocks, "Questions to Explore", analysis.questions_to_explore)
    _section(blocks, "Uncertainties / Not Stated", analysis.uncertainties)
    _section(blocks, "Suggested Next Topics", analysis.followups)
    _section(blocks, "Related Topics", analysis.related_topics)
    return blocks


def get_page_url(page_id: str) -> str:
    return f"https://www.notion.so/{page_id.replace('-', '')}"
