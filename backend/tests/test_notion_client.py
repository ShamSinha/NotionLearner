import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("NOTION_API_KEY", "test")
os.environ.setdefault("NOTION_DATABASE_ID", "test")
os.environ.setdefault("API_SECRET", "test")

from services.llm_processor import LearningAnalysis
from services.notion_client import (
    _apply_analysis_properties,
    _analysis_blocks,
    _summary_preview,
    create_learning_item,
    find_learning_item_by_url,
)


class NotionOutputTests(unittest.TestCase):
    def test_ai_summary_column_prefers_short_thesis(self):
        analysis = LearningAnalysis(
            summary="A much more detailed summary that belongs inside the page body.",
            category="Machine Learning",
            domain="AI",
            subtopic="World models",
            difficulty="intermediate",
            estimated_time_minutes=30,
            thesis="World models help agents predict abstract future states.",
        )
        properties = {}

        _apply_analysis_properties(properties, analysis, {"AI Summary"})

        content = properties["AI Summary"]["rich_text"][0]["text"]["content"]
        self.assertEqual(content, analysis.thesis)

    def test_ai_summary_fallback_is_capped(self):
        preview = _summary_preview("word " * 100)

        self.assertLessEqual(len(preview), 240)
        self.assertTrue(preview.endswith("…"))

    def test_raw_source_is_not_appended_by_default(self):
        client = MagicMock()
        client.pages.create.return_value = {"id": "page-123"}

        with patch("services.notion_client.get_client", return_value=client), patch(
            "services.notion_client._resolve_data_source",
            return_value=("source-1", "Name", {"Name", "URL", "Type", "Status"}),
        ), patch("services.notion_client.settings.notion_include_source_content", False):
            create_learning_item(
                title="Example",
                url="https://example.com",
                resource_type="article",
                transcript="Raw extracted source",
            )

        client.blocks.children.append.assert_not_called()

    def test_rich_learning_sections_are_rendered(self):
        analysis = LearningAnalysis(
            summary="A grounded summary.",
            category="Machine Learning",
            domain="AI",
            subtopic="Optimization",
            difficulty="intermediate",
            estimated_time_minutes=30,
            thesis="The central claim.",
            mental_model="A useful intuition.",
            key_takeaways=["Takeaway"],
            mechanism_steps=["First step"],
            misconceptions=["Wrong — corrected"],
            recall_questions=[{"question": "Why?", "answer": "Because."}],
            source_evidence=["02:10 — supporting evidence"],
        )

        blocks = _analysis_blocks(analysis)
        headings = [
            block["heading_3"]["rich_text"][0]["text"]["content"]
            for block in blocks
            if block["type"] == "heading_3"
        ]

        self.assertIn("Summary", headings)
        self.assertIn("Mental Model", headings)
        self.assertIn("Active Recall", headings)
        self.assertIn("Evidence From the Source", headings)

    def test_duplicate_lookup_uses_exact_url_filter(self):
        client = MagicMock()
        client.data_sources.query.return_value = {"results": [{"id": "page-123"}]}

        with patch("services.notion_client.get_client", return_value=client), patch(
            "services.notion_client._resolve_data_source",
            return_value=("source-1", "Name", {"Name", "URL"}),
        ):
            page_id = find_learning_item_by_url("https://example.com/lesson")

        self.assertEqual(page_id, "page-123")
        client.data_sources.query.assert_called_once_with(
            data_source_id="source-1",
            filter={
                "property": "URL",
                "url": {"equals": "https://example.com/lesson"},
            },
            page_size=1,
        )


if __name__ == "__main__":
    unittest.main()
