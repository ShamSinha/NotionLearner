import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("NOTION_API_KEY", "test")
os.environ.setdefault("NOTION_DATABASE_ID", "test")
os.environ.setdefault("API_SECRET", "test")

from services.llm_processor import (
    CategorizePayload,
    _chat_json,
    _clean_study_text,
    _chunks,
    _compact_notes,
    _without_timestamp_prefix,
    generate_source_title,
)


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class ChunkingTests(unittest.TestCase):
    def test_long_single_line_is_split_to_size(self):
        chunks = _chunks("x" * 9000, size=4500, overlap=200)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 4500 for chunk in chunks))

    def test_chunk_limit_preserves_start_and_end_coverage(self):
        source = "\n".join(
            f"SECTION-{index} " + ("x" * 480) for index in range(20)
        )

        chunks = _chunks(source, size=500, overlap=20, max_chunks=4)

        self.assertEqual(len(chunks), 4)
        self.assertIn("SECTION-0", chunks[0])
        self.assertIn("SECTION-19", chunks[-1])

    def test_compaction_preserves_all_details_for_small_input(self):
        encoded = _compact_notes(
            [{"key_points": ["entropy", "cross-entropy", "KL divergence"]}]
        )

        self.assertIn("entropy", encoded)
        self.assertIn("cross-entropy", encoded)
        self.assertIn("KL divergence", encoded)

    def test_non_video_timestamp_prefix_can_be_removed(self):
        self.assertEqual(
            _without_timestamp_prefix(["00:00 — Entropy measures uncertainty"]),
            ["Entropy measures uncertainty"],
        )

    def test_markdown_heading_markers_are_removed_for_notion(self):
        self.assertEqual(
            _clean_study_text("# Explanation\n\n**Self-check**\n- Question"),
            "Explanation\n\nSelf-check\n- Question",
        )


class StructuredOutputTests(unittest.TestCase):
    def test_invalid_json_is_retried_and_validated(self):
        valid = {
            "category": "Machine Learning",
            "subtopic": "Gradient descent",
            "domain": "Optimization",
            "followups": ["Momentum"],
        }
        create = MagicMock(
            side_effect=[_response("not json"), _response(json.dumps(valid))]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch("services.llm_processor._client", return_value=fake_client):
            result = _chat_json(
                "system",
                "user",
                "test-model",
                CategorizePayload,
                attempts=2,
            )

        self.assertEqual(result["category"], "Machine Learning")
        self.assertEqual(create.call_count, 2)
        response_format = create.call_args.kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(
            response_format["json_schema"]["name"], "CategorizePayload"
        )
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertIn("properties", response_format["json_schema"]["schema"])
        self.assertEqual(create.call_args.kwargs["reasoning_effort"], "none")
        retry_messages = create.call_args.kwargs["messages"]
        self.assertIn("Repair a JSON candidate", retry_messages[0]["content"])
        self.assertIn("not json", retry_messages[1]["content"])
        self.assertNotEqual(retry_messages[1]["content"], "user")

    @patch("services.llm_processor.get_categorize_model", return_value="qwen3:4b")
    @patch("services.llm_processor._chat_json", return_value={"title": "Latent World Models"})
    def test_grounded_title_uses_fast_model(self, chat_json, get_model):
        title = generate_source_title(
            "Repost",
            "A discussion of latent world models for planning.",
            "article",
            "https://reddit.com/r/ai/comments/abc/post/",
        )

        self.assertEqual(title, "Latent World Models")
        self.assertEqual(chat_json.call_args.kwargs["model"], "qwen3:4b")
        get_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
