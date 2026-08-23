import unittest

from services.content_extractor import (
    extract_from_url,
    extract_reddit_title,
    needs_generated_title,
)


class RedditTitleTests(unittest.TestCase):
    def test_post_component_title_wins_over_repost_label(self):
        page_html = (
            '<html><head><title>Repost</title></head><body>'
            '<shreddit-post post-title="Why world models matter"></shreddit-post>'
            "</body></html>"
        )

        self.assertEqual(
            extract_reddit_title(
                page_html,
                "Repost",
                "https://www.reddit.com/r/MachineLearning/comments/abc123/post/",
            ),
            "Why world models matter",
        )

    def test_og_title_is_cleaned(self):
        page_html = (
            '<meta property="og:title" '
            'content="A practical guide to JEPA : r/MachineLearning">'
        )

        self.assertEqual(
            extract_reddit_title(
                page_html,
                "Repost",
                "https://www.reddit.com/r/MachineLearning/comments/abc123/post/",
            ),
            "A practical guide to JEPA",
        )

    def test_reddit_extraction_uses_recovered_title(self):
        page_html = (
            '<html><head><meta property="og:title" '
            'content="Understanding latent world models : r/artificial"></head>'
            '<body><article>This post gives a sufficiently detailed explanation '
            "of latent world models and their use in planning.</article></body></html>"
        )

        extracted = extract_from_url(
            "https://www.reddit.com/r/artificial/comments/abc123/post/",
            page_title="Repost",
            page_html=page_html,
        )

        self.assertEqual(extracted.title, "Understanding latent world models")


class TitleFallbackTests(unittest.TestCase):
    def test_generic_reddit_label_needs_generated_title(self):
        self.assertTrue(needs_generated_title("Repost", "https://reddit.com/r/ai/post"))

    def test_pdf_filename_needs_generated_title(self):
        self.assertTrue(
            needs_generated_title("2405.12345.pdf", "https://example.com/2405.12345.pdf")
        )

    def test_real_title_does_not_need_generated_title(self):
        self.assertFalse(
            needs_generated_title(
                "Hierarchical World Models for Planning",
                "https://example.com/paper",
            )
        )


if __name__ == "__main__":
    unittest.main()
