import unittest

from services.source_url import unwrap_viewer_url


class SourceUrlTests(unittest.TestCase):
    def test_keeps_normal_https_url(self):
        url = "https://example.com/paper.pdf"
        self.assertEqual(unwrap_viewer_url(url), url)

    def test_unwraps_adobe_query_parameter(self):
        viewer = (
            "chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/viewer.html"
            "?file=https%3A%2F%2Fexample.com%2Fpapers%2Fworld-models.pdf"
        )
        self.assertEqual(
            unwrap_viewer_url(viewer),
            "https://example.com/papers/world-models.pdf",
        )

    def test_unwraps_pdf_url_stored_in_viewer_path(self):
        viewer = (
            "chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/"
            "https://example.com/paper.pdf"
        )
        self.assertEqual(unwrap_viewer_url(viewer), "https://example.com/paper.pdf")


if __name__ == "__main__":
    unittest.main()
