import unittest

from debug.check_image_url_in_source_page import _url_match_report


class ImageUrlSourcePageMatchTests(unittest.TestCase):
    def test_reports_original_and_resolved_url_matches_separately(self) -> None:
        markdown = "![Painting](https://origin.example/images/painting.jpg)"
        report = _url_match_report(
            markdown,
            image_url="https://oss.example/cache.png",
            original_url="https://origin.example/images/painting.jpg",
            resolved_url="https://resolved.example/another-image.webp",
        )

        self.assertFalse(report["url_checks"]["image_url"]["matched"])
        self.assertTrue(report["url_checks"]["original_url"]["matched"])
        self.assertFalse(report["url_checks"]["resolved_url"]["matched"])
        self.assertTrue(report["any_url_matched"])
        self.assertTrue(report["original_or_resolved_url_matched"])


if __name__ == "__main__":
    unittest.main()
