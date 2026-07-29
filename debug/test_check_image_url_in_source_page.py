import unittest

from debug.check_image_url_in_source_page import _match_summary, _url_match_report


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

    def test_match_summary_reports_per_url_field_counts(self) -> None:
        summary = _match_summary(
            [
                {
                    "image_url": "https://oss.example/cache.png",
                    "original_url": "https://origin.example/a.jpg",
                    "resolved_url": "https://origin.example/a.jpg",
                    "any_url_matched": True,
                    "original_or_resolved_url_matched": True,
                    "url_checks": {
                        "image_url": {"matched": False, "match_type": "none"},
                        "original_url": {"matched": True, "match_type": "substring"},
                        "resolved_url": {"matched": True, "match_type": "substring"},
                    },
                },
                {
                    "image_url": "https://oss.example/cache-2.png",
                    "original_url": None,
                    "resolved_url": None,
                    "any_url_matched": False,
                    "original_or_resolved_url_matched": False,
                    "url_checks": {
                        "image_url": {"matched": False, "match_type": "none"},
                    },
                },
            ]
        )

        self.assertEqual(summary["reader_success_count"], 2)
        self.assertEqual(summary["per_url_field"]["image_url"]["matched_count"], 0)
        self.assertEqual(summary["per_url_field"]["original_url"]["matched_count"], 1)
        self.assertEqual(summary["per_url_field"]["resolved_url"]["matched_rate_among_present_urls"], 1.0)


if __name__ == "__main__":
    unittest.main()
