import unittest

from utils.enhanced_reader import clean_raw_markdown


class EnhancedReaderMarkdownCleanTests(unittest.TestCase):
    def test_malformed_generic_image_does_not_leave_opener(self) -> None:
        cleaned, stats = clean_raw_markdown("before ![Image 10](")

        self.assertEqual(cleaned, "before ![Image 10]")
        self.assertEqual(stats["malformed_images_seen"], 1)
        self.assertEqual(stats["malformed_images_fixed"], 1)

    def test_truncated_image_chain_is_removed(self) -> None:
        source = "![Image 18: @davidjgraph]( 19: @antheas]( 20: @shijianjian]("

        cleaned, stats = clean_raw_markdown(source)

        self.assertEqual(cleaned, "![Image 18: @davidjgraph]")
        self.assertEqual(stats["malformed_images_seen"], 1)
        self.assertEqual(stats["malformed_images_fixed"], 1)

    def test_cleaning_stats_and_reference_section(self) -> None:
        source = (
            "正文中的 [Argentina](https://example.com/arg).\n\n"
            "## References\n\n"
            "1. [Source](https://example.com/source)\n\n"
            "## Next section\n\n"
            "继续正文。"
        )

        cleaned, stats = clean_raw_markdown(source)

        self.assertIn("正文中的 Argentina", cleaned)
        self.assertIn("## Next section", cleaned)
        self.assertNotIn("References", cleaned)
        self.assertEqual(stats["original_chars"], len(source))
        self.assertEqual(stats["cleaned_chars"], len(cleaned))
        self.assertEqual(stats["reference_sections_removed"], 1)


if __name__ == "__main__":
    unittest.main()
