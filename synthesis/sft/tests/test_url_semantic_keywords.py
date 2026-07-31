import unittest

from synthesis.sft.tools import _clean_url_for_keyword_prompt, _validate_url_keyword_hint


class UrlSemanticKeywordTests(unittest.TestCase):
    def test_cleanup_removes_cdn_noise_but_retains_path_words(self) -> None:
        cleaned = _clean_url_for_keyword_prompt(
            "https://media-cldnry.s-nbcnews.com/image/upload/"
            "t_nbcnews-fp-1024-512,f_auto,q_auto:best/newscms/2019_35/"
            "2989276/190829-conte-italian-politics-mc-1325.JPG"
        )
        self.assertIn("nbcnews", cleaned["tokens"])
        self.assertIn("conte", cleaned["tokens"])
        self.assertIn("italian", cleaned["tokens"])
        self.assertIn("politics", cleaned["tokens"])
        self.assertNotIn("newscms", cleaned["tokens"])
        self.assertNotIn("auto", cleaned["tokens"])

    def test_validation_drops_llm_added_words(self) -> None:
        hint = _validate_url_keyword_hint(
            "conte italian politics; giuseppe conte; 2019",
            allowed_tokens=["conte", "italian", "politics"],
        )
        self.assertEqual(hint, "conte italian politics")


if __name__ == "__main__":
    unittest.main()
