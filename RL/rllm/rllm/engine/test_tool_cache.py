import unittest

from rllm.engine.tool_cache import EpochToolCache, SampleToolCache


class ToolCacheTests(unittest.TestCase):
    def test_search_cache_is_hard_match_after_basic_normalization(self) -> None:
        cache = SampleToolCache()
        calls = 0

        def compute() -> dict:
            nonlocal calls
            calls += 1
            return {"ok": True, "results": [{"title": "Example"}]}

        cache.get_or_compute(
            "t2t_search",
            {"query": "Donald Trump", "lang": "en", "top_k": 5},
            compute,
        )
        cache.get_or_compute(
            "t2t_search",
            {"query": "  donald   trump ", "lang": "en", "top_k": 5},
            compute,
        )
        cache.get_or_compute(
            "t2t_search",
            {"query": "Mr. Donald Trump", "lang": "en", "top_k": 5},
            compute,
        )

        self.assertEqual(calls, 2)
        self.assertEqual(cache.hits, 1)

    def test_read_url_raw_cache_ignores_goal(self) -> None:
        cache = SampleToolCache()
        calls = 0

        def compute() -> dict:
            nonlocal calls
            calls += 1
            return {"url": "https://example.com/page", "content": "full page"}

        first = cache.fetch_document("enhanced_reader", "https://EXAMPLE.com/page", compute)
        second = cache.fetch_document("enhanced_reader", "https://example.com/page", compute)

        self.assertEqual(calls, 1)
        self.assertEqual(first["content"], second["content"])

    def test_firecrawl_fallback_document_skips_reader_on_later_calls(self) -> None:
        cache = SampleToolCache()
        document = {"url": "https://example.com/page", "content": "full page"}
        cache.fetch_document(
            "firecrawl",
            "https://example.com/page",
            lambda: document,
        )

        def reader_must_not_run():
            raise AssertionError("Enhanced Reader should not run after fallback cache hit")

        reused = cache.fetch_document(
            "enhanced_reader",
            "https://example.com/page",
            reader_must_not_run,
        )
        self.assertEqual(reused, document)

    def test_epoch_cache_reuses_explicit_sample_id(self) -> None:
        cache = EpochToolCache()
        first = cache.for_sample({"sample_id": "sample-1"}, "rollout-a")
        second = cache.for_sample({"sample_id": "sample-1"}, "rollout-b")
        other = cache.for_sample({"sample_id": "sample-2"}, "rollout-c")

        self.assertIs(first, second)
        self.assertIsNot(first, other)


if __name__ == "__main__":
    unittest.main()
