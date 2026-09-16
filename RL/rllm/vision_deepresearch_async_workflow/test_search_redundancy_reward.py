"""Unit tests for deterministic duplicate-search reward computation."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from vision_deepresearch_async_workflow.deepresearch_agent import MultiTurnReactAgent
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    _search_redundancy_metrics,
)
from vision_deepresearch_async_workflow.search_redundancy_reward import (
    SearchRedundancyConfig,
    canonicalize_url,
    compute_redundancy_from_embeddings,
    score_search_redundancy,
)


def _event(query: str, urls: list[str], *, round_number: int) -> dict:
    return {
        "tool_name": "t2t_search",
        "round": round_number,
        "query": query,
        "ok": True,
        "result_urls": urls,
    }


class SearchRedundancyRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SearchRedundancyConfig(mode="reward")
        self.urls_a = [
            "https://example.com/a?utm_source=x",
            "https://example.com/b",
            "https://example.com/c",
            "https://example.com/d",
            "https://example.com/e",
        ]
        self.urls_b = [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/x",
            "https://example.com/y",
            "https://example.com/z",
        ]

    def test_confirmed_repeat_costs_fixed_trajectory_penalty(self) -> None:
        trace = [
            _event("GOOGL historical prices", self.urls_a, round_number=1),
            _event("GOOGL stock price history", self.urls_b, round_number=2),
        ]
        vectors = {
            "googl historical prices": [1.0, 0.0],
            "googl stock price history": [0.94, 0.35],
        }

        result = compute_redundancy_from_embeddings(trace, vectors, self.config)

        self.assertTrue(result["scored"])
        self.assertEqual(result["confirmed_event_count"], 1)
        self.assertAlmostEqual(result["penalty"], 0.10)
        self.assertAlmostEqual(result["events"][0]["url_overlap"], 0.4)

    def test_new_year_is_exempt_even_with_same_serp(self) -> None:
        trace = [
            _event("Alphabet stock price July 24", self.urls_a, round_number=1),
            _event("Alphabet stock price July 24 2024", self.urls_a, round_number=2),
        ]
        vectors = {
            "alphabet stock price july 24": [1.0, 0.0],
            "alphabet stock price july 24 2024": [0.97, 0.24],
        }

        result = compute_redundancy_from_embeddings(trace, vectors, self.config)

        self.assertEqual(result["confirmed_event_count"], 0)
        self.assertEqual(result["hard_constraint_exempt_pair_count"], 1)
        self.assertEqual(result["penalty"], 0.0)

    def test_new_urls_prevent_penalty_despite_semantic_match(self) -> None:
        trace = [
            _event("Van Gogh birthplace", self.urls_a, round_number=1),
            _event(
                "Vincent van Gogh birthplace",
                [
                    "https://other.example/1",
                    "https://other.example/2",
                    "https://other.example/3",
                    "https://other.example/4",
                    "https://other.example/5",
                ],
                round_number=2,
            ),
        ]
        vectors = {
            "van gogh birthplace": [1.0, 0.0],
            "vincent van gogh birthplace": [0.99, 0.14],
        }

        result = compute_redundancy_from_embeddings(trace, vectors, self.config)

        self.assertEqual(result["confirmed_event_count"], 0)
        self.assertEqual(result["penalty"], 0.0)

    def test_multiple_repeats_still_cost_only_one_fixed_penalty(self) -> None:
        trace = [_event("same query", self.urls_a, round_number=index) for index in range(6)]
        vectors = {"same query": [1.0, 0.0]}

        result = compute_redundancy_from_embeddings(trace, vectors, self.config)

        self.assertEqual(result["confirmed_event_count"], 5)
        self.assertAlmostEqual(result["raw_penalty"], 0.10)
        self.assertAlmostEqual(result["penalty"], 0.10)
        self.assertEqual(_search_redundancy_metrics(result)["search_redundancy/cap_hit"], 1.0)

    def test_shadow_mode_records_events_without_changing_reward(self) -> None:
        config = SearchRedundancyConfig(mode="shadow")
        trace = [
            _event("repeat A", self.urls_a, round_number=1),
            _event("repeat B", self.urls_b, round_number=2),
        ]
        vectors = {"repeat a": [1.0, 0.0], "repeat b": [1.0, 0.0]}

        result = compute_redundancy_from_embeddings(trace, vectors, config)

        self.assertEqual(result["confirmed_event_count"], 1)
        self.assertAlmostEqual(result["raw_penalty"], 0.10)
        self.assertEqual(result["penalty"], 0.0)

    def test_training_metrics_keep_shadow_signal_and_do_not_leak_trace(self) -> None:
        config = SearchRedundancyConfig(mode="shadow")
        trace = [
            _event("repeat A", self.urls_a, round_number=1),
            _event("repeat B", self.urls_b, round_number=2),
        ]
        result = compute_redundancy_from_embeddings(
            trace,
            {"repeat a": [1.0, 0.0], "repeat b": [1.0, 0.0]},
            config,
        )

        metrics = _search_redundancy_metrics(result)

        self.assertEqual(metrics["search_redundancy/enabled"], 1.0)
        self.assertEqual(metrics["search_redundancy/mode_shadow"], 1.0)
        self.assertEqual(metrics["search_redundancy/scored"], 1.0)
        self.assertEqual(metrics["search_redundancy/confirmed_event_count"], 1.0)
        self.assertEqual(metrics["search_redundancy/raw_penalty"], 0.10)
        self.assertEqual(metrics["search_redundancy/applied_penalty"], 0.0)
        self.assertTrue(all(isinstance(value, float) for value in metrics.values()))
        self.assertNotIn("search_redundancy/events", metrics)

    def test_url_normalization_only_drops_tracking_parameters(self) -> None:
        self.assertEqual(
            canonicalize_url("HTTPS://Example.com/a?utm_source=x&page=2#section"),
            "https://example.com/a?page=2",
        )

    def test_off_mode_computes_but_never_applies_penalty(self) -> None:
        trace = [
            _event("repeat A", self.urls_a, round_number=1),
            _event("repeat B", self.urls_b, round_number=2),
        ]
        config = SearchRedundancyConfig(
            mode="off", embedding_url="http://embedding.invalid/embed"
        )
        with patch(
            "vision_deepresearch_async_workflow.search_redundancy_reward._request_embeddings",
            new=AsyncMock(return_value=([[1.0, 0.0], [1.0, 0.0]], 0.01, 0.0)),
        ) as request_embeddings:
            result = asyncio.run(score_search_redundancy(trace, config))

        request_embeddings.assert_awaited_once()
        self.assertTrue(result["scored"])
        self.assertTrue(result["embedding_request_executed"])
        self.assertTrue(result["embedding_response_valid"])
        self.assertEqual(result["confirmed_event_count"], 1)
        self.assertAlmostEqual(result["raw_penalty"], 0.10)
        self.assertEqual(result["penalty"], 0.0)
        metrics = _search_redundancy_metrics(result)
        self.assertEqual(metrics["search_redundancy/enabled"], 0.0)
        self.assertEqual(metrics["search_redundancy/observation_only"], 0.0)
        self.assertEqual(metrics["search_redundancy/reward_enabled"], 0.0)

    def test_failed_embedding_request_is_marked_as_attempted(self) -> None:
        trace = [
            _event("repeat A", self.urls_a, round_number=1),
            _event("repeat B", self.urls_b, round_number=2),
        ]
        with patch(
            "vision_deepresearch_async_workflow.search_redundancy_reward._request_embeddings",
            new=AsyncMock(side_effect=RuntimeError("embedding unavailable")),
        ):
            result = asyncio.run(
                score_search_redundancy(
                    trace,
                    SearchRedundancyConfig(
                        mode="reward",
                        embedding_url="http://embedding.invalid/embed",
                    ),
                )
            )

        self.assertEqual(result["reason"], "embedding_request_failed")
        self.assertTrue(result["embedding_request_executed"])
        self.assertFalse(result["embedding_response_valid"])

    def test_on_environment_alias_enables_reward_mode(self) -> None:
        previous = os.environ.get("SEARCH_REDUNDANCY_MODE")
        try:
            os.environ["SEARCH_REDUNDANCY_MODE"] = "on"
            self.assertEqual(SearchRedundancyConfig.from_env().mode, "reward")
        finally:
            if previous is None:
                os.environ.pop("SEARCH_REDUNDANCY_MODE", None)
            else:
                os.environ["SEARCH_REDUNDANCY_MODE"] = previous

    def test_trace_recovers_private_urls_from_resource_registry(self) -> None:
        agent = object.__new__(MultiTurnReactAgent)
        agent._search_trace = []
        agent._current_round = 3
        resources = {
            "page_a": SimpleNamespace(primary_url="https://example.com/a"),
            "page_b": SimpleNamespace(primary_url="https://example.com/b"),
        }
        context = SimpleNamespace(resolve_resource_id=resources.get)
        execution = SimpleNamespace(
            name="t2t_search",
            arguments={"query": "example query"},
            output={
                "ok": True,
                "results": [
                    {"source_page_id": "page_a"},
                    {"source_page_id": "page_b"},
                    {"source_page_id": "page_a"},
                ],
            },
        )

        agent._record_t2t_search_trace(execution, context)

        self.assertEqual(
            agent._search_trace,
            [
                {
                    "tool_name": "t2t_search",
                    "round": 3,
                    "query": "example query",
                    "ok": True,
                    "result_urls": ["https://example.com/a", "https://example.com/b"],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
