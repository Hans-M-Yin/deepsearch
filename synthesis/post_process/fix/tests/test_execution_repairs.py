"""Execution regressions exposed by v23, with no network requests."""

import unittest
from unittest.mock import Mock, patch

from synthesis.sft import tools
from synthesis.post_process.fix.schema import Candidate, Config
from synthesis.post_process.fix.search_cards import (
    compose_candidates,
    validate_composition,
)


class ExecutionRepairTests(unittest.TestCase):
    def test_timeout_retries_without_losing_exception(self):
        error = TimeoutError("search timed out")
        request = Mock(side_effect=[error, {"ok": True}])
        timings = {}
        with (
            patch.object(tools, "TOOL_TIMEOUT_RETRIES", 1),
            patch.object(tools, "tracked_sleep") as sleep,
            patch.object(tools, "_tool_retry_debug") as log,
        ):
            result = tools._run_search_with_timeout_retry(
                "t2i_search", request, timings=timings
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(request.call_count, 2)
        self.assertIs(sleep.call_args.kwargs["error"], error)
        self.assertIs(log.call_args.kwargs["error"], error)
        self.assertEqual(timings["serper_attempts"], 2)

    def test_timeout_exhaustion_preserves_original_error(self):
        error = TimeoutError("still timed out")
        request = Mock(side_effect=error)
        with (
            patch.object(tools, "TOOL_TIMEOUT_RETRIES", 1),
            patch.object(tools, "tracked_sleep") as sleep,
            patch.object(tools, "_tool_retry_debug"),
            self.assertRaises(TimeoutError) as caught,
        ):
            tools._run_search_with_timeout_retry("t2i_search", request)
        self.assertIs(caught.exception, error)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(sleep.call_count, 1)

    def test_non_timeout_is_not_retried(self):
        with patch.object(tools, "tracked_sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "bad request"):
                tools._run_search_with_timeout_retry(
                    "t2i_search", Mock(side_effect=ValueError("bad request"))
                )
            sleep.assert_not_called()

    def test_composition_repairs_typo_locally_and_records_reason(self):
        candidates = {
            "required": Candidate("required", {}, category="CONSISTENT"),
            "real_key": Candidate("real_key", {}, category="RELATED"),
        }
        model = Mock()
        model.ask.side_effect = [
            {"candidate_keys": ["required", "real_key_typo"]},
            {"candidate_keys": ["required", "real_key"]},
        ]
        selected, repairs = compose_candidates(
            {"required_keys": ["required"], "result_count": 2},
            candidates,
            model,
            Config(),
        )
        self.assertEqual([c.key for c in selected], ["required", "real_key"])
        self.assertIn("Unknown candidate keys", repairs[0]["error"])
        payload = model.ask.call_args.args[2]
        self.assertEqual(payload["allowed_candidate_keys"], list(candidates))
        self.assertEqual(payload["repair_attempt"], 1)

    def test_composition_retries_are_bounded(self):
        model = Mock()
        model.ask.return_value = {"candidate_keys": ["typo"]}
        with self.assertRaisesRegex(ValueError, "after 2 attempts.*Unknown"):
            compose_candidates(
                {"required_keys": ["good"], "result_count": 1},
                {"good": Candidate("good", {}, category="CONSISTENT")},
                model,
                Config(),
            )
        self.assertEqual(model.ask.call_count, 2)

    def test_available_negatives_cannot_be_replaced_by_filler(self):
        candidates = {
            key: Candidate(key, {}, category=category)
            for key, category in [
                ("p", "CONSISTENT"),
                ("n", "SIMILAR"),
                ("u", "UNRELATED"),
            ]
        }
        model = Mock()
        model.ask.side_effect = [
            {"candidate_keys": ["p", "u"]},
            {"candidate_keys": ["p", "n"]},
        ]
        selected, repairs = compose_candidates(
            {
                "required_keys": ["p"],
                "result_count": 2,
                "positive_limit": 2,
                "minimum_negative_count": 1,
            },
            candidates,
            model,
            Config(),
        )
        self.assertEqual([c.key for c in selected], ["p", "n"])
        self.assertEqual(len(repairs), 1)

    def test_composition_validation_distinguishes_failures(self):
        candidates = {
            "good": Candidate("good", {}, category="CONSISTENT"),
            "bad": Candidate("bad", {}, category="RELATED"),
        }
        for keys, required, count, message in [
            (["good", "good"], ["good"], 2, "Duplicate"),
            (["good"], ["good"], 2, "Expected 2"),
            (["bad"], ["good"], 1, "Missing required references"),
            (["bad"], [], 1, "lacks a verified positive"),
        ]:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_composition(
                    {"candidate_keys": keys}, candidates, required, count
                )
