"""Offline evaluator tests: no model, search or GPU requests."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from opensearch_vl import eval_t2i_alignment as evaluator


IDENTITY = {"alias": "fake", "revision": 1}


def verdict(label="aligned", ids=None):
    return json.dumps({"label": label, "matched_image_node_ids": [2] if ids is None else ids,
                       "reason": "query-specific explanation"})


def turn(query, number=0):
    return {"turn": number, "tool_call": json.dumps({"name": "t2i_search", "parameters": {"query": query}}),
            "tool_output": "Tool execution error: mock network failure still counts as search intent"}


def trace(case_id="a", queries=("A",), status="completed", category="image-interleave"):
    metadata = {"search_chain": {"nodes": [
        {"index": 0, "node_type": "image", "name": {"en": "input image"}},
        {"index": 1, "node_type": "text", "name": {"en": "text anchor A"}},
        {"index": 2, "node_type": "image", "name": {"en": "target image A"}},
        {"index": 3, "node_type": "text", "name": {"en": "text anchor B"}},
        {"index": 4, "node_type": "image", "name": {"en": "target image B"}},
    ], "edges": [{"source_index": 1, "target_index": 2, "relation": {"en": "portrait of A"}}]}}
    return {"case_id": case_id, "category": category, "status": status,
            "original_data": {"question": "Which attribute?", "answer": "SECRET_GOLD_ANSWER",
                              "images": ["SECRET_INPUT_IMAGE"], "source_metadata": json.dumps(metadata)},
            "final_response_text": "SECRET_AGENT_ANSWER",
            "turns": [turn(q, i) for i, q in enumerate(queries)]}


def write_trace(root, data, failure=False):
    directory = root / "failures" if failure else root
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (data["case_id"] + ("_failure.json" if failure else "_trajectory.json"))
    path.write_text(json.dumps(data))
    return path


def args(root, *extra):
    return evaluator.build_parser().parse_args([
        "--traj-dir", str(root), "--judge-model-alias", "fake", "--retry-delay", "0", *extra])


class ExtractionTests(unittest.TestCase):
    def test_responses_canonical_results_prevent_legacy_double_counting(self):
        data = trace(queries=())
        data["turns"] = [{**turn("duplicated legacy"), "tool_results": [
            {"name": "t2i_search", "arguments": {"query": "A"}, "executed": True},
            {"name": "t2i_search", "arguments": {"query": "B"}, "executed": True},
            {"name": "t2i_search", "arguments": {"query": "not executed"}, "executed": False},
        ]}]
        calls, invalid = evaluator.extract_queries(data)
        self.assertEqual([c["query"] for c in calls], ["A", "B"])
        self.assertEqual(invalid, 0)

    def test_proposals_finalization_and_invalid_queries_are_not_searches(self):
        data = trace(queries=())
        data["turns"] = [turn("A"), {"tool_call": turn("proposal")["tool_call"]},
                         {**turn("forced"), "tool_output_type": "max_turns_finalization"},
                         {**turn("bad format"), "tool_output_type": "error_format"}, turn(""), turn(42)]
        calls, invalid = evaluator.extract_queries(data)
        self.assertEqual([c["query"] for c in calls], ["A"])
        self.assertEqual(invalid, 2)

    def test_successful_retry_supersedes_stale_failure_and_other_failures_stay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, trace("a", ["stale"], "failed"), True)
            write_trace(root, trace("a", ["success"]))
            write_trace(root, trace("b", ["failed"], "failed"), True)
            cases, scan = evaluator.load_cases(root, {})
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["calls"][0]["query"], "success")
        self.assertEqual(cases[1]["status"], "failed")
        self.assertEqual(scan["superseded_files"], 1)

    def test_missing_metadata_and_placeholders_are_excluded(self):
        missing = trace("missing")
        missing["original_data"].pop("source_metadata")
        placeholder = trace("placeholder")
        meta = json.loads(placeholder["original_data"]["source_metadata"])
        meta["exclude_from_metrics"] = True
        placeholder["original_data"]["source_metadata"] = meta
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, missing)
            write_trace(root, placeholder)
            cases, _ = evaluator.load_cases(root, {})
        self.assertEqual([c["excluded_reason"] for c in cases], ["missing_construction_chain", "explicitly_ineligible"])

    def test_dataset_fallback_requires_same_question_and_preserves_saved_chain(self):
        source = trace()
        fallback = deepcopy(source["original_data"])
        missing = deepcopy(source)
        missing["original_data"].pop("source_metadata")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_trace(root, missing)
            cases, _ = evaluator.load_cases(root, {"a": fallback})
            self.assertEqual(cases[0]["reference_source"], "dataset_fallback")
            bad = {**fallback, "question": "A different question"}
            cases, _ = evaluator.load_cases(root, {"a": bad})
            self.assertEqual(cases[0]["excluded_reason"], "fallback_question_mismatch")
            path.write_text(json.dumps(source))
            cases, _ = evaluator.load_cases(root, {"a": bad})
            self.assertEqual(cases[0]["reference_source"], "trajectory")
            self.assertIsNone(cases[0]["excluded_reason"])

    def test_payload_is_blinded_and_marks_image_positions(self):
        data = trace()
        chain = evaluator.make_chain(json.loads(data["original_data"]["source_metadata"]))
        payload = evaluator.payload_for({"question": "Which?", "chain": chain}, "target A")
        serialized = json.dumps(payload)
        self.assertNotIn("SECRET", serialized)
        self.assertEqual(chain["nodes"][0]["position_type"], "initial")
        self.assertEqual(chain["nodes"][2]["position_type"], "intermediate")
        self.assertEqual(chain["nodes"][4]["position_type"], "final")
        self.assertEqual(chain["edges"][0]["relation"], "portrait of A")


class VerdictTests(unittest.TestCase):
    def test_json_and_fences_are_accepted_and_ids_deduplicated(self):
        parsed = evaluator.parse_verdict("```json\n" + verdict(ids=[2, 2, 4]) + "\n```", {0, 2, 4})
        self.assertEqual(parsed["matched_image_node_ids"], [2, 4])

    def test_invalid_labels_ids_or_reasons_are_rejected(self):
        bad = [verdict(ids=[99]), verdict(ids=[True]), verdict(ids=["2"]), verdict(ids=[]),
               verdict("off_chain", [2]), verdict("uncertain", [2]), verdict("yes", []),
               '{"label":"aligned","matched_image_node_ids":[2],"reason":""}', "prose not JSON"]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                evaluator.parse_verdict(raw, {0, 2, 4})

    def test_transport_and_format_errors_retry_then_succeed(self):
        judge = mock.Mock(side_effect=[TimeoutError("timeout"), "not JSON", verdict()])
        result = evaluator.judge_one({}, {2}, judge, 3, 0)
        self.assertEqual(result["label"], "aligned")
        self.assertEqual(result["attempts"], 3)

    def test_exhausted_errors_are_not_uncertain_or_off_chain(self):
        result = evaluator.judge_one({}, {2}, mock.Mock(return_value="bad"), 2, 0)
        self.assertEqual(result["label"], "judge_error")
        self.assertEqual(result["attempts"], 2)


class EndToEndTests(unittest.TestCase):
    def fixture(self, root):
        write_trace(root, trace("a", ["A", "A", "B"]))
        write_trace(root, trace("b", ["off"], "failed"), True)
        write_trace(root, trace("c", []))

    @staticmethod
    def judge(payload):
        query = payload["t2i_search_query"]
        if query == "off":
            return verdict("off_chain", [])
        return verdict(ids=[2] if query == "A" else [4])

    def test_counts_macro_micro_zero_call_cases_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            judge = mock.Mock(side_effect=self.judge)
            report = evaluator.evaluate(args(root), judge, IDENTITY)
            self.assertEqual(judge.call_count, 3)  # A twice is one judge request.
            counts = report["overall"]["counts"]
            rates = report["overall"]["rates"]
            self.assertEqual((counts["t2i_calls"], counts["aligned"], counts["off_chain"]), (4, 3, 1))
            self.assertEqual(rates["call_alignment_rate_confirmed"]["value"], .75)
            self.assertEqual(rates["macro_call_alignment_rate_among_searching_cases"]["value"], .5)
            self.assertAlmostEqual(rates["node_coverage_micro_confirmed"]["value"], 2/6)
            self.assertAlmostEqual(rates["node_coverage_macro_confirmed"]["value"], 1/3)
            self.assertEqual(report["by_status"]["failed"]["counts"]["t2i_calls"], 1)
            self.assertEqual(report["execution"]["judge_jobs_executed"], 3)
            unused = mock.Mock(side_effect=AssertionError("Should use cache"))
            resumed = evaluator.evaluate(args(root), unused, IDENTITY)
            unused.assert_not_called()
            self.assertEqual(resumed["execution"]["cache_hit_calls"], 4)
            cached_only = evaluator.evaluate(args(root, "--cache-only"))
            self.assertEqual(cached_only["overall"], report["overall"])
            with mock.patch.object(evaluator, "make_judge", side_effect=AssertionError("No backend")), \
                    mock.patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(evaluator.main(["--traj-dir", str(root), "--judge-model-alias", "fake", "--cache-only"]), 0)
            with (root / "t2i_alignment_eval/calls.csv").open(encoding="utf-8-sig") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 4)
            self.assertEqual(sum(r["label"] == "aligned" for r in csv_rows), 3)

    def test_changed_input_or_judge_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, trace("a", ["A"]))
            evaluator.evaluate(args(root), self.judge, IDENTITY)
            changed = trace("a", ["B"])
            write_trace(root, changed)
            judge = mock.Mock(side_effect=self.judge)
            evaluator.evaluate(args(root), judge, IDENTITY)
            self.assertEqual(judge.call_count, 1)
            judge.reset_mock()
            evaluator.evaluate(args(root), judge, {"alias": "fake", "revision": 2})
            self.assertEqual(judge.call_count, 1)

    def test_errors_remain_in_denominator_and_are_retried_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, trace("a", ["A", "B"]))
            judge = lambda p: verdict() if p["t2i_search_query"] == "A" else "bad JSON"
            report = evaluator.evaluate(args(root, "--max-attempts", "1"), judge, IDENTITY)
            self.assertEqual(report["overall"]["counts"]["judge_error"], 1)
            rates = report["overall"]["rates"]
            self.assertEqual(rates["call_alignment_rate_confirmed"]["value"], .5)
            self.assertEqual(rates["call_alignment_rate_upper_bound"]["value"], 1)
            self.assertEqual(rates["node_coverage_micro_upper_bound"]["value"], 1)
            self.assertEqual(report["overall"]["node_coverage_by_position_confirmed"]["initial"]["value"], 0)
            retry = mock.Mock(side_effect=self.judge)
            resumed = evaluator.evaluate(args(root), retry, IDENTITY)
            self.assertEqual(retry.call_count, 1)
            self.assertEqual(resumed["overall"]["counts"]["judge_error"], 0)

    def test_failed_rerun_invalidates_older_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, trace("a", ["A"]))
            evaluator.evaluate(args(root), self.judge, IDENTITY)
            evaluator.evaluate(args(root, "--rerun", "--max-attempts", "1"), lambda p: "bad", IDENTITY)
            retry = mock.Mock(side_effect=self.judge)
            evaluator.evaluate(args(root), retry, IDENTITY)
            self.assertEqual(retry.call_count, 1)

    def test_uncertain_and_initial_matches_are_separate_from_new_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, trace("a", ["initial", "uncertain"]))
            def judge(payload):
                return verdict(ids=[0]) if payload["t2i_search_query"] == "initial" else verdict("uncertain", [])
            report = evaluator.evaluate(args(root), judge, IDENTITY)
            rates = report["overall"]["rates"]
            self.assertEqual(rates["call_alignment_rate_confirmed"]["value"], .5)
            self.assertEqual(rates["new_image_call_alignment_rate_confirmed"]["value"], 0)
            self.assertEqual(report["overall"]["node_coverage_by_position_confirmed"]["initial"]["value"], 1)
            self.assertEqual(rates["node_coverage_micro_confirmed"]["value"], 0)
            self.assertEqual(rates["node_coverage_micro_upper_bound"]["value"], 1)

    def test_dry_run_has_no_judge_and_does_not_overwrite_main_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, trace())
            evaluator.evaluate(args(root), self.judge, IDENTITY)
            output = root / "t2i_alignment_eval"
            previous = (output / "report.json").read_bytes()
            judge = mock.Mock(side_effect=AssertionError("No API"))
            report = evaluator.evaluate(args(root, "--dry-run"), judge)
            judge.assert_not_called()
            self.assertEqual(report["overall"]["counts"]["pending"], 1)
            self.assertEqual((output / "report.json").read_bytes(), previous)
            plan = json.loads((output / "dry_run_plan.json").read_text())
            self.assertNotIn("SECRET", json.dumps(plan["example_judge_input"]))

    def test_input_only_and_zero_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = trace("a", [])
            meta = json.loads(data["original_data"]["source_metadata"])
            meta["search_chain"]["nodes"] = meta["search_chain"]["nodes"][:2]
            data["original_data"]["source_metadata"] = meta
            write_trace(root, data)
            with mock.patch.object(evaluator, "make_judge", side_effect=AssertionError("No calls")):
                report = evaluator.evaluate(args(root))
            self.assertEqual(report["overall"]["counts"]["input_only_cases"], 1)
            self.assertIsNone(report["overall"]["rates"]["call_alignment_rate_confirmed"]["value"])
            self.assertIsNone(report["overall"]["rates"]["node_coverage_micro_confirmed"]["value"])

    def test_truncated_cache_tail_does_not_corrupt_next_judgment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, trace("a", ["A"]))
            output = root / "t2i_alignment_eval"
            output.mkdir()
            cache = output / "judge_cache.jsonl"
            cache.write_text('{"cache_key":"interrupted')
            evaluator.evaluate(args(root), self.judge, IDENTITY)
            unused = mock.Mock(side_effect=AssertionError("Recovered result should be cached"))
            report = evaluator.evaluate(args(root), unused, IDENTITY)
            unused.assert_not_called()
            self.assertEqual(report["execution"]["cache_hit_calls"], 1)

    def test_zero_call_cases_and_categories_still_count_in_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, trace("a", ["A"], category="image-end"))
            write_trace(root, trace("b", [], category="image-interleave"))
            report = evaluator.evaluate(args(root), self.judge, IDENTITY)
            self.assertEqual(report["by_category"]["image-interleave"]["rates"]["node_coverage_micro_confirmed"]["value"], 0)
            self.assertEqual(report["overall"]["rates"]["search_trigger_rate_on_new_image_cases"]["value"], .5)
            self.assertEqual(report["overall"]["rates"]["node_coverage_micro_confirmed"]["value"], .25)

    def test_cli_returns_failure_for_exhausted_judge_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, trace())
            with mock.patch.object(evaluator, "make_judge", return_value=(lambda p: "invalid", IDENTITY)), \
                    mock.patch("sys.stdout", new=io.StringIO()):
                status = evaluator.main(["--traj-dir", str(root), "--judge-model-alias", "fake",
                                         "--max-attempts", "1", "--retry-delay", "0"])
            self.assertEqual(status, 1)
            report = json.loads((root / "t2i_alignment_eval/report.json").read_text())
            self.assertEqual(report["overall"]["counts"]["judge_error"], 1)
            self.assertEqual(report["overall"]["counts"]["off_chain"], 0)


if __name__ == "__main__":
    unittest.main()
