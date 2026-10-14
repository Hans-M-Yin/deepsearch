from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from synthesis.post_process.fix2.cases import prepare_cases_from_rows, write_public_cases
from synthesis.post_process.fix2.base_router import watch_base_evaluations
from synthesis.post_process.fix2.collector import build_collector_user_messages, collect_case
from synthesis.post_process.fix2.evaluation import judge_expert_records
from synthesis.post_process.fix2.filters import audit_record, quality_filter_details, summarize_filter_audit
from synthesis.post_process.fix2.io import iter_json_records, write_json
from synthesis.post_process.fix2.livevqa import prepare_livevqa_from_rows
from synthesis.post_process.fix2.mining import mine_base_failures, mine_base_hard_cases
from synthesis.post_process.fix2.publisher import BaseTrajectoryCostIndex, publish_base_failures
from synthesis.post_process.fix2.probe import prepare_evaluation_probe_from_rows
from synthesis.post_process.fix2.quality import build_quality_reviewer_messages, review_trajectory_quality
from synthesis.post_process.fix2.schema import CanonicalCase, CollectorConfig, FilterConfig, PublicCase
from synthesis.post_process.fix2.style import (
    HIDDEN_REFERENCE_OVERRIDE,
    build_fvqa_collector_base_prompt,
    build_legacy_style_examples_file,
)
from synthesis.post_process.fix2.task_queue import SqliteTaskQueue
from synthesis.post_process.fix2.trajectory import parse_trajectory
from synthesis.post_process.fix2.visual import review_blind_visual_requests
from synthesis.post_process.fix2.worker import (
    merge_atomic_teacher_results,
    merge_teacher_results,
    reconcile_teacher_results,
    run_teacher_workers,
)
from synthesis.sft.api_tools import RESPONSES_SYSTEM_PROMPT


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLq"
    "XQAAAABJRU5ErkJggg=="
)
_DATA_URL = "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")


def _action(name: str, arguments: dict) -> str:
    return f"<action>\n{json.dumps({'tool_name': name, 'arguments': arguments})}\n</action>"


def _accepted_record() -> dict:
    question = "Which animal is shown?"
    return {
        "case_id": "fvqa-1",
        "question_id": "fvqa-1",
        "question": question,
        "gold_answer": "otter",
        "input_images": [{"image_path": "/tmp/source.png"}],
        "source_metadata": {"collector_reference_mode": "hidden_reference"},
        "generation_summary": {"generation_complete": True},
        "raw_messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_path", "path": "/tmp/source.png"},
                ],
            },
            {"role": "assistant", "content": "I will find visually similar candidates.\n" + _action("t2i_search", {"query": "animal near water"})},
            {
                "role": "tool",
                "name": "t2i_search",
                "content": json.dumps({"ok": True, "results": [{"image_id": "img_a"}, {"image_id": "img_b"}]}),
            },
            {"role": "assistant", "content": "I need to inspect the first candidate rather than trust its metadata.\n" + _action("read_url", {"resource_id": "img_a", "goal": "Compare the visible animal."})},
            {"role": "tool", "name": "read_url", "content": json.dumps({"ok": True, "image_id": "opened_a"})},
            {"role": "user", "content": [{"type": "text", "text": "image_id=opened_a"}, {"type": "image_url", "image_url": {"url": _DATA_URL}}]},
            {"role": "assistant", "content": "The first candidate does not match the animal's body shape, so I reject it.\n" + _action("read_url", {"resource_id": "img_b", "goal": "Compare the visible animal."})},
            {"role": "tool", "name": "read_url", "content": json.dumps({"ok": True, "image_id": "opened_b"})},
            {"role": "user", "content": [{"type": "text", "text": "image_id=opened_b"}, {"type": "image_url", "image_url": {"url": _DATA_URL}}]},
            {"role": "assistant", "content": "The second candidate is visually consistent.\n" + _action("finish", {"answer": "otter"})},
        ],
    }


def _passing_quality_review() -> dict:
    return {
        "case_id": "fvqa-1",
        "status": "ok",
        "decision": "keep",
        "visual_verification_level": "strong",
        "evidence_sufficient": True,
        "internal_knowledge_suspected": False,
        "logical_gap": False,
        "reason_codes": [],
        "evidence": ["The opened candidate image is visibly consistent with the original."],
    }


class Fix2Tests(unittest.TestCase):
    def test_legacy_style_examples_use_nonfinal_assistant_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.json"
            output = root / "anchors.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "conversations": [
                                {"from": "human", "value": "old question"},
                                {
                                    "from": "gpt",
                                    "value": "<thinking>Need a broad clue first.</thinking><tool_call>{\"name\": \"t2t_search\", \"arguments\": {}}</tool_call>",
                                },
                                {"from": "gpt", "value": "<answer>old answer</answer>"},
                            ]
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = build_legacy_style_examples_file(source, output, max_examples=1, max_chars=1000)
            anchors = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["examples"], 1)
            self.assertIn("t2t_search", anchors["examples"][0])
            self.assertNotIn("old answer", anchors["examples"][0])

    def test_collector_prompt_excludes_label(self) -> None:
        case = PublicCase(
            case_id="x",
            question="What is in this image?",
            image_path="/tmp/image.png",
        )
        messages = build_collector_user_messages(case)
        serialized = json.dumps(messages)
        self.assertNotIn("secret otter", serialized)
        self.assertIn("What is in this image?", serialized)
        self.assertIn("No gold answer", HIDDEN_REFERENCE_OVERRIDE)

    def test_following_jsonl_ignores_only_an_inflight_final_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "details.jsonl"
            source.write_text(
                '{"case_id": "complete"}\n{"case_id": "inflight"',
                encoding="utf-8",
            )
            with self.assertRaises(json.JSONDecodeError):
                list(iter_json_records(source))
            self.assertEqual(
                list(iter_json_records(source, allow_incomplete_final_line=True)),
                [{"case_id": "complete"}],
            )

    def test_responses_collector_uses_native_tool_prompt(self) -> None:
        """The Responses worker must not inherit the manual-ReAct base prompt."""

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            image_path.write_bytes(_PNG)
            seen: dict[str, object] = {}

            def fake_build_agent_config(**kwargs: object) -> object:
                seen.update(kwargs)
                return object()

            result = SimpleNamespace(
                final_text="answer",
                messages=[],
                metadata={"generation_status": "finished", "generation_complete": True},
            )
            with (
                patch("synthesis.post_process.fix2.collector.build_agent_config", side_effect=fake_build_agent_config),
                patch("synthesis.post_process.fix2.collector.run_agent_session", return_value=result),
            ):
                collect_case(
                    PublicCase("case", "What is shown?", str(image_path)),
                    CollectorConfig(model="test", workdir=directory, api_mode="responses"),
                )

            prompt = str(seen["system_prompt"])
            self.assertIn("Tool-use tips:", prompt)
            self.assertIn(HIDDEN_REFERENCE_OVERRIDE, prompt)
            self.assertIn("do not use inner/world knowledge", prompt)
            self.assertIn("child with no world knowledge", prompt)
            self.assertIn("Avoid semantic search loops (binding)", prompt)
            self.assertIn("not merely the query wording", prompt)
            self.assertNotIn("will be provided to you separately", prompt)
            self.assertNotIn("The answer is provided to verify", prompt)
            self.assertIn("The question and image are the complete task input.", prompt)

    def test_fvqa_prompt_removes_manual_react_private_reference_assumption(self) -> None:
        from synthesis.sft.api_tools import DEFAULT_SYSTEM_PROMPT

        prompt = build_fvqa_collector_base_prompt(DEFAULT_SYSTEM_PROMPT, api_mode="manual_react")
        self.assertNotIn("will be provided to you", prompt)
        self.assertIn("No final answer, intermediate statement", prompt)

    def test_visual_provenance_and_gate_pass(self) -> None:
        record = _accepted_record()
        provenance = parse_trajectory(record)
        self.assertEqual(provenance.visual_candidate_ids, {"img_a", "img_b"})
        self.assertEqual(provenance.opened_candidate_ids, {"img_a", "img_b"})
        self.assertEqual(set(provenance.candidate_image_urls), {"img_a", "img_b"})
        decision, audit = audit_record(
            record,
            answer_review={"case_id": "fvqa-1", "is_correct": True},
            blind_visual_review={"case_id": "fvqa-1", "decision": "keep", "verdict": "CONSISTENT", "evidence": ["same visible body shape"]},
            trajectory_quality_review=_passing_quality_review(),
            config=FilterConfig(),
        )
        self.assertTrue(decision.accepted, audit)

    def test_default_filter_accepts_a_completed_30_turn_trace(self) -> None:
        record = _accepted_record()
        # The parsed trace counts assistant messages, so append harmless
        # completed reasoning turns to exercise the production default.
        record["raw_messages"] = [
            *record["raw_messages"][:-1],
            *({"role": "assistant", "content": "Evidence remains consistent."} for _ in range(26)),
            record["raw_messages"][-1],
        ]
        decision, _ = audit_record(
            record,
            answer_review={"case_id": "fvqa-1", "is_correct": True},
            blind_visual_review={"case_id": "fvqa-1", "decision": "keep", "verdict": "CONSISTENT", "evidence": ["same visible body shape"]},
            trajectory_quality_review=_passing_quality_review(),
            config=FilterConfig(),
        )
        self.assertTrue(decision.accepted)

    def test_native_function_trace_is_parsed(self) -> None:
        record = {
            "raw_messages": [
                {
                    "role": "assistant",
                    "content": "I will inspect candidate images before deciding.",
                    "tool_calls": [
                        {
                            "id": "call-search",
                            "type": "function",
                            "function": {"name": "t2i_search", "arguments": '{"query":"otter"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-search",
                    "name": "t2i_search",
                    "content": '{"ok":true,"results":[{"image_id":"candidate-1"}]}' ,
                },
                {
                    "role": "assistant",
                    "content": "I will open it for a visual comparison.",
                    "tool_calls": [
                        {
                            "id": "call-read",
                            "type": "function",
                            "function": {"name": "read_url", "arguments": '{"resource_id":"candidate-1"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-read",
                    "name": "read_url",
                    "content": '{"ok":true,"image_id":"opened-1"}',
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "image_id=opened-1"},
                        {"type": "image_url", "image_url": {"url": _DATA_URL}},
                    ],
                },
            ]
        }
        provenance = parse_trajectory(record)
        self.assertEqual([event.name for event in provenance.tool_events], ["t2i_search", "read_url"])
        self.assertEqual(provenance.opened_candidate_ids, {"candidate-1"})
        self.assertEqual(provenance.candidate_image_urls, {"candidate-1": _DATA_URL})

    def test_visual_gate_allows_one_verified_candidate_without_forced_rejection(self) -> None:
        record = _accepted_record()
        record["raw_messages"] = record["raw_messages"][:6] + record["raw_messages"][-1:]
        decision, _audit = audit_record(
            record,
            answer_review={"case_id": "fvqa-1", "is_correct": True},
            blind_visual_review={"case_id": "fvqa-1", "decision": "keep", "verdict": "CONSISTENT", "evidence": ["visible"]},
            trajectory_quality_review=_passing_quality_review(),
            config=FilterConfig(),
        )
        self.assertTrue(decision.accepted)

    def test_visual_gate_can_require_multiple_candidates_for_negative_branch(self) -> None:
        record = _accepted_record()
        record["raw_messages"] = record["raw_messages"][:6] + record["raw_messages"][-1:]
        decision, _audit = audit_record(
            record,
            answer_review={"case_id": "fvqa-1", "is_correct": True},
            blind_visual_review={"case_id": "fvqa-1", "decision": "keep", "verdict": "CONSISTENT", "evidence": ["visible"]},
            trajectory_quality_review=_passing_quality_review(),
            config=FilterConfig(
                require_visual_compare=True,
                min_opened_visual_candidates=2,
                require_candidate_rejection=True,
            ),
        )
        self.assertFalse(decision.accepted)
        self.assertIn("insufficient_opened_visual_candidates", decision.reasons)

    def test_nonvisual_trace_can_pass_with_quality_review(self) -> None:
        record = _accepted_record()
        record["raw_messages"] = [
            record["raw_messages"][0],
            {
                "role": "assistant",
                "content": "The visible sign supplies a name, so I will verify that name on an external page.\n"
                + _action("t2t_search", {"query": "name printed on visible sign"}),
            },
            {
                "role": "tool",
                "name": "t2t_search",
                "content": json.dumps({"ok": True, "results": [{"source_page_id": "p1"}]}),
            },
            {
                "role": "assistant",
                "content": "I need the page text before relying on the search result.\n"
                + _action("read_url", {"resource_id": "p1", "goal": "Verify the visible sign name."}),
            },
            {
                "role": "tool",
                "name": "read_url",
                "content": json.dumps({"ok": True, "content": "The named place is in the requested region."}),
            },
            {"role": "assistant", "content": "The inspected page closes the requested fact."},
        ]
        quality = {**_passing_quality_review(), "visual_verification_level": "weak"}
        decision, audit = audit_record(
            record,
            answer_review={"case_id": "fvqa-1", "is_correct": True},
            blind_visual_review=None,
            trajectory_quality_review=quality,
            config=FilterConfig(),
        )
        self.assertTrue(decision.accepted, audit)
        self.assertEqual(audit["metrics"]["trajectory_quality_visual_verification_level"], "weak")

    def test_quality_reviewer_is_label_blind_and_persists_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            image.write_bytes(_PNG)
            record = _accepted_record()
            record["input_images"] = [{"image_path": str(image)}]
            record["raw_messages"][0]["content"][1]["path"] = str(image)
            record["gold_answer"] = "secret otter"
            input_path = root / "raw.jsonl"
            output_path = root / "quality.jsonl"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            response = json.dumps(_passing_quality_review())
            with patch(
                "synthesis.post_process.fix2.quality.LLM_WORKER.generate",
                return_value=SimpleNamespace(content=response),
            ) as generate:
                result = review_trajectory_quality(input_path, output_path)
            self.assertEqual(result["written"], 1)
            review = next(iter_json_records(output_path))
            self.assertEqual(review["status"], "ok")
            request = generate.call_args.args[0]
            serialized = json.dumps([message.to_dict() for message in request.messages])
            self.assertNotIn("secret otter", serialized)
            self.assertIn("Which animal is shown?", serialized)
            self.assertIn("Original image", serialized)
            self.assertIn("visible_trajectory", serialized)
            self.assertEqual(build_quality_reviewer_messages(record)[0].role, "system")

    def test_quality_reviewer_parallel_workers_preserve_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            image.write_bytes(_PNG)
            first = _accepted_record()
            second = _accepted_record()
            first.update({"case_id": "first", "question_id": "first", "input_images": [{"image_path": str(image)}]})
            second.update({"case_id": "second", "question_id": "second", "input_images": [{"image_path": str(image)}]})
            for row in (first, second):
                row["raw_messages"][0]["content"][1]["path"] = str(image)
            input_path = root / "raw.jsonl"
            output_path = root / "quality.jsonl"
            input_path.write_text("\n".join(json.dumps(row) for row in (first, second)) + "\n", encoding="utf-8")
            response = json.dumps({key: value for key, value in _passing_quality_review().items() if key != "case_id"})
            with patch("synthesis.post_process.fix2.quality.LLM_WORKER.generate", return_value=SimpleNamespace(content=response)) as generate:
                result = review_trajectory_quality(input_path, output_path, max_workers=2)
            self.assertEqual(result["written"], 2)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual([row["case_id"] for row in iter_json_records(output_path)], ["first", "second"])

    def test_expert_answer_judge_parallel_workers_preserve_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = {"case_id": "first", "question_id": "first", "extracted_answer": "one"}
            second = {"case_id": "second", "question_id": "second", "extracted_answer": "two"}
            input_path = root / "raw.jsonl"
            output_path = root / "answers.jsonl"
            input_path.write_text("\n".join(json.dumps(row) for row in (first, second)) + "\n", encoding="utf-8")
            cases = {
                "first": CanonicalCase("first", "First?", "one", "/tmp/a.png"),
                "second": CanonicalCase("second", "Second?", "two", "/tmp/b.png"),
            }
            with patch("synthesis.post_process.fix2.evaluation.judge_answer", return_value={"acc": 1}) as judge:
                result = judge_expert_records(input_path, output_path, model_alias="test", private_cases=cases, max_workers=2)
            self.assertEqual(result["written"], 2)
            self.assertEqual(judge.call_count, 2)
            self.assertEqual([row["case_id"] for row in iter_json_records(output_path)], ["first", "second"])

    def test_filter_audit_summary_counts_decisions_and_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "audit.jsonl"
            audit_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "decision": "keep",
                            "reasons": [],
                            "answer_review": {"is_correct": True},
                            "trajectory_quality_review": {"status": "ok", "decision": "keep"},
                            "metrics": {"trajectory_quality_visual_verification_level": "strong"},
                        },
                        {
                            "decision": "filter",
                            "reasons": ["expert_answer_not_independently_correct", "trajectory_quality_review_filter"],
                            "answer_review": {"is_correct": False},
                            "trajectory_quality_review": {"status": "ok", "decision": "filter"},
                            "metrics": {"trajectory_quality_visual_verification_level": "weak"},
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            summary = summarize_filter_audit(audit_path)
            self.assertEqual(summary["decision"], {"keep": 1, "filter": 1})
            self.assertEqual(summary["rejection_reasons"]["expert_answer_not_independently_correct"], 1)
            self.assertEqual(summary["independent_answer_review"], {"correct": 1, "incorrect": 1})
            details = quality_filter_details(audit_path)
            self.assertEqual(len(details), 1)
            self.assertFalse(details[0]["answer_independently_correct"])
            self.assertEqual(details[0]["quality_review"]["decision"], "filter")

    def test_blind_visual_reviewer_parallel_workers_preserve_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            image.write_bytes(_PNG)
            first = _accepted_record()
            second = _accepted_record()
            first.update({"case_id": "first", "question_id": "first", "input_images": [{"image_path": str(image)}]})
            second.update({"case_id": "second", "question_id": "second", "input_images": [{"image_path": str(image)}]})
            input_path = root / "raw.jsonl"
            output_path = root / "visual.jsonl"
            input_path.write_text("\n".join(json.dumps(row) for row in (first, second)) + "\n", encoding="utf-8")
            response = json.dumps({"decision": "keep", "verdict": "CONSISTENT", "evidence": ["same visible form"]})
            with patch("synthesis.post_process.fix2.visual.LLM_WORKER.generate", return_value=SimpleNamespace(content=response)) as generate:
                result = review_blind_visual_requests(input_path, output_path, model_alias="test", max_workers=2)
            self.assertEqual(result["written"], 2)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual([row["case_id"] for row in iter_json_records(output_path)], ["first", "second"])

    def test_mining_maps_run_infer_index_case_id(self) -> None:
        case = CanonicalCase("native-id", "Q", "A", "/tmp/i.png", source_index=7)
        selected, manifest, counts = mine_base_failures(
            [case],
            [{"case_id": "case_7", "acc": 0, "model_answer": "wrong"}],
        )
        self.assertEqual([item.case_id for item in selected], ["native-id"])
        self.assertEqual(manifest[0].case_id, "native-id")
        self.assertEqual(manifest[0].source_metadata["base_run_case_id"], "case_7")
        self.assertEqual(counts["wrong"], 1)

    def test_mining_includes_correct_expensive_cases_without_changing_wrong_bucket(self) -> None:
        cases = [
            CanonicalCase("slow", "Slow Q", "Slow A", "/tmp/slow.png", source_index=3),
            CanonicalCase("wrong", "Wrong Q", "Wrong A", "/tmp/wrong.png", source_index=4),
            CanonicalCase("cheap", "Cheap Q", "Cheap A", "/tmp/cheap.png", source_index=5),
        ]
        selected, manifest, counts = mine_base_hard_cases(
            cases,
            [
                {"case_id": "case_3", "acc": 1, "model_answer": "right but slow"},
                {"case_id": "case_4", "acc": 0, "model_answer": "wrong"},
                {"case_id": "case_5", "acc": 1, "model_answer": "right and cheap"},
            ],
            base_tool_call_counts={"case_3": 6, "case_4": 2, "case_5": 5},
            min_correct_tool_calls=6,
        )
        self.assertEqual([item.case_id for item in selected], ["slow", "wrong"])
        self.assertEqual([item.reason for item in manifest], ["base_model_correct_but_tool_expensive", "base_model_incorrect"])
        self.assertEqual(manifest[0].source_metadata["selection_bucket"], "base_expensive_correct")
        self.assertEqual(manifest[0].source_metadata["base_tool_call_count"], 6)
        self.assertEqual(manifest[1].source_metadata["selection_bucket"], "base_incorrect")
        self.assertEqual(counts["correct"], 2)
        self.assertEqual(counts["expensive_correct"], 1)
        self.assertEqual(counts["below_cost_threshold"], 1)
        self.assertEqual(counts["wrong"], 1)

    def test_base_cost_index_and_publisher_keep_cost_metadata_out_of_teacher_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_dir = root / "base"
            trajectory_dir.mkdir()
            (trajectory_dir / "case_3_trajectory.json").write_text(
                json.dumps({"case_id": "case_3", "tool_call_count": 6}),
                encoding="utf-8",
            )
            index = BaseTrajectoryCostIndex(trajectory_dir)
            self.assertEqual(index.refresh(), {"discovered": 1, "indexed": 1, "malformed": 0})
            self.assertEqual(index.tool_call_counts, {"case_3": 6})
            self.assertEqual(index.refresh(), {"discovered": 0, "indexed": 0, "malformed": 0})

            queue = SqliteTaskQueue(root / "queue.sqlite")
            case = CanonicalCase("slow", "Q", "A", "/tmp/image.png", source_index=3)
            result = publish_base_failures(
                [case],
                [{"case_id": "case_3", "acc": 1, "model_answer": "right but slow"}],
                queue=queue,
                base_tool_call_counts=index.tool_call_counts,
                min_correct_tool_calls=6,
            )
            self.assertEqual(result["published"], 1)
            self.assertEqual(result["expensive_correct"], 1)
            task = queue.lease("teacher_collect", worker_id="test", lease_s=60)
            assert task is not None
            serialized = json.dumps(task.payload)
            self.assertNotIn("tool_call_count", serialized)
            self.assertNotIn("right but slow", serialized)
            self.assertNotIn("gold_answer", serialized)

    def test_prepare_and_stream_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "cases.jsonl"
            result = prepare_cases_from_rows(
                [{"id": "r1", "question": "Q?", "answer": "A", "image": [_PNG]}],
                output_jsonl=output,
                image_dir=root / "images",
                split="train",
            )
            self.assertEqual(result["written"], 1)
            record = next(iter_json_records(output))
            self.assertEqual(record["case_id"], "r1")
            self.assertTrue(Path(record["image_path"]).is_file())
            self.assertNotIn("acceptable_answers", record)
            array_path = root / "array.json"
            array_path.write_text('[{"case_id":"one"},{"case_id":"two"}]', encoding="utf-8")
            self.assertEqual([x["case_id"] for x in iter_json_records(array_path)], ["one", "two"])
            large_array = root / "large_array.json"
            large_array.write_text(json.dumps([{"case_id": "large", "text": "x" * (130 * 1024)}]), encoding="utf-8")
            self.assertEqual(next(iter_json_records(large_array))["case_id"], "large")

    def test_prepare_resumes_selected_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "cases.jsonl"
            rows = [
                {"id": f"r{index}", "question": f"Q{index}?", "answer": f"A{index}", "image": [_PNG]}
                for index in range(3)
            ]
            first = prepare_cases_from_rows(
                rows,
                output_jsonl=output,
                image_dir=root / "images",
                split="train",
                limit=1,
            )
            self.assertEqual(first["written"], 1)
            resumed = prepare_cases_from_rows(
                rows,
                output_jsonl=output,
                image_dir=root / "images",
                split="train",
                limit=2,
                resume=True,
            )
            self.assertEqual(resumed["written"], 1)
            self.assertEqual(resumed["skipped_resume"], 1)
            records = list(iter_json_records(output))
            self.assertEqual([record["case_id"] for record in records], ["r0", "r1"])
            self.assertEqual([record["source_index"] for record in records], [0, 1])

    def test_prepare_parallel_materialization_preserves_order_and_deduplicates(self) -> None:
        """Parallel image work must not change manifest ordering or resume keys."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "cases.jsonl"
            rows = [
                {"id": "first", "question": "What is this?", "answer": "A", "image": [_PNG]},
                {"id": "second", "question": "What else is this?", "answer": "B", "image": [_PNG]},
                # Same wording plus image as first: this must be removed by
                # the coordinator even if workers finish in a different order.
                {"id": "duplicate", "question": "What is this?", "answer": "A", "image": [_PNG]},
            ]
            result = prepare_cases_from_rows(
                rows,
                output_jsonl=output,
                image_dir=root / "images",
                split="train",
                materialize_workers=2,
            )
            self.assertEqual(result, {"input": 3, "written": 2, "duplicate": 1, "skipped_resume": 0})
            records = list(iter_json_records(output))
            self.assertEqual([record["case_id"] for record in records], ["first", "second"])
            self.assertEqual([record["source_index"] for record in records], [0, 1])
            self.assertTrue(all(Path(record["image_path"]).is_file() for record in records))

    def test_livevqa_direct_sampling_is_1_to_3_sha_deduplicated_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "input_images"
            images.mkdir()

            def image(name: str, marker: bytes) -> str:
                path = images / name
                # The PNG decoder recognizes the valid prefix; the distinct
                # trailing markers give every fixture an exact unique SHA256.
                path.write_bytes(_PNG + marker)
                return str(path)

            level1_image = image("level1.png", b"level1")
            level2_images = [image(f"level2-{index}.png", f"level2-{index}".encode()) for index in range(3)]
            rows = [
                {
                    "id": "l1",
                    "question": "Who is shown?",
                    "ground_truth_list": ["Alice", "A. Example"],
                    "images": [{"url": level1_image}],
                    "level": "Level 1",
                    "source": "news",
                    "topic": "person",
                },
                # Exact duplicate pixels of Level 1: it is eligible at the
                # question level but cannot survive the sole SHA256 gate.
                {
                    "id": "l2-duplicate",
                    "question": "Why is the person there?",
                    "ground_truth_list": ["An event"],
                    "images": [{"url": level1_image}],
                    "level": "Level 2",
                },
                *[
                    {
                        "id": f"l2-{index}",
                        "question": f"What multi-hop fact {index}?",
                        "ground_truth_list": [f"answer-{index}", f"alias-{index}"],
                        "images": [{"url": path}],
                        "level": 2,
                        "source": "arxiv",
                        "topic": "count",
                    }
                    for index, path in enumerate(level2_images)
                ],
            ]
            private = root / "private.jsonl"
            public = root / "public.jsonl"
            report = root / "selection.json"
            result = prepare_livevqa_from_rows(
                rows,
                output_cases=private,
                output_public_cases=public,
                selection_report=report,
                image_dir=root / "materialized",
                level1_count=1,
                seed=17,
                download_workers=2,
            )
            self.assertEqual(result["selected_level1"], 1)
            self.assertEqual(result["selected_level2"], 3)
            self.assertEqual(result["written"], 4)
            self.assertGreaterEqual(result["duplicate_image_sha256"], 1)
            private_rows = list(iter_json_records(private))
            self.assertEqual(len(private_rows), 4)
            self.assertEqual({row["source_metadata"]["livevqa_level"] for row in private_rows}, {1, 2})
            self.assertEqual(len({row["source_metadata"]["source_image_sha256"] for row in private_rows}), 4)
            self.assertTrue(all("acceptable_answers" in row for row in private_rows))
            public_rows = list(iter_json_records(public))
            self.assertEqual(len(public_rows), 4)
            self.assertTrue(all("acceptable_answers" not in row for row in public_rows))

            resumed = prepare_livevqa_from_rows(
                rows,
                output_cases=private,
                output_public_cases=public,
                selection_report=report,
                image_dir=root / "materialized",
                level1_count=1,
                seed=17,
                download_workers=2,
                resume=True,
            )
            self.assertEqual(resumed["written"], 0)
            self.assertEqual(resumed["selected_level1"], 1)
            self.assertEqual(resumed["selected_level2"], 3)

    def test_livevqa_resume_expands_a_same_seed_manifest_without_rewriting_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def image_url(tag: str) -> str:
                return "data:image/png;base64," + base64.b64encode(_PNG + tag.encode("utf-8")).decode("ascii")

            rows = [
                {
                    "id": f"l1-{index}",
                    "question": f"Which Level 1 entity {index}?",
                    "ground_truth_list": [f"l1-answer-{index}"],
                    "images": [{"url": image_url(f"l1-{index}")}],
                    "level": 1,
                }
                for index in range(2)
            ] + [
                {
                    "id": f"l2-{index}",
                    "question": f"Which Level 2 relation {index}?",
                    "ground_truth_list": [f"l2-answer-{index}"],
                    "images": [{"url": image_url(f"l2-{index}")}],
                    "level": 2,
                }
                for index in range(6)
            ]
            private = root / "private.jsonl"
            public = root / "public.jsonl"
            common = {
                "output_cases": private,
                "output_public_cases": public,
                "image_dir": root / "images",
                "seed": 23,
                "download_workers": 2,
            }
            initial = prepare_livevqa_from_rows(rows, level1_count=1, **common)
            initial_ids = {row["case_id"] for row in iter_json_records(private)}
            self.assertEqual(initial["written"], 4)

            expanded = prepare_livevqa_from_rows(rows, level1_count=2, resume=True, **common)
            expanded_rows = list(iter_json_records(private))
            self.assertEqual(expanded["written"], 4)
            self.assertEqual(expanded["selected_level1"], 2)
            self.assertEqual(expanded["selected_level2"], 6)
            self.assertTrue(initial_ids.issubset({row["case_id"] for row in expanded_rows}))
            self.assertEqual(len(expanded_rows), 8)
            self.assertEqual(len(list(iter_json_records(public))), 8)

            with self.assertRaisesRegex(ValueError, "same SAMPLE_SEED"):
                prepare_livevqa_from_rows(rows, level1_count=2, resume=True, **{**common, "seed": 24})

    def test_livevqa_reads_selected_images_from_local_source_zip(self) -> None:
        """LiveVQA img_path points inside an archive, not a Hub image URL."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_dir = root / "archives"
            archive_dir.mkdir()
            paths = ["paper_images/sample-l1.png", *(f"paper_images/sample-l2-{index}.png" for index in range(3))]
            with zipfile.ZipFile(archive_dir / "paper_images.zip", "w") as archive:
                for index, path in enumerate(paths):
                    archive.writestr(path, _PNG + f"zip-{index}".encode())
            rows = [
                {
                    "id": "zip-l1",
                    "question": "Which paper is shown?",
                    "Ground_Truth_List": ["Paper"],
                    "img_path": paths[0],
                    "source": "Arxiv",
                    "level": 1,
                },
                *[
                    {
                        "id": f"zip-l2-{index}",
                        "question": f"Which derived value {index}?",
                        "Ground_Truth_List": [str(index)],
                        "img_path": path,
                        "source": "Arxiv",
                        "level": 2,
                    }
                    for index, path in enumerate(paths[1:])
                ],
            ]
            private = root / "private.jsonl"
            result = prepare_livevqa_from_rows(
                rows,
                output_cases=private,
                output_public_cases=root / "public.jsonl",
                image_dir=root / "images",
                archive_cache_dir=archive_dir,
                level1_count=1,
                seed=3,
            )
            self.assertEqual(result["written"], 4)
            records = list(iter_json_records(private))
            self.assertTrue(all(Path(record["image_path"]).is_file() for record in records))

    def test_public_manifest_resumes_by_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "public.jsonl"
            first = CanonicalCase("first", "First?", "secret-a", "/tmp/first.png")
            second = CanonicalCase("second", "Second?", "secret-b", "/tmp/second.png")
            self.assertEqual(write_public_cases([first], output), 1)
            self.assertEqual(write_public_cases([first, second], output, resume=True), 1)
            records = list(iter_json_records(output))
            self.assertEqual([record["case_id"] for record in records], ["first", "second"])
            self.assertNotIn("gold_answer", records[0])

    def test_evaluation_probe_samples_only_judged_test_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_rows = [
                {"data_id": f"fvqa_test_{index}", "question": f"Q{index}", "answer": f"A{index}", "images": [_PNG]}
                for index in range(4)
            ]
            evaluation_rows = [
                {"case_id": "fvqa_test_0", "acc": 0, "model_answer": "wrong zero"},
                {"case_id": "fvqa_test_1", "acc": 1, "model_answer": "right"},
                {"case_id": "fvqa_test_2", "acc": 0, "model_answer": "wrong two"},
                {"case_id": "fvqa_test_3", "error": "judge unavailable"},
            ]
            result = prepare_evaluation_probe_from_rows(
                source_rows,
                evaluation_rows,
                output_cases=root / "private.jsonl",
                output_public_cases=root / "public.jsonl",
                output_failure_manifest=root / "failure.jsonl",
                output_selection_report=root / "selection.json",
                image_dir=root / "images",
                limit=2,
                seed=7,
            )
            self.assertEqual(result["eligible_wrong"], 2)
            self.assertEqual(result["public_cases"], 2)
            private_rows = list(iter_json_records(root / "private.jsonl"))
            self.assertEqual({row["case_id"] for row in private_rows}, {"fvqa_test_0", "fvqa_test_2"})
            public_text = (root / "public.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("A0", public_text)
            self.assertNotIn("wrong zero", public_text)
            selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(selection["source_split"], "test")
            self.assertIn("not_for_sft", selection["purpose"])

    def test_base_router_does_not_rejudge_unchanged_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "case_0_trajectory.json").write_text("{}", encoding="utf-8")
            with (
                patch(
                    "opensearch_vl.eval_infer_with_llm.run_eval",
                    return_value={"total": 1},
                ) as run_eval,
                patch(
                    "synthesis.post_process.fix2.base_router.time.sleep",
                    side_effect=[None, StopIteration],
                ),
            ):
                with self.assertRaises(StopIteration):
                    watch_base_evaluations(
                        trajectory_dir=root,
                        answer_file="unused.parquet",
                        output_path=root / "report.json",
                        judge_model_alias="unused",
                        poll_s=0.01,
                    )
            self.assertEqual(run_eval.call_count, 1)

    def test_queue_is_idempotent_leased_and_public_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = SqliteTaskQueue(Path(directory) / "queue.sqlite")
            public = PublicCase("case-1", "What is shown?", "/tmp/image.png").to_dict()
            self.assertTrue(queue.enqueue("teacher_collect", "case-1", public, max_attempts=2))
            self.assertFalse(queue.enqueue("teacher_collect", "case-1", public, max_attempts=2))
            with self.assertRaises(ValueError):
                queue.enqueue("teacher_collect", "private", {**public, "gold_answer": "secret"})
            first = queue.lease("teacher_collect", worker_id="worker-a", lease_s=60)
            self.assertIsNotNone(first)
            assert first is not None
            self.assertNotIn("gold_answer", first.payload)
            self.assertEqual(queue.fail(first, error_text="transient", retryable=True), "pending")
            second = queue.lease("teacher_collect", worker_id="worker-b", lease_s=60)
            self.assertEqual(second.attempt, 2)
            queue.complete(second, result_path="/tmp/result.json")
            self.assertEqual(queue.stats("teacher_collect"), {"succeeded": 1, "total": 1})

    def test_queue_can_immediately_requeue_leases_after_a_controlled_worker_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = SqliteTaskQueue(Path(directory) / "queue.sqlite")
            public = PublicCase("case-1", "What is shown?", "/tmp/image.png").to_dict()
            self.assertTrue(queue.enqueue("test", "case-1", public, max_attempts=3))
            lease = queue.lease("test", worker_id="old-worker", lease_s=3600)
            self.assertIsNotNone(lease)
            self.assertEqual(queue.requeue_leased("test", reason="controlled_restart"), 1)
            self.assertEqual(queue.stats("test"), {"pending": 1, "total": 1})
            retried = queue.lease("test", worker_id="new-worker", lease_s=3600)
            self.assertIsNotNone(retried)
            assert retried is not None
            self.assertEqual(retried.attempt, 2)

    def test_atomic_teacher_results_can_recover_a_lost_local_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "new-local-queue.sqlite"
            queue = SqliteTaskQueue(queue_path)
            public = PublicCase("case-1", "What is shown?", "/tmp/image.png").to_dict()
            self.assertTrue(queue.enqueue("recover", "case-1", public, max_attempts=3))
            self.assertTrue(queue.enqueue("recover", "case-2", {**public, "case_id": "case-2"}, max_attempts=3))
            result_dir = root / "teacher_results" / "cases"
            result_dir.mkdir(parents=True)
            (result_dir / "case-1.json").write_text(json.dumps({"case_id": "case-1", "raw_messages": []}), encoding="utf-8")
            (result_dir / "broken.json").write_text("not json", encoding="utf-8")

            outcome = reconcile_teacher_results(queue_path=queue_path, results_dir=root / "teacher_results", queue_name="recover")
            self.assertEqual(outcome["recovered"], 1)
            self.assertEqual(outcome["invalid"], 1)
            self.assertEqual(queue.stats("recover"), {"pending": 1, "succeeded": 1, "total": 2})

    def test_publisher_cap_is_global_across_incremental_evaluator_polls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                CanonicalCase(f"case-{index}", f"Q{index}", f"A{index}", f"/tmp/{index}.png", source_index=index)
                for index in range(3)
            ]
            queue = SqliteTaskQueue(root / "queue.sqlite")
            first = publish_base_failures(
                cases,
                [{"case_id": "case_0", "acc": 0}, {"case_id": "case_1", "acc": 0}],
                queue=queue,
                max_cases=2,
            )
            self.assertEqual(first["published"], 2)
            # A later evaluator snapshot can contain an earlier, previously
            # unseen failure.  The durable cap must still prevent task 2.
            second = publish_base_failures(
                cases,
                [{"case_id": "case_2", "acc": 0}, {"case_id": "case_0", "acc": 0}],
                queue=queue,
                max_cases=2,
            )
            self.assertEqual(second["published"], 0)
            self.assertEqual(second["hard_cap_stopped"], 2)
            self.assertEqual(queue.stats("teacher_collect"), {"pending": 2, "total": 2})

    def test_publisher_and_parallel_worker_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                CanonicalCase(f"case-{index}", f"Q{index}", f"A{index}", f"/tmp/{index}.png", source_index=index)
                for index in range(3)
            ]
            queue_path = root / "queue.sqlite"
            queue = SqliteTaskQueue(queue_path)
            publish = publish_base_failures(
                cases,
                [{"case_id": f"case_{index}", "acc": 0} for index in range(3)],
                queue=queue,
            )
            self.assertEqual(publish["published"], 3)

            def fake_collect(public_case: PublicCase, _config: CollectorConfig) -> dict:
                self.assertNotIn("gold_answer", public_case.to_dict())
                return {"case_id": public_case.case_id, "question": public_case.question, "raw_messages": []}

            with patch("synthesis.post_process.fix2.worker.collect_case", side_effect=fake_collect):
                outcome = run_teacher_workers(
                    queue_path=queue_path,
                    results_dir=root / "results",
                    collector_config=CollectorConfig(model="unused", workdir=str(root / "runtime")),
                    workers=2,
                    idle_exit_s=0.01,
                    poll_s=0.005,
                )
            self.assertEqual(outcome["succeeded"], 3)
            self.assertEqual(outcome["queue_succeeded"], 3)
            merged = merge_teacher_results(queue_path=queue_path, output_jsonl=root / "merged.jsonl")
            self.assertEqual(merged["merged"], 3)
            self.assertEqual(len(list(iter_json_records(root / "merged.jsonl"))), 3)

    def test_atomic_results_merge_does_not_depend_on_local_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_dir = root / "teacher_results" / "cases"
            write_json(cases_dir / "z.json", {"case_id": "case-z", "raw_messages": []})
            write_json(cases_dir / "a.json", {"case_id": "case-a", "raw_messages": []})

            output = root / "finalization" / "teacher_raw.jsonl"
            merged = merge_atomic_teacher_results(results_dir=root / "teacher_results", output_jsonl=output)

            self.assertEqual(merged, {"scanned": 2, "merged": 2, "invalid": 0, "duplicate_case_id": 0})
            self.assertEqual([item["case_id"] for item in iter_json_records(output)], ["case-a", "case-z"])

    def test_worker_captures_system_exit_without_terminating_the_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "queue.sqlite"
            queue = SqliteTaskQueue(queue_path)
            public = PublicCase("case-1", "What is shown?", "/tmp/image.png").to_dict()
            self.assertTrue(queue.enqueue("teacher_collect", "case-1", public, max_attempts=2))

            with patch("synthesis.post_process.fix2.worker.collect_case", side_effect=SystemExit(1)):
                outcome = run_teacher_workers(
                    queue_path=queue_path,
                    results_dir=root / "results",
                    collector_config=CollectorConfig(model="unused", workdir=str(root / "runtime")),
                    workers=1,
                    idle_exit_s=0.01,
                    poll_s=0.005,
                )

            self.assertEqual(outcome["retryable_error"], 1)
            self.assertEqual(outcome["failed"], 1)
            self.assertEqual(outcome["queue_failed"], 1)


if __name__ == "__main__":
    unittest.main()
