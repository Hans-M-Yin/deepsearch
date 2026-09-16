"""Unit tests for text-grounded statement partial credit."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vision_deepresearch_async_workflow import partial_credit_reward
from vision_deepresearch_async_workflow.partial_credit_reward import (
    PartialCreditConfig,
    build_text_only_judge_trace,
    extract_judge_statement_assessments,
    score_statement_partial_credit,
    validate_judge_verdicts,
)


def _step(
    *,
    action_type: str,
    response: str,
    observation: str = "",
    tool_call: str = "",
):
    return SimpleNamespace(
        action=SimpleNamespace(
            action={"type": action_type, "tool_call": tool_call}
        ),
        model_response=response,
        observation=observation,
    )


class PartialCreditRewardTest(unittest.IsolatedAsyncioTestCase):
    def test_judge_prompt_rejects_cross_entity_evidence_merging(self) -> None:
        prompt = " ".join(partial_credit_reward._JUDGE_SYSTEM_PROMPT.split())
        self.assertIn("Josef Mengele is never evidence for Johann Kremer", prompt)
        self.assertIn("both anchors and the relation", prompt)
        self.assertIn("unambiguous alias", prompt)
        self.assertIn("explicit statement that contradicts", prompt)

    def test_trace_hides_final_answer_and_marks_images(self) -> None:
        trace, records, count = build_text_only_judge_trace(
            [
                _step(
                    action_type="tool_call",
                    response="I will search for evidence.",
                    observation="<tool_response>Fact <image> retained text</tool_response>",
                ),
                _step(
                    action_type="final_answer",
                    response="SECRET FINAL ANSWER",
                ),
            ],
            question="Which item is shown in <image>?",
        )

        self.assertEqual(count, 2)
        self.assertTrue(records[1]["is_tool_turn"])
        self.assertTrue(records[2]["is_final"])
        self.assertIn("VISUAL_OBSERVATION_OMITTED", trace)
        self.assertIn("ANSWER_OMITTED", trace)
        self.assertNotIn("SECRET FINAL ANSWER", trace)

    def test_validation_requires_immediate_grounded_adoption(self) -> None:
        config = PartialCreditConfig(mode="shadow", min_confidence=0.85)
        statements = [{"statement_id": "hop_1", "statement": "Alpha links to Beta."}]
        records = {
            1: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "The archive confirms Alpha links to Beta in 1990.",
            },
            2: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "More material.",
            },
            3: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "Late material.",
            },
        }
        payload = {
            "statements": [
                {
                    "statement_id": "hop_1",
                    "verdict": "verified_and_adopted",
                    "supporting_turn": 1,
                    "adoption_turn": 2,
                    "evidence_quote": "Alpha links to Beta in 1990",
                    "confidence": 0.91,
                },
                {
                    "statement_id": "hop_1",
                    "verdict": "verified_and_adopted",
                    "supporting_turn": 2,
                    "adoption_turn": 3,
                    "evidence_quote": "More material",
                    "confidence": 0.99,
                },
            ]
        }

        events, rejected = validate_judge_verdicts(
            payload,
            statements=statements,
            turn_records=records,
            config=config,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["supporting_turn"], 1)
        self.assertEqual(rejected, 1)

    def test_validation_accepts_two_evidence_turns_and_later_adoption(self) -> None:
        config = PartialCreditConfig(mode="shadow")
        statements = [
            {
                "statement_id": "hop_1",
                "statement": "Alpha links to Gamma.",
                "source_object": "Alpha",
                "target_object": "Gamma",
            }
        ]
        records = {
            1: {"is_tool_turn": True, "is_final": False, "observation": "Alpha is Beta."},
            2: {"is_tool_turn": True, "is_final": False, "observation": "Unrelated result."},
            3: {"is_tool_turn": True, "is_final": False, "observation": "Beta links to Gamma."},
            4: {"is_tool_turn": True, "is_final": False, "observation": "Further search result."},
        }
        events, rejected = validate_judge_verdicts(
            {
                "statements": [
                    {
                        "statement_id": "hop_1",
                        "verdict": "established_and_used",
                        "evidence_turns": [1, 3],
                        "adoption_turn": 4,
                    }
                ]
            },
            statements=statements,
            turn_records=records,
            config=config,
        )

        self.assertEqual(rejected, 0)
        self.assertEqual(events[0]["evidence_turns"], [1, 3])
        self.assertEqual(events[0]["adoption_turn"], 4)

    def test_validation_leaves_entity_semantics_to_judge(self) -> None:
        statements = [
            {
                "statement_id": "hop_2",
                "statement": "Johann Kremer conducted experiments at Auschwitz-Birkenau.",
                "source_object": "Johann Kremer",
                "target_object": "Auschwitz-Birkenau",
            }
        ]
        records = {
            1: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": (
                    "Josef Mengele conducted medical experiments at "
                    "Auschwitz-Birkenau."
                ),
            },
            2: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "The First Auschwitz Trial had forty defendants.",
            },
            3: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "Later search result.",
            },
        }

        events, rejected = validate_judge_verdicts(
            {
                "statements": [
                    {
                        "statement_id": "hop_2",
                        "verdict": "established_and_used",
                        "evidence_turns": [1, 2],
                        "adoption_turn": 3,
                    }
                ]
            },
            statements=statements,
            turn_records=records,
            config=PartialCreditConfig(mode="shadow"),
        )

        # The deterministic layer validates trace structure only.  The judge
        # prompt, which sees the source/target anchors, owns semantic entity
        # substitution and alias resolution.
        self.assertEqual(rejected, 0)
        self.assertEqual(events[0]["evidence_turns"], [1, 2])

    def test_validation_allows_entry_hop_with_only_target_object(self) -> None:
        statements = [
            {
                "statement_id": "hop_0",
                "statement": "The tapestry is the Royal Banner of Scotland.",
                "source_object": "",
                "target_object": "Royal Banner of Scotland",
            }
        ]
        records = {
            1: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "The image result identifies the Royal Banner of Scotland.",
            },
            2: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "Later search result.",
            },
        }

        events, rejected = validate_judge_verdicts(
            {
                "statements": [
                    {
                        "statement_id": "hop_0",
                        "verdict": "established_and_used",
                        "evidence_turns": [1],
                        "adoption_turn": 2,
                    }
                ]
            },
            statements=statements,
            turn_records=records,
            config=PartialCreditConfig(mode="shadow"),
        )

        self.assertEqual(rejected, 0)
        self.assertEqual(len(events), 1)

    def test_diagnostic_assessments_cover_every_statement(self) -> None:
        statements = [
            {"statement_id": "hop_1", "statement": "Alpha links to Beta."},
            {"statement_id": "hop_2", "statement": "Beta links to Gamma."},
        ]
        assessments = extract_judge_statement_assessments(
            {
                "statements": [
                    {
                        "statement_id": "hop_1",
                        "verdict": "not_verified",
                        "reason": "The trace names Alpha but has no textual evidence for its link to Beta.",
                    }
                ]
            },
            statements=statements,
        )

        self.assertEqual(len(assessments), 2)
        self.assertEqual(assessments[0]["verdict"], "not_verified")
        self.assertIn("no textual evidence", assessments[0]["reason"])
        self.assertIsNone(assessments[0]["evidence_turns"])
        self.assertIsNone(assessments[0]["adoption_turn"])
        self.assertEqual(assessments[1]["verdict"], "missing_assessment")

    def test_diagnostic_assessments_keep_safe_turn_references(self) -> None:
        assessments = extract_judge_statement_assessments(
            {
                "statements": [
                    {
                        "statement_id": "hop_1",
                        "verdict": "established_and_used",
                        "evidence_turns": [3, 7],
                        "adoption_turn": 8,
                        "reason": "The two tool observations establish the link and the next action uses it.",
                    }
                ]
            },
            statements=[{"statement_id": "hop_1", "statement": "Alpha links to Beta."}],
        )

        self.assertEqual(assessments[0]["evidence_turns"], [3, 7])
        self.assertEqual(assessments[0]["adoption_turn"], 8)

    async def test_benchmark_without_partial_credit_metadata_skips(self) -> None:
        result = await score_statement_partial_credit(
            task={"question": "Benchmark question", "answer": "answer", "extra_info": {}},
            steps=[],
            config=PartialCreditConfig(mode="shadow"),
        )

        self.assertFalse(result["scored"])
        self.assertEqual(result["reason"], "missing_partial_credit_metadata")

    async def test_shadow_scores_one_earliest_statement_without_reward(self) -> None:
        task = {
            "question": "Find the final entity.",
            "answer": "Final",
            "extra_info": {
                "partial_credit": {
                    "version": 1,
                    # This mirrors pandas/pyarrow's representation after the
                    # metadata parquet round-trip.
                    "statements": np.array([
                        {
                            "statement_id": "hop_1",
                            "statement": "Alpha links to Beta through a 1990 event.",
                        },
                        # A malformed source can repeat a statement. It is
                        # still eligible exactly once in the judge request.
                        {
                            "statement_id": "hop_1_duplicate",
                            "statement": "Alpha links to Beta through a 1990 event.",
                        },
                    ], dtype=object),
                }
            },
        }
        steps = [
            _step(
                action_type="tool_call",
                response="Search Alpha 1990 event",
                observation=(
                    "<tool_response>The archive says Alpha links to Beta "
                    "through a 1990 event.</tool_response>"
                ),
                tool_call="search Alpha 1990 event",
            ),
            _step(
                action_type="tool_call",
                response="Search Beta history using the 1990 event",
                observation="<tool_response>Beta history result.</tool_response>",
                tool_call="search Beta history",
            ),
        ]

        async def fake_request(_prompt, _config):
            return (
                json.dumps(
                    {
                        "statements": [
                            {
                                "statement_id": "hop_1",
                                "verdict": "verified_and_adopted",
                                "supporting_turn": 1,
                                "adoption_turn": 2,
                                "evidence_quote": (
                                    "Alpha links to Beta through a 1990 event"
                                ),
                                "confidence": 0.95,
                            }
                        ]
                    }
                ),
                0.01,
                0.0,
            )

        config = PartialCreditConfig(
            mode="shadow", statement_reward=0.05, include_judge_reasons=True
        )
        with patch.object(
            partial_credit_reward, "_request_judge", new=fake_request
        ):
            result = await score_statement_partial_credit(
                task=task,
                steps=steps,
                config=config,
            )
            reward_result = await score_statement_partial_credit(
                task=task,
                steps=steps,
                config=PartialCreditConfig(
                    mode="reward", statement_reward=0.05
                ),
            )

        self.assertTrue(result["scored"])
        self.assertEqual(result["eligible_statement_count"], 1)
        self.assertEqual(result["duplicate_statement_skipped_count"], 1)
        self.assertEqual(result["verified_statement_count"], 1)
        self.assertEqual(result["raw_bonus"], 0.05)
        self.assertEqual(result["bonus"], 0.0)
        self.assertEqual(
            result["judge_statement_assessments"][0]["verdict"],
            "verified_and_adopted",
        )
        self.assertEqual(reward_result["bonus"], 0.05)

    async def test_judge_receives_anchors_without_exact_prefilter(self) -> None:
        task = {
            "question": "Find Beta.",
            "extra_info": {
                "partial_credit": {
                    "version": 3,
                    "statements": [
                        {
                            "statement_id": "hop_0",
                            "statement": "The answer is Beta.",
                            "source_object": None,
                            "target_object": "Beta",
                        }
                    ],
                }
            },
        }
        steps = [
            _step(
                action_type="tool_call",
                response="Search Alpha",
                observation="<tool_response>Only Alpha appears here.</tool_response>",
                tool_call="search Alpha",
            )
        ]

        prompts: list[str] = []

        async def fake_request(prompt, _config):
            prompts.append(prompt)
            return (
                json.dumps(
                    {
                        "statements": [
                            {
                                "statement_id": "hop_0",
                                "verdict": "not_established",
                                "evidence_turns": [],
                                "adoption_turn": None,
                            }
                        ]
                    }
                ),
                0.01,
                0.0,
            )

        with patch.object(
            partial_credit_reward, "_request_judge", new=fake_request
        ):
            result = await score_statement_partial_credit(
                task=task,
                steps=steps,
                config=PartialCreditConfig(mode="shadow"),
            )

        self.assertTrue(result["scored"])
        self.assertEqual(result["verified_statement_count"], 0)
        self.assertEqual(len(prompts), 1)
        self.assertIn('"source_object": ""', prompts[0])
        self.assertIn('"target_object": "Beta"', prompts[0])

    async def test_chunked_judge_deduplicates_statement_reward(self) -> None:
        task = {
            "question": "Find Alpha.",
            "extra_info": {
                "partial_credit": {
                    "statements": [
                        {"statement_id": "hop_1", "statement": "Alpha links to Beta."}
                    ]
                }
            },
        }
        steps = [
            _step(
                action_type="tool_call",
                response=f"Search evidence turn {index}",
                observation=f"Tool evidence at turn {index}.",
                tool_call=f"search turn {index}",
            )
            for index in range(1, 18)
        ]
        prompts: list[str] = []

        async def fake_request(prompt, _config):
            prompts.append(prompt)
            if "--- Turn 16 ---" in prompt:
                evidence_turn, adoption_turn = 16, 17
            else:
                evidence_turn, adoption_turn = 1, 2
            return (
                json.dumps(
                    {
                        "statements": [
                            {
                                "statement_id": "hop_1",
                                "verdict": "established_and_used",
                                "evidence_turns": [evidence_turn],
                                "adoption_turn": adoption_turn,
                            }
                        ]
                    }
                ),
                0.01,
                0.0,
            )

        with patch.object(
            partial_credit_reward, "_request_judge", new=fake_request
        ):
            result = await score_statement_partial_credit(
                task=task,
                steps=steps,
                config=PartialCreditConfig(
                    mode="reward", trace_chunk_steps=15, statement_reward=0.05
                ),
            )

        self.assertEqual(len(prompts), 2)
        self.assertTrue(all("hop_1" in prompt for prompt in prompts))
        self.assertEqual(result["trace_chunk_count"], 2)
        self.assertEqual(result["trace_chunk_scored_count"], 2)
        self.assertEqual(result["chunk_valid_verdict_count"], 2)
        self.assertEqual(result["verified_statement_count"], 1)
        self.assertEqual(result["chunk_valid_duplicate_statement_count"], 1)
        self.assertEqual(result["chunk_valid_duplicate_occurrence_count"], 1)
        self.assertEqual(result["raw_bonus"], 0.05)
        self.assertEqual(result["events"][0]["evidence_turns"], [1])
        self.assertEqual(result["events"][0]["chunk_index"], 0)


if __name__ == "__main__":
    unittest.main()
