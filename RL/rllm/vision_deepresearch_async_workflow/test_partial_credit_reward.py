"""Unit tests for text-grounded statement partial credit."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vision_deepresearch_async_workflow import partial_credit_reward
from vision_deepresearch_async_workflow.partial_credit_reward import (
    PartialCreditConfig,
    build_text_only_judge_trace,
    extract_judge_statement_assessments,
    score_statement_partial_credit,
    validate_judge_verdicts,
)
from rllm.utils.partial_credit_shaping import (
    apply_credit_turn_advantage_shaping,
    build_credit_turn_token_mask,
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
    async def test_failed_chunk_discards_prior_progress_and_stops_scoring(self):
        task = {"question": "Find Beta.", "extra_info": {"partial_credit": {
            "statements": [{"statement_id": "hop_1", "statement": "Alpha links to Beta."}],
        }}}
        steps = [_step(action_type="tool", response="Search Alpha.",
                       observation="Alpha links to Beta.", tool_call="t2t_search") for _ in range(3)]
        from unittest.mock import AsyncMock
        scorer = AsyncMock(side_effect=[
            {"scored": True, "reason": "scored", "events": [{"statement_id": "hop_1", "credit_turn": 1}]},
            {"scored": False, "reason": "judge_request_failed", "judge_error_type": "TimeoutError"},
        ])
        with patch.object(partial_credit_reward, "_score_trace_chunk", new=scorer):
            result = await score_statement_partial_credit(
                task=task, steps=steps,
                config=PartialCreditConfig(mode="reward", trace_chunk_steps=1),
            )
        self.assertEqual(scorer.await_count, 2)
        self.assertEqual(result["reason"], "judge_request_failed")
        self.assertFalse(result["scored"])
        self.assertEqual(result["raw_bonus"], 0)
        self.assertEqual(result["bonus"], 0)
        self.assertEqual(result["events"], [])

    def test_progress_scores_furthest_hit_not_hit_count(self) -> None:
        order = [{"statement_id": name} for name in ("entry", "middle", "last")]
        for hits, position in [([], 0), (["entry"], 1), (["middle"], 2),
                               (["entry", "middle"], 2), (["last"], 3),
                               (["entry", "middle", "last"], 3),
                               (["middle", "middle", "unknown"], 2)]:
            with self.subTest(hits=hits):
                actual, progress = partial_credit_reward.statement_progress(
                    [{"statement_id": hit} for hit in hits], order
                )
                self.assertEqual(actual, position)
                self.assertAlmostEqual(0.3 * progress, 0.1 * position)
        self.assertEqual(partial_credit_reward.statement_progress([], []), (0, 0.0))

    def test_credit_turn_mask_aligns_only_selected_assistant_block(self) -> None:
        # Three assistant actions separated by tool observations.  Padding at
        # the end must never be selected even when the final turn is credited.
        response_mask = torch.tensor(
            [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0]
        )
        token_mask, aligned = build_credit_turn_token_mask(
            response_mask, (2,), expected_turn_count=3
        )
        self.assertTrue(aligned)
        self.assertEqual(
            token_mask.to(dtype=torch.int64).tolist(),
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        )

    def test_credit_turn_mask_fails_closed_on_turn_count_mismatch(self) -> None:
        token_mask, aligned = build_credit_turn_token_mask(
            torch.tensor([1, 1, 0, 1]),
            (2,),
            expected_turn_count=3,
        )
        self.assertFalse(aligned)
        self.assertFalse(bool(token_mask.any()))

    def test_advantage_shaping_zeroes_negative_prefix_and_positive_suffix(self) -> None:
        advantages = torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ]
        )
        response_mask = torch.tensor(
            [[1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0]]
        )
        shaped, stats = apply_credit_turn_advantage_shaping(
            advantages,
            response_mask,
            credit_turns_by_row=[(2,), (1,)],
            expected_turn_counts=[2, 2],
            is_correct_by_row=[False, False],
        )
        self.assertEqual(
            shaped.tolist(),
            [[0.0, 0.0, -1.0, 0.0, 0.0, -1.0], [0.5, 0.5, 0.5, 0.0, 0.0, 0.5]],
        )
        self.assertEqual(stats["negative_zeroed_token_count"], 4.0)
        self.assertEqual(stats["positive_doubled_token_count"], 0.0)

    def test_advantage_shaping_masks_wrong_positive_suffix_after_last_credit(self) -> None:
        advantages = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        response_mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1]])

        shaped, stats = apply_credit_turn_advantage_shaping(
            advantages,
            response_mask,
            credit_turns_by_row=[(2,)],
            expected_turn_counts=[3],
            is_correct_by_row=[False],
        )

        # Turn 2 and the earlier setup turn stay unchanged. The
        # later action turn (typically further search or the wrong answer) is
        # neutralized. Tool-observation positions remain outside the mask.
        self.assertTrue(
            torch.equal(
                shaped,
                torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]]),
            )
        )
        self.assertEqual(stats["incorrect_positive_credit_rows"], 1.0)
        self.assertEqual(stats["incorrect_positive_suffix_masked_rows"], 1.0)
        self.assertEqual(stats["incorrect_positive_suffix_token_count"], 2.0)
        self.assertEqual(stats["incorrect_positive_suffix_zeroed_token_count"], 2.0)

    def test_advantage_shaping_keeps_correct_positive_suffix(self) -> None:
        advantages = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        response_mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1]])

        shaped, stats = apply_credit_turn_advantage_shaping(
            advantages,
            response_mask,
            credit_turns_by_row=[(2,)],
            expected_turn_counts=[3],
            is_correct_by_row=[True],
        )

        self.assertTrue(
            torch.equal(
                shaped,
                advantages,
            )
        )
        self.assertEqual(stats["incorrect_positive_suffix_masked_rows"], 0.0)

    def test_negative_prefix_includes_uncredited_turns_but_not_suffix(self) -> None:
        mask = torch.tensor([[1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]])
        advantages = torch.full(mask.shape, -0.2)
        shaped, stats = apply_credit_turn_advantage_shaping(
            advantages, mask, [(1, 3)], [4], is_correct_by_row=[False],
        )
        expected = advantages.clone()
        expected[0, [0, 1, 3, 5, 6, 7]] = 0
        self.assertTrue(torch.equal(shaped, expected))
        self.assertTrue(torch.equal(advantages, torch.full(mask.shape, -0.2)))
        self.assertEqual(stats["incorrect_negative_prefix_zeroed_token_count"], 6)

    def test_prefix_suffix_shaping_skips_missing_credit_and_misalignment(self) -> None:
        mask = torch.tensor([[1, 0, 1, 0, 1, 0]] * 4)
        advantages = torch.tensor([[-0.2] * 6, [0.2] * 6, [-0.2] * 6, [0.] * 6])
        shaped, stats = apply_credit_turn_advantage_shaping(
            advantages, mask, [(), (), (2,), (2,)], [3, 3, 4, 3],
            is_correct_by_row=[False] * 4,
        )
        self.assertTrue(torch.equal(shaped, advantages))
        self.assertEqual(stats["alignment_mismatch_rows"], 1)

    def test_shaping_skips_unknown_outcomes(self) -> None:
        advantages = torch.tensor([[-0.2, -0.2, -0.2]])
        shaped, _ = apply_credit_turn_advantage_shaping(
            advantages, torch.tensor([[1, 0, 1]]), [(1,)], [2],
        )
        self.assertTrue(torch.equal(shaped, advantages))

    def test_correct_negative_and_zero_advantages_are_unchanged(self) -> None:
        advantages = torch.tensor([[-0.2] * 5, [0.0] * 5])
        mask = torch.tensor([[1, 1, 0, 1, 1]] * 2)
        shaped, stats = apply_credit_turn_advantage_shaping(
            advantages, mask, [(1, 2), (1, 2)], [2, 2],
            is_correct_by_row=[True, True],
        )
        self.assertTrue(torch.equal(shaped, advantages))
        self.assertEqual(stats["correct_rows_skipped"], 2)
        self.assertEqual(stats["negative_zeroed_token_count"], 0)

    def test_judge_prompt_rejects_cross_entity_evidence_merging(self) -> None:
        prompt = " ".join(partial_credit_reward._JUDGE_SYSTEM_PROMPT.split())
        self.assertIn("Josef Mengele is never evidence for Johann Kremer", prompt)
        self.assertIn("core target and the relation", prompt)
        self.assertIn("Aliases, including a former name", prompt)
        self.assertIn("explicit statement that contradicts", prompt)
        self.assertIn("the assistant acts BEFORE receiving that turn's tool result", prompt)
        self.assertIn("credit_turn", prompt)

    def test_judge_prompt_prefers_earliest_sufficient_evidence(self) -> None:
        prompt = " ".join(partial_credit_reward._JUDGE_SYSTEM_PROMPT.split())
        for requirement in (
            "A sufficient search snippet and sufficient read_url text have equal evidential standing",
            "choose the earliest sufficient one",
            "choose the fewest turns, then the earliest turn numbers",
            "this use check does not move credit to a later confirmation",
            "not [4] or [3,4]",
            "state what that result was missing",
            "The retrospective original-image exception in Rule 6 remains separate",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, prompt)

    def test_judge_prompt_keeps_statement_identity_and_causal_evidence(self) -> None:
        prompt = " ".join(partial_credit_reward._JUDGE_SYSTEM_PROMPT.split())
        for requirement in (
            "Do not replace the supplied statement",
            "Do not add requirements from the next step",
            "Gold: [this statement's relation]",
            "Every evidence turn must be no later than ``credit_turn``",
            "A final-answer turn is not adoption",
            'say "not shown in this slice or memory"',
            "pixels omitted; identification unverified",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, prompt)
        self.assertIn("80 words", partial_credit_reward._JUDGE_REASON_INSTRUCTION)

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

    def test_validation_requires_explicit_grounded_credit_turn(self) -> None:
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
                    "evidence_turns": [1],
                    "credit_turn": 1,
                    "adoption_turn": 2,
                    "evidence_quote": "Alpha links to Beta in 1990",
                    "confidence": 0.91,
                },
                {
                    "statement_id": "hop_1",
                    "verdict": "verified_and_adopted",
                    "evidence_turns": [2],
                    "credit_turn": 2,
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
        self.assertEqual(events[0]["credit_turn"], 1)
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
                        "credit_turn": 3,
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
        self.assertEqual(events[0]["credit_turn"], 3)
        self.assertEqual(events[0]["adoption_turn"], 4)

    def test_validation_rejects_shifted_or_final_credit_assignment(self) -> None:
        statements = [{"statement_id": "hop_1", "statement": "Alpha links to Beta."}]
        records = {
            1: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "Alpha links to Beta.",
            },
            2: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "I will now search Beta's later history.",
            },
            3: {
                "is_tool_turn": False,
                "is_final": True,
                "observation": "",
            },
        }
        payload = {
            "statements": [
                {
                    "statement_id": "hop_1",
                    "verdict": "established_and_used",
                    "evidence_turns": [1],
                    "credit_turn": 2,
                    "adoption_turn": 3,
                },
                {
                    "statement_id": "hop_1",
                    "verdict": "established_and_used",
                    "evidence_turns": [1],
                    "credit_turn": 1,
                    "adoption_turn": 3,
                },
            ]
        }

        events, rejected = validate_judge_verdicts(
            payload,
            statements=statements,
            turn_records=records,
            config=PartialCreditConfig(mode="shadow"),
        )

        self.assertEqual(events, [])
        self.assertEqual(rejected, 2)

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
                        "credit_turn": 2,
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
                        "credit_turn": 1,
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

    def test_validation_allows_judge_verified_omitted_visual_evidence(self) -> None:
        """The judge may credit an image-derived transition under Rule 6.

        Image bytes are intentionally absent from the compact judge trace, so
        the deterministic layer must preserve a structurally valid verdict
        rather than rejecting it merely because its evidence turn has the
        omission marker.
        """
        statements = [
            {
                "statement_id": "hop_0",
                "statement": "The pictured tapestry is the Royal Banner of Scotland.",
                "source_object": "",
                "target_object": "Royal Banner of Scotland",
            }
        ]
        records = {
            1: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": (
                    "[VISUAL_OBSERVATION_OMITTED: the rollout policy received an "
                    "image, but the judge did not.]"
                ),
            },
            2: {
                "is_tool_turn": True,
                "is_final": False,
                "observation": "The next search investigates the Royal Banner of Scotland.",
            },
        }

        events, rejected = validate_judge_verdicts(
            {
                "statements": [
                    {
                        "statement_id": "hop_0",
                        "verdict": "established_and_used",
                        "evidence_turns": [1],
                        "credit_turn": 1,
                        "adoption_turn": 2,
                    }
                ]
            },
            statements=statements,
            turn_records=records,
            config=PartialCreditConfig(mode="reward"),
        )

        self.assertEqual(rejected, 0)
        self.assertEqual(events, [
            {
                "statement_id": "hop_0",
                "evidence_turns": [1],
                "credit_turn": 1,
                "adoption_turn": 2,
            }
        ])

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
        self.assertIsNone(assessments[0]["credit_turn"])
        self.assertIsNone(assessments[0]["adoption_turn"])
        self.assertEqual(assessments[1]["verdict"], "missing_assessment")

    def test_multi_turn_credit_validation_and_deduplicated_shaping(self) -> None:
        statements = [{"statement_id": "hop_0"}]
        item = {"statement_id": "hop_0", "verdict": "established_and_used",
                "evidence_turns": [1, 2, 3, 4, 5], "credit_turns": [1, 2, 3, 4, 5],
                "credit_turn": 5, "adoption_turn": 6}
        records = {n: {"is_tool_turn": True, "is_final": False, "observation": "evidence"} for n in range(1, 8)}
        events, rejected = validate_judge_verdicts(
            {"statements": [item]}, statements=statements, turn_records=records,
            config=PartialCreditConfig(),
        )
        self.assertEqual(rejected, 0)
        self.assertEqual(events[0]["credit_turns"], [1, 2, 3, 4, 5])
        self.assertEqual(partial_credit_reward.statement_progress(events, statements), (1, 1.0))
        turns = partial_credit_reward.collect_credit_turns(events + events + [{"credit_turn": 3}], 7)
        self.assertEqual(turns, (1, 2, 3, 4, 5))
        mask = torch.tensor([[1, 0] * 7, [1, 0] * 7])
        advantages = torch.tensor([[0.5] * 14, [-1.0] * 14])
        shaped, _ = apply_credit_turn_advantage_shaping(
            advantages, mask, [turns, turns], [7, 7], is_correct_by_row=[False, False],
        )
        self.assertEqual(shaped[0].tolist(), [0.5, 0.5] * 5 + [0.0, 0.5] * 2)
        self.assertEqual(shaped[1].tolist(), [0.0, -1.0] * 5 + [-1.0] * 4)
        for update in [
            {"credit_turns": [1, 2, 3, 4, 5, 6]},
            {"credit_turns": [1, 1, 5]},
            {"credit_turns": [1, 6]},
            {"credit_turns": [1, 2]},
            {"evidence_turns": [1, 5, 6]},
            {"adoption_turn": 5},
            {"credit_turns": [True, 5]},
        ]:
            with self.subTest(update=update):
                actual, rejected = validate_judge_verdicts(
                    {"statements": [dict(item, **update)]}, statements=statements,
                    turn_records=records, config=PartialCreditConfig(),
                )
                self.assertEqual(actual, [])
                self.assertEqual(rejected, 1)

    def test_diagnostics_preserve_full_reason_and_evidence(self) -> None:
        reason = "Gold: Alpha. Evidence: " + "long explanation " * 80
        item = {"statement_id": "hop_0", "verdict": "established_and_used",
                "reason": reason, "evidence_turns": [1, 2, 3, 4, 5, 6],
                "credit_turn": 1, "adoption_turn": 2}
        statements = [{"statement_id": "hop_0"}]
        result = extract_judge_statement_assessments({"statements": [item]}, statements=statements)[0]
        self.assertEqual(result["reason"], reason)
        self.assertEqual(result["evidence_turns"], item["evidence_turns"])
        records = {n: {"is_tool_turn": True, "is_final": False, "observation": "evidence"} for n in (1, 2)}
        for evidence, adoption, expected in [
            ([1, 2, 3, 4, 5, 6], 2, "evidence_turn_count_out_of_range"),
            ([1], 1, "adoption_not_after_credit"),
            ([1], 2, "adoption_turn_is_final"),
        ]:
            with self.subTest(expected=expected):
                records[2]["is_final"] = expected == "adoption_turn_is_final"
                errors = {}
                events, rejected = validate_judge_verdicts(
                    {"statements": [dict(item, evidence_turns=evidence, adoption_turn=adoption)]},
                    statements=statements, turn_records=records,
                    config=PartialCreditConfig(), rejection_reasons=errors,
                )
                self.assertEqual(events, [])
                self.assertEqual(rejected, 1)
                self.assertIn(expected, errors["hop_0"])

    def test_diagnostic_assessments_keep_safe_turn_references(self) -> None:
        assessments = extract_judge_statement_assessments(
            {
                "statements": [
                    {
                        "statement_id": "hop_1",
                        "verdict": "established_and_used",
                        "evidence_turns": [3, 7],
                        "credit_turn": 7,
                        "adoption_turn": 8,
                        "reason": "The two tool observations establish the link and the next action uses it.",
                    }
                ]
            },
            statements=[{"statement_id": "hop_1", "statement": "Alpha links to Beta."}],
        )

        self.assertEqual(assessments[0]["evidence_turns"], [3, 7])
        self.assertEqual(assessments[0]["credit_turn"], 7)
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
                                "evidence_turns": [1],
                                "credit_turn": 1,
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
        self.assertEqual(result["raw_bonus"], 0.15)
        self.assertEqual(result["bonus"], 0.0)
        self.assertEqual(
            result["judge_statement_assessments"][0]["verdict"],
            "verified_and_adopted",
        )
        self.assertEqual(reward_result["bonus"], 0.15)
        with patch.object(partial_credit_reward, "_request_judge", new=fake_request):
            correct_result = await score_statement_partial_credit(
                task=task, steps=steps, is_correct=True,
                config=PartialCreditConfig(mode="reward"),
            )
        self.assertEqual(correct_result["bonus"], 0.0)
        self.assertEqual(correct_result["raw_bonus"], 0.15)
        self.assertTrue(correct_result["reward_suppressed_correct"])
        self.assertEqual(len(correct_result["events"]), 1)

        # Judge limits must not turn the first of three statements into 100%
        # progress. Legacy per-statement reward must not affect the new score.
        limited_task = {
            "question": task["question"],
            "extra_info": {"partial_credit": {"statements": [
                {"statement_id": "hop_1", "statement": "Alpha links to Beta through a 1990 event."},
                {"statement_id": "hop_2", "statement": "Beta links to Gamma."},
                {"statement_id": "hop_3", "statement": "Gamma links to Delta."},
            ]}},
        }
        with patch.object(partial_credit_reward, "_request_judge", new=fake_request):
            limited = await score_statement_partial_credit(
                task=limited_task, steps=steps,
                config=PartialCreditConfig(mode="reward", max_statements=1,
                                           statement_reward=9.0, trajectory_reward_cap=0.3),
            )
        self.assertEqual(limited["progress_statement_count"], 3)
        self.assertEqual(limited["furthest_verified_statement_position"], 1)
        self.assertAlmostEqual(limited["bonus"], 0.1)

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
                                "credit_turn": None,
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
                                "credit_turn": evidence_turn,
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
        self.assertEqual(result["raw_bonus"], 0.15)
        self.assertEqual(result["events"][0]["evidence_turns"], [1])
        self.assertEqual(result["events"][0]["credit_turn"], 1)
        self.assertEqual(result["events"][0]["chunk_index"], 0)


if __name__ == "__main__":
    unittest.main()
