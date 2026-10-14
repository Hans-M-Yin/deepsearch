"""Regression contracts for the new sequential augmentation flow."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from synthesis.post_process.fix import progress, rollout, rewrite, search_cards
from synthesis.post_process.fix.errors import SkipAugmentation, ReviewRejected
from synthesis.post_process.fix.pipeline import process_record
from synthesis.post_process.fix.schema import Candidate, Config, Plan
from synthesis.post_process.fix.storage import Cache
from synthesis.post_process.fix.trajectory_index import (
    index_record,
    image_bindings,
)
from synthesis.post_process.fix.tests.test_pipeline import fixture
from synthesis.post_process.fix.tests.test_integration import FakeModel, FakeBackend
from synthesis.post_process.fix.tests.writer_fixture import observation


class NaturalLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.record, self.source, self.target = fixture(self.root)
        self.window = index_record(self.record)[0]
        self.config = Config()
        self.anchor = {
            "anchor_image_id": "image_0",
            "anchor_image_ids": ["image_0"],
            "viewed_before_conclusion_ids": [],
            "target": "building",
            "entity": "Target",
        }

    def candidate(self, key, category):
        return Candidate(
            key,
            {
                "image_id": key,
                "source_page_id": "page_" + key,
                "title": "Building " + key,
                "source": "site",
            },
            image_paths=[str(self.target)],
            category=category,
            verdict="YES" if category in search_cards.POSITIVE else "NO",
            evidence=["Concrete paired roof comparison"],
        )

    def test_three_wrong_choices_feed_exact_writer_text_into_next_selection(self):
        cs = [self.candidate(str(i), "SIMILAR") for i in range(3)] + [
            self.candidate("correct", "CONSISTENT")
        ]
        plan = Plan("direct", cs, [], metadata={"flow_version": "selector_writer_v1"})
        events, selector_payloads, writer_payloads = [], [], []
        model = Mock()

        def ask(alias, prompt, payload, images=()):
            if prompt == rollout.REASONS:
                return {
                    "cards": [
                        {
                            "image_id": c["image_id"],
                            "reasoning": "The title suggests a building to inspect.",
                        }
                        for c in payload["search_results"]
                    ]
                }
            if prompt == rollout.SELECT:
                events.append("select")
                selector_payloads.append(copy.deepcopy(payload))
                self.assertFalse(images)
                self.assertNotIn("category", json.dumps(payload))
                self.assertNotIn("designated", json.dumps(payload))
                self.assertNotIn("references_to_check", payload)
                return {
                    "image_id": payload["uninspected_image_ids"][0],
                    "reasoning": "Check this building lead.",
                }
            if prompt == rewrite.AFTER:
                events.append("write")
                writer_payloads.append(copy.deepcopy(payload))
                self.assertEqual(len(images), 2)
                return observation(
                    "Visible roof comparison for "
                    + payload["selected_card"]["image_id"],
                    payload["verdict"],
                )
            self.fail("Unexpected stage")

        model.ask.side_effect = ask
        steps = rollout.run_loop(
            self.record, self.window, plan, self.anchor, self.source, model, self.config
        )
        self.assertEqual(events, ["select", "write"] * 4)
        self.assertEqual(plan.branch, "contrast")
        self.assertEqual(plan.metadata["wrong_count"], 3)
        self.assertEqual(len(steps), 4)
        for i in range(1, 4):
            previous = steps[i - 1]
            self.assertEqual(
                selector_payloads[i]["previous_observations"][-1]["observation"],
                previous["after"],
            )
            self.assertIn(
                previous["after"], json.dumps(writer_payloads[i]["trajectory_history"])
            )
        self.assertTrue(
            all(x["mode"] == "autonomous" for x in plan.metadata["selection_modes"])
        )

    def test_unrelated_not_given_a_rationale_or_selectable(self):
        cards = [
            self.candidate("bad", "UNRELATED"),
            self.candidate("good", "CONSISTENT"),
        ]
        model = Mock()

        def ask(alias, prompt, payload, images=()):
            if prompt == rollout.REASONS:
                self.assertEqual(
                    [c["image_id"] for c in payload["search_results"]], ["good"]
                )
                return {
                    "cards": [
                        {"image_id": "good", "reasoning": "Inspect the building card."}
                    ]
                }
            self.assertEqual(payload["uninspected_image_ids"], ["good"])
            return {"image_id": "bad", "reasoning": "Inspect it."}

        model.ask.side_effect = ask
        with self.assertRaisesRegex(ValueError, "unavailable"):
            rollout.run_loop(
                self.record,
                self.window,
                Plan("direct", cards, []),
                self.anchor,
                self.source,
                model,
                self.config,
            )

    def test_reuse_original_correct_never_calls_writer_or_repeats_image(self):
        record, source, target = fixture(self.root, True)
        backend, model = FakeBackend(target), FakeModel(wrong_first=True)
        output, audit = process_record(
            record, self.root, self.root, self.config, backend, model, Cache(self.root)
        )
        self.assertEqual(audit["status"], "rewritten", audit)
        w = audit["windows"][0]
        # Insertion-only polishing must not rewrite the original continuation.
        original_turns = record["conversations"]
        new_turns = output["conversations"]
        for old_index, new_index in w["splice"]["continuation_map"].items():
            old_index = int(old_index)
            old_text = original_turns[old_index]["value"]
            new_text = new_turns[new_index]["value"]
            if old_index == index_record(record)[0].start:
                self.assertTrue(new_text.endswith(old_text.split(">", 1)[1]))
            else:
                self.assertEqual(new_text, old_text)
        self.assertEqual(w["plan"]["metadata"]["reuse_original_read"], "image_0")
        self.assertEqual(model.observed_verdicts, ["NO"])
        self.assertEqual(len(output["images"]), len(record["images"]) + 1)
        original_obs = record["conversations"][4]
        self.assertEqual(output["conversations"].count(original_obs), 1)
        self.assertEqual(w["splice"]["replaced"], 0)
        image_bindings(output)

    def test_another_positive_is_an_explicit_skip_for_viewed_record(self):
        old = self.candidate("old_image_0", "CONSISTENT")
        other = self.candidate("new", "SAME_ENTITY")
        plan = Plan(
            "direct",
            [other, old],
            [],
            metadata={"original_correct_views": {old.key: "image_0"}},
        )
        model = Mock()
        model.ask.side_effect = [
            {
                "cards": [
                    {
                        "image_id": c.public["image_id"],
                        "reasoning": "Inspect its building title.",
                    }
                    for c in plan.first
                ]
            },
            {"image_id": "new", "reasoning": "Check this building lead."},
        ]
        with self.assertRaises(SkipAugmentation) as caught:
            rollout.run_loop(
                self.record,
                self.window,
                plan,
                self.anchor,
                self.source,
                model,
                self.config,
            )
        self.assertEqual(caught.exception.reason, "different_correct_candidate")
        self.assertEqual(model.ask.call_count, 2)

    def test_unviewed_same_entity_is_terminal_without_exact_photo_claim_requirement(
        self,
    ):
        c = self.candidate("only", "SAME_ENTITY")
        plan = Plan("direct", [c], [])
        model = Mock()
        model.ask.side_effect = [
            {"cards": [{"image_id": "only", "reasoning": "Inspect the building."}]},
            {"image_id": "only", "reasoning": "Inspect this lead."},
            observation(
                "The same distinctive tower is visible from a different angle."
            ),
        ]
        steps = rollout.run_loop(
            self.record, self.window, plan, self.anchor, self.source, model, self.config
        )
        self.assertEqual(plan.positive_key, "only")
        self.assertEqual(len(steps), 1)

    def test_uncertain_writer_stops_loop_without_fabricated_rejection(self):
        cs = [self.candidate("wrong", "SIMILAR"), self.candidate("right", "CONSISTENT")]
        plan = Plan("direct", cs, [])
        model = Mock()
        model.ask.side_effect = [
            {
                "cards": [
                    {"image_id": c.key, "reasoning": "Inspect this lead."} for c in cs
                ]
            },
            {"image_id": "wrong", "reasoning": "Inspect this lead."},
            observation("Different pose, identity unresolved", "CONTINUE"),
        ]
        with self.assertRaises(ReviewRejected):
            rollout.run_loop(
                self.record,
                self.window,
                plan,
                self.anchor,
                self.source,
                model,
                self.config,
            )
        self.assertEqual(model.ask.call_count, 3)
        self.assertEqual(len(plan.attempts), 1)
        self.assertEqual(plan.metadata["steps"], [])
        self.assertEqual(
            plan.metadata["writer_assessments"][0]["assessment"]["decision"],
            "UNCERTAIN",
        )

    def test_three_references_retained_without_mandatory_three_checks(self):
        old = [
            self.candidate("old_image_" + str(i), "CONSISTENT" if i == 0 else "RELATED")
            for i in range(3)
        ]
        anchor = {**self.anchor, "anchor_image_ids": ["image_0", "image_1", "image_2"]}
        plan = search_cards.build_search_plan(
            old, [], anchor, self.window, Mock(), self.config
        )
        self.assertEqual(len(plan.first), 3)
        self.assertEqual(plan.metadata["retained_reference_keys"], [c.key for c in old])
        self.assertNotIn("required_reference_keys", plan.metadata)
        self.assertFalse(search_cards.is_convergent(plan))

    def test_composer_cannot_drop_required_reference(self):
        old = [self.candidate("old_image_0", "CONSISTENT")]
        pool = [self.candidate(str(i), "RELATED") for i in range(5)]
        model = Mock()
        model.ask.return_value = {"candidate_keys": [c.key for c in pool]}
        with self.assertRaisesRegex(ValueError, "required references"):
            search_cards.build_search_plan(
                old, pool, self.anchor, self.window, model, self.config
            )

    def test_positive_cap_and_unrelated_filler(self):
        old = [self.candidate("old_image_0", "CONSISTENT")]
        pool = [self.candidate("p" + str(i), "CONSISTENT") for i in range(8)]
        pool += [self.candidate("f" + str(i), "UNRELATED") for i in range(3)]
        plan = search_cards.build_search_plan(
            old, pool, self.anchor, self.window, Mock(), self.config
        )
        self.assertEqual(len(plan.first), 5)
        self.assertEqual(
            sum(c.category in search_cards.POSITIVE for c in plan.first), 2
        )
        self.assertEqual(sum(c.category == "UNRELATED" for c in plan.first), 3)

    def test_required_positives_override_cap(self):
        old = [self.candidate("old_image_" + str(i), "CONSISTENT") for i in range(3)]
        pool = [
            self.candidate("p", "CONSISTENT"),
            self.candidate("n", "SIMILAR"),
            self.candidate("f", "UNRELATED"),
        ]
        anchor = {**self.anchor, "anchor_image_ids": ["image_0", "image_1", "image_2"]}
        plan = search_cards.build_search_plan(
            old, pool, anchor, self.window, Mock(), self.config
        )
        self.assertEqual(len(plan.first), 5)
        self.assertEqual(
            sum(c.category in search_cards.POSITIVE for c in plan.first), 3
        )
        self.assertTrue({c.key for c in old} <= {c.key for c in plan.first})

    def test_negative_priority_and_no_positive_padding(self):
        old = [self.candidate("old_image_0", "CONSISTENT")]
        pool = [self.candidate("p" + str(i), "CONSISTENT") for i in range(7)]
        pool.append(self.candidate("n", "RELATED"))
        plan = search_cards.build_search_plan(
            old, pool, self.anchor, self.window, Mock(), self.config
        )
        self.assertEqual(len(plan.first), 3)
        self.assertEqual(
            sum(c.category in search_cards.POSITIVE for c in plan.first), 2
        )

    def test_display_order_is_reproducible_and_not_positive_first(self):
        old = [self.candidate("old_image_0", "CONSISTENT")]
        pool = [self.candidate(str(i), "RELATED") for i in range(4)]
        ranks = set()
        for seed in range(20):
            config = Config(seed=seed)
            plans = [
                search_cards.build_search_plan(
                    old, pool, self.anchor, self.window, Mock(), config
                )
                for _ in range(2)
            ]
            keys = [[c.key for c in p.first] for p in plans]
            self.assertEqual(keys[0], keys[1])
            self.assertEqual(set(keys[0]), {c.key for c in old + pool})
            ranks.add(keys[0].index("old_image_0"))
        self.assertGreater(len(ranks), 1)

    def test_context_editor_receives_complete_pre_search_evidence(self):
        from synthesis.post_process.fix import context_patch
        from synthesis.post_process.fix.trajectory_index import messages, role, content

        model = FakeModel()
        original_ask = model.ask
        captured = []

        def ask(alias, prompt, payload, *args, **kwargs):
            if prompt == context_patch.PROMPT:
                captured.append(payload["evidence_history_before_search"])
            return original_ask(alias, prompt, payload, *args, **kwargs)

        model.ask = ask
        _, audit = process_record(
            self.record,
            self.root,
            self.root,
            self.config,
            FakeBackend(self.target),
            model,
            Cache(self.root),
        )
        self.assertEqual(audit["status"], "rewritten", audit)
        self.assertEqual(
            captured,
            [
                [
                    {"role": role(m), "text": content(m)}
                    for m in messages(self.record)[: self.window.call_index]
                ]
            ],
        )

    def test_no_progress_skips_without_network(self):
        model, backend = Mock(), Mock()
        model.ask.return_value = {"useful": False, "reason": "Search was abandoned"}
        output, audit = process_record(
            self.record,
            self.root,
            self.root,
            self.config,
            backend,
            model,
            Cache(self.root),
        )
        self.assertEqual(output, self.record)
        self.assertEqual(audit["status"], "unchanged")
        self.assertEqual(audit["windows"][0]["skip_reason"], "no_search_progress")
        backend.search.assert_not_called()

    def test_convergence_requires_five_visual_positives(self):
        cards = [
            self.candidate(str(i), "SAME_ENTITY" if i % 2 else "CONSISTENT")
            for i in range(5)
        ]
        self.assertTrue(search_cards.is_convergent(Plan("direct", cards, [])))
        self.assertFalse(search_cards.is_convergent(Plan("direct", cards[:4], [])))
        cards[-1].category = "SIMILAR"
        self.assertFalse(search_cards.is_convergent(Plan("direct", cards, [])))

    def test_page_returning_an_image_counts_as_actual_view(self):
        record, _, _ = fixture(self.root, True)
        turn = record["conversations"][3]
        turn["value"] = turn["value"].replace('"image_0"', '"page_0"')
        window = index_record(record)[0]
        self.assertIn("image_0", window.reads)
        self.assertEqual(window.reads["image_0"]["resource_id"], "page_0")

    def test_progress_retains_three_citations_and_actual_tool_dependencies(self):
        model = Mock()
        model.ask.return_value = {
            **self.anchor,
            "useful": True,
            "safe": True,
            "anchor_image_ids": ["image_1", "image_2", "image_3"],
            "dependency_image_ids": [],
        }
        result = progress.assess_progress(
            self.record, self.window, self.source, model, self.config
        )
        self.assertEqual(
            result["anchor_image_ids"], ["image_0", "image_1", "image_2", "image_3"]
        )


if __name__ == "__main__":
    unittest.main()
