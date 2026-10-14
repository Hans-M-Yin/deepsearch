"""Regressions for strict composition, natural selection and failure separation."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from synthesis.post_process.fix.candidate_sets import controlled_set, rng_for
from synthesis.post_process.fix.matching import same_metadata
from synthesis.post_process.fix.pipeline import process_record
from synthesis.post_process.fix.recovery import retry_query
from synthesis.post_process.fix.schema import Config, Plan
from synthesis.post_process.fix.selection import choose_checks
from synthesis.post_process.fix.storage import Cache
from synthesis.post_process.fix.trajectory_index import index_record, parse_observation
from synthesis.post_process.fix.tests.test_pipeline import fixture, candidates
from synthesis.post_process.fix.tests.test_integration import FakeModel, FakeBackend


class RefinementTests(unittest.TestCase):
    def test_target_evidence_policy_shared_across_visual_stages(self):
        from synthesis.post_process.fix.evidence_policy import TARGET_EVIDENCE
        from synthesis.post_process.fix.visual_evidence import PROMPT as VERIFY
        from synthesis.post_process.fix.precheck import PROMPT as PRECHECK
        from synthesis.post_process.fix.rewrite import AFTER
        from synthesis.post_process.fix.validate import JUDGE, BLIND_REVIEW

        for prompt in (VERIFY, PRECHECK, AFTER, JUDGE, BLIND_REVIEW):
            self.assertIn(TARGET_EVIDENCE, prompt)

    def test_lightweight_pool_keeps_same_entity_photos(self):
        for c in self.pool:
            c.category, c.verdict = "SAME_ENTITY", "CONTINUE"
        for c in self.pool[:2]:
            c.category, c.verdict = "SIMILAR", "NO"
        first, meta = controlled_set(
            self.old, self.pool, self.old[0], [], Config(), rng_for(1, "a", 1)
        )
        self.assertEqual(meta["composition_mode"], "lightweight")
        self.assertEqual(len(first), 5)
        self.assertEqual(sum(c.verdict == "NO" for c in first), 2)
        self.assertEqual(sum(c.verdict == "CONTINUE" for c in first), 2)

    def test_precheck_downgrade_precedes_next_selection_and_writer(self):
        from synthesis.post_process.fix.precheck import PROMPT as PRECHECK
        from synthesis.post_process.fix.selection import SELECT
        from synthesis.post_process.fix.rewrite import AFTER, write_trace

        plan = Plan(
            "direct",
            [self.old[0], *self.pool[:4]],
            [],
            metadata={"local_visual_target": "Find the same portrait"},
        )
        selected = []
        payloads = []

        def ask(alias, system, payload, images=()):
            if system == SELECT:
                if selected:
                    history = payload["previous_observations"][0]
                    self.assertIn("do not reject identity", history["conclusion"])
                c = self.pool[0] if not selected else self.old[0]
                selected.append(c)
                return {
                    "image_id": c.public["image_id"],
                    "reasoning": "Check this lead",
                }
            if system == PRECHECK:
                self.assertNotIn("entity", payload)
                return {
                    "category": "DIFFERENT_PHOTO"
                    if len(selected) == 1
                    else "CONSISTENT",
                    "evidence": [
                        "Different photograph, identity uncertain"
                        if len(selected) == 1
                        else "Same photograph"
                    ],
                }
            if system == AFTER:
                from synthesis.post_process.fix.tests.writer_fixture import observation

                payloads.append(payload)
                return observation(
                    "Comparison grounded in the observed evidence", payload["verdict"]
                )
            raise AssertionError(system)

        model = Mock()
        model.ask.side_effect = ask
        choose_checks(
            plan, self.window, self.source, model, Config(blind_visual_review=True)
        )
        self.assertEqual(plan.metadata["wrong_count"], 0)
        self.assertEqual(plan.metadata["inconclusive_count"], 1)
        self.assertEqual(plan.metadata["composition_mode"], "lightweight")
        write_trace(
            self.record,
            self.window,
            plan,
            {"target": "portrait"},
            self.source,
            model,
            Config(),
        )
        self.assertEqual(payloads[0]["category"], "DIFFERENT_PHOTO")
        self.assertEqual(payloads[0]["verdict"], "CONTINUE")

    def test_visual_categories_need_no_confidence_score(self):
        from synthesis.post_process.fix.visual_evidence import verify

        model, backend = Mock(), Mock()
        for category, verdict in (
            ("CONSISTENT", "YES"),
            ("SIMILAR", "NO"),
            ("UNRELATED", "NO"),
            ("UNRESOLVED", "UNCERTAIN"),
        ):
            model.ask.return_value = {
                "category": category,
                "evidence": ["Visible comparison"],
            }
            c = copy.deepcopy(self.old[0])
            verify(
                c, self.source, "question", "target", "entity", backend, model, Config()
            )
            self.assertEqual((c.category, c.verdict), (category, verdict))
        model.ask.return_value = {
            "verdict": "YES",
            "confidence": 0.99,
            "evidence": ["old schema"],
        }
        with self.assertRaisesRegex(ValueError, "category"):
            verify(
                c, self.source, "question", "target", "entity", backend, model, Config()
            )

    def test_negative_category_preference_and_fallback(self):
        from synthesis.post_process.fix.candidate_sets import sample_negatives

        self.pool[0].category = "UNRELATED"
        self.pool[1].category = "RELATED"
        selected = sample_negatives(
            self.pool, 3, rng_for(1, "a", 1), Config(similar_negative_probability=1)
        )
        self.assertTrue(all(c.category == "SIMILAR" for c in selected))
        selected = sample_negatives(
            self.pool, 3, rng_for(1, "a", 1), Config(similar_negative_probability=0)
        )
        self.assertEqual(selected[0].category, "RELATED")
        self.assertTrue(all(c.category != "UNRELATED" for c in selected))
        self.assertEqual(len({c.key for c in selected}), 3)

    def test_legacy_image_observation(self):
        self.assertEqual(
            parse_observation(
                '{"image_id":"img_3"}\nThe image is shown below:\n<image>'
            ),
            {"image_id": "img_3"},
        )
        for suffix in (
            '\n{"another":1}',
            "\narbitrary text",
            "\nThe image is shown below:",
        ):
            with self.assertRaises(ValueError):
                parse_observation('{"image_id":"img_3"}' + suffix)

    def test_unreadable_historical_cards_are_replaced_before_selection(self):
        for c in self.old[1:]:
            c.verdict = "UNCERTAIN"
            c.image_paths = []
        result, meta = controlled_set(
            self.old, self.pool, self.old[0], [], Config(), rng_for(1, "a", 1)
        )
        self.assertEqual(meta["replacement_count"], 4)
        self.assertTrue(all(c.verdict in {"YES", "NO"} for c in result))
        short, _ = controlled_set(
            self.old, [], self.old[0], [], Config(), rng_for(1, "a", 1)
        )
        self.assertEqual(short, [self.old[0]])

    def test_selector_receives_only_local_target_not_private_anchor(self):
        plan = Plan(
            "direct",
            [self.old[0], *self.pool[:4]],
            [],
            metadata={
                "local_visual_target": "Identify the left driver marked MAX",
                "anchor_image_id": "private-anchor",
            },
        )
        model = Mock()
        model.ask.return_value = {
            "image_id": self.old[0].public["image_id"],
            "reasoning": "Compare the left driver.",
        }
        choose_checks(
            plan, self.window, self.source, model, Config(blind_visual_review=False)
        )
        payload = model.ask.call_args.args[2]
        self.assertEqual(
            payload["local_visual_target"], plan.metadata["local_visual_target"]
        )
        self.assertNotIn("private-anchor", str(payload))
        self.assertNotIn("verdict", str(payload))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.record, self.source, self.target = fixture(self.root, True)
        self.window = index_record(self.record)[0]
        self.old, self.pool = candidates(self.window, self.target)

    def test_partial_replacement_cannot_leave_four_positives(self):
        for c in self.old:
            c.verdict = "YES"
        result, meta = controlled_set(
            self.old,
            self.pool,
            self.old[0],
            [],
            Config(replacement_counts=[1]),
            rng_for(1, "a", 1),
        )
        self.assertEqual(meta["composition_mode"], "lightweight")
        self.assertEqual(len(result), 5)

    def test_all_controlled_sets_obey_exact_counts(self):
        for c in self.old:
            c.verdict = "YES"
        pool = self.pool + copy.deepcopy(self.old[1:])
        for seed in range(30):
            result, meta = controlled_set(
                self.old,
                pool,
                self.old[0],
                [],
                Config(replacement_counts=[1, 2, 3, 4]),
                rng_for(seed, "a", 1),
            )
            self.assertEqual(len(result), 5)
            self.assertIn(sum(c.verdict == "YES" for c in result), {1, 2})
            self.assertEqual(
                sum(c.verdict == "YES" for c in result), meta["positive_count"]
            )
            self.assertTrue(all(c.verdict in {"YES", "NO"} for c in result))

    def test_preserved_list_is_a_separate_explicit_exception(self):
        for c in self.old:
            c.verdict = "YES"
            c.image_paths = [str(self.target)]
        result, meta = controlled_set(
            self.old,
            self.pool,
            self.old[0],
            [],
            Config(replacement_counts=[0]),
            rng_for(1, "a", 1),
        )
        self.assertEqual(result, self.old)
        self.assertEqual(meta["composition_mode"], "unchanged")

    def test_legacy_image_wrapper_survives_full_pipeline(self):
        for message in self.record["conversations"]:
            if message["from"] == "observation" and "<image>" in message["value"]:
                message["value"] = message["value"].replace(
                    "<image>", "The image is shown below:\n<image>"
                )
        _, audit = process_record(
            self.record,
            self.root,
            self.root,
            Config(
                branch_weights={
                    "convergent": 0,
                    "direct": 1,
                    "contrast": 0,
                    "recovery": 0,
                }
            ),
            FakeBackend(self.target),
            FakeModel(),
            Cache(self.root),
        )
        self.assertEqual(audit["status"], "rewritten", audit)

    def test_natural_first_correct_choice_is_not_forced_to_reject(self):
        plan = Plan(
            "contrast",
            [self.old[0], *self.pool[:4]],
            [self.pool[0], self.old[0]],
            positive_key=self.old[0].key,
        )
        model = FakeModel()
        choose_checks(plan, self.window, self.source, model, Config())
        self.assertEqual(plan.branch, "direct")
        self.assertEqual(len(plan.attempts), 1)
        self.assertEqual(plan.metadata["requested_branch"], "contrast")
        self.assertNotIn("verdict", str(model.before_payloads))

    def test_guided_candidate_must_be_obeyed_or_rejected(self):
        plan = Plan(
            "contrast", [self.old[0], *self.pool[:4]], [], positive_key=self.old[0].key
        )
        model = Mock()

        def pick(alias, system, payload, images):
            chosen = next(
                c for c in payload["uninspected_image_ids"] if c.startswith("new_")
            )
            return {"image_id": chosen, "reasoning": "Inspect this lead."}

        model.ask.side_effect = pick
        with self.assertRaisesRegex(ValueError, "designated image ID"):
            choose_checks(
                plan, self.window, self.source, model, Config(blind_visual_review=False)
            )
        self.assertEqual(model.ask.call_count, 4)

    def test_existing_multi_view_still_gets_progress_audit(self):
        window = copy.deepcopy(self.window)
        window.reads["another"] = window.reads["image_0"]
        model, backend = Mock(), Mock()
        model.ask.return_value = {"useful": False, "reason": "No progress"}
        with patch(
            "synthesis.post_process.fix.pipeline.index_record", return_value=[window]
        ):
            output, audit = process_record(
                self.record,
                self.root,
                self.root,
                Config(),
                backend,
                model,
                Cache(self.root),
            )
        self.assertEqual(output, self.record)
        self.assertEqual(audit["windows"][0]["skip_reason"], "no_search_progress")
        model.ask.assert_called_once()
        backend.search.assert_not_called()

    def test_retry_region_validation(self):
        plan = Plan("recovery", self.pool[:5], self.pool[:2])
        model = Mock()
        model.ask.return_value = {
            "strategy": "full",
            "reason": "Recover missing context.",
        }
        result = retry_query(self.window, plan, self.source, "img_1", model, Config())
        self.assertEqual(result["arguments"]["region"], [0, 0, 1000, 1000])
        self.assertEqual(result["arguments"]["image"], "img_1")
        model.ask.return_value = {
            "strategy": "target_crop",
            "region": [100, 100, 110, 110],
            "reason": "Tiny crop",
        }
        with self.assertRaisesRegex(ValueError, "too-small"):
            retry_query(self.window, plan, self.source, "img_1", model, Config())

    def test_conservative_normalized_matching(self):
        self.assertTrue(
            same_metadata(
                {"title": "A &amp; B!", "source": "Site"},
                {"title": "a & b", "source": "site"},
            )
        )
        self.assertFalse(
            same_metadata(
                {"title": "Building A", "source": "Site"},
                {"title": "Building B", "source": "Site"},
            )
        )

    def test_review_service_error_is_not_a_content_rejection(self):
        config = Config(
            branch_weights={"convergent": 0, "direct": 1, "contrast": 0, "recovery": 0}
        )
        with patch(
            "synthesis.post_process.fix.pipeline.semantic_review",
            side_effect=TimeoutError("offline"),
        ):
            _, audit = process_record(
                self.record,
                self.root,
                self.root,
                config,
                FakeBackend(self.target),
                FakeModel(),
                Cache(self.root),
            )
        self.assertEqual(audit["windows"][0]["failure_kind"], "model_or_tool_error")
        self.assertEqual(audit["windows"][0]["failure_stage"], "review")

    def test_distinct_judge_requirement_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "distinct judge"):
            Config(require_distinct_judge=True).validate()

    def test_blind_review_disagreement_rejects_without_labels(self):
        from synthesis.post_process.fix.validate import BLIND_REVIEW

        model = FakeModel()
        original_ask = model.ask

        def ask(alias, system, payload, images=()):
            if system == BLIND_REVIEW:
                self.assertEqual(set(payload), {"question"})
                return {
                    "comparisons": [
                        {"category": "UNRESOLVED", "evidence": ["Not enough detail"]}
                    ]
                }
            return original_ask(alias, system, payload, images)

        model.ask = ask
        config = Config(
            blind_visual_review=True,
            branch_weights={"convergent": 0, "direct": 1, "contrast": 0, "recovery": 0},
        )
        output, audit = process_record(
            self.record,
            self.root,
            self.root,
            config,
            FakeBackend(self.target),
            model,
            Cache(self.root),
        )
        self.assertEqual(output, self.record)
        self.assertEqual(audit["windows"][0]["failure_kind"], "review_rejected")

    def test_final_review_reuses_blind_checks_and_keeps_semantic_judge(self):
        from synthesis.post_process.fix.validate import BLIND_REVIEW, JUDGE

        model = FakeModel()
        ask = model.ask
        calls = []

        def tracked(alias, system, payload, images=()):
            calls.append(system)
            if system == BLIND_REVIEW:
                raise AssertionError("Redundant blind review")
            if system == JUDGE:
                self.assertIn("updated", payload)
                self.assertNotIn("original", payload)
            return ask(alias, system, payload, images)

        model.ask = tracked
        _, audit = process_record(
            self.record,
            self.root,
            self.root,
            Config(
                branch_weights={
                    "convergent": 0,
                    "direct": 1,
                    "contrast": 0,
                    "recovery": 0,
                }
            ),
            FakeBackend(self.target),
            model,
            Cache(self.root),
        )
        self.assertEqual(audit["status"], "rewritten")
        self.assertIn(JUDGE, calls)
        self.assertEqual(
            audit["windows"][0]["convergence"]["basis"],
            "five_final_cards_match_local_target",
        )


if __name__ == "__main__":
    unittest.main()
