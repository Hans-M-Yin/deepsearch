"""Text-only adopted cards and legitimate repeat image reads remain eligible."""

import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from synthesis.post_process.fix.errors import SkipAugmentation
from synthesis.post_process.fix.progress import assess_progress
from synthesis.post_process.fix.schema import Config
from synthesis.post_process.fix.trajectory_index import (
    index_record,
    candidate_read_occurrences,
)
from synthesis.post_process.fix.pipeline import process_record
from synthesis.post_process.fix.storage import Cache
from synthesis.post_process.fix.tests.test_pipeline import fixture, action
from synthesis.post_process.fix.tests.test_integration import FakeModel, FakeBackend


class ProgressRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.record, self.source, self.target = fixture(self.root)
        self.window = index_record(self.record)[0]
        self.valid = {
            "useful": True,
            "safe": True,
            "anchor_image_ids": ["image_0"],
            "dependency_image_ids": [],
            "viewed_before_conclusion_ids": [],
            "target": "building",
            "entity": "Target",
            "reason": "Adopted title and page",
        }

    def test_empty_adopted_ids_repaired_without_requiring_an_image_view(self):
        model = Mock()
        invalid = {**self.valid, "anchor_image_ids": []}
        model.ask.side_effect = [invalid, dict(self.valid)]
        result = assess_progress(self.record, self.window, self.source, model, Config())
        self.assertEqual(result["anchor_image_ids"], ["image_0"])
        self.assertEqual(result["viewed_before_conclusion_ids"], [])
        self.assertEqual(result["progress_audit_attempts"], 2)
        payload = model.ask.call_args_list[1].args[2]
        self.assertIn("regardless", payload["validation_error"])
        self.assertEqual(payload["invalid_response"], invalid)
        self.assertIn(
            {"image_id": "image_0", "title": "Target", "page_ids": ["page_0"]},
            payload["candidate_reference_map"],
        )

    def test_failed_repair_stops_without_inventing_anchor(self):
        model = Mock()
        model.ask.return_value = {**self.valid, "anchor_image_ids": []}
        with self.assertRaisesRegex(ValueError, "bounded repair"):
            assess_progress(self.record, self.window, self.source, model, Config())
        self.assertEqual(model.ask.call_count, 2)

    def test_no_progress_and_unsafe_are_not_retried(self):
        for response, error in [
            ({"useful": False}, SkipAugmentation),
            ({**self.valid, "safe": False}, ValueError),
        ]:
            model = Mock()
            model.ask.return_value = response
            with self.assertRaises(error):
                assess_progress(self.record, self.window, self.source, model, Config())
            model.ask.assert_called_once()

    def test_unknown_id_is_not_silently_accepted(self):
        model = Mock()
        model.ask.return_value = {**self.valid, "anchor_image_ids": ["invented"]}
        with self.assertRaises(ValueError):
            assess_progress(
                self.record,
                self.window,
                self.source,
                model,
                Config(progress_audit_retries=0),
            )
        model.ask.assert_called_once()

    def test_dependency_and_explicit_prose_citation_join_required_set(self):
        self.record["conversations"].append(
            {
                "from": "gpt",
                "value": "The caption at page_3 supplies another lead; page_40 is unavailable.",
            }
        )
        model = Mock()
        model.ask.return_value = {**self.valid, "dependency_image_ids": ["image_2"]}
        result = assess_progress(self.record, self.window, self.source, model, Config())
        self.assertEqual(
            result["required_reference_ids"], ["image_0", "image_2", "image_3"]
        )
        self.assertEqual(result["anchor_image_ids"], result["required_reference_ids"])

    def test_invalid_dependencies_get_bounded_schema_repair(self):
        for dependencies in [
            None,
            ["page_0"],
            ["unknown"],
            ["image_1", "image_1"],
            [{}],
        ]:
            with self.subTest(dependencies=dependencies):
                model = Mock()
                model.ask.side_effect = [
                    {**self.valid, "dependency_image_ids": dependencies},
                    dict(self.valid),
                ]
                result = assess_progress(
                    self.record, self.window, self.source, model, Config()
                )
                self.assertEqual(result["progress_audit_attempts"], 2)
                self.assertIn(
                    "dependency_image_ids",
                    model.ask.call_args.args[2]["validation_error"],
                )

    def test_pipeline_retains_extra_dependency_and_original_keywords(self):
        from synthesis.post_process.fix import progress

        class Model(FakeModel):
            def ask(self, alias, system, payload, images=()):
                result = super().ask(alias, system, payload, images)
                if system == progress.PROMPT:
                    result["dependency_image_ids"] = ["image_2"]
                return result

        observation = self.record["conversations"][2]
        data = json.loads(observation["value"])
        data["results"][2]["image_url_keywords"] = "essential archive caption"
        observation["value"] = json.dumps(data)
        output, audit = process_record(
            self.record,
            self.root,
            self.root,
            Config(),
            FakeBackend(self.target),
            Model(),
            Cache(self.root),
        )
        self.assertEqual(audit["status"], "rewritten", audit)
        window = audit["windows"][0]
        self.assertEqual(
            window["anchor"]["required_reference_ids"], ["image_0", "image_2"]
        )
        self.assertIn(
            "old_image_2", window["plan"]["metadata"]["retained_reference_keys"]
        )
        results = json.loads(output["conversations"][2]["value"])["results"]
        self.assertTrue(
            any(
                c.get("image_url_keywords") == "essential archive caption"
                for c in results
            )
        )

    def repeated_record(self, consecutive=False):
        record, source, target = fixture(self.root, True)
        insertion = 5 if consecutive else len(record["conversations"]) - 1
        extra = [
            action(
                "read_url",
                {"resource_id": "image_0", "goal": "Inspect the lower-right signature"},
            ),
            {"from": "observation", "value": '{"image_id":"img_3"}\n<image>'},
        ]
        if not consecutive:
            extra = [
                action("t2t_search", {"query": "building signature"}),
                {"from": "observation", "value": '{"results":[]}'},
            ] + extra
        record["conversations"][insertion:insertion] = extra
        record["images"].append(str(target))
        return record, source, target

    def test_consecutive_and_later_reads_preserve_first_and_all_occurrences(self):
        for consecutive in [False, True]:
            with self.subTest(consecutive=consecutive):
                record, source, _ = self.repeated_record(consecutive)
                window = index_record(record)[0]
                read = window.reads["image_0"]
                self.assertEqual(read["turn"], 3)
                self.assertEqual(
                    [o["turn"] for o in candidate_read_occurrences(read)],
                    [3, 5 if consecutive else 9],
                )
                model = Mock()
                model.ask.return_value = {
                    **self.valid,
                    "viewed_before_conclusion_ids": ["image_0"],
                }
                assess_progress(record, window, source, model, Config())
                observed = model.ask.call_args.args[2]["actual_image_reads"]["image_0"][
                    "observations"
                ]
                self.assertEqual(len(observed), 2)
                self.assertIn("signature", observed[1]["action"])

    def test_pipeline_retains_both_original_reads_and_observations(self):
        record, _, target = self.repeated_record()
        output, audit = process_record(
            record,
            self.root,
            self.root,
            Config(),
            FakeBackend(target),
            FakeModel(),
            Cache(self.root),
        )
        self.assertEqual(audit["status"], "rewritten", audit)
        w = audit["windows"][0]
        self.assertEqual(w["plan"]["metadata"]["original_read_turn"], 3)
        self.assertEqual(w["splice"]["replaced"], 0)
        self.assertEqual(output["images"], record["images"])
        for i in [4, 10]:
            self.assertEqual(output["conversations"][i], record["conversations"][i])
        self.assertIn(
            "Inspect the lower-right signature", output["conversations"][9]["value"]
        )


if __name__ == "__main__":
    unittest.main()
