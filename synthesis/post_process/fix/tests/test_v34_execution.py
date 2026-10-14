import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from synthesis.post_process.fix import global_refine
from synthesis.post_process.fix.pipeline import process_record
from synthesis.post_process.fix.rollout import inspection_reasons
from synthesis.post_process.fix.schema import Candidate, Config
from synthesis.post_process.fix.storage import Cache
from synthesis.post_process.fix.tests.test_integration import FakeBackend, FakeModel
from synthesis.post_process.fix.tests.test_pipeline import fixture
from synthesis.post_process.fix.trajectory_index import messages
from synthesis.post_process.fix.validate import validate_reasoning_only


class V34ExecutionTests(unittest.TestCase):
    def test_pipeline_accepts_prefix_reasoning_refinement(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            record, _, target = fixture(root)
            model = FakeModel()
            base = model.ask

            def ask(alias, prompt, payload, *args, **kwargs):
                response = base(alias, prompt, payload, *args, **kwargs)
                if prompt == global_refine.PROMPT:
                    response["turns"][0]["reasoning"] = (
                        "I need a caption to identify this building."
                    )
                return response

            model.ask = ask
            updated, audit = process_record(
                record, root, root, Config(), FakeBackend(target), model, Cache(root)
            )
            self.assertEqual(audit["status"], "rewritten", audit)
            self.assertIn("I need a caption", messages(updated)[1]["value"])
            self.assertEqual(updated["conversations"][0], record["conversations"][0])

    def test_refinement_protects_all_non_reasoning_content(self):
        record = {
            "images": ["a.png"],
            "conversations": [
                {"from": "human", "value": "Question <image>"},
                {
                    "from": "gpt",
                    "value": '<thinking>Reason</thinking><tool_call>{"name":"x","arguments":{}}</tool_call>',
                },
                {"from": "observation", "value": "Result"},
                {
                    "from": "gpt",
                    "value": "<thinking>Reason</thinking><answer>A</answer>",
                },
            ],
        }
        updated = copy.deepcopy(record)
        updated["conversations"][1]["value"] = updated["conversations"][1][
            "value"
        ].replace("Reason", "Better reason")
        self.assertEqual(validate_reasoning_only(record, updated), [])
        for index in range(4):
            changed = copy.deepcopy(record)
            changed["conversations"][index]["value"] += (
                " changed" if index in (0, 2) else "<answer>Different</answer>"
            )
            self.assertTrue(validate_reasoning_only(record, changed), index)
        for kind in ["images", "count", "order", "role", "tool"]:
            changed = copy.deepcopy(record)
            if kind == "images":
                changed["images"] = ["other.png"]
            elif kind == "count":
                changed["conversations"].pop()
            elif kind == "order":
                changed["conversations"].reverse()
            elif kind == "role":
                changed["conversations"][1]["from"] = "human"
            else:
                changed["conversations"][1]["value"] = changed["conversations"][1][
                    "value"
                ].replace('"x"', '"y"')
            self.assertTrue(validate_reasoning_only(record, changed), kind)

    def test_rationale_repairs_are_bounded_and_text_only(self):
        candidates = [
            Candidate(k, {"image_id": k, "title": "Lead"}, category="SIMILAR")
            for k in ["a", "b"]
        ]
        good = {
            "cards": [
                {"image_id": k, "reasoning": "Inspect this lead."} for k in ["a", "b"]
            ]
        }
        bad_responses = [
            {"cards": [{"image_id": "typo", "reasoning": "Inspect."}]},
            {"cards": [good["cards"][0], good["cards"][0]]},
            {"cards": [good["cards"][0]]},
            {"cards": [None]},
            {"cards": [{"image_id": [], "reasoning": "Inspect."}]},
        ]
        for bad in bad_responses:
            with self.subTest(bad=bad):
                model = Mock()
                model.ask.side_effect = [bad, good]
                audit = {}
                result = inspection_reasons(
                    candidates,
                    SimpleNamespace(question="Q"),
                    [],
                    model,
                    Config(),
                    audit,
                )
                self.assertEqual(set(result), {"a", "b"})
                self.assertEqual(model.ask.call_count, 2)
                payload = model.ask.call_args.args[2]
                self.assertEqual(payload["allowed_image_ids"], ["a", "b"])
                self.assertNotIn("category", str(payload))
                self.assertEqual(len(model.ask.call_args.args), 3)
                self.assertEqual(
                    [x["valid"] for x in audit["inspection_reason_attempts"]],
                    [False, True],
                )
        model.ask.side_effect = None
        model.ask.return_value = bad_responses[0]
        model.reset_mock()
        with self.assertRaises(ValueError):
            inspection_reasons(
                candidates, SimpleNamespace(question="Q"), [], model, Config()
            )
        self.assertEqual(model.ask.call_count, 2)
