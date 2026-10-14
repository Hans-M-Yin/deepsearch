import copy
import json
import tempfile
import unittest
from pathlib import Path

from synthesis.post_process.fix.schema import Candidate, Config
from synthesis.post_process.fix.trajectory_index import index_record, messages, content
from synthesis.post_process.fix.candidate_sets import (
    build_plan,
    rng_for,
    finish_recovery,
)
from synthesis.post_process.fix.splice import assemble
from synthesis.post_process.fix.context_patch import apply_reasoning
from synthesis.post_process.fix.validate import validate_structure
from synthesis.post_process.fix.storage import Cache


def action(name, args, reason="Reason"):
    return {
        "from": "gpt",
        "value": "<thinking>"
        + reason
        + "</thinking>\n<tool_call>"
        + json.dumps({"name": name, "arguments": args})
        + "</tool_call>",
    }


def fixture(root, viewed=False):
    from PIL import Image

    source = root / "source.png"
    target = root / "target.png"
    Image.new("RGB", (64, 64), "red").save(source)
    Image.new("RGB", (64, 64), "blue").save(target)
    cards = [
        {
            "title": "Target" if i == 0 else f"Other {i}",
            "source": "site",
            "image_id": f"image_{i}",
            "source_page_id": f"page_{i}",
        }
        for i in range(5)
    ]
    turns = [
        {"from": "human", "value": "Which city?\n<image>"},
        action("i2i_search", {"region": [0, 0, 1000, 1000]}),
        {"from": "observation", "value": json.dumps({"ok": True, "results": cards})},
    ]
    paths = [str(source)]
    if viewed:
        turns.extend(
            [
                action("read_url", {"resource_id": "image_0"}),
                {"from": "observation", "value": '{"image_id":"img_2"}\n<image>'},
            ]
        )
        paths.append(str(target))
    turns.extend(
        [
            action("read_url", {"resource_id": "page_0"}),
            {"from": "observation", "value": '{"text":"Target is in City"}'},
            {
                "from": "gpt",
                "value": "<thinking>Supported.</thinking><answer>City</answer>",
            },
        ]
    )
    specs = [
        {
            "type": "function",
            "function": {"name": "i2i_search", "parameters": {"properties": {}}},
        }
    ]
    record = {
        "id": "fixture",
        "system": "<tools>" + json.dumps(specs) + "</tools>",
        "conversations": turns,
        "images": paths,
    }
    return record, source, target


def candidates(window, target):
    historical = [
        Candidate("old_" + c["image_id"], c.copy(), historical=True)
        for c in window.candidates
    ]
    historical[0].verdict = "YES"
    historical[0].category = "CONSISTENT"
    historical[0].image_paths = [str(target)]
    historical[0].evidence = ["Matching distinctive window and roof"]
    pool = [
        Candidate(
            f"n{i}",
            {
                "title": f"Similar {i}",
                "image_id": f"new_{i}",
                "source_page_id": f"newpage_{i}",
                "source": "site",
            },
            image_paths=[str(target)],
            verdict="NO",
            category="SIMILAR",
            evidence=["Different window layout"],
        )
        for i in range(7)
    ]
    return historical, pool


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def plan(self, branch="contrast", viewed=False, seed=42):
        record, source, target = fixture(self.root, viewed)
        window = index_record(record)[0]
        historical, pool = candidates(window, target)
        config = Config(
            branch_weights={
                k: float(k == branch)
                for k in ("convergent", "direct", "contrast", "recovery")
            }
        )
        plan = build_plan(
            window,
            historical,
            pool,
            {"anchor_image_id": "image_0"},
            {"eligible": branch == "convergent"},
            config,
            rng_for(seed, "r", 1),
        )
        plan.metadata["anchor_image_id"] = "image_0"
        return record, source, target, window, historical, pool, plan

    def assemble(self, record, window, plan):
        steps = []
        for i, candidate in enumerate(plan.attempts):
            if plan.branch == "recovery" and i == plan.retry_after:
                steps.append(
                    {
                        "kind": "retry",
                        "before": "The inspected candidates failed; retry the original.",
                    }
                )
            steps.append(
                {
                    "kind": "read",
                    "candidate_key": candidate.key,
                    "before": "Inspect this candidate's distinctive features.",
                    "after": "Matches the distinctive roof."
                    if candidate.verdict == "YES"
                    else "The window layout differs.",
                }
            )
        patches = {
            window.end: apply_reasoning(
                messages(record)[window.end], "The entity is verified; read its source."
            )
        }
        return assemble(record, window, plan, steps, patches)

    def test_index_preserves_page_boundary_and_image_bindings(self):
        for viewed in [False, True]:
            record, _, _ = fixture(self.root, viewed)
            w = index_record(record)[0]
            self.assertEqual(w.end, 5 if viewed else 3)
            self.assertEqual(len(w.reads), int(viewed))
            self.assertIn("page_0", w.protected_ids)
            self.assertEqual(w.source_image, record["images"][0])

    def test_contrast_with_and_without_original_view(self):
        for viewed in [False, True]:
            record, _, _, w, _, _, plan = self.plan(viewed=viewed)
            updated, audit = self.assemble(record, w, plan)
            self.assertEqual(validate_structure(record, updated, w, plan, audit), [])
            self.assertEqual(
                content(messages(updated)[-1]), content(messages(record)[-1])
            )
            self.assertEqual(len(updated["images"]), 1 + len(plan.attempts))

    def test_direct_and_convergent(self):
        record, _, _, w, _, _, plan = self.plan("direct")
        self.assertEqual(len(plan.attempts), 1)
        updated, audit = self.assemble(record, w, plan)
        self.assertEqual(validate_structure(record, updated, w, plan, audit), [])
        record, _, _, w, _, _, plan = self.plan("convergent")
        updated, audit = assemble(record, w, plan, [], {})
        self.assertEqual(updated, record)
        self.assertEqual(validate_structure(record, updated, w, plan, audit), [])

    def test_randomization_includes_zero_one_two_and_varied_ranks(self):
        counts, ranks = set(), set()
        for seed in range(50):
            for branch in ["direct", "contrast"]:
                _, _, _, _, _, _, plan = self.plan(branch, seed=seed)
                counts.add(len(plan.attempts) - 1)
                ranks.add(
                    next(
                        i
                        for i, c in enumerate(plan.first)
                        if c.key == plan.positive_key
                    )
                )
        self.assertEqual(counts, {0, 1, 2})
        self.assertEqual(ranks, set(range(5)))

    def test_recovery_has_real_second_result_and_explicit_query_image(self):
        record, _, target, w, historical, pool, plan = self.plan(
            "recovery", viewed=True
        )
        positive = copy.deepcopy(historical[0])
        positive.key = "retry_positive"
        positive.public["image_id"] = "retry_image"
        plan.retry_call = {
            "name": "i2i_search",
            "arguments": {"image": "img_1", "region": [0, 0, 1000, 1000]},
        }
        finish_recovery(plan, [positive, *pool], historical[0], rng_for(42, "retry", 1))
        updated, audit = self.assemble(record, w, plan)
        self.assertEqual(validate_structure(record, updated, w, plan, audit), [])
        self.assertEqual(len(plan.attempts), 3)
        self.assertIn('"image": "img_1"', json.dumps(updated).replace('\\"', '"'))

    def test_unknown_images_never_become_negatives(self):
        record, _, target = fixture(self.root)
        w = index_record(record)[0]
        historical, pool = candidates(w, target)
        for c in pool:
            c.verdict = "UNCERTAIN"
        config = Config(
            branch_weights={"convergent": 0, "direct": 0, "contrast": 1, "recovery": 0}
        )
        with self.assertRaises(ValueError):
            build_plan(
                w,
                historical,
                pool,
                {"anchor_image_id": "image_0"},
                {},
                config,
                rng_for(42, "r", 1),
            )

    def test_changed_answer_is_rejected(self):
        record, _, _, w, _, _, plan = self.plan()
        updated, audit = self.assemble(record, w, plan)
        messages(updated)[-1]["value"] = "<answer>Wrong city</answer>"
        self.assertIn(
            "Final answer changed", validate_structure(record, updated, w, plan, audit)
        )

    def test_cache_and_atomic_checkpoint(self):
        cache = Cache(self.root)
        self.assertEqual(
            cache.get_or_compute("s", {"a": 1}, lambda: {"b": 2}), {"b": 2}
        )
        self.assertEqual(
            cache.get_or_compute("s", {"a": 1}, lambda: self.fail("Cache miss")),
            {"b": 2},
        )
        self.assertNotEqual(
            cache.get_or_compute("s", {"a": 2}, lambda: {"b": 3}), {"b": 2}
        )

    def test_context_patch_protects_tools_and_answers(self):
        message = action("read_url", {"resource_id": "page_0"})
        patched = apply_reasoning(message, "Confirmed the target.")
        self.assertIn('"resource_id": "page_0"', content(patched))
        with self.assertRaises(ValueError):
            apply_reasoning(message, "<tool_call>fabricated</tool_call>")

    def test_implicit_followup_search_keeps_original_image(self):
        record, _, _, w, _, _, plan = self.plan(viewed=False)
        record["conversations"][-1:-1] = [
            action("i2i_search", {}),
            {"from": "observation", "value": '{"results":[]}'},
        ]
        updated, audit = self.assemble(record, w, plan)
        self.assertTrue(audit["source_repairs"])
        self.assertEqual(
            next(iter(audit["source_repairs"].values()))["arguments"]["image"], "img_1"
        )
        self.assertEqual(validate_structure(record, updated, w, plan, audit), [])

    def test_hidden_crop_is_not_guessed_for_followup_search(self):
        record, _, _, w, _, _, plan = self.plan(viewed=False)
        record["conversations"][1] = action(
            "i2i_search", {"region": [100, 100, 400, 400]}
        )
        record["conversations"][-1:-1] = [
            action("i2i_search", {}),
            {"from": "observation", "value": '{"results":[]}'},
        ]
        with self.assertRaisesRegex(ValueError, "unrecorded crop"):
            self.assemble(record, w, plan)


if __name__ == "__main__":
    unittest.main()
