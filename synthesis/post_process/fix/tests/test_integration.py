"""Exercise real stage wiring with deterministic model/tool transports."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from synthesis.post_process.fix import (
    eligibility,
    visual_evidence,
    rewrite,
    context_patch,
    validate,
)
from synthesis.post_process.fix.schema import Config, Candidate
from synthesis.post_process.fix.storage import Cache, write_json, read_json
from synthesis.post_process.fix.pipeline import process_record
from synthesis.post_process.fix.trajectory_index import messages, content
from synthesis.post_process.fix.tests.test_pipeline import fixture


class FakeModel:
    def __init__(self, approve=True, convergent=False, wrong_first=False):
        self.approve, self.convergent = approve, convergent
        self.before_payloads = []
        self.wrong_first = wrong_first
        self.observed_verdicts = []

    def ask(self, alias, system, payload, images=()):
        from synthesis.post_process.fix import progress, search_cards, rollout

        if system == progress.PROMPT:
            return {
                "useful": True,
                "safe": True,
                "anchor_image_ids": ["image_0"],
                "dependency_image_ids": [],
                "viewed_before_conclusion_ids": list(payload["actual_image_reads"]),
                "entity": "Target",
                "target": "building",
                "reason": "Used the building identification",
            }
        if system == search_cards.COMPOSE:
            required = payload["required_keys"]
            others = [
                c["key"]
                for c in sorted(
                    payload["candidates"],
                    key=lambda c: c["category"] not in {"RELATED", "SIMILAR"},
                )
                if c["key"] not in required
            ]
            return {
                "candidate_keys": required
                + others[: payload["result_count"] - len(required)]
            }
        if system == rollout.REASONS:
            return {
                "cards": [
                    {
                        "image_id": c["image_id"],
                        "reasoning": "Inspect the building suggested by this card.",
                    }
                    for c in payload["search_results"]
                ]
            }
        from synthesis.post_process.fix.candidate_relevance import PROMPT as COMPETITION

        if system == COMPETITION:
            return {
                "cards": [
                    {
                        "image_id": c["image_id"],
                        "competitive": True,
                        "reason": "Specific matching visual clue",
                    }
                    for c in payload["search_results"]
                ]
            }
        from synthesis.post_process.fix.selection import SELECT
        from synthesis.post_process.fix.recovery import PROMPT as RECOVERY
        from synthesis.post_process.fix.precheck import PROMPT as PRECHECK

        if system == PRECHECK:
            return {
                "category": "CONSISTENT"
                if self.selected_title == "Target"
                else "SIMILAR",
                "evidence": ["visible comparison"],
            }

        if system in (SELECT, rollout.SELECT):
            self.before_payloads.append(payload)
            cards = [
                c
                for c in payload["search_results"]
                if c["image_id"] in payload["uninspected_image_ids"]
            ]
            preferred = [
                c
                for c in cards
                if (
                    c["title"] != "Target"
                    if self.wrong_first and not payload["previous_observations"]
                    else c["title"] == "Target"
                )
            ]
            # These fixtures exercise retained-read reuse, independently of
            # public display order. The real Selector receives no such hint.
            preferred.sort(key=lambda c: c["image_id"] != "image_0")
            self.selected_title = (preferred or cards)[0]["title"]
            return {
                "image_id": (preferred or cards)[0]["image_id"],
                "reasoning": "Inspect the visible lead.",
            }
        if system == RECOVERY:
            return {
                "strategy": "same",
                "reason": "Retry the original image after inconsistent leads.",
            }
        if system == validate.BLIND_REVIEW:
            return {
                "comparisons": [
                    {
                        "category": "CONSISTENT" if v == "YES" else "SIMILAR",
                        "evidence": ["visible comparison"],
                    }
                    for v in self.observed_verdicts
                ]
            }
        if system == eligibility.ANCHOR_PROMPT:
            return {
                "safe": True,
                "anchor_image_id": "image_0",
                "entity": "Target",
                "target": "building",
            }
        if system == eligibility.CONVERGENCE_PROMPT:
            return {
                "supporting_image_ids": [f"image_{i}" for i in range(5)],
                "independent_sources": 5,
                "competing_entity": False,
                "target_unambiguous": self.convergent,
            }
        if system == visual_evidence.PROMPT:
            yes = self.convergent or payload["candidate_title"] == "Target"
            return {
                "category": "CONSISTENT" if yes else "SIMILAR",
                "evidence": ["Matching roof" if yes else "Different roof"],
            }
        if system == rewrite.BEFORE:
            self.before_payloads.append(payload)
            return {
                "reasoning": "I will inspect the candidate suggested by this title."
            }
        if system == rewrite.AFTER:
            from synthesis.post_process.fix.tests.writer_fixture import observation

            self.observed_verdicts.append(payload["verdict"])
            return observation(
                "The roof matches."
                if payload["verdict"] == "YES"
                else "The roof differs; reject this candidate.",
                payload["verdict"],
            )
        if system == rewrite.RETRY:
            return {
                "reasoning": "The inspected candidates did not match; retry the original image."
            }
        if system == context_patch.PROMPT:
            return {
                "trace_patches": [
                    {
                        "step_id": step["step_id"],
                        "before": step["before"],
                        **({"after": step["after"]} if step["kind"] == "read" else {}),
                    }
                    for step in payload["editable_trace"]
                ]
            }
        if system == validate.JUDGE:
            return {
                "pass": self.approve,
                "issues": [] if self.approve else ["Mismatch"],
            }
        from synthesis.post_process.fix import global_refine

        if system == global_refine.PROMPT:
            return {
                "severe_errors": [],
                "turns": [
                    {
                        "assistant_id": row["assistant_id"],
                        "issues": [],
                        "reasoning": None,
                    }
                    for row in payload["trajectory"]
                    if "assistant_id" in row
                ],
            }
        raise AssertionError("Unknown model stage")


class FakeBackend:
    def __init__(self, image):
        self.image, self.searches, self.reads = str(image), [], []

    def search(self, source, call, metadata=None, nonce="first"):
        self.searches.append((source, call, nonce))
        return [
            Candidate(
                f"{nonce}_{i}",
                {
                    "title": "Target" if i == 0 else f"Wrong {i}",
                    "source": "site",
                    "image_id": f"{nonce}_image_{i}",
                    "source_page_id": f"{nonce}_page_{i}",
                },
            )
            for i in range(15)
        ]

    def read(self, candidate):
        if not candidate.image_paths:
            self.reads.append(candidate.key)
            candidate.image_paths = [self.image]
            candidate.read_observation = '{"image_id":"img_1"}'
        return candidate


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        # This fake backend resolves legacy IDs without real URL resources.
        # Real title recovery is covered separately in test_title_recovery.
        recovery = patch(
            "synthesis.post_process.fix.title_recovery.recover_anchor",
            return_value={"status": "not_needed"},
        )
        recovery.start()
        self.addCleanup(recovery.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_natural_outcomes_and_original_view_states(self):
        for branch in ["convergent", "direct", "contrast"]:
            for viewed in (
                [False] if branch == "convergent" else [False, True, "after_page"]
            ):
                with self.subTest(branch=branch, viewed=viewed):
                    original, source, target = fixture(self.root, viewed)
                    if viewed == "after_page":
                        turns = original["conversations"]
                        turns[3:7] = turns[5:7] + turns[3:5]
                    if branch == "convergent":
                        obs = json.loads(original["conversations"][2]["value"])
                        for i, card in enumerate(obs["results"]):
                            card["source"] = f"independent site {i}"
                        original["conversations"][2]["value"] = json.dumps(obs)
                    model, backend = (
                        FakeModel(
                            convergent=branch == "convergent",
                            wrong_first=branch == "contrast",
                        ),
                        FakeBackend(target),
                    )
                    config = Config(
                        branch_weights={
                            k: float(k == branch)
                            for k in ["convergent", "direct", "contrast", "recovery"]
                        }
                    )
                    output, audit = process_record(
                        original,
                        self.root,
                        self.root / branch,
                        config,
                        backend,
                        model,
                        Cache(self.root / "cache"),
                    )
                    if branch == "convergent":
                        self.assertEqual(output, original)
                        self.assertEqual(len(backend.searches), 1)
                        self.assertEqual(audit["windows"][0]["status"], "convergent")
                    else:
                        self.assertEqual(audit["status"], "rewritten", audit)
                        self.assertEqual(audit["windows"][0]["branch"], branch)
                        self.assertEqual(
                            content(messages(output)[-1]),
                            content(messages(original)[-1]),
                        )
                        self.assertTrue(model.before_payloads)
                        for payload in model.before_payloads:
                            self.assertNotIn("verdict", payload)
                            self.assertNotIn("evidence", payload)
                        if viewed:
                            self.assertNotIn("old_image_0", backend.reads)
                    if branch == "recovery":
                        self.assertEqual(
                            [s[2] for s in backend.searches], ["first", "retry"]
                        )
                        self.assertEqual(
                            backend.searches[1][1]["arguments"]["image"], "img_1"
                        )

    def test_failed_independent_review_preserves_original(self):
        original, _, target = fixture(self.root, True)
        config = Config(
            branch_weights={"convergent": 0, "direct": 1, "contrast": 0, "recovery": 0}
        )
        updated, audit = process_record(
            original,
            self.root,
            self.root,
            config,
            FakeBackend(target),
            FakeModel(approve=False),
            Cache(self.root / "cache"),
        )
        self.assertEqual(updated, original)
        self.assertEqual(audit["status"], "quarantined")

    def test_cli_export_and_resume(self):
        from synthesis.post_process.fix.run import main

        record, _, _ = fixture(self.root)
        source, output, work = (
            self.root / "input.json",
            self.root / "output.json",
            self.root / "work",
        )
        write_json(source, [record])
        argv = [
            "--input",
            str(source),
            "--workdir",
            str(work),
            "--output",
            str(output),
            "--mode",
            "augment",
        ]
        with patch(
            "synthesis.post_process.fix.pipeline.process_record",
            return_value=(record, {"status": "rewritten", "windows": []}),
        ) as process:
            self.assertEqual(main(argv), 0)
            self.assertEqual(main(argv), 0)
            self.assertEqual(process.call_count, 1)
        self.assertEqual(read_json(output), [record])
        with self.assertRaises(ValueError):
            main(
                [
                    "--input",
                    str(source),
                    "--workdir",
                    str(work),
                    "--output",
                    str(source),
                    "--mode",
                    "augment",
                ]
            )

    def test_full_export_keeps_previous_selected_batches(self):
        from synthesis.post_process.fix.run import main

        record, _, _ = fixture(self.root)
        source, output, work = (
            self.root / "input.json",
            self.root / "output.json",
            self.root / "work",
        )
        write_json(source, [{**record, "id": "first"}, {**record, "id": "second"}])

        def process(record, *args):
            return {**record, "id": record["id"] + "_augmented"}, {
                "status": "rewritten",
                "windows": [],
            }

        argv = [
            "--input",
            str(source),
            "--workdir",
            str(work),
            "--output",
            str(output),
            "--mode",
            "augment",
        ]
        with patch(
            "synthesis.post_process.fix.pipeline.process_record", side_effect=process
        ):
            main(argv + ["--indices", "0"])
            main(argv + ["--indices", "1"])
        self.assertEqual(
            [r["id"] for r in read_json(output)],
            ["first_augmented", "second_augmented"],
        )


class BackendTests(unittest.TestCase):
    def test_private_limit_does_not_change_default(self):
        from synthesis.sft import tools

        lookup = Mock(
            return_value=[
                {"title": str(i), "imageUrl": f"https://example.com/{i}.jpg"}
                for i in range(20)
            ]
        )
        private = tools.i2i_search(
            "https://example.com/source.jpg",
            visual_lookup=lookup,
            top_k=15,
            private_result_limit=15,
        )
        default = tools.i2i_search(
            "https://example.com/source.jpg", visual_lookup=lookup, top_k=15
        )
        self.assertEqual(len(private["matches"]), 15)
        self.assertEqual(len(default["matches"]), 5)
        self.assertEqual(tools.MAX_SEARCH_RESULTS, 5)

    def test_serper_num_requested_only_in_private_mode(self):
        from synthesis.sft import tools

        response = Mock()
        response.json.return_value = {"organic": [{"title": str(i)} for i in range(20)]}
        with (
            patch.object(tools, "acquire_serper_api_key", return_value=("test", {})),
            patch.object(tools.requests, "post", return_value=response) as post,
        ):
            results = tools._image_search_via_serper(
                "https://example.com/a.jpg", top_k=15, requested_results=15
            )
            self.assertEqual(post.call_args.kwargs["json"]["num"], 15)
            self.assertEqual(len(results), 15)
            tools._image_search_via_serper("https://example.com/a.jpg")
            self.assertNotIn("num", post.call_args.kwargs["json"])

    def test_real_adapter_registers_page_and_image_resources(self):
        from synthesis.post_process.fix.replay_i2i import ToolBackend

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = ToolBackend(root, Config(), Cache(root))
            ctx = backend._context("test")
            compact = ctx.postprocess_search_output(
                "i2i_search",
                {
                    "ok": True,
                    "matches": [
                        {
                            "title": "A",
                            "source": "site",
                            "link": "https://example.com/page",
                            "imageUrl": "https://example.com/photo.jpg",
                        }
                    ],
                },
                timings={},
            )
            item = compact["results"][0]
            self.assertIsNotNone(ctx.resolve_resource_id(item["image_id"]))
            self.assertIsNotNone(ctx.resolve_resource_id(item["source_page_id"]))


if __name__ == "__main__":
    unittest.main()
