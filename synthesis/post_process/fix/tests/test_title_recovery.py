import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

from synthesis.post_process.fix.storage import Cache
from synthesis.post_process.fix.title_recovery import search_title, recover_anchor
from synthesis.post_process.fix.schema import Candidate, Config


@dataclass
class Resource:
    resource_id: str
    search_tool: str = "t2i_search"


class TitleRecoveryTests(unittest.TestCase):
    def test_source_aligned_reference_keeps_real_visual_category(self):
        for category, viewed, expected in [
            ("SAME_ENTITY", False, "recovered"),
            ("SAME_ENTITY", True, "recovered"),
            ("RELATED", False, "recovered"),
            ("SIMILAR", False, "recovered"),
            ("UNRELATED", False, "recovered"),
            ("RELATED", True, "unresolved"),
            ("UNRESOLVED", False, "unresolved"),
        ]:
            with self.subTest(category=category, viewed=viewed):
                old = Candidate(
                    "old_image_old",
                    {
                        "image_id": "image_old",
                        "source_page_id": "page_old",
                        "title": "Historical reference",
                    },
                    historical=True,
                )
                found = Candidate(
                    "found",
                    {
                        "image_id": "new",
                        "source_page_id": "page_old",
                        "title": "Recovered reference",
                    },
                    image_paths=["real.jpg"],
                    category=category,
                    verdict="CONTINUE" if category == "SAME_ENTITY" else "NO",
                )
                historical = [old]
                with (
                    patch(
                        "synthesis.post_process.fix.title_recovery.search_title",
                        return_value=[found],
                    ),
                    patch(
                        "synthesis.post_process.fix.provenance.page_alignment",
                        return_value=True,
                    ),
                    patch(
                        "synthesis.post_process.fix.visual_evidence.verify_pool",
                        return_value=[],
                    ),
                ):
                    result = recover_anchor(
                        historical,
                        None,
                        {
                            "anchor_image_id": "image_old",
                            "viewed_before_conclusion_ids": ["image_old"]
                            if viewed
                            else [],
                        },
                        {},
                        None,
                        None,
                        Config(),
                        "source",
                    )
                self.assertEqual(result["status"], expected)
                if expected == "recovered":
                    self.assertEqual(historical[0].category, category)
                    self.assertEqual(historical[0].public["source_page_id"], "page_old")
                    self.assertEqual(
                        result["binding_basis"], "source_aligned_reference_image"
                    )

    def test_cross_page_requires_consistent_image_and_retains_old_page(self):
        for category, verdict, expected in [
            ("CONSISTENT", "YES", "recovered"),
            ("SAME_ENTITY", "CONTINUE", "unresolved"),
            ("SIMILAR", "NO", "unresolved"),
        ]:
            original = Candidate(
                "old_image_old",
                {
                    "image_id": "image_old",
                    "source_page_id": "page_old",
                    "title": "Original",
                },
                historical=True,
            )
            candidate = Candidate(
                "mirror",
                {
                    "image_id": "image_new",
                    "source_page_id": "page_mirror",
                    "title": "Mirror",
                },
                resources=[
                    {"kind": "image", "source_page_url": "https://mirror.test/file"}
                ],
            )
            historical = [original]

            def verify(items, *args):
                self.assertEqual(items, [candidate])
                candidate.category, candidate.verdict = category, verdict
                candidate.image_paths = ["verified.jpg"]
                return []

            with (
                patch(
                    "synthesis.post_process.fix.title_recovery.search_title",
                    return_value=[candidate],
                ),
                patch(
                    "synthesis.post_process.fix.provenance.page_alignment",
                    return_value=False,
                ),
                patch(
                    "synthesis.post_process.fix.visual_evidence.verify_pool",
                    side_effect=verify,
                ),
            ):
                audit = recover_anchor(
                    historical,
                    None,
                    {"anchor_image_id": "image_old"},
                    {},
                    None,
                    None,
                    Config(),
                    "source",
                )
            self.assertEqual(audit["status"], expected)
            if expected == "recovered":
                self.assertEqual(historical[0].public["source_page_id"], "page_old")
                self.assertEqual(
                    historical[0].resources[0]["source_page_url"],
                    "https://mirror.test/file",
                )
                self.assertEqual(
                    audit["binding_basis"], "visual_match_original_page_retained"
                )
            else:
                self.assertIs(historical[0], original)

    def test_common_candidate_schema_and_private_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            context = Mock()
            context.resolve_resource_id.side_effect = lambda rid: Resource(rid)
            backend = SimpleNamespace(
                cache=Cache(root), _context=Mock(return_value=context)
            )
            output = {
                "ok": True,
                "results": [
                    {
                        "title": "Original title",
                        "source": "Wikipedia",
                        "image_id": "image_x",
                        "source_page_id": "page_x",
                        "rank": 1,
                        "query": "private recovery query",
                    }
                ],
            }
            with patch(
                "synthesis.sft.api_tools.execute_tool_call",
                return_value=SimpleNamespace(output=output),
            ) as call:
                candidates = search_title(backend, "Original title")
                cached = search_title(backend, "Original title")
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.args[0], "t2i_search")
                self.assertEqual(candidates[0].public, cached[0].public)
                self.assertEqual(
                    set(candidates[0].public),
                    {"title", "source", "image_id", "source_page_id"},
                )
                self.assertEqual(
                    candidates[0].resources[0]["search_tool"], "t2i_search"
                )

    def test_empty_title_rejected(self):
        with self.assertRaises(ValueError):
            search_title(None, " ")

    def test_anchor_fallback_requires_alignment_and_visual_match(self):
        original = Candidate(
            "old_image_old",
            {"image_id": "image_old", "source_page_id": "page_old", "title": "old"},
            historical=True,
        )
        candidate = Candidate(
            "found",
            {"image_id": "image_new", "source_page_id": "page_old", "title": "new"},
            resources=[{"kind": "image"}],
        )
        historical = [original]

        def verify(items, *args):
            items[0].verdict = "YES"
            items[0].category = "CONSISTENT"
            items[0].image_paths = ["verified.jpg"]
            return []

        with (
            patch(
                "synthesis.post_process.fix.title_recovery.search_title",
                return_value=[candidate],
            ) as search,
            patch(
                "synthesis.post_process.fix.provenance.page_alignment",
                return_value=True,
            ),
            patch(
                "synthesis.post_process.fix.visual_evidence.verify_pool",
                side_effect=verify,
            ),
        ):
            audit = recover_anchor(
                historical,
                None,
                {"anchor_image_id": "image_old"},
                {},
                None,
                None,
                Config(),
                "source",
            )
            self.assertEqual(audit["status"], "recovered")
            self.assertEqual(historical[0].public["source_page_id"], "page_old")
            self.assertEqual(historical[0].public["image_id"], "image_new")
            self.assertTrue(historical[0].key.endswith("old_image_old"))
            self.assertEqual(
                recover_anchor(
                    historical,
                    None,
                    {"anchor_image_id": "image_old"},
                    {},
                    None,
                    None,
                    Config(),
                    "source",
                )["status"],
                "not_needed",
            )
            self.assertEqual(search.call_count, 1)
