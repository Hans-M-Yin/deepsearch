import unittest
from types import SimpleNamespace

from synthesis.post_process.fix.provenance import page_alignment
from synthesis.post_process.fix.context_patch import validate_trace_patches
from synthesis.post_process.fix.rewrite import checked_reasoning


class JointRefinementTests(unittest.TestCase):
    def test_provenance_requires_same_page_evidence(self):
        old = {"source_page_id": "page_old"}
        candidate = SimpleNamespace(
            resources=[
                {
                    "kind": "image",
                    "image_url": "https://c8.alamy.com/comp/2GCW275/photo.jpg",
                    "source_page_url": "https://www.alamy.com/photo.html",
                }
            ]
        )

        def record(page, photo):
            return {
                "conversations": [
                    {
                        "from": "observation",
                        "value": '{"page_id":"'
                        + page
                        + '","content":"Image ID:'
                        + photo
                        + '"}',
                    }
                ]
            }

        self.assertTrue(page_alignment(old, candidate, record("page_old", "2GCW275")))
        self.assertFalse(page_alignment(old, candidate, record("other", "2GCW275")))
        self.assertFalse(page_alignment(old, candidate, record("page_old", "OTHER")))
        self.assertFalse(page_alignment(old, candidate, {"conversations": []}))

    def test_trace_edit_is_atomic_and_preserves_actions(self):
        steps = [
            {"kind": "read", "candidate_key": "a", "before": "Before", "after": "After"}
        ]
        result, merged = validate_trace_patches(
            {
                "trace_patches": [
                    {
                        "step_id": "inserted_step_000000",
                        "after": "Concrete match evidence.",
                    }
                ],
                "transition_includes_final_check": True,
            },
            steps,
        )
        self.assertTrue(merged)
        self.assertEqual(steps[0]["after"], "After")
        self.assertEqual(result[0]["candidate_key"], "a")
        for patch in (
            {"step_id": "inserted_step_000000", "candidate_key": "b"},
            {
                "step_id": "inserted_step_000000",
                "after": "The preserved next step is useful.",
            },
            {"step_id": "inserted_step_999999", "after": "Bad ID"},
        ):
            with self.assertRaises(ValueError):
                validate_trace_patches({"trace_patches": [patch]}, steps)

    def test_writer_conflict_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "Unsafe writer evidence"):
            checked_reasoning(
                {
                    "unsafe": True,
                    "reason": "Identity unsupported",
                    "reasoning": "Ignore",
                }
            )
