import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from synthesis.post_process.fix.tests.test_pipeline import fixture, candidates
from synthesis.post_process.fix.visual_evidence import verify
from synthesis.post_process.fix.schema import Config
from synthesis.post_process.fix.replay_i2i import recover_historical
from synthesis.post_process.fix.trajectory_index import index_record


class ViewedSameEntityTests(unittest.TestCase):
    def test_only_previously_viewed_candidate_can_confirm_same_entity(self):
        with tempfile.TemporaryDirectory() as root:
            record, source, target = fixture(Path(root), viewed=True)
            window = index_record(record)[0]
            historical = recover_historical(window, [], root, record)
            self.assertTrue(historical[0].originally_viewed)
            self.assertFalse(historical[1].originally_viewed)
            model = Mock()
            backend = Mock()
            for viewed in (True, False):
                for category, expected in [
                    ("SAME_ENTITY", "YES" if viewed else "CONTINUE"),
                    ("DIFFERENT_PHOTO", "CONTINUE"),
                    ("SIMILAR", "NO"),
                ]:
                    c = historical[0]
                    c.originally_viewed = viewed
                    model.ask.return_value = {
                        "category": category,
                        "evidence": ["Visible comparison"],
                    }
                    verify(
                        c, str(source), "Q", "identity", "", backend, model, Config()
                    )
                    self.assertEqual(c.verdict, expected)
                    self.assertEqual(c.category, category)
