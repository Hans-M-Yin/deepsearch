import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rc = load("audit_consistency", "rllm/utils/rollout_consistency.py")
worker = load("actor_tito_audit", "verl/verl/utils/tito_audit.py")


class TrainingAuditTest(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([0, 10, 11, 20, 21, 30, 31, 40, 0])
        self.attention = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 0])
        self.response = self.ids[3:].clone()
        self.mask = torch.tensor([1, 1, 0, 0, 1, 0])
        self.records = [dict(token_ids=[20, 21], prompt_length=2, prompt_fingerprint=rc.token_fingerprint([10, 11])),
                        dict(token_ids=[40], prompt_length=6, prompt_fingerprint=rc.token_fingerprint([10, 11, 20, 21, 30, 31]))]

    def verify(self):
        return rc.verify_tito_training_row(self.ids, self.attention, self.response, self.mask, self.records)

    def test_padded_multiturn_passes(self):
        self.assertEqual(self.verify(), 2)

    def test_observation_corruption_fails_even_when_actions_match(self):
        self.ids[5] = self.response[2] = 999
        with self.assertRaisesRegex(ValueError, "prefix differs"):
            self.verify()

    def test_action_corruption_fails(self):
        self.ids[3] = self.response[0] = 999
        with self.assertRaisesRegex(ValueError, "action IDs differ"):
            self.verify()

    def test_missing_fingerprint_fails(self):
        self.records[0].pop("prompt_fingerprint")
        with self.assertRaisesRegex(ValueError, "missing prompt fingerprint"):
            self.verify()

    def test_fatal_suffix_not_claimed_as_verified(self):
        self.mask[4] = 0
        self.assertEqual(self.verify(), 1)

    def test_actor_transport_and_reordering(self):
        ids = torch.stack([self.ids, self.ids + 100])
        attn = torch.stack([self.attention, self.attention])
        expected = worker.sequence_fingerprints(ids, attn)
        data = SimpleNamespace(batch=dict(input_ids=ids.flip(0), attention_mask=attn,
                                         responses=ids.flip(0)[:, 3:]),
                               non_tensor_batch=dict(tito_sequence_fingerprint=expected[::-1]))
        worker.verify_actor_input(data, "test_update")
        data.batch["input_ids"][0, 1] = 999
        with self.assertRaisesRegex(ValueError, "sequence changed"):
            worker.verify_actor_input(data, "test_update")


if __name__ == "__main__":
    unittest.main()
