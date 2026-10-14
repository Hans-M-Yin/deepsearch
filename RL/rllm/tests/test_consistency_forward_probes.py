import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "consistency_debug", Path(__file__).resolve().parents[1] / "verl/verl/utils/consistency_debug.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProbeTest(unittest.TestCase):
    def test_raw_monitor_preserves_training_state(self):
        data = SimpleNamespace(meta_info={"temperature": .7})
        original = torch.tensor([[-.1, -.2]])
        rng = torch.get_rng_state().clone()

        def compute(data, calculate_entropy):
            self.assertEqual(data.meta_info["temperature"], 1.0)
            self.assertFalse(calculate_entropy)
            self.assertFalse(torch.is_grad_enabled())
            torch.rand(2)
            return original - .2, None

        actor = SimpleNamespace(compute_log_prob=compute)
        raw = module.raw_monitor_logprobs(actor, data, original)
        self.assertTrue(torch.allclose(raw, original - .2))
        self.assertEqual(data.meta_info["temperature"], .7)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        data.meta_info["temperature"] = 1.0
        self.assertIs(module.raw_monitor_logprobs(actor, data, original), original)
        data.meta_info["temperature"] = .7
        with patch.object(actor, "compute_log_prob", side_effect=ValueError("failed")):
            with self.assertRaises(ValueError):
                module.raw_monitor_logprobs(actor, data, original)
        self.assertEqual(data.meta_info["temperature"], .7)

    def test_restore_on_success_and_failure(self):
        for fail in (False, True):
            model = SimpleNamespace(patched=True)
            data = SimpleNamespace(meta_info={"temperature": .7, "max_token_len": 100})
            original = data.meta_info.copy()
            actor = SimpleNamespace(use_fused_kernels=True, actor_module=[model])
            seen = []

            def compute(data, calculate_entropy):
                seen.append((actor.use_fused_kernels, data.meta_info.copy()))
                torch.rand(1)
                if fail and not actor.use_fused_kernels:
                    raise ValueError("probe failure")
                return torch.zeros(1, 2), None

            actor.compute_log_prob = compute
            config = SimpleNamespace(use_dynamic_bsz=True, ppo_max_token_len_per_gpu=50,
                                     ppo_micro_batch_size_per_gpu=1)
            fake = SimpleNamespace(
                patch_fused_forward=lambda m: setattr(m, "patched", True),
                unpatch_fused_forward=lambda m: setattr(m, "patched", False),
            )
            rng = torch.get_rng_state().clone()
            with patch.dict(sys.modules, {"verl.models.mcore.model_forward_fused": fake}), patch.object(module, "emit"):
                if fail:
                    with self.assertRaises(ValueError):
                        module.forward_probes(actor, data, config, False)
                else:
                    result = module.forward_probes(actor, data, config, False)
                    self.assertEqual(len(result), 3)
            self.assertEqual(data.meta_info, original)
            self.assertTrue(actor.use_fused_kernels and model.patched)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(seen[1][1]["max_token_len"], 50)
            self.assertFalse(seen[2][0])


if __name__ == "__main__":
    unittest.main()
