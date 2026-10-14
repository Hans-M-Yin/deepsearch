"""CPU tests execute the installed repository's actual verl TIS functions."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("token_tis", ROOT / "rllm/utils/token_tis.py")
tis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tis)


def load_verl_functions():
    # Avoid importing GPU/distributed dependencies for a CPU numerical test.
    tree = ast.parse((ROOT / "verl/verl/trainer/ppo/rollout_corr_helper.py").read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (
        "compute_rollout_correction_weights", "compute_is_metrics")]
    functional = SimpleNamespace(masked_mean=lambda x, m, axis=None:
                                 (x * m).sum(dim=axis) / (m.sum(dim=axis) + 1e-8))
    namespace = {"torch": torch, "verl_F": functional, "SAFETY_BOUND": 20.0}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "verl_tis_functions", "exec"), namespace)
    return SimpleNamespace(**namespace)


class TokenTISTest(unittest.TestCase):
    def test_actual_ppo_loss_uses_tis_weights(self):
        tree = ast.parse((ROOT / "verl/verl/trainer/ppo/core_algos.py").read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("agg_loss", "compute_policy_loss_vanilla")]
        for node in funcs:
            node.decorator_list = []
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        namespace = {"torch": torch, "AlgoConfig": type("AlgoConfig", (), {}),
                     "verl_F": SimpleNamespace(masked_mean=lambda x, m, axis=None:
                                                (x * m).sum(dim=axis) / (m.sum(dim=axis) + 1e-8))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + funcs, type_ignores=[])),
                     "actual_ppo_loss", "exec"), namespace)
        config = SimpleNamespace(clip_ratio=.2, clip_ratio_low=None, clip_ratio_high=None,
                                 get=lambda key, default=None: default)
        old = torch.tensor([[-1., -1., -1.]])
        current = old.clone().requires_grad_()
        advantage = torch.tensor([[1., -1., 1.]])
        mask = torch.tensor([[1., 1., 0.]])
        weights = torch.tensor([[2., .5, 0.]])
        loss, _ = namespace["compute_policy_loss_vanilla"](
            old, current, advantage, mask, "token-mean", config, weights)
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[-1., .25, 0.]]))

    def setUp(self):
        self.mock = patch.dict(sys.modules, {
            "verl.trainer.ppo.rollout_corr_helper": load_verl_functions()
        })
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_weights_clipping_direction_and_mask(self):
        old = torch.tensor([[math.log(.2), math.log(.4), math.log(.05), float("nan")]], requires_grad=True)
        behavior = torch.tensor([[math.log(.2), math.log(.1), math.log(.2), float("nan")]])
        mask = torch.tensor([[1, 1, 1, 0]])
        weights, stats = tis.compute_token_tis(old, behavior, mask, mask)
        torch.testing.assert_close(weights, torch.tensor([[1., 2., .25, 0.]]))
        self.assertFalse(weights.requires_grad)
        self.assertAlmostEqual(stats["tis/clipped_fraction"], 1 / 3, places=6)

    def test_reject_missing_and_nonfinite_active_tokens(self):
        old = torch.zeros(1, 2)
        mask = torch.ones(1, 2)
        with self.assertRaisesRegex(ValueError, "ALIGNMENT"):
            tis.compute_token_tis(old, old, torch.tensor([[1, 0]]), mask)
        with self.assertRaisesRegex(ValueError, "ALIGNMENT"):
            tis.compute_token_tis(old, torch.tensor([[0., float("nan")]]), mask, mask)

    def test_equal_policies_are_identity(self):
        old = torch.tensor([[-1., -2.]])
        mask = torch.ones_like(old)
        weights, _ = tis.compute_token_tis(old, old, mask, mask)
        torch.testing.assert_close(weights, mask)


if __name__ == "__main__":
    unittest.main()
