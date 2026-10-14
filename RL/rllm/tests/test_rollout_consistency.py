import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "rollout_consistency", Path(__file__).resolve().parents[1] / "rllm/utils/rollout_consistency.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConsistencyTest(unittest.TestCase):
    def test_examples_are_bounded_per_reason_and_include_windows(self):
        ids, mask, records = [], [], []
        for i in range(5):
            ids.extend([1, 2, 9, 0])
            mask.extend([1, 1, 1, 0])
            records.append({"token_ids": [1, 2], "log_probs": [-1., -2.]})
        ids.extend([4, 5])
        mask.extend([1, 1])
        records.append({"token_ids": [4, 6], "log_probs": [-1., -2.]})
        report = {}
        module.align_rollout_logprobs(torch.tensor(ids), torch.tensor(mask), records, diagnostics=report)
        self.assertEqual(len(report["examples"]), 4)
        last = report["examples"][-1]
        self.assertEqual(last["reason"], "token_mismatch_or_unverified_cut")
        self.assertEqual(last["first_difference_from_span_start"], 1)
        self.assertEqual(last["training_window_ids"], [4, 5])
        self.assertEqual(last["generated_window_ids"], [4, 6])

    def test_exact_tail_prefix_and_failure_categories(self):
        records = [{"token_ids": [1, 2, 3], "log_probs": [-1., -2., -3.]}]
        report = {}
        values, valid = module.align_rollout_logprobs(
            torch.tensor([1, 2]), torch.ones(2), records,
            allow_tail_prefix=True, diagnostics=report)
        self.assertEqual(valid.tolist(), [True, True])
        self.assertEqual(values.tolist(), [-1., -2.])
        self.assertEqual(report["counts"], {"tail_prefix_aligned": 2})
        # A fatal mask or padding boundary is not a physical truncation.
        for ids, mask in [([1, 2, 9], [1, 1, 0]), ([9, 1, 2], [1, 1, 1]),
                          ([1, 4], [1, 1])]:
            _, valid = module.align_rollout_logprobs(
                torch.tensor(ids), torch.tensor(mask), records, allow_tail_prefix=True)
            self.assertFalse(valid.any())
        report = {}
        _, valid = module.align_rollout_logprobs(
            torch.tensor([9, 1, 2, 3]), torch.ones(4), records, diagnostics=report)
        self.assertEqual(valid.tolist(), [False, True, True, True])
        self.assertEqual(report["counts"], {"extra_span_tokens_unverified_origin": 1})

    def test_missing_length_nonfinite_diagnostics(self):
        for record, reason in [({}, "missing_record_or_logprobs"),
                               ({"token_ids": [1, 2], "log_probs": [-1.]}, "token_logprob_length_mismatch"),
                               ({"token_ids": [1, 2], "log_probs": [-1., float("nan")]}, "rollout_record_nonfinite")]:
            report = {}
            _, valid = module.align_rollout_logprobs(torch.tensor([1, 2]), torch.ones(2),
                                                     [record], diagnostics=report)
            self.assertFalse(valid.any())
            self.assertEqual(report["counts"], {reason: 2})

    def test_prefix_audit_and_diagnostic_report(self):
        response = torch.tensor([1, 2, 8, 3])
        mask = torch.tensor([1, 1, 0, 1])
        ids = torch.tensor([0, 9, 1, 2, 8, 3])
        attn = torch.tensor([0, 1, 1, 1, 1, 1])
        records = [
            {"token_ids": [1, 2], "log_probs": [-1., -2.], "prompt_fingerprint": module.token_fingerprint([9])},
            {"token_ids": [3], "log_probs": [-1.], "prompt_fingerprint": module.token_fingerprint([9, 1, 2, 7])},
        ]
        audit = module.audit_prompt_prefixes(ids, attn, response, mask, records)
        self.assertEqual(audit.tolist(), [1, 1, -1, 0])
        old = torch.zeros(1, 4)
        report = module.diagnostic_report({
            "response_mask": mask[None], "attention_mask": attn[None],
            "input_ids": ids[None], "responses": response[None],
            "old_log_probs": old, "diagnostic_repeat_log_probs": old.clone(),
            "diagnostic_prompt_equal": audit[None],
        })
        self.assertEqual(report["prompt_prefix_different_tokens"], 1)
        self.assertEqual(report["response_input_mismatch_tokens"], 0)
        self.assertEqual(report["repeat/vs_old/rollout_logprob_abs_diff_mean"], 0)

    def test_observations_template_padding_and_missing_turn(self):
        response = torch.tensor([9, 1, 2, 8, 7, 3, 4, 0])
        mask = torch.tensor([1, 1, 1, 0, 0, 1, 1, 0])
        values, valid = module.align_rollout_logprobs(response, mask, [
            {"token_ids": [1, 2], "log_probs": [-1., -2.]},
            {"token_ids": [3, 4], "log_probs": [-3., -4.]},
        ])
        self.assertEqual(valid.tolist(), [False, True, True, False, False, True, True, False])
        stats = module.consistency_metrics(values + 0.1, values, valid, mask)
        self.assertEqual(stats["training/rollout_logprob_coverage"], 0.8)
        self.assertAlmostEqual(stats["training/rollout_logprob_abs_diff_mean"], 0.1, places=6)
        _, valid = module.align_rollout_logprobs(response, mask, [{}, {"token_ids": [3, 4], "log_probs": [-3., -4.]}])
        self.assertEqual(int(valid.sum()), 2)

    def test_rewritten_truncated_and_nonfinite_are_excluded(self):
        for record in [
            {"token_ids": [1, 3], "log_probs": [-1., -2.]},
            {"token_ids": [1, 2, 3], "log_probs": [-1., -2., -3.]},
            {"token_ids": [1, 2], "log_probs": [-1., float("nan")]},
        ]:
            _, valid = module.align_rollout_logprobs(torch.tensor([1, 2]), torch.ones(2), [record])
            self.assertFalse(valid.any())

    def test_no_data_and_one_token(self):
        x = torch.tensor([0.])
        stats = module.consistency_metrics(x, x, x.bool(), torch.ones(1))
        self.assertEqual(stats["training/rollout_logprob_coverage"], 0)
        self.assertNotIn("training/rollout_probs_diff_mean", stats)
        stats = module.consistency_metrics(x, x, torch.ones(1).bool(), torch.ones(1))
        self.assertEqual(stats["training/rollout_probs_diff_std"], 0)

    def test_fatal_suffix_and_ambiguous_match(self):
        records = [{"token_ids": [1, 2], "log_probs": [-1., -2.]},
                   {"token_ids": [3], "log_probs": [-3.]}]
        _, valid = module.align_rollout_logprobs(
            torch.tensor([1, 2, 9, 3]), torch.tensor([1, 1, 0, 0]), records)
        self.assertEqual(valid.tolist(), [True, True, False, False])
        _, valid = module.align_rollout_logprobs(
            torch.tensor([1, 2, 1, 2]), torch.ones(4), records[:1])
        self.assertFalse(valid.any())


if __name__ == "__main__":
    unittest.main()
