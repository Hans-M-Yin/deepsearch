"""Decoupled token TIS for the custom workflow trainer (not PPO bypass)."""
import math

import torch


def validate_tis_config(config):
    rollout = config.actor_rollout_ref.rollout
    mode = (rollout.get("engine_kwargs", {}).get("vllm", {}) or {}).get("logprobs_mode")
    if rollout.name != "vllm" or mode != "processed_logprobs":
        raise ValueError("Token TIS requires vLLM processed_logprobs; raw monitoring values are not behavior logprobs.")
    if not rollout.calculate_log_probs or not rollout.do_sample or float(rollout.temperature) <= 0:
        raise ValueError("Token TIS requires stochastic sampling and calculate_log_probs=True.")
    if float(rollout.top_p) != 1.0 or int(rollout.top_k) != -1:
        raise ValueError("Initial token TIS integration requires top_p=1 and top_k=-1 (no support truncation).")
    if float(rollout.get("repetition_penalty", 1.0)) != 1.0:
        raise ValueError("Initial token TIS integration requires repetition_penalty=1.")
    actor = config.actor_rollout_ref.actor
    if actor.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("Token TIS integration requires the existing vanilla PPO loss.")
    correction = config.algorithm.get("rollout_correction", {}) or {}
    if correction.get("bypass_mode", False) or correction.get("use_policy_gradient", False):
        raise ValueError("Token TIS must not be combined with bypass/policy-gradient mode.")


def compute_token_tis(old, behavior, valid, response_mask, threshold=2.0):
    """Fail closed on missing sampled-token logprobs; never change reward/mask.

    Use verl's token TIS implementation: stop_grad(min(exp(old-behavior), cap)).
    The PPO ratio remains current/old, independent from this correction.
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

    if not math.isfinite(threshold) or threshold < 1:
        raise ValueError("TIS threshold must be finite and >= 1.")
    if not (old.shape == behavior.shape == valid.shape == response_mask.shape):
        raise ValueError("TIS logprobs and masks must have identical shapes.")
    active = response_mask.bool()
    invalid = active & (~valid.bool() | ~torch.isfinite(old) | ~torch.isfinite(behavior))
    if invalid.any():
        raise ValueError(
            f"[TOKEN_TIS_ALIGNMENT_ERROR] {int(invalid.sum())}/{int(active.sum())} active tokens "
            "lack aligned finite behavior/old logprobs. Refusing to train with invented weights."
            f" unaligned={int((active & ~valid.bool()).sum())}"
            f" old_nonfinite={int((active & ~torch.isfinite(old)).sum())}"
            f" aligned_behavior_nonfinite={int((active & valid.bool() & ~torch.isfinite(behavior)).sum())}."
            " See the rollout logprob alignment warning for categories (counts may overlap)."
        )
    if not active.any():
        raise ValueError("TIS batch contains no active response tokens.")
    with torch.no_grad():
        log_ratio = torch.where(active, old.float() - behavior.float(), 0.0)
        weights, stats = compute_rollout_correction_weights(
            log_ratio, active, rollout_is="token", rollout_is_threshold=threshold,
            rollout_is_batch_normalize=False,
        )
    metrics = {"tis/" + key: value for key, value in stats.items()}
    metrics.update({"tis/enabled": 1, "tis/coverage": 1.0, "tis/threshold": threshold,
                    "tis/weight_mean": weights[active].mean().item(),
                    "tis/clipped_fraction": (log_ratio[active] > math.log(threshold)).float().mean().item()})
    return weights, metrics
