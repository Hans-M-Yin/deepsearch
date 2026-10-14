"""Token-aligned process-credit shaping for cumulative agent trajectories.

The partial-credit judge returns one-based ``credit_turn`` values.  In the
cumulative training representation, ``response_mask`` alternates between
assistant action blocks (1) and tool/environment observations (0).  This
module reconstructs the action block for each credited turn from that mask
and uses the last credited turn to split the learnable prefix and suffix.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _positive_int_set(value: Any) -> set[int] | None:
    """Return positive integer turn IDs, or ``None`` for malformed input."""

    if value is None:
        return set()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        return None
    turns: set[int] = set()
    try:
        for item in value:
            turn = int(item)
            if turn <= 0:
                return None
            turns.add(turn)
    except (TypeError, ValueError):
        return None
    return turns


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def build_credit_turn_token_mask(
    response_mask: torch.Tensor,
    credit_turns: Any,
    expected_turn_count: Any,
) -> tuple[torch.Tensor, bool]:
    """Map one-based agent turns to their valid assistant-action tokens.

    A returned ``False`` means the cumulative response no longer has the same
    number of action blocks as the original trajectory (for example after a
    truncation or a tokenization mismatch).  The caller must then skip
    shaping rather than guessing an alignment.
    """

    if response_mask.ndim != 1:
        raise ValueError(
            "response_mask must be one-dimensional, got "
            f"shape={tuple(response_mask.shape)}"
        )

    target = torch.zeros_like(response_mask, dtype=torch.bool)
    turns = _positive_int_set(credit_turns)
    expected = _positive_int(expected_turn_count)
    if turns is None or expected is None:
        return target, False
    if not turns:
        return target, True
    if expected == 0 or max(turns) > expected:
        return target, False

    action_mask = response_mask.bool()
    action_turn = 0
    was_action = False
    for token_index, is_action in enumerate(action_mask.tolist()):
        if is_action and not was_action:
            action_turn += 1
        if is_action and action_turn in turns:
            target[token_index] = True
        was_action = is_action

    if action_turn != expected:
        # Do not partially retain a target mask: a shifted action block would
        # update the wrong decision token(s).
        return torch.zeros_like(target), False
    return target, True


def build_credit_prefix_token_mask(
    response_mask: torch.Tensor,
    credit_turns: Any,
    expected_turn_count: Any,
) -> tuple[torch.Tensor, bool]:
    """Select assistant-action tokens through the last credited turn.

    The last credited action is included in the prefix. As with
    ``build_credit_turn_token_mask``, any
    action-block mismatch fails closed.
    """

    if response_mask.ndim != 1:
        raise ValueError(
            "response_mask must be one-dimensional, got "
            f"shape={tuple(response_mask.shape)}"
        )

    target = torch.zeros_like(response_mask, dtype=torch.bool)
    turns = _positive_int_set(credit_turns)
    expected = _positive_int(expected_turn_count)
    if turns is None or expected is None:
        return target, False
    if not turns:
        return target, True
    if expected == 0 or max(turns) > expected:
        return target, False

    last_credit_turn = max(turns)
    action_mask = response_mask.bool()
    action_turn = 0
    was_action = False
    for token_index, is_action in enumerate(action_mask.tolist()):
        if is_action and not was_action:
            action_turn += 1
        if is_action and action_turn <= last_credit_turn:
            target[token_index] = True
        was_action = is_action

    if action_turn != expected:
        return torch.zeros_like(target), False
    return target, True


def apply_credit_turn_advantage_shaping(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    credit_turns_by_row: Sequence[Any],
    expected_turn_counts: Sequence[Any],
    *,
    eligible_rows: torch.Tensor | None = None,
    is_correct_by_row: Sequence[Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply sign-based prefix/suffix shaping to outcome-incorrect rows.

    Through the last credited action (inclusive), negative advantages become
    zero. After that action, positive advantages become zero. All other values
    remain unchanged; credited actions no longer receive a multiplier.
    Correct rows, unknown outcomes, and rows without credit remain unchanged.

    ``response_mask`` is authoritative, so tool observations and padding can
    never be shaped.  ``eligible_rows`` allows callers to exclude padding or
    later-filtered rows while retaining the original batch layout.
    """

    if advantages.shape != response_mask.shape:
        raise ValueError(
            "advantages and response_mask must have identical shapes, got "
            f"{tuple(advantages.shape)} and {tuple(response_mask.shape)}"
        )
    if advantages.ndim != 2:
        raise ValueError(
            "advantages must be two-dimensional, got "
            f"shape={tuple(advantages.shape)}"
        )
    row_count = advantages.shape[0]
    if len(credit_turns_by_row) != row_count or len(expected_turn_counts) != row_count:
        raise ValueError(
            "partial-credit row metadata does not match batch size: "
            f"rows={row_count}, credit_turns={len(credit_turns_by_row)}, "
            f"expected_turn_counts={len(expected_turn_counts)}"
        )
    if is_correct_by_row is not None and len(is_correct_by_row) != row_count:
        raise ValueError(
            "is_correct_by_row does not match batch size: "
            f"rows={row_count}, is_correct={len(is_correct_by_row)}"
        )
    if eligible_rows is None:
        eligible = torch.ones(row_count, dtype=torch.bool, device=advantages.device)
    else:
        eligible = eligible_rows.to(device=advantages.device, dtype=torch.bool).reshape(-1)
        if eligible.numel() != row_count:
            raise ValueError(
                "eligible_rows does not match batch size: "
                f"rows={row_count}, eligible_rows={eligible.numel()}"
            )

    shaped = advantages.clone()
    stats = {
        "rows": float(row_count),
        "eligible_rows": float(eligible.sum().item()),
        "correct_rows_skipped": 0.0,
        "credit_turn_rows": 0.0,
        "alignment_mismatch_rows": 0.0,
        "target_token_count": 0.0,
        "negative_zeroed_token_count": 0.0,
        "positive_doubled_token_count": 0.0,
        "zero_advantage_target_token_count": 0.0,
        "incorrect_positive_credit_rows": 0.0,
        "incorrect_positive_suffix_masked_rows": 0.0,
        "incorrect_positive_suffix_token_count": 0.0,
        "incorrect_positive_suffix_zeroed_token_count": 0.0,
        "incorrect_negative_prefix_masked_rows": 0.0,
        "incorrect_negative_prefix_token_count": 0.0,
        "incorrect_negative_prefix_zeroed_token_count": 0.0,
    }

    for row_index in range(row_count):
        if not bool(eligible[row_index].item()):
            continue
        if is_correct_by_row is None:
            continue
        if bool(is_correct_by_row[row_index]):
            stats["correct_rows_skipped"] += 1.0
            continue
        turns = _positive_int_set(credit_turns_by_row[row_index])
        if turns is None:
            stats["alignment_mismatch_rows"] += 1.0
            continue
        if not turns:
            continue
        token_mask, aligned = build_credit_turn_token_mask(
            response_mask[row_index],
            turns,
            expected_turn_counts[row_index],
        )
        if not aligned:
            stats["alignment_mismatch_rows"] += 1.0
            continue

        # The helper derives the mask from response_mask, but retain this
        # intersection as a hard invariant if its implementation changes.
        token_mask &= response_mask[row_index].bool()
        stats["credit_turn_rows"] += 1.0
        stats["target_token_count"] += float(token_mask.sum().item())
        zero = token_mask & (shaped[row_index] == 0)
        stats["zero_advantage_target_token_count"] += float(zero.sum().item())
        prefix_mask, prefix_aligned = build_credit_prefix_token_mask(
            response_mask[row_index],
            turns,
            expected_turn_counts[row_index],
        )
        if not prefix_aligned:
            # The earlier credit mask was aligned, so this branch should only
            # be reachable if helper semantics drift.  Never guess a cutoff.
            stats["alignment_mismatch_rows"] += 1.0
            continue
        # Previous credit-only rules are intentionally disabled for this
        # experiment:
        # shaped[row_index][token_mask & (advantages[row_index] < 0)] = 0.0
        # shaped[row_index][token_mask & (advantages[row_index] > 0)] *= 2.0
        negative_prefix = prefix_mask & (advantages[row_index] < 0)
        if bool(negative_prefix.any()):
            shaped[row_index][negative_prefix] = 0.0
            count = float(negative_prefix.sum().item())
            stats["negative_zeroed_token_count"] += count
            stats["incorrect_negative_prefix_masked_rows"] += 1.0
            stats["incorrect_negative_prefix_token_count"] += float(prefix_mask.sum().item())
            stats["incorrect_negative_prefix_zeroed_token_count"] += count

        if not bool((advantages[row_index][response_mask[row_index].bool()] > 0).any()):
            continue
        stats["incorrect_positive_credit_rows"] += 1.0
        suffix_mask = response_mask[row_index].bool() & ~prefix_mask
        suffix_token_count = int(suffix_mask.sum().item())
        if suffix_token_count == 0:
            continue
        zeroed_suffix = suffix_mask & (advantages[row_index] > 0)
        shaped[row_index][zeroed_suffix] = 0.0
        stats["incorrect_positive_suffix_masked_rows"] += 1.0
        stats["incorrect_positive_suffix_token_count"] += float(suffix_token_count)
        stats["incorrect_positive_suffix_zeroed_token_count"] += float(
            zeroed_suffix.sum().item()
        )

    return shaped, stats
