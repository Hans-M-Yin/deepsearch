"""Monitoring only: align sampled tokens without altering policy losses."""
import math
import hashlib
import json

import torch


def token_fingerprint(tokens):
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def verify_tito_training_row(input_ids, attention_mask, response, response_mask, records):
    """Fail closed on each active action AND its complete unpadded prefix.

    Fatal trailing turns with zero loss are not treated as verified actions.
    Does not require logprobs (validation rollouts may not request them).
    """
    offset = len(input_ids) - len(response)
    if offset < 0 or not torch.equal(input_ids[offset:], response):
        raise ValueError("[TITO_TRAIN_ALIGNMENT_ERROR] response/input suffix mismatch")
    active = response_mask.bool().tolist()
    starts = [i for i, value in enumerate(active) if value and (i == 0 or not active[i-1])]
    ends = [i+1 for i, value in enumerate(active) if value and (i+1 == len(active) or not active[i+1])]
    if len(starts) > len(records):
        raise ValueError("[TITO_TRAIN_ALIGNMENT_ERROR] missing rollout records")
    for turn, (start, end, record) in enumerate(zip(starts, ends, records), 1):
        prefix = input_ids[:offset+start][attention_mask[:offset+start].bool()]
        if not record or not record.get("prompt_fingerprint"):
            raise ValueError(f"[TITO_TRAIN_ALIGNMENT_ERROR] turn={turn} missing prompt fingerprint")
        if len(prefix) != record.get("prompt_length") or token_fingerprint(prefix) != record["prompt_fingerprint"]:
            raise ValueError(f"[TITO_TRAIN_ALIGNMENT_ERROR] turn={turn} rollout/training prefix differs")
        if response[start:end].tolist() != record.get("token_ids"):
            raise ValueError(f"[TITO_TRAIN_ALIGNMENT_ERROR] turn={turn} action IDs differ")
    return len(starts)


def audit_prompt_prefixes(input_ids, attention_mask, response, response_mask, records):
    """Compare adapter-expanded prompt to reconstructed training prefix.

    This does NOT inspect vLLM's internally reprocessed pixels or position IDs.
    -1 = unverified, 0 = different, 1 = equal, on sampled tokens only.
    """
    audit = torch.full(response.shape, -1, dtype=torch.int8)
    active = response_mask.bool().tolist()
    starts = [i for i, x in enumerate(active) if x and (i == 0 or not active[i - 1])]
    ends = [i + 1 for i, x in enumerate(active) if x and (i == len(active) - 1 or not active[i + 1])]
    if len(starts) > len(records):
        return audit
    offset = len(input_ids) - len(response)
    for start, end, record in zip(starts, ends, records):
        if not record or not record.get("prompt_fingerprint"):
            continue
        _, valid = align_rollout_logprobs(response[start:end], response_mask[start:end], [record])
        locations = valid.nonzero().flatten()
        if not len(locations):
            continue
        first = start + int(locations[0])
        prefix = input_ids[:offset + first][attention_mask[:offset + first].bool()]
        equal = token_fingerprint(prefix) == record["prompt_fingerprint"]
        audit[start:end][valid] = int(equal)
    return audit


def align_rollout_logprobs(response, response_mask, records, *, allow_tail_prefix=False, diagnostics=None):
    """Match whole generations inside their corresponding assistant spans.

    No fuzzy matching: rewritten generations are excluded. With explicit
    allow_tail_prefix, an exact prefix reaching the physical response boundary
    is accepted; internal cuts and shifted prefixes remain unverified. Extra
    template tokens, observations and padding never receive invented logprobs.
    Missing records remain placeholders so later turns cannot shift backwards.
    """
    ids = response.tolist()
    mask = response_mask.bool().tolist()
    values = torch.zeros(len(ids), dtype=torch.float32)
    valid = torch.zeros(len(ids), dtype=torch.bool)
    def report(reason, turn, start, end):
        if diagnostics is None:
            return
        counts = diagnostics.setdefault("counts", {})
        counts[reason] = counts.get(reason, 0) + end - start
        examples = diagnostics.setdefault("examples", [])
        if reason != "tail_prefix_aligned" and sum(x["reason"] == reason for x in examples) < 3:
            span_start, span_end = spans[turn]
            record = (records[turn] or {}) if turn < len(records) else {}
            generated = record.get("token_ids") or []
            training = ids[span_start:span_end]
            first = next((j for j, (a, b) in enumerate(zip(training, generated)) if a != b),
                         min(len(training), len(generated)))
            lo, hi = max(0, first - 8), first + 9
            examples.append({
                "turn": turn, "start": start, "end": end, "reason": reason,
                "span_start": span_start, "span_end": span_end,
                "training_length": len(training), "generated_length": len(generated),
                "logprob_length": len(record.get("log_probs") or []),
                "at_response_boundary": span_end == len(ids),
                "first_difference_from_span_start": first,
                "window_start": lo,
                "training_window_ids": training[lo:hi],
                "generated_window_ids": generated[lo:hi],
                "training_tail_ids": training[-8:], "generated_tail_ids": generated[-8:],
                "unmatched_head_ids": ids[start:min(end, start + 16)],
                "unmatched_tail_ids": ids[max(start, end - 16):end],
            })
    spans = []
    for i, active in enumerate(mask):
        if active and (i == 0 or not mask[i - 1]):
            spans.append([i, i + 1])
        elif active:
            spans[-1][1] = i + 1
    if len(spans) > len(records):
        for turn, (start, end) in enumerate(spans):
            report("span_record_count_mismatch", turn, start, end)
        return values, valid
    for turn, ((start, end), record) in enumerate(zip(spans, records)):
        record = record or {}
        tokens, probs = record.get("token_ids"), record.get("log_probs")
        if not tokens or probs is None:
            report("missing_record_or_logprobs", turn, start, end)
            continue
        if len(tokens) != len(probs):
            report("token_logprob_length_mismatch", turn, start, end)
            continue
        if not all(math.isfinite(p) for p in probs):
            report("rollout_record_nonfinite", turn, start, end)
            continue
        matches = [i for i in range(start, end - len(tokens) + 1)
                   if ids[i] == tokens[0] and ids[i:i + len(tokens)] == tokens]
        if len(matches) != 1:
            # Only at the physical response cut boundary, starting exactly at
            # the assistant span start. Never search arbitrary short suffixes
            # or infer a cut from the last active token (it may be fatal-masked).
            if (not matches and allow_tail_prefix and end == len(ids)
                    and 0 < end - start < len(tokens)
                    and ids[start:end] == tokens[:end - start]):
                values[start:end] = torch.tensor(probs[:end - start], dtype=torch.float32)
                valid[start:end] = True
                report("tail_prefix_aligned", turn, start, end)
                continue
            report("ambiguous_match" if matches else "token_mismatch_or_unverified_cut", turn, start, end)
            continue
        i = matches[0]
        values[i:i + len(tokens)] = torch.tensor(probs, dtype=torch.float32)
        valid[i:i + len(tokens)] = True
        # Extra tokens are NOT automatically classified as template tokens:
        # origin must be established before changing the policy loss mask.
        if i > start:
            report("extra_span_tokens_unverified_origin", turn, start, i)
        if i + len(tokens) < end:
            report("extra_span_tokens_unverified_origin", turn, i + len(tokens), end)
    return values, valid


def consistency_metrics(actor, rollout, valid, response_mask):
    """Report sampled-token differences, not full-vocabulary KL or IS weights."""
    active = response_mask.bool()
    compared = active & valid.bool() & torch.isfinite(actor) & torch.isfinite(rollout)
    count = int(compared.sum().item())
    result = {
        "training/rollout_logprob_compared_tokens": count,
        "training/rollout_logprob_coverage": count / max(int(active.sum().item()), 1),
    }
    if not count:
        return result
    delta = (actor.float() - rollout.float())[compared]
    absolute = delta.abs()
    prob_diff = (actor.float().exp() - rollout.float().exp())[compared].abs()
    result.update({
        "training/rollout_logprob_diff_mean": delta.mean().item(),
        "training/rollout_logprob_abs_diff_mean": absolute.mean().item(),
        "training/rollout_logprob_abs_diff_p95": torch.quantile(absolute, 0.95).item(),
        "training/rollout_logprob_abs_diff_max": absolute.max().item(),
        "training/rollout_probs_diff_mean": prob_diff.mean().item(),
        "training/rollout_probs_diff_max": prob_diff.max().item(),
        "training/rollout_probs_diff_std": prob_diff.std(unbiased=False).item(),
    })
    return result


def diagnostic_report(batch):
    """Separate trainer nondeterminism/fusion/batching from cross-engine error."""
    mask = batch["response_mask"].bool()
    old = batch["old_log_probs"]
    result = {
        "active_tokens": int(mask.sum()),
        "actor_nonfinite_tokens": int((mask & ~torch.isfinite(old)).sum()),
        "mask_on_padding_tokens": int((mask & ~batch["attention_mask"][:, -mask.shape[1]:].bool()).sum()),
        "response_input_mismatch_tokens": int((mask & (batch["responses"] != batch["input_ids"][:, -mask.shape[1]:])).sum()),
    }
    rollout = batch.get("diagnostic_rollout_log_probs")
    if "diagnostic_prompt_equal" in batch:
        audit = batch["diagnostic_prompt_equal"]
        for name, value in (("equal", 1), ("different", 0), ("unverified", -1)):
            result["prompt_prefix_" + name + "_tokens"] = int((mask & (audit == value)).sum())
    valid = batch.get("diagnostic_rollout_log_probs_mask")
    for name in ("baseline", "actor_raw", "repeat", "actor_batching", "unfused"):
        value = old if name == "baseline" else batch.get("diagnostic_" + name + "_log_probs")
        if value is None:
            continue
        for target_name, target, target_mask in (
            ("vs_old", old, mask), ("vs_rollout", rollout, valid)
        ):
            if target is None or target_mask is None:
                continue
            for key, val in consistency_metrics(value, target, target_mask, mask).items():
                result[name + "/" + target_name + "/" + key.removeprefix("training/")] = val
    return result
