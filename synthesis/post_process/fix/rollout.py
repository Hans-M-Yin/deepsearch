"""Sequential Selector -> actual image -> Writer -> updated history loop."""

from .candidate_relevance import visible_card
from .search_cards import SELECTABLE, POSITIVE
from .selection import selection_history
from .rewrite import checked_reasoning, write_after
from .process_leakage import check_process_leakage, IMAGE_LABEL_INSTRUCTION
from .errors import SkipAugmentation
from .timing import timed_call
import random
from .storage import digest
from .evidence_attribution import EVIDENCE_ATTRIBUTION

REASONS = (
    """For EVERY selectable search card, explain a plausible reason to open
it using ONLY the question, available trajectory history and visible card text.
This is not a competition filter: a weaker lead remains selectable. Do not claim
to have seen candidate images, infer identities from world knowledge, introduce
names/terms absent from the supplied text, or claim a title confirms a match.
If text is vague, honestly explain what inspecting it could resolve; do not
fabricate specificity. Return JSON {"cards":[{"image_id":"...","reasoning":"..."}]}.
"""
    + IMAGE_LABEL_INSTRUCTION
    + EVIDENCE_ATTRIBUTION
)

SELECT = (
    """Choose ONE uninspected candidate to open, based ONLY on the question,
visible search cards and trajectory history observed so far. Use the supplied
possible inspection reasons as hypotheses, not verified facts. Do not use extra
world knowledge or introduce entities/technical terms absent from the supplied
text. No unseen candidate images or correct-answer labels are supplied. Explain
why this lead is useful now without claiming it matches. Past selection reasoning
and subsequent image observations are chronological; account for rejected leads.
Return JSON {"image_id":"an available ID", "reasoning":"natural pre-view reasoning"}.
Also return equally_plausible_ids (a list including image_id and ALL candidates comparably worth opening when text does not establish a meaningful preference), preference_evidence (a concrete comparative advantage, or an empty string for a tie), and exclusions (a list of {image_id, evidence_quote, reason}). If there is no decisive text-grounded advantage, do not manufacture a unique best candidate: return the tied IDs. A program will sample among them without knowing which is correct, using their supplied inspection reasons.
Exclude a candidate only for an explicit contradiction supported by a verbatim quote from the question or observed history. An unknown country cannot rule out Japan or Spain; different country names in titles do not determine the target country. A passenger-train/truck title may be less direct than observed passenger/freight involvement, but lower priority alone is not proof of irrelevance. Specificity or source authority can justify a preference, not a claim that alternatives are wrong. Use no extra world knowledge. Never infer that an unviewed candidate is visually wrong. For a tie, return preference_evidence=""; otherwise explicitly compare the selected lead with the strongest alternatives using available evidence.
Do not mention internal fields, editing, quotas, labels or future results.
Complete output examples (replace illustrative IDs with available IDs):
Tie: {"image_id":"A","reasoning":"Opening this lead could resolve the target.","equally_plausible_ids":["A","B"],"preference_evidence":"","exclusions":[]}
Preference: {"image_id":"A","reasoning":"This title directly addresses the observed clue.","equally_plausible_ids":["A"],"preference_evidence":"A explicitly mentions the observed clue, while B is generic.","exclusions":[]}
For exclusions, copy evidence_quote exactly from the question or observed trajectory history, not from an unobserved candidate title. Omit unsupported exclusions rather than paraphrasing evidence. If validation_feedback is provided, return a complete corrected JSON object; keep the same evidence limits and do not mention validation in agent reasoning.
"""
    + IMAGE_LABEL_INSTRUCTION
    + EVIDENCE_ATTRIBUTION
)


def choose_from_assessment(response, remaining, reasons, history, question, seed, step):
    """Sample text-plausible ties without consulting private visual categories."""
    rid = response.get("image_id")
    if rid not in remaining:
        raise ValueError("Selector chose an unavailable or repeated image")
    exclusions = response.get("exclusions", [])
    if not isinstance(exclusions, list):
        raise ValueError("exclusions must be a list")
    evidence_text = question + "\n" + str(history)
    excluded = set()
    for index, item in enumerate(exclusions):
        if not isinstance(item, dict):
            raise ValueError(f"exclusions[{index}] must be an object")
        key, quote = item.get("image_id"), item.get("evidence_quote")
        if not isinstance(key, str) or key not in remaining:
            raise ValueError(f"exclusions[{index}].image_id is unavailable")
        if key in excluded:
            raise ValueError(f"exclusions[{index}].image_id is duplicated")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(
                f"exclusions[{index}].evidence_quote must be a nonempty string"
            )
        if quote not in evidence_text:
            raise ValueError(
                f"exclusions[{index}].evidence_quote is not a verbatim quote from the question or observed history"
            )
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError(f"exclusions[{index}].reason must be a nonempty string")
        excluded.add(key)
    ties = response.get("equally_plausible_ids", [rid])
    if (
        not isinstance(ties, list)
        or any(not isinstance(k, str) or k not in remaining for k in ties)
        or len(set(ties)) != len(ties)
        or rid not in ties
        or excluded.intersection(ties)
    ):
        raise ValueError("Invalid equally plausible candidates")
    if len(ties) > 1:
        if response.get("preference_evidence"):
            raise ValueError("Tied candidates cannot have a decisive preference")
        rng = random.Random(digest([seed, question, step, sorted(ties)]))
        rid = rng.choice(sorted(ties))
        before = checked_reasoning({"reasoning": reasons[rid]})
        mode = "plausible_tie_sample"
    else:
        before = checked_reasoning(response)
        mode = "autonomous"
    return rid, before, mode


def select_with_repair(
    model,
    config,
    payload,
    remaining,
    reasons,
    history,
    question,
    step,
    audit,
    max_repairs=2,
):
    """Retry only invalid Selector output, without exposing private labels."""
    request = dict(payload)
    for attempt in range(max_repairs + 1):
        response = timed_call("selector", model.ask, config.model, SELECT, request)
        entry = {"step": step, "attempt": attempt + 1, "response": response}
        audit.setdefault("selector_validation_attempts", []).append(entry)
        try:
            if not isinstance(response, dict):
                raise ValueError("Selector response must be a JSON object")
            if not isinstance(response.get("image_id"), str):
                raise ValueError("image_id must be an available string ID")
            result = choose_from_assessment(
                response, remaining, reasons, history, question, config.seed, step
            )
            check_process_leakage(result[1])
        except ValueError as exc:
            entry.update(valid=False, error=str(exc))
            if attempt == max_repairs:
                raise ValueError(
                    f"Selector validation failed after {attempt + 1} attempts: {exc}"
                ) from exc
            request = {
                **payload,
                "validation_feedback": {
                    "attempt": attempt + 1,
                    "error": str(exc),
                    "previous_response": response,
                    "instruction": "Correct the specified output inconsistency using only the supplied evidence. Return the full JSON object, with natural pre-view reasoning and no validation commentary.",
                },
            }
        else:
            entry["valid"] = True
            return response, result


def validate_inspection_reasons(response, allowed):
    if not isinstance(response, dict) or not isinstance(response.get("cards"), list):
        raise ValueError("Expected JSON object with cards list")
    reasons = {}
    for row in response["cards"]:
        if not isinstance(row, dict):
            raise ValueError("Each candidate rationale must be a JSON object")
        rid = row.get("image_id")
        if not isinstance(rid, str) or rid not in allowed:
            raise ValueError(f"Unavailable candidate rationale ID: {rid!r}")
        if rid in reasons:
            raise ValueError(f"Duplicate candidate rationale ID: {rid}")
        reasons[rid] = checked_reasoning(row)
        check_process_leakage(reasons[rid])
    missing = allowed - set(reasons)
    if missing:
        raise ValueError(f"Missing candidate rationale IDs: {sorted(missing)}")
    return reasons


def inspection_reasons(candidates, window, history, model, config, audit=None):
    cards = [
        visible_card(c)
        for c in candidates
        if c.category in SELECTABLE and not c.text_reference_only
    ]
    payload = {
        "question": window.question,
        "trajectory_history": history,
        "search_results": cards,
    }
    allowed = {c["image_id"] for c in cards}
    for attempt in range(config.context_patch_retries + 1):
        response = model.ask(config.model, REASONS, payload)
        entry = {"attempt": attempt + 1, "response": response}
        if audit is not None:
            audit.setdefault("inspection_reason_attempts", []).append(entry)
        try:
            reasons = validate_inspection_reasons(response, allowed)
        except ValueError as exc:
            entry.update(valid=False, error=str(exc))
            if attempt == config.context_patch_retries:
                raise
            payload = {
                **payload,
                "allowed_image_ids": sorted(allowed),
                "previous_invalid_response": response,
                "validation_error": str(exc),
                "repair_instruction": "Return a complete valid JSON object with cards. Copy each allowed image_id exactly once. Do not guess or reconstruct IDs. Preserve evidence limits; do not mention validation in reasoning.",
            }
        else:
            entry["valid"] = True
            return reasons


def run_loop(record, window, plan, anchor, source, model, config):
    history = selection_history(record, window)
    reasons = timed_call(
        "inspection_reasons",
        inspection_reasons,
        plan.first,
        window,
        history,
        model,
        config,
        audit=plan.metadata,
    )
    plan.metadata["inspection_reasons"] = reasons
    plan.metadata["selection_reasoning"] = {}
    plan.metadata["selection_modes"] = []
    plan.metadata["steps"] = steps = []
    observations = []
    remaining = {
        c.public["image_id"]: c
        for c in plan.first
        if c.category in SELECTABLE and not c.text_reference_only
    }
    viewed_ids = set(anchor["viewed_before_conclusion_ids"])
    original_correct = {
        c.key: rid
        for c in plan.first
        for rid in viewed_ids
        if c.key.endswith("old_" + rid) and c.category in POSITIVE
    }
    # Determine this independently of composition: another correct viewed card
    # may have been omitted if it was not a supporting reference.
    original_correct.update(plan.metadata.get("original_correct_views", {}))
    while remaining:
        payload = {
            "question": window.question,
            "search_results": [visible_card(c) for c in plan.first],
            "uninspected_image_ids": list(remaining),
            "candidate_reasons": {rid: reasons[rid] for rid in remaining},
            "trajectory_history": history,
            "previous_observations": observations,
        }
        response, (rid, before, mode) = select_with_repair(
            model,
            config,
            payload,
            remaining,
            reasons,
            history,
            window.question,
            len(plan.attempts),
            plan.metadata,
        )
        plan.metadata.setdefault("selection_assessments", []).append(response)
        check_process_leakage(before)
        c = remaining.pop(rid)
        plan.attempts.append(c)
        plan.metadata["selection_reasoning"][c.key] = before
        plan.metadata["selection_modes"].append(
            {"step": len(plan.attempts), "candidate_key": c.key, "mode": mode}
        )
        if c.category in POSITIVE and original_correct:
            if c.key not in original_correct:
                raise SkipAugmentation(
                    "different_correct_candidate",
                    {
                        "selected_key": c.key,
                        "original_correct_keys": list(original_correct),
                    },
                )
            # Stop immediately BEFORE the retained original image check. Its tool
            # observation and subsequent reasoning remain in their original place.
            plan.metadata["reuse_original_read"] = original_correct[c.key]
            plan.metadata["terminal_selection_reasoning"] = before
            plan.positive_key = c.key
            break
        assessment = {}
        plan.metadata.setdefault("writer_assessments", []).append(
            {
                "candidate_key": c.key,
                "inspection_index": len(steps),
                "assessment": assessment,
            }
        )
        after = timed_call(
            "writer_observation",
            write_after,
            c,
            window,
            anchor,
            source,
            model,
            config,
            history=history,
            selection_reasoning=before,
            assessment=assessment,
        )
        check_process_leakage(after)
        steps.append(
            {"kind": "read", "candidate_key": c.key, "before": before, "after": after}
        )
        # These are the exact generated words the next selector receives, not a
        # verifier's hidden labels or a separately synthesized retrospective trace.
        history = history + [
            {
                "role": "assistant",
                "content": before,
                "action": {"name": "read_url", "arguments": {"resource_id": rid}},
            },
            {"role": "tool", "content": "Candidate image displayed for " + rid},
            {"role": "assistant", "content": after},
        ]
        observations.append(
            {
                "card": visible_card(c),
                "selection_reasoning": before,
                "observation": after,
            }
        )
        if c.category in POSITIVE:
            plan.positive_key = c.key
            break
    if not plan.positive_key:
        raise SkipAugmentation("candidates_exhausted")
    plan.branch = (
        "contrast" if any(c.verdict == "NO" for c in plan.attempts) else "direct"
    )
    plan.metadata["wrong_count"] = sum(c.verdict == "NO" for c in plan.attempts)
    return steps
