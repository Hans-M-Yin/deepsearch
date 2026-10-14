"""Choose candidates without exposing private labels or future image evidence."""

from .rewrite import checked_reasoning
from .precheck import check_selected
from .trajectory_index import messages, content, role
from .process_leakage import check_process_leakage, IMAGE_LABEL_INSTRUCTION
from .candidate_relevance import visible_card

SELECT = """Choose the next candidate image to inspect using ONLY the question,
reference image, visible search cards and observations from images already read.
You are not given unseen candidate images or correctness labels. Compare plausible
leads and pick the most useful uninspected image; do not deliberately choose a bad
one. Return JSON {"image_id":"an uninspected visible ID", "reasoning":"brief
reason for selecting it, without claiming to have seen it"}."""

SELECT += """
STRICT EVIDENCE BOUNDARY: Do not use ANY additional world knowledge, memorized
facts, event-to-location associations, or recognition-based identity guesses.
Every named entity and specialist term in your reasoning must already appear in
the supplied question, trajectory history, visible candidate text, or previous
observations. Do not introduce a new person, venue, location, event, organization,
date, or technical term absent from that text, even if you believe it is obvious
or true. Describe visible shapes, colors and arrangements in ordinary language;
do not name an unfamiliar stadium or person from appearance or memory.
For example, a card naming '2021 FAI Cup Final' does NOT authorize adding 'Aviva
Stadium' if that venue has not appeared in the supplied text. A matching title is
a lead to inspect, not confirmation that the photo depicts that event. Attribute
textual claims to the card and distinguish them from directly visible features.
Only compare the supplied evidence; leave missing identities unnamed until an
actual observation establishes them. Never output internal payload field names.
"""

SELECT += """
Focus on local_visual_target: the original purpose of THIS reverse-image search.
The full question is context only. Verify that target before pursuing downstream
facts (such as a podium sponsor). Do not prefer a card merely because it might
answer a later subquestion. No adopted identity or future answer is supplied."""

SELECT += """
trajectory_history contains only messages up to and including the current search
call, never its original results or later continuation. Use it to understand prior
evidence, unresolved hypotheses and why this search was initiated. Prior assistant
claims are hypotheses, not automatically verified facts. Continue naturally without
mentioning editing, original trajectories or preserved steps. Historical <image>
placeholders are not attached images; do not invent visual details from them.
Current search_results supersede any earlier search cards for this selection.
"""


SELECT += """
When references_to_check contains two IDs, inspect both captioned leads to verify
their visual support before continuing. Choose their order using visible evidence;
other plausible candidates may also be inspected. These IDs do not authorize
claiming a match before opening the image. Explain the source-based reason for
comparison naturally, never mention the field name or an imposed checking quota.
If previous observations already include a match and another check is requested,
choose the most useful remaining corroborating or competing lead from its text.
Do not assume the next candidate matches or fails. Explain the remaining evidence
question naturally; no quotas, designated references or preselected candidates.
"""


def selection_history(record, window):
    if record is None:
        return []
    return [
        {"role": role(message), "content": content(message)}
        for message in messages(record)[: window.call_index + 1]
    ]


SELECT += IMAGE_LABEL_INSTRUCTION


def select_visible(
    candidates,
    inspected,
    history,
    window,
    source,
    model,
    config,
    target=None,
    trajectory_history=None,
    reference_ids=None,
    designated_id=None,
):
    allowed = {
        c.public["image_id"]: c
        for c in candidates
        if c.key not in inspected and getattr(c, "category", None) != "UNRELATED"
    }
    if not allowed:
        raise ValueError("No inspectable candidates remain")
    payload = {
        "question": window.question,
        "local_visual_target": target,
        "search_results": [visible_card(c) for c in candidates],
        "uninspected_image_ids": list(allowed),
        "previous_observations": history,
        "trajectory_history": trajectory_history or [],
        "references_to_check": reference_ids or [],
    }
    prompt = SELECT
    if designated_id is not None:
        if designated_id not in allowed:
            raise ValueError("Designated candidate is unavailable")
        payload["designated_image_id"] = designated_id
        prompt += """
For this step inspect designated_image_id. Write a genuine pre-inspection reason
using ONLY its visible card, the reference and previous observations. You have
not seen this candidate and are NOT told it matches. Do not claim it is correct,
invent supporting clues, or mention designation, quotas, instructions or fields.
A cautious reason to test an uncertain lead is sufficient. If no evidence-based
reason exists, return {"unsafe":true,"reason":"No supported inspection rationale"}.
Return the designated ID in image_id, with reasoning as usual.
"""
    retries = getattr(config, "context_patch_retries", 1)
    for attempt in range(retries + 1):
        response = model.ask(config.model, prompt, payload, [source])
        try:
            if designated_id is not None and response.get("image_id") != designated_id:
                raise ValueError("Return the designated image ID")
            reasoning = checked_reasoning(response)
            check_process_leakage(reasoning)
            break
        except ValueError as exc:
            if attempt == retries:
                raise
            payload = {
                **payload,
                "repair_attempt": attempt + 1,
                "invalid_response": response,
                "validation_error": str(exc),
                "repair_instruction": "Explain the choice using visible evidence only. Never mention internal field names or repair instructions.",
            }
    if response.get("image_id") not in allowed:
        raise ValueError("Selector returned an unavailable/already-inspected candidate")
    return allowed[response["image_id"]], reasoning


def select_sequence(
    candidates,
    window,
    source,
    model,
    config,
    *,
    limit,
    require_match,
    target=None,
    prechecks=None,
    trajectory_history=None,
    required_reference_keys=None,
    selection_audit=None,
):
    attempts, reasons, history = [], {}, []
    for _ in range(limit):
        designated = None
        if require_match and len(attempts) >= 2:
            seen = {c.key for c in attempts}
            remaining = [
                c
                for c in candidates
                if c.key not in seen
                and c.verdict == "YES"
                and (
                    c.category == "CONSISTENT"
                    or (c.category == "SAME_ENTITY" and c.originally_viewed)
                )
            ]
            required = set(required_reference_keys or [])
            if required:
                remaining = [c for c in remaining if c.key in required]
            if not remaining:
                raise ValueError(
                    "No verified uninspected candidate for guided confirmation"
                )
            designated = remaining[0].public["image_id"]
        candidate, reason = select_visible(
            candidates,
            {c.key for c in attempts},
            history,
            window,
            source,
            model,
            config,
            target,
            trajectory_history,
            reference_ids=[
                c.public["image_id"]
                for c in candidates
                if c.key in (required_reference_keys or [])
            ],
            designated_id=designated,
        )
        if selection_audit is not None:
            selection_audit.append(
                {
                    "step": len(attempts) + 1,
                    "candidate_key": candidate.key,
                    "mode": "guided" if designated else "autonomous",
                }
            )
        if config.blind_visual_review:
            result = check_selected(candidate, window, target, source, model, config)
            if prechecks is not None:
                prechecks[candidate.key] = result
        if candidate.verdict not in {"YES", "NO", "CONTINUE"}:
            raise ValueError("Selected candidate lacks conclusive visual evidence")
        attempts.append(candidate)
        reasons[candidate.key] = reason
        if candidate.verdict == "YES":
            if not require_match:
                raise ValueError("Unexpected positive in recovery first list")
            required = set(required_reference_keys or [])
            confirmed = {c.key for c in attempts if c.verdict == "YES"}
            if not required or required <= confirmed:
                return attempts, reasons
        history.append(
            {
                "card": visible_card(candidate),
                "observed_visual_evidence": candidate.evidence,
                "conclusion": "This image confirms the same entity, but is a different photograph."
                if candidate.verdict == "YES" and candidate.category == "SAME_ENTITY"
                else "This inspected image matches the reference."
                if candidate.verdict == "YES"
                else "Different identity"
                if candidate.verdict == "NO"
                else "Not the same photograph; do not reject identity. Continue seeking the exact reference.",
            }
        )
    if require_match:
        raise ValueError(
            "Natural selection did not confirm a match within the check budget"
        )
    return attempts, reasons


def choose_checks(plan, window, source, model, config, record=None):
    requested = plan.branch
    attempts, reasons = select_sequence(
        plan.first,
        window,
        source,
        model,
        config,
        limit=2
        if requested == "recovery"
        else config.max_selection_checks
        + bool(plan.metadata.get("required_reference_keys")),
        require_match=requested != "recovery",
        target=plan.metadata.get("local_visual_target"),
        prechecks=plan.metadata.setdefault("blind_prechecks", {}),
        trajectory_history=selection_history(record, window),
        required_reference_keys=plan.metadata.get("required_reference_keys"),
        selection_audit=plan.metadata.setdefault("selection_modes", []),
    )
    plan.attempts = attempts
    plan.metadata["requested_branch"] = requested
    plan.metadata["selection_reasoning"] = reasons
    if requested != "recovery":
        plan.positive_key = attempts[-1].key
        plan.branch = (
            "direct" if all(c.verdict == "YES" for c in attempts) else "contrast"
        )
        plan.metadata["wrong_count"] = sum(c.verdict == "NO" for c in attempts)
        plan.metadata["contrast_achieved"] = any(c.verdict == "NO" for c in attempts)
        plan.metadata["contrast_shortfall"] = (
            requested == "contrast" and not plan.metadata["contrast_achieved"]
        )
        plan.metadata["inconclusive_count"] = sum(
            c.verdict == "CONTINUE" for c in attempts
        )
        if any(c.verdict == "CONTINUE" for c in plan.first):
            plan.metadata["composition_mode"] = "lightweight"
    return plan


def choose_retry_checks(plan, window, source, model, config, record=None):
    attempts, reasons = select_sequence(
        plan.second,
        window,
        source,
        model,
        config,
        limit=config.max_selection_checks,
        require_match=True,
        target=plan.metadata.get("local_visual_target"),
        prechecks=plan.metadata.setdefault("blind_prechecks", {}),
        trajectory_history=selection_history(record, window),
        selection_audit=plan.metadata.setdefault("retry_selection_modes", []),
    )
    plan.attempts = plan.attempts[: plan.retry_after] + attempts
    plan.positive_key = attempts[-1].key
    plan.metadata["selection_reasoning"].update(reasons)
    if any(c.verdict == "CONTINUE" for c in plan.second):
        plan.metadata.pop("retry_positive_count", None)
        plan.metadata["retry_composition_mode"] = "lightweight"
    return plan
