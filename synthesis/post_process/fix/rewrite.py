"""Write each decision with only evidence available at that point in time."""

from .trajectory_index import content, messages
from .evidence_policy import TARGET_EVIDENCE
from .evidence_attribution import EVIDENCE_ATTRIBUTION
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from .process_leakage import check_image_labels, IMAGE_LABEL_INSTRUCTION
from .errors import ReviewRejected

BEFORE = """Write the assistant reasoning immediately BEFORE opening the specified
candidate image. You see only available search cards and prior observations.
Explain a plausible choice using those cards and the original reference. Do not
claim to have inspected the candidate or know whether it matches. Do not mention
augmentation, labels, predetermined order, or the future answer. Return JSON
{"reasoning":"..."}. Keep it concise, in the original trajectory's language."""

AFTER = """Write the assistant reasoning after inspecting the candidate image.
IMAGE 0 is the original and the other images are the inspected candidate.
Use the supplied validated evidence, comparing the question-relevant target.
Reject a demonstrably wrong identity or accept a supported match. Preserve
uncertainty. No claims about unopened pictures or future results. Return the
structured JSON assessment specified below. Keep reasoning concise, in the trajectory's language."""

AFTER += """
For SAME_ENTITY with verdict YES, confirm the same person/object using concrete
identity evidence and the source caption, while explicitly acknowledging this is
a different photograph. Do NOT claim an exact-photo match or keep searching merely
because poses differ. Matching an image does not verify all source-page claims;
the source caption is a textual lead, not proof of an event, date or location.
For SAME_ENTITY with verdict CONTINUE or DIFFERENT_PHOTO, do NOT reject identity.
Explain what remains unresolved for the LOCAL objective; continue only if that
objective is not established, never merely to find the exact photo. SAME_ENTITY supports the same subject;
DIFFERENT_PHOTO leaves identity uncertain. Clothes, pose and grooming changes
alone cannot establish a different person. Never strengthen the supplied evidence.
Describe images naturally as the original and the opened candidate, not IMAGE 0/1."""

AFTER += """
When rejecting a candidate or continuing because it is not the reference photo,
do not merely say 'not a match', 'different features' or 'not the same photograph'.
Explain specific paired observations: what is visible in the original versus
what is visible in the candidate, and where. Prefer 2-3 discriminative details
when clearly supported; a single decisive detail is better than invented extras.
For identity exclusion, prioritize visible facial structure/proportions or
object geometry and distinctive components. Hair, garment patterns and watch
chains primarily support photo correspondence, not person-identity exclusion.
These are comparison directions, not a checklist: never invent a difference.
Do not treat a feature outside the crop, occluded, or too small to see as absent.
Then state what those observations establish: different photograph, uncertain
identity, or different subject. Pose, clothing and grooming can establish photo
differences without establishing different identity. Preserve that distinction.
Give the resulting next action naturally, without formulaic Accept/Reject labels.
Specific evidence is more important than extreme brevity; avoid repetitive lists."""

AFTER += TARGET_EVIDENCE + EVIDENCE_ATTRIBUTION
AFTER += """
Use supplied visual labels as constraints against overclaiming, not as a verdict
to rationalize. If a claimed difference lacks target-relevant support, do not
invent a justification: state what can and cannot be established from the image.
Organize the explanation around the decisive evidence and its implication, not
an inventory of clothes, buttons and background details. A same-photo match may
be sufficient with multiple exact correspondences; do not fabricate facial detail.
"""

RETRY = """Write a short rationale for repeating reverse-image search on the
original query image/region, after the inspected candidates failed. Seeing two
mismatches does NOT prove all five fail. Say the results inspected have not
identified the target; do not assert unseen results are wrong. You do not know
what the retry will return. Return JSON {"reasoning":"..."}."""


AFTER += """
Compare the two actual images, not just the supplied classification. Establish
which question is being decided: same photograph, same person/object instance,
or same category. Give paired observations and explain WHY the feature separates
that target, accounting for viewpoint, lighting, age and changeable appearance.
Clothes, grooming or pose alone cannot establish different identity. Likewise,
failure to prove different identity does not establish the same identity.
Do not claim 'different facial structure' without specifying supported visible
differences. Make the inference understandable without the images, but never
invent details to make it persuasive. A nonmatching photograph can simply remain
an unconfirmed identity lead. If supplied labels materially contradict what you
can support, return {"unsafe":true,"reason":"specific evidence conflict"} rather
than rationalizing the label or silently changing the planned verdict.
"""


BEFORE += IMAGE_LABEL_INSTRUCTION
AFTER += IMAGE_LABEL_INSTRUCTION
RETRY += IMAGE_LABEL_INSTRUCTION

AFTER += """
Return a structured assessment in addition to reasoning:
{"reasoning":"natural agent reasoning", "decision":"ACCEPT|REJECT|UNCERTAIN",
"local_goal_satisfied":true, "established_facts":["supported fact"],
"unresolved_questions":["remaining question"], "decision_evidence":"specific paired evidence"}.
local_goal_satisfied is a JSON boolean. Decide from the actual pictures and the
LOCAL objective, independently of the supplied category/verdict. ACCEPT means
the local objective is supported; REJECT requires a supported target-relevant
contradiction; UNCERTAIN means neither is established. Different photo alone is
not REJECT. Do not fabricate facial distinctions to justify a negative label.
For identity/operator identification, a different photo may suffice. For a
specific scene's color, pose or wording, require the relevant detail to be visible;
same identity alone does not suffice. A readable target poster in a different
photo is useful; a different bus number alone does not refute a fleet/operator.
Do not keep searching just to reproduce the reference composition. Report label
conflicts honestly; the caller will handle them. established_facts must contain
only findings from this inspection and visible prior evidence, not future facts.
"""


def validate_observation(result, candidate, *, allow_inconclusive=False):
    """Fail closed when the actual inspection cannot support the planned branch."""
    if not isinstance(result, dict):
        raise ValueError("Writer response must be a JSON object")
    if result.get("unsafe"):
        raise ReviewRejected("Writer visual evidence conflict: " + str(result))
    reasoning = checked_reasoning(result)
    decision = result.get("decision")
    satisfied = result.get("local_goal_satisfied")
    if decision not in {"ACCEPT", "REJECT", "UNCERTAIN"} or type(satisfied) is not bool:
        raise ValueError("Writer must return decision and boolean local_goal_satisfied")
    for key in ("established_facts", "unresolved_questions"):
        if not isinstance(result.get(key), list) or any(
            not isinstance(x, str) or not x.strip() for x in result[key]
        ):
            raise ValueError("Invalid writer " + key)
    if (
        not isinstance(result.get("decision_evidence"), str)
        or not result["decision_evidence"].strip()
    ):
        raise ValueError("Missing writer decision_evidence")
    positive = candidate.category in {"CONSISTENT", "SAME_ENTITY"}
    if satisfied != (decision == "ACCEPT"):
        raise ValueError("Inconsistent writer decision and local_goal_satisfied")
    if (
        allow_inconclusive
        and candidate.verdict == "CONTINUE"
        and decision == "UNCERTAIN"
    ):
        return reasoning
    if (
        decision == "UNCERTAIN"
        or positive != satisfied
        or (not positive and decision != "REJECT")
    ):
        raise ReviewRejected(
            "Writer evidence does not support planned visual decision: " + str(result)
        )
    return reasoning


def checked_reasoning(result):
    if result.get("unsafe"):
        raise ValueError("Unsafe writer evidence: " + str(result.get("reason", "")))
    text = result.get("reasoning", "")
    if (
        not isinstance(text, str)
        or not text.strip()
        or any(
            t in text
            for t in (
                "<tool_call",
                "<action",
                "<answer",
                "<image>",
                "<think",
                "</think",
            )
        )
    ):
        raise ValueError("Writer returned invalid reasoning")
    check_image_labels(text)
    return text.strip()


def write_trace(record, window, plan, anchor, source, model, config):
    # AFTER consumes only this candidate's evidence, never previous writer text.
    # Generate independently, then assemble in the actual inspection order.
    with ThreadPoolExecutor(max_workers=config.writer_workers) as executor:
        futures = [
            executor.submit(
                copy_context().run,
                write_after,
                candidate,
                window,
                anchor,
                source,
                model,
                config,
            )
            for candidate in plan.attempts
        ]
        return assemble_trace(record, window, plan, source, model, config, futures)


def write_after(
    candidate,
    window,
    anchor,
    source,
    model,
    config,
    *,
    history=None,
    selection_reasoning=None,
    assessment=None,
):
    payload = {
        "question": window.question,
        "target": anchor["target"],
        "selected_card": candidate.public,
        "verdict": candidate.verdict,
        "category": candidate.category,
        "evidence": candidate.evidence,
        "trajectory_history": history or [],
        "selection_reasoning": selection_reasoning,
    }
    for attempt in range(config.context_patch_retries + 1):
        result = model.ask(
            config.model, AFTER, payload, [source, *candidate.image_paths]
        )
        if assessment is not None and isinstance(result, dict):
            assessment.clear()
            assessment.update(result)
        try:
            return validate_observation(
                result, candidate, allow_inconclusive=assessment is None
            )
        except ReviewRejected:
            # Never ask the model to rationalize a substantive evidence conflict.
            raise
        except ValueError as exc:
            if attempt == config.context_patch_retries:
                raise
            payload = {
                **payload,
                "validation_feedback": str(exc),
                "previous_response": result,
                "repair_instruction": "Return a complete valid JSON assessment. Repair format only; do not change evidence to satisfy the planned label.",
            }


def assemble_trace(record, window, plan, source, model, config, futures):
    history = []
    steps = []
    visible = [c.public for c in plan.first]
    for i, candidate in enumerate(plan.attempts):
        if plan.branch == "recovery" and i == plan.retry_after:
            retry = checked_reasoning(
                model.ask(
                    config.model,
                    RETRY,
                    {
                        "question": window.question,
                        "history": history,
                        "call": plan.retry_call,
                        "strategy_reason": plan.metadata.get("retry_reason"),
                    },
                    [source],
                )
            )
            steps.append({"kind": "retry", "before": retry})
            visible = [c.public for c in plan.second]
            history.append({"retry_results": visible})
        before = plan.metadata.get("selection_reasoning", {}).get(candidate.key)
        if before is None:
            before = checked_reasoning(
                model.ask(
                    config.model,
                    BEFORE,
                    {
                        "question": window.question,
                        "prefix": [
                            content(m) for m in messages(record)[: window.call_index]
                        ],
                        "search_results": visible,
                        "prior_checks": history,
                        "selected_card": candidate.public,
                    },
                    [source],
                )
            )
        after = futures[i].result()
        step = {
            "kind": "read",
            "candidate_key": candidate.key,
            "before": before,
            "after": after,
        }
        steps.append(step)
        history.append({"card": candidate.public, "observation": after})
    return steps
