"""Structural invariants plus an independent multimodal continuity review."""

import re
from pathlib import Path
from .trajectory_index import (
    messages,
    content,
    role,
    call,
    image_bindings,
    ASSISTANT,
    OBSERVATION,
    parse_observation,
    RESOURCE_FIELDS,
)
from .splice import remap_text
from .visual_evidence import CLASSIFICATION
from .evidence_policy import TARGET_EVIDENCE
from .review_images import build_review_images
from .process_leakage import check_image_labels
from .context_patch import PROTECTED


def validate_reasoning_only(before, after):
    """Global editing may change prose, but no message or protected payload."""
    errors = []
    if {k: v for k, v in before.items() if k not in {"messages", "conversations"}} != {
        k: v for k, v in after.items() if k not in {"messages", "conversations"}
    }:
        errors.append("Refinement changed record metadata or images")
    old, new = messages(before), messages(after)
    if len(old) != len(new):
        return errors + ["Refinement changed message count"]
    for index, (a, b) in enumerate(zip(old, new)):
        if role(a) not in ASSISTANT:
            if a != b:
                errors.append(f"Refinement changed non-assistant turn {index}")
            continue
        text_key = "value" if "value" in a else "content"
        if {k: v for k, v in a.items() if k != text_key} != {
            k: v for k, v in b.items() if k != text_key
        }:
            errors.append(f"Refinement changed assistant metadata/role at {index}")
        if [m.group(0) for m in PROTECTED.finditer(content(a))] != [
            m.group(0) for m in PROTECTED.finditer(content(b))
        ] or content(a).count("<image>") != content(b).count("<image>"):
            errors.append(f"Refinement changed protected assistant content at {index}")
    return errors


def validate_structure(
    original, updated, window, plan, splice_audit, *, pre_refinement=None
):
    image_bindings(updated)
    old, new = messages(original), messages(updated)
    errors = []
    if plan.branch == "convergent":
        if updated != original:
            errors.append("Convergent branch must preserve the original record")
        if plan.attempts or not plan.metadata.get("convergence", {}).get("eligible"):
            errors.append("Invalid zero-view branch")
        return errors
    # Check assembled prose, not just newly generated fragments. Tool arguments
    # and observations are immutable data and must not be rewritten as prose.
    for index, message in enumerate(new):
        if role(message) in ASSISTANT:
            prose = re.sub(
                r"<(tool_call|action)>.*?</\1>", "", content(message), flags=re.S
            )
            try:
                check_image_labels(prose)
            except ValueError:
                errors.append(f"Internal image-label leakage in assistant turn {index}")
    if not 1 <= len(plan.first) <= 5 or (
        plan.branch == "recovery" and not 1 <= len(plan.second) <= 5
    ):
        errors.append("Visible search results must have one to five entries")
    controlled = plan.metadata.get("composition_mode") == "controlled"
    for candidates, expected in (
        (plan.first, plan.metadata.get("positive_count") if controlled else None),
        (plan.second, plan.metadata.get("retry_positive_count")),
    ):
        if expected is not None and (
            expected not in {1, 2}
            or sum(c.verdict == "YES" for c in candidates) != expected
            or any(c.verdict not in {"YES", "NO"} for c in candidates)
        ):
            errors.append("Controlled candidate set violates positive/negative counts")
    all_public = plan.first + plan.second
    text_only = {c.key for c in all_public if c.text_reference_only}
    retained = set(plan.metadata.get("retained_reference_keys", []))
    if text_only != set(plan.metadata.get("text_only_reference_keys", [])):
        errors.append("Text-only reference metadata does not match candidate flags")
    historical_cards = {c["image_id"]: c for c in window.candidates}
    for c in all_public:
        if not c.text_reference_only:
            continue
        rid = c.public.get("image_id")
        original_card = historical_cards.get(rid)
        read = window.reads.get(rid, {})
        if (
            plan.metadata.get("flow_version") != "selector_writer_v1"
            or c.key not in retained
            or not c.historical
            or not c.key.endswith("old_" + str(rid))
            or c.originally_viewed
            or read.get("paths")
            or not original_card
            or any(
                c.public.get(k) != original_card.get(k)
                for k in ("title", "source", "image_id", "source_page_id")
            )
            or c.category != "UNRESOLVED"
            or c.verdict != "UNCERTAIN"
        ):
            errors.append("Invalid retained text-only historical reference")
    if any(c.text_reference_only or c.key in text_only for c in plan.attempts):
        errors.append("Text-only reference cannot be inspected as an image")
    if plan.positive_key in text_only or any(
        s.get("candidate_key") in text_only for s in plan.metadata.get("steps", [])
    ):
        errors.append("Text-only reference cannot supply visual confirmation")
    if plan.metadata.get("flow_version") == "selector_writer_v1":
        retained = set(plan.metadata.get("retained_reference_keys", []))
        if not retained <= {c.key for c in plan.first}:
            errors.append("A relied-on historical reference was removed")
        if len({c.key for c in plan.attempts}) != len(plan.attempts):
            errors.append("Natural inspection loop repeated a candidate")
        if any(c.category == "UNRELATED" for c in plan.attempts):
            errors.append("Natural inspection loop selected an excluded card")
        expected_steps = (
            plan.attempts[:-1]
            if plan.metadata.get("reuse_original_read")
            else plan.attempts
        )
        if [s.get("candidate_key") for s in plan.metadata.get("steps", [])] != [
            c.key for c in expected_steps
        ]:
            errors.append("Inspection trace differs from actual selection order")
        if plan.metadata.get("reuse_original_read"):
            turn = plan.metadata.get("original_read_turn")
            if (
                not isinstance(turn, int)
                or not 0 <= turn < len(old) - 1
                or "<image>" not in content(old[turn + 1])
                or turn in splice_audit.get("removed_indices", [])
            ):
                errors.append("Terminal original image check was not preserved")
    category_labels = {
        "CONSISTENT": "YES",
        "SIMILAR": "NO",
        "RELATED": "NO",
        "UNRELATED": "NO",
        "SAME_ENTITY": "CONTINUE",
        "DIFFERENT_PHOTO": "CONTINUE",
    }

    def expected_verdict(c):
        if c.text_reference_only:
            # The provenance/usage checks above bound this exception. Do not
            # generally allow unresolved images in executable candidate sets.
            return "UNCERTAIN"
        historical_view = (
            c.originally_viewed
            and c.historical
            and any(c.key.endswith("old_" + rid) for rid in window.reads)
        )
        return (
            "YES"
            if c.category == "SAME_ENTITY"
            and (
                historical_view
                or plan.metadata.get("flow_version") == "selector_writer_v1"
            )
            else category_labels.get(c.category)
        )

    if any(expected_verdict(c) != c.verdict for c in all_public):
        errors.append("Candidate category is unresolved or inconsistent with its label")
    provided = {
        c.public.get(k) for c in all_public for k in RESOURCE_FIELDS if c.public.get(k)
    }
    if not set(window.protected_ids) <= provided:
        errors.append("A historical resource required by the continuation was removed")
    for candidates in (plan.first, plan.second):
        ids = [c.public.get("image_id") for c in candidates]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate candidate image IDs")
    if (
        not plan.attempts
        or plan.attempts[-1].verdict != "YES"
        or plan.attempts[-1].key != plan.positive_key
    ):
        errors.append("Missing terminal correct visual confirmation")
    required_refs = set(plan.metadata.get("required_reference_keys", []))
    if required_refs and not required_refs <= {
        c.key for c in plan.attempts if c.verdict == "YES"
    }:
        errors.append("Both historical references require visual confirmation")
    if any(
        c.verdict not in {"NO", "CONTINUE"}
        and not (required_refs and c.verdict == "YES")
        for c in plan.attempts[:-1]
    ):
        errors.append("Non-negative candidate used as rejection")
    if plan.branch == "recovery" and any(
        c.verdict not in {"NO", "CONTINUE"} for c in plan.first
    ):
        errors.append(
            "Recovery first list contains an unverified or matching candidate"
        )
    for index, c in enumerate(plan.attempts):
        visible = (
            plan.second
            if plan.branch == "recovery" and index >= plan.retry_after
            else plan.first
        )
        if c.key not in {x.key for x in visible}:
            errors.append("Viewed candidate is absent from preceding search")
        if (
            not c.evidence
            or not c.image_paths
            or any(not Path(p).is_file() for p in c.image_paths)
        ):
            errors.append("Missing actual visual evidence")
    mapping = splice_audit["id_mapping"]
    expected_prefix = old[: window.observation_index]
    if pre_refinement is not None:
        errors.extend(validate_reasoning_only(pre_refinement, updated))
        # The snapshot must itself satisfy the strict splice-prefix contract.
        prefix = messages(pre_refinement)[: window.observation_index]
    else:
        prefix = new[: window.observation_index]
    if prefix != expected_prefix:
        errors.append("Prefix modified")
    for i in range(window.end, len(old)):
        if i in splice_audit.get("removed_indices", []):
            continue
        a, b = old[i], new[splice_audit["continuation_map"][str(i)]]
        if role(a) != role(b):
            errors.append("Continuation role changed")
        if role(a) in OBSERVATION and remap_text(content(a), mapping) != content(b):
            errors.append("Original observation changed")
        if role(a) in ASSISTANT:
            ac = call(remap_text(content(a), mapping))
            if str(i) in splice_audit.get("source_repairs", {}):
                ac = splice_audit["source_repairs"][str(i)]
            bc = call(content(b))
            if ac != bc:
                errors.append("Original downstream tool action changed")

    def answers(turns):
        return re.findall(
            r"<answer>.*?</answer>", "\n".join(content(m) for m in turns), re.S
        )

    if answers(old) != answers(new):
        errors.append("Final answer changed")
    # New resources must have appeared in a prior search observation.
    declared = set()
    for m in new:
        if role(m) in OBSERVATION:
            try:
                payload = parse_observation(content(m))
                for c in payload.get("matches", payload.get("results", [])):
                    if isinstance(c, dict):
                        declared.update(c.get(k) for k in RESOURCE_FIELDS if c.get(k))
                if payload.get("image_id"):
                    declared.add(payload["image_id"])
            except (ValueError, TypeError, AttributeError):
                pass
        elif role(m) in ASSISTANT:
            action = call(content(m))
            if action:
                for value in action["arguments"].values():
                    if (
                        isinstance(value, str)
                        and value.startswith(splice_audit["namespace"])
                        and value not in declared
                    ):
                        errors.append("New resource referenced before declaration")
    return sorted(set(errors))


JUDGE = """Independently review an augmented image-search SFT trajectory.
Use image_evidence_map to associate IMAGE indices with the reference, candidate
checks, and zero-based turns / image placeholders in the updated trajectory.
All retained trajectory images, including later original tool images, are attached.
An image may serve multiple turns. Only treat it as observed at its mapped turn;
having all images available to the reviewer does not give the agent foreknowledge.
Check the question-relevant entity, concrete visual rejection/acceptance, no
foreknowledge before a candidate image is opened, no leaked augmentation labels,
natural candidate selection order, continuity of the patched original reasoning,
and whether the accepted entity supports the original downstream facts/answer.
Multiple search titles agreeing does not resolve a different local target.
Do not rubber-stamp supplied verifier labels or the original answer.
Consecutive assistant reasoning turns are allowed by this dataset's format;
their adjacency alone is not a semantic error. Still reject factual conflicts,
missing evidence, or incoherent reasoning within or between those turns.
Return JSON {"pass":true|false, "issues":["..."], "reason":"..."}.
Fail if evidence is insufficient, the continuation is inconsistent, or the
writer says all unseen candidates fail after inspecting only a subset."""

JUDGE += """
A candidate may confirm the same entity in a different
photograph. Do not reject solely for lack of exact-photo correspondence. Require
actual identity evidence and truthful wording; this does not license transferring
event-specific or photograph-specific details between different photographs.
"""

JUDGE += """
For a rejection or a decision to inspect another candidate, require concrete
paired visual observations, not just 'different photo' or generic differences.
Check that fine details are actually visible, not hallucinated or outside a crop.
Distinguish evidence for a different photograph from evidence for a different
identity. A cautious decision to seek a closer match is not an error merely
because a reviewer could make a stronger identity judgment. Reject unsupported
identity exclusions, but do not demand a categorical identity rejection."""

JUDGE += TARGET_EVIDENCE
JUDGE += """
Reject a subject-identity exclusion supported only by incidental photo differences.
Do not reward a longer detail list unless it actually supports the target decision.
Accept an explicit 'different person/object' conclusion when paired, visible,
identity-discriminating observations support it after accounting for viewpoint
and image quality. Do not require hedging merely because the conclusion concerns
identity. Reject unsupported certainty, not well-supported identity exclusion.
"""


JUDGE += """
Check the actual merged reasoning at insertion boundaries: final visual acceptance
must retain concrete observed matching evidence, not merely 'verified'. Reject
unsupported identity exclusions or confirmations; a different photo does not
establish a different person, and inability to exclude does not establish identity.
Check that later 'first confirmation' claims reflect all earlier image checks.
Do not accept invented discriminative details added solely to make prose convincing.

Compact failure examples (apply to ALL assistant turns, including unchanged original turns; a plausible final answer does not excuse invalid reasoning):
- Event conflation: the input photo shows a Chiefs game, but the question asks about a later touchdown celebration. Rejecting a Patriots celebration candidate because it is not the input game is invalid; track the two events separately.
- Unclosed evidence chain: Zeus's mother is established as Rhea, but the altar source identifies Tellus and never establishes the required connection. Saying 'that distinction does not matter' does not close the gap. A source saying 'goose or swan' also does not alone justify a certain 'swan'; require additional distinguishing evidence or explicit uncertainty.
- Temporal contradiction: after identifying a Super Bowl LIV (2020) photo, describing a 2000 dress appearance as happening 'later' reverses the chronology, even if the person and final answer are correct.
- Editing leakage: 'as already laid out in the draft' and 'the answer retains the trajectory's conclusion' describe editing rather than solving. Reject such assistant narration even when inherited from the original trajectory.
- Internal-label leakage: prose such as 'I will inspect candidate fix_...' exposes construction bookkeeping. Refer naturally to the result's title or visible position; resource IDs inside legitimate tool calls are allowed.
For any such failure, return pass=false and identify the zero-based turn and specific unsupported inference or offending wording in issues. These are failure patterns, not facts to import into other trajectories.
"""

BLIND_REVIEW = (
    """Independently compare the question-relevant target in IMAGE 0
with each following candidate image. No titles, prior labels, trajectory or
answer are provided. """
    + CLASSIFICATION
    + """
Return JSON {"comparisons":[{"category":"CONSISTENT|SIMILAR|RELATED|UNRELATED|SAME_ENTITY|DIFFERENT_PHOTO|UNRESOLVED",
"evidence":["visible matching features or contradictions"]}]} in image order."""
)


def semantic_review(
    original, updated, window, plan, source, model, config, dataset_root="."
):
    blind = None
    if config.blind_visual_review:
        prechecks = plan.metadata.get("blind_prechecks", {})
        if all(c.key in prechecks for c in plan.attempts):
            blind = {
                "comparisons": [prechecks[c.key] for c in plan.attempts],
                "reused_prewrite_checks": True,
            }
        else:
            blind = model.ask(
                config.judge_model,
                BLIND_REVIEW,
                {"question": window.question},
                [source, *[p for c in plan.attempts for p in c.image_paths]],
            )
        comparisons = blind.get("comparisons", [])
        if len(comparisons) != len(plan.attempts) or any(
            r.get("category") != c.category or not r.get("evidence")
            for r, c in zip(comparisons, plan.attempts)
        ):
            return {
                "pass": False,
                "issues": ["Blind visual review disagrees or is uncertain"],
                "blind_review": blind,
            }
    review_paths, evidence_map = build_review_images(
        updated, source, plan.attempts, dataset_root
    )
    result = model.ask(
        config.judge_model,
        JUDGE,
        {
            "question": window.question,
            "preservation_contract": "Structural validation already verified original downstream tool calls, tool observations and final answer are unchanged. Evaluate factual support and continuity independently; preservation does not imply correctness.",
            "updated": messages(updated),
            "candidate_image_order": [c.public for c in plan.attempts],
            "image_evidence_map": evidence_map,
        },
        review_paths,
    )
    result["blind_review"] = blind
    result["image_evidence_map"] = evidence_map
    return result
