"""Causal, visual-evidence-aware acceptance gates for FVQA writer output."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import append_jsonl, iter_json_records
from .quality import validate_trajectory_quality_review
from .schema import CanonicalCase, Decision, FilterConfig
from .trajectory import has_rejection_language, message_text, parse_trajectory
from .visual import build_blind_visual_request, validate_blind_visual_verdict


_PRIVATE_LANGUAGE = (
    "gold answer",
    "ground truth",
    "private reference",
    "provided answer",
    "reference answer",
    "hop_chain",
    "hop chain",
)


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("case_id") or record.get("question_id") or record.get("sample_id") or "").strip()


def _answer_review_passes(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("error"):
        return False
    return value.get("is_correct") in {True, 1, "1", "true", "True", "yes"} or value.get("acc") in {True, 1, "1"}


def _initial_prompt_is_clean(record: dict[str, Any], *, gold_answers: Iterable[str]) -> bool:
    question = str(record.get("question") or "").strip()
    normalized_gold = [str(answer or "").strip().casefold() for answer in gold_answers]
    normalized_gold = [answer for answer in normalized_gold if answer]
    for message in record.get("raw_messages") or []:
        if not isinstance(message, dict) or str(message.get("role") or "").lower() != "user":
            continue
        text = message_text(message.get("content"))
        if not text:
            continue
        # Follow-up image messages are allowed, but the first natural-language
        # prompt must be the public question and must not contain the answer.
        if text.strip() == question:
            lowered = text.casefold()
            return not any(answer in lowered for answer in normalized_gold)
        return False
    return False


def _contains_private_language(provenance_texts: list[str]) -> bool:
    transcript = "\n".join(provenance_texts).casefold()
    return any(token in transcript for token in _PRIVATE_LANGUAGE)


def _has_textual_evidence(provenance: Any) -> bool:
    for event in provenance.tool_events:
        if event.name != "read_url" or event.ok is False:
            continue
        # An image read returns just image_id. A page read produces text or a
        # public page observation; either supports the later factual claim.
        if event.output and not event.output.get("image_id"):
            return True
    return False


def audit_record(
    record: dict[str, Any],
    *,
    answer_review: dict[str, Any] | None,
    blind_visual_review: dict[str, Any] | None,
    trajectory_quality_review: dict[str, Any] | None = None,
    config: FilterConfig,
    gold_answers: Iterable[str] = (),
    gold_answer: str | None = None,
) -> tuple[Decision, dict[str, Any]]:
    """Run non-generative gates and return a detailed auditable decision.

    ``gold_answer`` remains accepted for older FVQA callers. New multi-answer
    sources provide ``gold_answers``; both are private leakage-audit inputs.
    """

    config.validate()
    reasons: list[str] = []
    effective_gold_answers = tuple(gold_answers)
    if gold_answer:
        effective_gold_answers = (*effective_gold_answers, gold_answer)
    provenance = parse_trajectory(record)
    summary = record.get("generation_summary") or {}
    if record.get("collection_error"):
        reasons.append("collection_error")
    if str((record.get("source_metadata") or {}).get("collector_reference_mode") or "") != "hidden_reference":
        reasons.append("collector_not_marked_hidden_reference")
    if not _initial_prompt_is_clean(record, gold_answers=effective_gold_answers):
        reasons.append("initial_prompt_not_label_safe")
    if not bool(summary.get("generation_complete")):
        reasons.append("generation_incomplete")
    if config.require_correct_answer and not _answer_review_passes(answer_review):
        reasons.append("expert_answer_not_independently_correct")
    if config.require_trajectory_quality_review:
        quality_ok, quality_reason = validate_trajectory_quality_review(trajectory_quality_review)
        if not quality_ok:
            reasons.append(quality_reason)
    if provenance.assistant_turns > config.max_assistant_turns:
        reasons.append("too_many_assistant_turns")
    if provenance.tool_errors >= config.reject_tool_errors_at_or_above:
        reasons.append("too_many_tool_errors")
    if _contains_private_language(provenance.assistant_texts):
        reasons.append("private_supervision_language")

    visual_calls = [event for event in provenance.tool_events if event.name in {"i2i_search", "t2i_search"} and event.ok is not False]
    quality_visual_level = (
        str((trajectory_quality_review or {}).get("visual_verification_level") or "").lower()
        if isinstance(trajectory_quality_review, dict)
        else ""
    )
    # Strong visual evidence is valuable enough to preserve its stricter
    # causal/blind-verification gate even in the mixed-diversity export. Weak
    # and nonvisual records are instead governed by the label-blind quality
    # review above, unless the caller explicitly requests the visual subset.
    visual_branch_required = config.require_visual_compare or quality_visual_level == "strong"
    if visual_branch_required:
        if not visual_calls:
            reasons.append("no_visual_search")
        if len(provenance.opened_candidate_ids) < config.min_opened_visual_candidates:
            reasons.append("insufficient_opened_visual_candidates")
        request = build_blind_visual_request(record, provenance)
        if request is None:
            reasons.append("opened_candidate_images_not_recoverable")
        if config.require_candidate_rejection and not has_rejection_language(provenance.assistant_texts):
            reasons.append("no_grounded_candidate_rejection")
        if config.require_blind_visual_review:
            valid, reason = validate_blind_visual_verdict(blind_visual_review)
            if not valid:
                reasons.append(reason)
    else:
        request = None
    if config.require_external_fact_evidence and not _has_textual_evidence(provenance):
        reasons.append("missing_external_text_evidence")

    metrics = {
        "assistant_turns": provenance.assistant_turns,
        "tool_errors": provenance.tool_errors,
        "tool_calls": dict(Counter(event.name for event in provenance.tool_events)),
        "visual_search_calls": len(visual_calls),
        "visual_candidates_returned": len(provenance.visual_candidate_ids),
        "opened_visual_candidates": len(provenance.opened_candidate_ids),
        "opened_candidate_images": len(provenance.candidate_image_urls),
        "has_candidate_rejection_language": has_rejection_language(provenance.assistant_texts),
        "has_textual_evidence": _has_textual_evidence(provenance),
        "trajectory_quality_visual_verification_level": quality_visual_level,
        "visual_branch_required": visual_branch_required,
    }
    audit = {
        "case_id": _record_id(record),
        "decision": "keep" if not reasons else "filter",
        "reasons": reasons,
        "metrics": metrics,
        "provenance": provenance.to_dict(),
        "blind_visual_request": request.to_dict() if request is not None else None,
        "answer_review": answer_review,
        "trajectory_quality_review": trajectory_quality_review,
        "blind_visual_review": blind_visual_review,
    }
    return Decision(audit["decision"], reasons, metrics), audit


def _review_map(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = _record_id(row)
        if identifier:
            values[identifier] = row
    return values


def filter_records(
    raw_input: str | Path,
    accepted_output: str | Path,
    rejected_output: str | Path,
    audit_output: str | Path,
    *,
    answer_reviews: Iterable[dict[str, Any]],
    trajectory_quality_reviews: Iterable[dict[str, Any]],
    blind_reviews: Iterable[dict[str, Any]],
    config: FilterConfig,
    private_cases: dict[str, CanonicalCase],
) -> dict[str, int]:
    """Apply gates and write immutable raw rows for ShareGPT conversion."""

    for target in (accepted_output, rejected_output, audit_output):
        path = Path(target)
        if path.exists() and path.stat().st_size:
            raise FileExistsError(f"Filter output exists; choose a fresh output directory: {path}")
    answer_map = _review_map(answer_reviews)
    quality_map = _review_map(trajectory_quality_reviews)
    blind_map = _review_map(blind_reviews)
    counts: Counter[str] = Counter()
    for record in iter_json_records(raw_input):
        case_id = _record_id(record)
        if not case_id:
            counts["missing_case_id"] += 1
            continue
        decision, audit = audit_record(
            record,
            answer_review=answer_map.get(case_id),
            blind_visual_review=blind_map.get(case_id),
            trajectory_quality_review=quality_map.get(case_id),
            config=config,
            gold_answers=(
                private_cases[case_id].acceptable_answers or (private_cases[case_id].gold_answer,)
                if case_id in private_cases
                else ()
            ),
        )
        enriched = dict(record)
        enriched["answer_judge"] = answer_map.get(case_id) or {"is_correct": False, "status": "missing_independent_review"}
        enriched["trajectory_quality_judge"] = quality_map.get(case_id) or {
            "decision": "filter",
            "status": "missing_independent_review",
        }
        enriched["fix2_filter"] = audit
        append_jsonl(audit_output, audit)
        if decision.accepted:
            append_jsonl(accepted_output, enriched)
            counts["accepted"] += 1
        else:
            append_jsonl(rejected_output, enriched)
            counts["rejected"] += 1
            for reason in decision.reasons:
                counts[f"reason:{reason}"] += 1
    return dict(counts)


def summarize_filter_audit(audit_input: str | Path) -> dict[str, Any]:
    """Summarize immutable filter audit rows without exposing private labels."""

    decisions: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    rejection_reason_combinations: Counter[str] = Counter()
    answer_reviews: Counter[str] = Counter()
    quality_reviews: Counter[str] = Counter()
    visual_levels: Counter[str] = Counter()
    total = 0
    for row in iter_json_records(audit_input):
        total += 1
        decisions[str(row.get("decision") or "unknown")] += 1
        reasons = [str(reason) for reason in row.get("reasons") or []]
        for reason in reasons:
            rejection_reasons[str(reason)] += 1
        if str(row.get("decision") or "") == "filter":
            rejection_reason_combinations[" + ".join(sorted(reasons)) or "unspecified"] += 1

        answer = row.get("answer_review")
        if not isinstance(answer, dict):
            answer_reviews["missing"] += 1
        elif answer.get("error"):
            answer_reviews["error"] += 1
        elif _answer_review_passes(answer):
            answer_reviews["correct"] += 1
        else:
            answer_reviews["incorrect"] += 1

        quality = row.get("trajectory_quality_review")
        if not isinstance(quality, dict):
            quality_reviews["missing"] += 1
        else:
            quality_reviews[f"{quality.get('status') or 'unknown'}/{quality.get('decision') or 'unknown'}"] += 1

        level = str((row.get("metrics") or {}).get("trajectory_quality_visual_verification_level") or "unknown")
        visual_levels[level] += 1

    return {
        "total": total,
        "decision": dict(decisions),
        "rejection_reasons": dict(rejection_reasons),
        "rejection_reason_combinations": dict(rejection_reason_combinations),
        "independent_answer_review": dict(answer_reviews),
        "trajectory_quality_review": dict(quality_reviews),
        "visual_verification_level": dict(visual_levels),
    }


def quality_filter_details(audit_input: str | Path) -> list[dict[str, Any]]:
    """Return concise, label-free evidence for trajectories rejected by quality review."""

    details: list[dict[str, Any]] = []
    for row in iter_json_records(audit_input):
        reasons = [str(reason) for reason in row.get("reasons") or []]
        if "trajectory_quality_review_filter" not in reasons:
            continue
        review = row.get("trajectory_quality_review") or {}
        metrics = row.get("metrics") or {}
        details.append(
            {
                "case_id": str(row.get("case_id") or ""),
                "filter_reasons": reasons,
                "answer_independently_correct": _answer_review_passes(row.get("answer_review")),
                "quality_review": {
                    "decision": review.get("decision"),
                    "reason": review.get("reason"),
                    "evidence_sufficient": review.get("evidence_sufficient"),
                    "internal_knowledge_suspected": review.get("internal_knowledge_suspected"),
                    "logical_gap": review.get("logical_gap"),
                    "visual_verification_level": review.get("visual_verification_level"),
                    "evidence": review.get("evidence"),
                },
                "trajectory_metrics": {
                    "assistant_turns": metrics.get("assistant_turns"),
                    "tool_calls": metrics.get("tool_calls"),
                    "opened_visual_candidates": metrics.get("opened_visual_candidates"),
                    "has_textual_evidence": metrics.get("has_textual_evidence"),
                },
            }
        )
    return details
