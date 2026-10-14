"""Construct a reproducible expert queue from base-model difficulty signals."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .schema import CanonicalCase, FailureCandidate


def _case_id(value: dict[str, Any]) -> str:
    return str(value.get("case_id") or value.get("question_id") or value.get("sample_id") or value.get("id") or "").strip()


def _resolve_case(
    case_map: Mapping[str, CanonicalCase],
    source_index_map: Mapping[int, CanonicalCase],
    case_id: str,
) -> CanonicalCase | None:
    case = case_map.get(case_id)
    if case is None and case_id.startswith("case_"):
        try:
            case = source_index_map.get(int(case_id.removeprefix("case_")))
        except ValueError:
            return None
    return case


def _base_tool_call_count(
    case: CanonicalCase,
    base_run_case_id: str,
    base_tool_call_counts: Mapping[str, int],
) -> int | None:
    """Look up a base cost under both evaluator and canonical identifiers."""

    candidate_keys = [base_run_case_id, case.case_id]
    if case.source_index is not None:
        candidate_keys.append(f"case_{case.source_index}")
    for key in candidate_keys:
        value = base_tool_call_counts.get(key)
        if value is not None:
            return int(value)
    return None


def mine_base_hard_cases(
    cases: Iterable[CanonicalCase],
    evaluation_rows: Iterable[dict[str, Any]],
    *,
    base_tool_call_counts: Mapping[str, int] | None = None,
    min_correct_tool_calls: int = 0,
    max_cases: int | None = None,
) -> tuple[list[CanonicalCase], list[FailureCandidate], dict[str, int]]:
    """Select wrong answers plus correct but tool-expensive base trajectories.

    The latter bucket is useful for distilling a more efficient tool policy.
    Its routing metadata remains outside the PublicCase sent to the writer.
    """

    if min_correct_tool_calls < 0:
        raise ValueError("min_correct_tool_calls must be nonnegative")
    if min_correct_tool_calls and base_tool_call_counts is None:
        raise ValueError("base_tool_call_counts is required when min_correct_tool_calls is positive")

    case_map = {case.case_id: case for case in cases}
    source_index_map = {case.source_index: case for case in cases if case.source_index is not None}
    base_tool_call_counts = base_tool_call_counts or {}
    selected: list[CanonicalCase] = []
    manifest: list[FailureCandidate] = []
    counts = {
        "eval_rows": 0,
        "wrong": 0,
        "correct": 0,
        "expensive_correct": 0,
        "below_cost_threshold": 0,
        "missing_base_trajectory": 0,
        "missing_case": 0,
        "evaluator_error": 0,
    }
    for row in evaluation_rows:
        counts["eval_rows"] += 1
        if row.get("error"):
            counts["evaluator_error"] += 1
            continue
        acc = row.get("acc", row.get("is_correct"))
        is_correct = acc in {1, True, "1", "true", "True", "yes"}
        is_wrong = acc in {0, False, "0", "false", "False", "no"}
        if not is_correct and not is_wrong:
            counts["evaluator_error"] += 1
            continue
        case_id = _case_id(row)
        case = _resolve_case(case_map, source_index_map, case_id)
        if case is None:
            counts["missing_case"] += 1
            continue
        tool_call_count = _base_tool_call_count(case, case_id, base_tool_call_counts)

        if is_correct:
            counts["correct"] += 1
            if not min_correct_tool_calls:
                continue
            if tool_call_count is None:
                counts["missing_base_trajectory"] += 1
                continue
            if tool_call_count < min_correct_tool_calls:
                counts["below_cost_threshold"] += 1
                continue
            selected.append(case)
            manifest.append(
                FailureCandidate(
                    case_id=case.case_id,
                    reason="base_model_correct_but_tool_expensive",
                    base_answer=str(row.get("model_answer") or row.get("base_answer") or ""),
                    evaluator_confidence=(float(row["confidence"]) if row.get("confidence") is not None else None),
                    source_run=str(row.get("source_run") or ""),
                    source_metadata={
                        "raw_judge": row.get("raw_judge", ""),
                        "base_run_case_id": case_id,
                        "selection_bucket": "base_expensive_correct",
                        "base_tool_call_count": tool_call_count,
                        "min_correct_tool_calls": min_correct_tool_calls,
                    },
                )
            )
            counts["expensive_correct"] += 1
            if max_cases is not None and len(selected) >= max_cases:
                break
            continue

        selected.append(case)
        manifest.append(
            FailureCandidate(
                case_id=case.case_id,
                reason=str(row.get("reasoning") or row.get("reason") or "base_model_incorrect"),
                base_answer=str(row.get("model_answer") or row.get("base_answer") or ""),
                evaluator_confidence=(float(row["confidence"]) if row.get("confidence") is not None else None),
                source_run=str(row.get("source_run") or ""),
                source_metadata={
                    "raw_judge": row.get("raw_judge", ""),
                    "base_run_case_id": case_id,
                    "selection_bucket": "base_incorrect",
                    "base_tool_call_count": tool_call_count,
                },
            )
        )
        counts["wrong"] += 1
        if max_cases is not None and len(selected) >= max_cases:
            break
    return selected, manifest, counts


def mine_base_failures(
    cases: Iterable[CanonicalCase],
    evaluation_rows: Iterable[dict[str, Any]],
    *,
    max_cases: int | None = None,
) -> tuple[list[CanonicalCase], list[FailureCandidate], dict[str, int]]:
    """Backward-compatible wrong-answer-only selection helper."""

    return mine_base_hard_cases(cases, evaluation_rows, max_cases=max_cases)
