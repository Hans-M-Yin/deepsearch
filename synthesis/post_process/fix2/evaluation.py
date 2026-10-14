"""Independent answer review after collection, never during writing."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping

from opensearch_vl.eval_infer_with_llm import judge_answer

from .io import append_jsonl, iter_json_records, load_completed_ids
from .schema import CanonicalCase


def _format_reference_answers(case: CanonicalCase) -> str:
    """Render one-or-more private references for the independent judge.

    The writer never sees this value. Keeping alternatives distinct prevents a
    LiveVQA answer such as one accepted alias from being evaluated against a
    fabricated comma-joined target.
    """

    answers = tuple(dict.fromkeys(case.acceptable_answers or (case.gold_answer,)))
    answers = tuple(answer for answer in answers if answer.strip())
    if len(answers) <= 1:
        return answers[0] if answers else ""
    return "Any one of these reference answers is acceptable:\n" + "\n".join(
        f"- {answer}" for answer in answers
    )


def judge_expert_records(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    model_alias: str,
    private_cases: Mapping[str, CanonicalCase],
    max_tokens: int = 1024,
    prompt_style: str = "local",
    resume: bool = False,
    max_workers: int = 1,
) -> dict[str, int]:
    """Evaluate final answers only; this is the first component allowed to see gold."""

    if Path(output_jsonl).exists() and not resume:
        raise FileExistsError(f"Answer-review output exists; use --resume or choose a new path: {output_jsonl}")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    completed = load_completed_ids(output_jsonl, require_success=True) if resume else set()
    counts = {"seen": 0, "written": 0, "skipped_resume": 0, "errors": 0}

    def review_one(record: dict[str, Any], case_id: str) -> tuple[dict[str, Any], bool]:
        if record.get("collection_error"):
            return {"case_id": case_id, "is_correct": False, "error": record["collection_error"]}, True
        private_case = private_cases.get(case_id)
        if private_case is None:
            return {"case_id": case_id, "is_correct": False, "error": "missing_private_case"}, True
        try:
            result = judge_answer(
                private_case.question,
                _format_reference_answers(private_case),
                str(record.get("extracted_answer") or ""),
                judge_model_alias=model_alias,
                judge_max_tokens=max_tokens,
                prompt_style=prompt_style,
            )
            return {
                "case_id": case_id,
                "is_correct": bool(result.get("acc")),
                "reference_answer_count": len(private_case.acceptable_answers or (private_case.gold_answer,)),
                **result,
            }, False
        except Exception as error:
            return {"case_id": case_id, "is_correct": False, "error": f"{error.__class__.__name__}: {error}"}, True

    def persist(value: tuple[dict[str, Any], bool]) -> None:
        row, has_error = value
        append_jsonl(output_jsonl, row)
        counts["written"] += int(not has_error)
        counts["errors"] += int(has_error)

    if max_workers == 1:
        for record in iter_json_records(input_jsonl):
            case_id = str(record.get("case_id") or record.get("question_id") or "").strip()
            if not case_id:
                continue
            counts["seen"] += 1
            if case_id in completed:
                counts["skipped_resume"] += 1
                continue
            persist(review_one(record, case_id))
        return counts

    # Parallelize network-bound judgement while keeping the append-only audit
    # file in input order.  Ordered writes make a crash/restart deterministic.
    next_index = 0
    next_to_write = 0
    outcomes: dict[int, tuple[dict[str, Any], bool]] = {}
    pending: dict[Any, int] = {}

    def collect_done(*, block: bool) -> None:
        nonlocal next_to_write
        if not pending:
            return
        if block:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in pending if future.done()}
        for future in done:
            index = pending.pop(future)
            outcomes[index] = future.result()
        while next_to_write in outcomes:
            persist(outcomes.pop(next_to_write))
            next_to_write += 1

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fix2-answer") as pool:
        for record in iter_json_records(input_jsonl):
            case_id = str(record.get("case_id") or record.get("question_id") or "").strip()
            if not case_id:
                continue
            counts["seen"] += 1
            if case_id in completed:
                counts["skipped_resume"] += 1
                continue
            pending[pool.submit(review_one, record, case_id)] = next_index
            next_index += 1
            if len(pending) >= max_workers * 2:
                collect_done(block=True)
        while pending:
            collect_done(block=True)
    return counts
