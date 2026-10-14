"""Reproducible evaluation-only expert probes over completed base failures."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

from .cases import prepare_cases_from_rows, load_cases, write_public_cases
from .io import append_jsonl, write_json
from .schema import FailureCandidate


def _case_id(value: dict[str, Any]) -> str:
    return str(value.get("case_id") or value.get("question_id") or value.get("sample_id") or value.get("id") or "").strip()


def _is_wrong(value: dict[str, Any]) -> bool:
    if value.get("error"):
        return False
    return value.get("acc", value.get("is_correct")) in {0, False, "0", "false", "False", "no"}


def _source_case_id(row: dict[str, Any]) -> str:
    for key in ("data_id", "case_id", "question_id", "sample_id", "id", "qid"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def prepare_evaluation_probe_from_rows(
    source_rows: Iterable[dict[str, Any]],
    evaluation_rows: Iterable[dict[str, Any]],
    *,
    output_cases: str | Path,
    output_public_cases: str | Path,
    output_failure_manifest: str | Path,
    output_selection_report: str | Path,
    image_dir: str | Path,
    limit: int = 20,
    seed: int = 20260918,
    source_dataset: str = "fvqa_test_expert_probe",
) -> dict[str, int]:
    """Sample independently-judged test failures and materialize only their images.

    This is intentionally evaluation-only.  The resulting canonical manifest
    has ``split='test'`` and must never be passed to SFT conversion/merge.
    The public writer manifest omits both gold answers and base judge output.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    targets = [Path(output_cases), Path(output_public_cases), Path(output_failure_manifest), Path(output_selection_report)]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite evaluation-probe artifacts: {existing}")

    wrong_by_id: dict[str, dict[str, Any]] = {}
    counts = {"eval_rows": 0, "eligible_wrong": 0, "evaluator_error": 0, "missing_source_case": 0}
    for row in evaluation_rows:
        counts["eval_rows"] += 1
        if row.get("error"):
            counts["evaluator_error"] += 1
            continue
        if not _is_wrong(row):
            continue
        case_id = _case_id(row)
        if case_id:
            wrong_by_id.setdefault(case_id, dict(row))
    counts["eligible_wrong"] = len(wrong_by_id)
    if len(wrong_by_id) < limit:
        raise ValueError(f"Only {len(wrong_by_id)} independently judged base failures are available; need {limit}")

    chosen_ids = random.Random(seed).sample(sorted(wrong_by_id), limit)
    source_by_id = {_source_case_id(dict(row)): dict(row) for row in source_rows}
    chosen_rows: list[dict[str, Any]] = []
    for case_id in chosen_ids:
        source = source_by_id.get(case_id)
        if source is None:
            counts["missing_source_case"] += 1
            continue
        chosen_rows.append(source)
    if len(chosen_rows) != limit:
        raise ValueError(f"Selected base failures missing from source Parquet: {counts['missing_source_case']}")

    materialized = prepare_cases_from_rows(
        chosen_rows,
        output_jsonl=output_cases,
        image_dir=image_dir,
        split="test",
        source_dataset=source_dataset,
        allow_nontrain=True,
        progress=True,
    )
    selected_cases = load_cases(output_cases)
    case_map = {case.case_id: case for case in selected_cases}
    for case_id in chosen_ids:
        if case_id not in case_map:
            raise ValueError(f"Selected test case was not materialized: {case_id}")
        row = wrong_by_id[case_id]
        append_jsonl(
            output_failure_manifest,
            FailureCandidate(
                case_id=case_id,
                reason=str(row.get("reasoning") or row.get("reason") or "base_model_incorrect"),
                base_answer=str(row.get("model_answer") or row.get("base_answer") or ""),
                evaluator_confidence=(float(row["confidence"]) if row.get("confidence") is not None else None),
                source_run="completed_test_evaluation",
                source_metadata={"raw_judge": row.get("raw_judge", ""), "base_run_case_id": case_id},
            ).to_dict(),
        )
    public_count = write_public_cases(selected_cases, output_public_cases)
    write_json(
        output_selection_report,
        {
            "purpose": "evaluation_only_expert_probe_not_for_sft",
            "source_split": "test",
            "seed": seed,
            "requested": limit,
            "selected_case_ids": chosen_ids,
            "source_dataset": source_dataset,
            "counts": {**counts, **{f"materialize_{key}": value for key, value in materialized.items()}, "public_cases": public_count},
        },
    )
    return {**counts, **{f"materialize_{key}": value for key, value in materialized.items()}, "public_cases": public_count}


def prepare_evaluation_probe_from_parquet(
    input_parquet: str | Path,
    evaluation_rows: Iterable[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, int]:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("evaluation probe requires pandas plus a Parquet engine") from error
    frame = pd.read_parquet(input_parquet)
    return prepare_evaluation_probe_from_rows(frame.to_dict(orient="records"), evaluation_rows, **kwargs)
