"""Remove questions whose final wording used rule-based composition.

Examples:

    # Inspect only
    python -m synthesis.vqa.filter_fallback_questions --vqa-dir /path/to/vqa

    # Rewrite questions.jsonl in place, keeping samples.jsonl unchanged
    python -m synthesis.vqa.filter_fallback_questions \
        --vqa-dir /path/to/vqa --in-place
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .batch_runner import VqaBatchRunner


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            yield record


def _load_samples(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _iter_jsonl(path) or ():
        sample_id = str(record.get("sample_id") or "").strip()
        if sample_id:
            records[sample_id] = record
    return records


def _load_legacy_fallback_sample_ids(path: Path) -> set[str]:
    """Recover compose failures from old warnings.jsonl files."""

    sample_ids: set[str] = set()
    for record in _iter_jsonl(path) or ():
        if str(record.get("stage") or "").strip() != "compose_question":
            continue
        sample_id = str(record.get("sample_id") or "").strip()
        if sample_id:
            sample_ids.add(sample_id)
    return sample_ids


def _question_is_fallback(
    question: dict[str, Any],
    *,
    sample: dict[str, Any] | None,
    legacy_fallback_sample_ids: set[str],
) -> tuple[bool, list[str], str | None]:
    if question.get("writer_fallback_used") is True:
        stages = question.get("writer_fallback_stages") or ["compose_question"]
        if isinstance(stages, str):
            stages = [stages]
        return True, [str(item) for item in stages], question.get("writer_fallback_reason")

    if sample is not None:
        used, stages, reason = VqaBatchRunner._writer_fallback_info(sample)
        if used:
            return used, stages, reason

    sample_id = str(question.get("sample_id") or "").strip()
    if sample_id in legacy_fallback_sample_ids:
        return True, ["compose_question"], "legacy_warning_record"
    return False, [], None


def _rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def filter_questions(*, vqa_dir: Path, in_place: bool) -> dict[str, Any]:
    questions_path = vqa_dir / "questions.jsonl"
    samples_path = vqa_dir / "samples.jsonl"
    warnings_path = vqa_dir / "warnings.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"questions.jsonl does not exist: {questions_path}")

    samples = _load_samples(samples_path)
    legacy_fallback_sample_ids = _load_legacy_fallback_sample_ids(warnings_path)
    questions = list(_iter_jsonl(questions_path) or ())
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for question in questions:
        sample_id = str(question.get("sample_id") or "").strip()
        is_fallback, stages, reason = _question_is_fallback(
            question,
            sample=samples.get(sample_id),
            legacy_fallback_sample_ids=legacy_fallback_sample_ids,
        )
        if is_fallback:
            removed.append(
                {
                    "question_id": question.get("question_id"),
                    "sample_id": sample_id,
                    "stages": stages,
                    "reason": reason,
                }
            )
        else:
            kept.append(question)

    if in_place:
        _rewrite_jsonl(questions_path, kept)

    return {
        "questions_path": str(questions_path.resolve()),
        "in_place": in_place,
        "total_questions": len(questions),
        "removed_questions": len(removed),
        "kept_questions": len(kept),
        "removed": removed,
        "legacy_warning_sample_count": len(legacy_fallback_sample_ids),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vqa-dir",
        type=Path,
        required=True,
        help="Directory containing questions.jsonl and, preferably, samples.jsonl.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite questions.jsonl after removing fallback-composed questions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = filter_questions(vqa_dir=args.vqa_dir.resolve(), in_place=args.in_place)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
