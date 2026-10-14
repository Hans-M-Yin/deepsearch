"""Print a compact, paste-friendly summary of one local fix2 finalization."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .io import iter_json_records


_FILES = {
    "raw": "teacher_raw.jsonl",
    "answer_reviews": "teacher_answer_reviews.jsonl",
    "quality_reviews": "teacher_trajectory_quality_reviews.jsonl",
    "blind_reviews": "teacher_blind_visual_reviews.jsonl",
    "accepted": "teacher_accepted_raw.jsonl",
    "rejected": "teacher_rejected_raw.jsonl",
    "audit": "teacher_filter_audit.jsonl",
}


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("question_id") or row.get("sample_id") or "").strip()


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_json_records(path)) if path.is_file() else []


def _map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {identifier: row for row in rows if (identifier := _case_id(row))}


def _answer_passes(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict) or row.get("error"):
        return False
    return row.get("is_correct") in {True, 1, "1", "true", "True", "yes"} or row.get("acc") in {True, 1, "1"}


def summarize(state_dir: str | Path) -> dict[str, Any]:
    root = Path(state_dir)
    datasets = {name: _rows(root / filename) for name, filename in _FILES.items()}
    answers = _map(datasets["answer_reviews"])
    quality = _map(datasets["quality_reviews"])
    blind = _map(datasets["blind_reviews"])
    audit = _map(datasets["audit"])
    case_ids = sorted(set(_map(datasets["raw"])) | set(audit))
    reason_counts = Counter(reason for row in audit.values() for reason in row.get("reasons") or [])
    output: dict[str, Any] = {
        "state_dir": str(root),
        "files_present": {name: (root / filename).is_file() for name, filename in _FILES.items()},
        "counts": {name: len(rows) for name, rows in datasets.items()},
        "answer_judge": {
            "correct": sum(_answer_passes(row) for row in answers.values()),
            "incorrect_or_error": sum(not _answer_passes(row) for row in answers.values()),
        },
        "trajectory_quality": {
            "by_status_and_decision": dict(
                sorted(
                    Counter(
                        f"{str(row.get('status') or 'missing')}/{str(row.get('decision') or 'missing')}"
                        for row in quality.values()
                    ).items()
                )
            ),
            "visual_verification_level": dict(
                sorted(Counter(str(row.get("visual_verification_level") or "missing") for row in quality.values()).items())
            ),
        },
        "filter": {
            "decision": dict(sorted(Counter(str(row.get("decision") or "") for row in audit.values()).items())),
            "rejection_reasons": dict(sorted(reason_counts.items())),
        },
        "cases": [],
    }
    for case_id in case_ids:
        answer = answers.get(case_id) or {}
        quality_row = quality.get(case_id) or {}
        blind_row = blind.get(case_id) or {}
        audit_row = audit.get(case_id) or {}
        output["cases"].append(
            {
                "case_id": case_id,
                "answer_correct": _answer_passes(answer),
                "answer_reason": str(answer.get("reasoning") or answer.get("error") or ""),
                "quality": {
                    "status": quality_row.get("status"),
                    "decision": quality_row.get("decision"),
                    "visual_verification_level": quality_row.get("visual_verification_level"),
                    "reason_codes": quality_row.get("reason_codes") or [],
                    "reason": quality_row.get("reason") or quality_row.get("error") or "",
                },
                "blind_visual": {
                    "decision": blind_row.get("decision"),
                    "verdict": blind_row.get("verdict"),
                    "reason": blind_row.get("reason") or blind_row.get("error") or "",
                },
                "filter": {
                    "decision": audit_row.get("decision"),
                    "reasons": audit_row.get("reasons") or [],
                },
            }
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.state_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
