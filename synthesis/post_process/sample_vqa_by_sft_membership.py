#!/usr/bin/env python3
"""Sample a VQA directory by membership in a final SFT dataset.

The SFT artifact used by this project is a large JSON array rather than a
JSONL file.  Membership is determined by unique ``question_id``.  The output
is a self-contained VQA directory containing the selected ``questions.jsonl``
and matching ``samples.jsonl`` records, plus a compact selection manifest.

Example:

    python synthesis/post_process/sample_vqa_by_sft_membership.py \
        --source-vqa-dir runs/.../vqa/0803_batch_1 \
        --sft-file data/sharegpt_dataset_refined_v3_rewritten_full_24k_complete/trajectories_sharegpt.json \
        --output-dir runs/.../vqa/0803_batch_1/rl_sft_overlap_2k_nonoverlap_2k \
        --count 2000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _iter_sft_records(value: Any) -> Iterator[dict[str, Any]]:
    """Yield trajectory objects from either a JSON array or nested arrays."""

    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_sft_records(item)


def _sft_question_id(record: dict[str, Any]) -> str | None:
    source_metadata = record.get("source_metadata")
    if not isinstance(source_metadata, dict):
        source_metadata = {}
    question_record = source_metadata.get("question_record")
    if not isinstance(question_record, dict):
        question_record = {}
    for value in (
        record.get("question_id"),
        source_metadata.get("question_id"),
        question_record.get("question_id"),
    ):
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _load_sft_question_ids(path: Path) -> tuple[set[str], dict[str, int]]:
    """Load unique SFT question IDs and report duplicate/missing records."""

    # The current final SFT artifact is a 1GB JSON array.  Loading it once is
    # dependency-free and keeps this utility usable in the existing image.
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    question_ids: set[str] = set()
    total = 0
    missing = 0
    duplicate = 0
    for record in _iter_sft_records(payload):
        total += 1
        question_id = _sft_question_id(record)
        if question_id is None:
            missing += 1
            continue
        if question_id in question_ids:
            duplicate += 1
        question_ids.add(question_id)
    return question_ids, {
        "records": total,
        "unique_question_ids": len(question_ids),
        "duplicate_question_id_records": duplicate,
        "missing_question_id_records": missing,
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-vqa-dir", type=Path, required=True)
    parser.add_argument("--sft-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000, help="Items per pool.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    source_dir = args.source_vqa_dir.expanduser().resolve()
    sft_file = args.sft_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    questions_path = source_dir / "questions.jsonl"
    samples_path = source_dir / "samples.jsonl"
    for path in (questions_path, samples_path, sft_file):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose another path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = _load_jsonl(questions_path)
    question_ids: list[str] = []
    questions_by_id: dict[str, dict[str, Any]] = {}
    duplicate_source_ids: list[str] = []
    for index, question in enumerate(questions):
        value = question.get("question_id")
        if value is None or not str(value).strip():
            raise ValueError(f"Missing question_id at questions.jsonl line {index + 1}")
        question_id = str(value).strip()
        question_ids.append(question_id)
        if question_id in questions_by_id:
            duplicate_source_ids.append(question_id)
        questions_by_id[question_id] = question

    sft_ids, sft_stats = _load_sft_question_ids(sft_file)
    source_id_set = set(question_ids)
    in_sft_indices = [index for index, qid in enumerate(question_ids) if qid in sft_ids]
    not_in_sft_indices = [index for index, qid in enumerate(question_ids) if qid not in sft_ids]
    if len(in_sft_indices) < args.count or len(not_in_sft_indices) < args.count:
        raise ValueError(
            "Insufficient records: "
            f"in_sft={len(in_sft_indices)}, not_in_sft={len(not_in_sft_indices)}, "
            f"requested_each={args.count}"
        )

    rng = random.Random(args.seed)
    selected_in_sft = set(rng.sample(in_sft_indices, args.count))
    selected_not_in_sft = set(rng.sample(not_in_sft_indices, args.count))
    selected_indices = sorted(selected_in_sft | selected_not_in_sft)
    selected_questions = [questions[index] for index in selected_indices]

    _write_jsonl(output_dir / "questions.jsonl", selected_questions)
    with (output_dir / "source_indices.txt").open("w", encoding="utf-8") as handle:
        for index in selected_indices:
            handle.write(f"{index}\n")

    selected_sample_ids = {
        str(question["sample_id"])
        for question in selected_questions
        if question.get("sample_id") is not None
    }
    if len(selected_sample_ids) != len(selected_questions):
        raise ValueError("Every selected question must have a unique sample_id")

    found_sample_ids: set[str] = set()
    selected_samples_path = output_dir / "samples.jsonl"
    with samples_path.open("r", encoding="utf-8") as source_handle, selected_samples_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(source_handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                raise ValueError(f"Expected JSON object at {samples_path}:{line_number}")
            sample_id = sample.get("sample_id")
            if sample_id is None or str(sample_id) not in selected_sample_ids:
                continue
            sample_id = str(sample_id)
            if sample_id in found_sample_ids:
                raise ValueError(f"Duplicate sample_id in source samples.jsonl: {sample_id}")
            output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            found_sample_ids.add(sample_id)

    missing_sample_ids = sorted(selected_sample_ids - found_sample_ids)
    if missing_sample_ids:
        raise ValueError(
            f"Missing {len(missing_sample_ids)} selected samples; first IDs: "
            f"{missing_sample_ids[:5]}"
        )

    manifest_records: list[dict[str, Any]] = []
    for index in selected_indices:
        question = questions[index]
        question_id = str(question["question_id"])
        pool = "in_sft" if index in selected_in_sft else "not_in_sft"
        manifest_records.append(
            {
                "source_index": index,
                "question_id": question_id,
                "sample_id": question.get("sample_id"),
                "path_id": question.get("path_id"),
                "pool": pool,
            }
        )
    _write_jsonl(output_dir / "selection_manifest.jsonl", manifest_records)

    report = {
        "source_vqa_dir": str(source_dir),
        "sft_file": str(sft_file),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "requested_per_pool": args.count,
        "source_question_count": len(questions),
        "source_unique_question_id_count": len(source_id_set),
        "source_duplicate_question_id_count": len(duplicate_source_ids),
        "source_in_sft_count": len(in_sft_indices),
        "source_not_in_sft_count": len(not_in_sft_indices),
        "selected_in_sft_count": len(selected_in_sft),
        "selected_not_in_sft_count": len(selected_not_in_sft),
        "selected_total": len(selected_indices),
        "selected_sample_count": len(found_sample_ids),
        "sft_membership": sft_stats,
        "files": {
            "questions": str(output_dir / "questions.jsonl"),
            "samples": str(output_dir / "samples.jsonl"),
            "manifest": str(output_dir / "selection_manifest.jsonl"),
            "source_indices": str(output_dir / "source_indices.txt"),
        },
    }
    with (output_dir / "selection_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
