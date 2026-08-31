#!/usr/bin/env python3
"""Build an OpenSearch-VL inference dataset from a VQA batch directory.

Example:
    python scripts/build_opensearch_dataset_from_vqa.py --vqa-dir /path/to/vqa_run
    python scripts/build_opensearch_dataset_from_vqa.py --vqa-dir /path/to/vqa_run --output /path/to/questions.parquet
    python scripts/build_opensearch_dataset_from_vqa.py --vqa-dir /path/to/vqa_run --offset 100 --limit 300 --seed 42
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any


def _normalize_image_reference(image_url: str) -> str:
    """Convert local absolute paths to ``file://`` URLs for API backends."""

    if image_url.startswith(("http://", "https://", "data:", "file://")):
        return image_url

    candidate = Path(image_url).expanduser()
    if candidate.is_absolute():
        return candidate.resolve().as_uri()
    return image_url


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"questions.jsonl line {line_number} is not a JSON object"
                )
            records.append(payload)
    return records


def _question_text(record: dict[str, Any]) -> str:
    return str(
        record.get("final_question")
        or record.get("question")
        or record.get("polished_question")
        or record.get("draft_question")
        or ""
    )


def _build_row(record: dict[str, Any]) -> dict[str, object]:
    question = _question_text(record)
    image_url = str(record.get("image_url") or "").strip()
    normalized_image_url = _normalize_image_reference(image_url) if image_url else ""

    row: dict[str, object] = {
        "data_id": record.get("sample_id") or record.get("question_id"),
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "category": "vqa_batch_question",
        "data_source": "synthesis.vqa.run_batch",
        "question": question,
        "prompt": [{"content": question}],
        "images": [{"url": normalized_image_url}] if normalized_image_url else [],
        "answer": record.get("answer", ""),
        "draft_question": record.get("draft_question"),
        "final_question": record.get("final_question"),
        "polished_question": record.get("polished_question"),
        "status": record.get("status"),
    }

    source_record = dict(record)
    if normalized_image_url:
        source_record["image_url"] = normalized_image_url
    row["source_metadata"] = json.dumps(source_record, ensure_ascii=False)
    return row


def _build_rows(records: list[dict[str, Any]]) -> list[dict[str, object]]:
    return [_build_row(record) for record in records]


def _select_records(
    records: list[dict[str, Any]],
    *,
    offset: int = 0,
    limit: int = 0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Skip ``offset`` records, then randomly sample up to ``limit`` records.

    ``offset`` is a zero-based position in the original ``questions.jsonl``
    order. A zero ``limit`` keeps all records after the offset. When a
    seed is supplied, the selected benchmark is reproducible.
    """

    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit < 0:
        raise ValueError("limit must be non-negative; use 0 for all remaining records")

    candidates = records[offset:]
    if limit == 0 or limit >= len(candidates):
        return candidates

    sampler = random.Random(seed)
    return sampler.sample(candidates, k=limit)


def write_parquet(rows: list[dict[str, object]], output_path: Path) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Writing parquet requires pandas plus a parquet backend such as "
            "pyarrow or fastparquet."
        ) from exc

    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write parquet. Make sure pyarrow or fastparquet is installed."
        ) from exc
    buffer.seek(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buffer.getvalue())


def write_json(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vqa-dir",
        required=True,
        help="Directory produced by synthesis.vqa.run_batch.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output dataset path. Defaults to <vqa-dir>/opensearch_questions.parquet.",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "json"],
        default="parquet",
        help="Dataset format to write.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Zero-based start position in questions.jsonl. Defaults to 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of records to randomly sample after offset. 0 keeps all remaining records.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible benchmark sampling.",
    )
    args = parser.parse_args()

    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.limit < 0:
        parser.error("--limit must be non-negative; use 0 for all remaining records")

    vqa_dir = Path(args.vqa_dir).expanduser().resolve()
    questions_path = vqa_dir / "questions.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"questions.jsonl does not exist: {questions_path}")

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        suffix = "json" if args.format == "json" else "parquet"
        output_path = vqa_dir / f"opensearch_questions.{suffix}"

    records = _load_jsonl(questions_path)
    selected_records = _select_records(
        records,
        offset=args.offset,
        limit=args.limit,
        seed=args.seed,
    )
    rows = _build_rows(selected_records)
    if args.format == "json":
        write_json(rows, output_path)
    else:
        write_parquet(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
